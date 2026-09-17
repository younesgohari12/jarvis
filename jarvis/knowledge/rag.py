from __future__ import annotations

import hashlib
import json
import math
import sqlite3
import threading
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from jarvis.tools.files import FileInspection, FileManager
from jarvis.utils.text import cosine_sparse, hashed_features, normalize_text, tokenize


@dataclass(frozen=True, slots=True)
class RAGHit:
    document: str
    path: str
    chunk: str
    score: float
    relevance: float = 0.0
    confidence: float = 0.0
    recency: float = 0.0
    source: str = "local_document"
    bm25_score: float = 0.0
    vector_score: float = 0.0
    rerank_score: float = 0.0


class RAGStore:
    """Local hybrid RAG: keyword + BM25 + project-owned hashed vectors + rerank."""

    VECTOR_SIZE = 512

    def __init__(self, database: Path, max_documents: int = 500) -> None:
        database.parent.mkdir(parents=True, exist_ok=True)
        self.connection = sqlite3.connect(database, check_same_thread=False)
        self.connection.row_factory = sqlite3.Row
        self.connection.execute("PRAGMA foreign_keys=ON")
        self.lock = threading.RLock()
        self.max_documents = max(10, max_documents)
        with self.connection:
            self.connection.executescript(
                """
                PRAGMA journal_mode=WAL;
                PRAGMA synchronous=NORMAL;
                CREATE TABLE IF NOT EXISTS rag_documents (
                    id INTEGER PRIMARY KEY,
                    path TEXT NOT NULL UNIQUE,
                    name TEXT NOT NULL,
                    sha256 TEXT NOT NULL,
                    indexed_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
                );
                CREATE TABLE IF NOT EXISTS rag_chunks (
                    id INTEGER PRIMARY KEY,
                    document_id INTEGER NOT NULL REFERENCES rag_documents(id) ON DELETE CASCADE,
                    position INTEGER NOT NULL,
                    text TEXT NOT NULL,
                    tokens TEXT NOT NULL,
                    token_count INTEGER NOT NULL DEFAULT 0,
                    vector TEXT NOT NULL DEFAULT '{}'
                );
                CREATE INDEX IF NOT EXISTS idx_rag_chunks_document ON rag_chunks(document_id);
                """
            )
            columns = {
                str(row[1]) for row in self.connection.execute("PRAGMA table_info(rag_chunks)")
            }
            if "token_count" not in columns:
                self.connection.execute(
                    "ALTER TABLE rag_chunks ADD COLUMN token_count INTEGER NOT NULL DEFAULT 0"
                )
            if "vector" not in columns:
                self.connection.execute(
                    "ALTER TABLE rag_chunks ADD COLUMN vector TEXT NOT NULL DEFAULT '{}'"
                )

    @staticmethod
    def _chunks(text: str, size: int = 1100, overlap: int = 140) -> list[str]:
        clean = "\n".join(line.rstrip() for line in text.splitlines()).strip()
        if not clean:
            return []
        result: list[str] = []
        start = 0
        while start < len(clean) and len(result) < 400:
            end = min(len(clean), start + size)
            if end < len(clean):
                boundary = max(clean.rfind("\n", start, end), clean.rfind(". ", start, end))
                if boundary > start + size // 2:
                    end = boundary + 1
            result.append(clean[start:end].strip())
            if end >= len(clean):
                break
            start = max(start + 1, end - overlap)
        return [value for value in result if value]

    def index_text(self, path: Path, text: str) -> dict[str, object]:
        resolved = path.expanduser().resolve()
        chunks = self._chunks(text)
        if not chunks:
            raise ValueError("Document has no indexable text")
        digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
        with self.lock, self.connection:
            existing = self.connection.execute(
                "SELECT id, sha256 FROM rag_documents WHERE path = ?", (str(resolved),)
            ).fetchone()
            if existing and existing["sha256"] == digest:
                return {"indexed": False, "unchanged": True, "chunks": 0, "path": str(resolved)}
            if existing:
                document_id = int(existing["id"])
                self.connection.execute("DELETE FROM rag_chunks WHERE document_id = ?", (document_id,))
                self.connection.execute(
                    "UPDATE rag_documents SET name = ?, sha256 = ?, indexed_at = CURRENT_TIMESTAMP WHERE id = ?",
                    (resolved.name, digest, document_id),
                )
            else:
                cursor = self.connection.execute(
                    "INSERT INTO rag_documents(path, name, sha256) VALUES (?, ?, ?)",
                    (str(resolved), resolved.name, digest),
                )
                document_id = int(cursor.lastrowid)
            self.connection.executemany(
                """
                INSERT INTO rag_chunks(
                    document_id, position, text, tokens, token_count, vector
                ) VALUES (?, ?, ?, ?, ?, ?)
                """,
                [
                    (
                        document_id,
                        position,
                        chunk,
                        json.dumps(tokenize(chunk), ensure_ascii=False),
                        len(tokenize(chunk)),
                        json.dumps(
                            {str(key): value for key, value in hashed_features(chunk, self.VECTOR_SIZE).items()},
                            separators=(",", ":"),
                        ),
                    )
                    for position, chunk in enumerate(chunks)
                ],
            )
            overflow = self.connection.execute(
                "SELECT id FROM rag_documents ORDER BY indexed_at DESC LIMIT -1 OFFSET ?",
                (self.max_documents,),
            ).fetchall()
            if overflow:
                self.connection.executemany(
                    "DELETE FROM rag_documents WHERE id = ?",
                    [(int(row["id"]),) for row in overflow],
                )
        return {"indexed": True, "unchanged": False, "chunks": len(chunks), "path": str(resolved)}

    def index_inspection(self, inspection: FileInspection) -> dict[str, object]:
        if inspection.kind == "text":
            return self.index_text(inspection.path, inspection.preview)
        text = "\n\n".join(
            f"{entry.name}\n{entry.text_preview or ''}"
            for entry in inspection.zip_entries
        )
        return self.index_text(inspection.path, text)

    def index_file(self, files: FileManager, path: str | Path) -> dict[str, object]:
        return self.index_inspection(files.inspect(path))

    def search(self, query: str, limit: int = 5) -> tuple[RAGHit, ...]:
        from jarvis.knowledge.retrieval_v20 import self_contained, expand_query, content_signature
        if self_contained(query):
            return ()
        query=expand_query(query)
        query_sequence = tokenize(query)
        query_tokens = set(query_sequence)
        if not query_tokens:
            return ()
        with self.lock:
            rows = self.connection.execute(
                """
                SELECT d.name, d.path, d.indexed_at, c.text, c.tokens,
                       c.token_count, c.vector
                FROM rag_chunks c JOIN rag_documents d ON d.id = c.document_id
                ORDER BY d.indexed_at DESC LIMIT 5000
                """
            ).fetchall()
        if not rows:
            return ()
        document_frequency = {token: 0 for token in query_tokens}
        parsed: list[tuple[sqlite3.Row, list[str], set[str]]] = []
        for row in rows:
            try:
                raw_tokens = json.loads(row["tokens"])
            except (json.JSONDecodeError, TypeError):
                raw_tokens = []
            sequence = [str(token) for token in raw_tokens] if isinstance(raw_tokens, list) else []
            if not sequence:
                sequence = tokenize(str(row["text"]))
            token_set = set(sequence)
            parsed.append((row, sequence, token_set))
            for token in query_tokens & token_set:
                document_frequency[token] += 1
        average_length = sum(len(sequence) for _row, sequence, _set in parsed) / max(1, len(parsed))
        query_vector = hashed_features(query, self.VECTOR_SIZE)
        normalized_query = normalize_text(query)
        raw: list[tuple[sqlite3.Row, float, float, float, float]] = []
        maximum_bm25 = 0.0
        now = datetime.now(UTC)
        for row, sequence, token_set in parsed:
            if not (query_tokens & token_set):
                vector_payload = str(row["vector"] or "{}")
                try:
                    candidate_vector = {
                        int(key): float(value) for key, value in json.loads(vector_payload).items()
                    }
                except (json.JSONDecodeError, TypeError, ValueError, AttributeError):
                    candidate_vector = hashed_features(str(row["text"]), self.VECTOR_SIZE)
                vector_score = max(0.0, cosine_sparse(query_vector, candidate_vector))
                if vector_score < 0.16:
                    continue
            else:
                try:
                    candidate_vector = {
                        int(key): float(value) for key, value in json.loads(str(row["vector"] or "{}")).items()
                    }
                except (json.JSONDecodeError, TypeError, ValueError, AttributeError):
                    candidate_vector = hashed_features(str(row["text"]), self.VECTOR_SIZE)
                if not candidate_vector:
                    candidate_vector = hashed_features(str(row["text"]), self.VECTOR_SIZE)
                vector_score = max(0.0, cosine_sparse(query_vector, candidate_vector))

            counts: dict[str, int] = {}
            for token in sequence:
                counts[token] = counts.get(token, 0) + 1
            bm25 = 0.0
            length = max(1, len(sequence))
            for token in query_sequence:
                frequency = counts.get(token, 0)
                if not frequency:
                    continue
                inverse = math.log(
                    1.0 + (len(parsed) - document_frequency.get(token, 0) + 0.5)
                    / (document_frequency.get(token, 0) + 0.5)
                )
                denominator = frequency + 1.2 * (
                    1.0 - 0.75 + 0.75 * length / max(1.0, average_length)
                )
                bm25 += inverse * frequency * 2.2 / denominator
            maximum_bm25 = max(maximum_bm25, bm25)
            normalized_text = normalize_text(str(row["text"]))
            overlap = len(query_tokens & token_set) / max(1, len(query_tokens))
            phrase = 1.0 if normalized_query and normalized_query in normalized_text else 0.0
            keyword = min(1.0, overlap * 0.8 + phrase * 0.2)
            try:
                indexed = datetime.fromisoformat(str(row["indexed_at"]).replace("Z", "+00:00"))
                if indexed.tzinfo is None:
                    indexed = indexed.replace(tzinfo=UTC)
                days = max(0.0, (now - indexed).total_seconds() / 86_400.0)
                recency = math.exp(-days / 365.0)
            except ValueError:
                recency = 0.5
            raw.append((row, bm25, vector_score, keyword, recency))

        hits: list[RAGHit] = []
        for row, bm25, vector_score, keyword, recency in raw:
            bm25_normalized = bm25 / maximum_bm25 if maximum_bm25 else 0.0
            base = (
                0.47 * bm25_normalized + 0.28 * vector_score
                + 0.17 * keyword + 0.08 * recency
            )
            length_quality = min(1.0, len(str(row["text"])) / 240.0)
            rerank = min(1.0, base * 0.9 + length_quality * 0.1)
            confidence = max(0.0, min(0.99, 0.18 + rerank * 0.8))
            hits.append(
                RAGHit(
                    str(row["name"]), str(row["path"]), str(row["text"]),
                    rerank, relevance=base, confidence=confidence, recency=recency,
                    bm25_score=bm25_normalized, vector_score=vector_score,
                    rerank_score=rerank,
                )
            )
        # Absolute evidence support matters: normalized BM25 alone makes the best
        # irrelevant chunk look strong when every candidate is poor.
        from jarvis.agent.deliberation import content_terms
        meaningful=set(content_terms(query))
        selected=[]; seen=set()
        for hit in sorted(hits,key=lambda value:value.score,reverse=True):
            sig=content_signature(hit.chunk)
            support=meaningful & set(content_terms(hit.chunk))
            if sig in seen or (meaningful and not support): continue
            if hit.relevance<.20: continue
            seen.add(sig); selected.append(hit)
            if len(selected)>=max(1,min(10,limit)): break
        return tuple(selected)

    def stats(self) -> dict[str, int]:
        with self.lock:
            documents = int(self.connection.execute("SELECT COUNT(*) FROM rag_documents").fetchone()[0])
            chunks = int(self.connection.execute("SELECT COUNT(*) FROM rag_chunks").fetchone()[0])
        return {"documents": documents, "chunks": chunks}

    def close(self) -> None:
        with self.lock:
            self.connection.close()
