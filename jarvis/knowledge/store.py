from __future__ import annotations

import json
import math
import sqlite3
import threading
from collections import Counter
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from jarvis.utils.text import normalize_text, tokenize


class KnowledgeError(RuntimeError):
    pass


@dataclass(frozen=True, slots=True)
class KnowledgeHit:
    document_id: int
    title: str
    content: str
    source: str
    score: float


class KnowledgeStore:
    """Bounded SQLite documents with an in-process TF-IDF ranker and optional FTS5."""

    def __init__(
        self,
        path: Path,
        seed_path: Path,
        seed_version: int,
        max_documents: int = 500,
    ) -> None:
        self.path = path
        self.max_documents = max(20, int(max_documents))
        self._lock = threading.RLock()
        self._closed = False
        path.parent.mkdir(parents=True, exist_ok=True)
        try:
            self._connection = sqlite3.connect(path, timeout=3.0, check_same_thread=False)
            self._connection.row_factory = sqlite3.Row
            self._connection.execute("PRAGMA journal_mode=WAL")
            self._connection.execute("PRAGMA synchronous=NORMAL")
            self._migrate()
            self._seed(seed_path, seed_version)
        except sqlite3.Error as exc:
            raise KnowledgeError(f"Cannot initialize knowledge store: {exc}") from exc

    @staticmethod
    def _now() -> str:
        return datetime.now(UTC).isoformat(timespec="seconds")

    def _migrate(self) -> None:
        with self._connection:
            self._connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS metadata (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS documents (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    external_id TEXT UNIQUE,
                    title TEXT NOT NULL,
                    content TEXT NOT NULL,
                    keywords TEXT NOT NULL DEFAULT '',
                    source TEXT NOT NULL DEFAULT 'user',
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_documents_source ON documents(source);
                """
            )
            try:
                self._connection.execute(
                    "CREATE VIRTUAL TABLE IF NOT EXISTS documents_fts USING fts5(title, content, keywords, tokenize='unicode61')"
                )
                self._fts_available = True
            except sqlite3.OperationalError:
                self._fts_available = False

    def _seed(self, path: Path, version: int) -> None:
        current = self._connection.execute(
            "SELECT value FROM metadata WHERE key = 'seed_version'"
        ).fetchone()
        if current and str(current["value"]) == str(version):
            return
        try:
            with path.open("r", encoding="utf-8") as handle:
                payload = json.load(handle)
        except (OSError, json.JSONDecodeError) as exc:
            raise KnowledgeError(f"Cannot load knowledge seed: {exc}") from exc
        entries = payload.get("entries", []) if isinstance(payload, dict) else []
        with self._lock, self._connection:
            for item in entries:
                if not isinstance(item, dict):
                    continue
                answers = item.get("answers", {})
                content = "\n".join(
                    f"[{language}] {text}" for language, text in answers.items() if str(text).strip()
                )
                self._upsert_locked(
                    external_id=f"builtin:{item.get('id', item.get('title', 'item'))}",
                    title=str(item.get("title", "Jarvis knowledge")),
                    content=content,
                    keywords=" ".join(str(value) for value in item.get("keywords", [])),
                    source="builtin",
                )
            self._connection.execute(
                "INSERT OR REPLACE INTO metadata(key, value) VALUES ('seed_version', ?)",
                (str(version),),
            )
            self._rebuild_fts_locked()
            self._enforce_limit_locked()

    def _upsert_locked(
        self, external_id: str | None, title: str, content: str, keywords: str, source: str
    ) -> int:
        now = self._now()
        self._connection.execute(
            """
            INSERT INTO documents(external_id, title, content, keywords, source, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(external_id) DO UPDATE SET title=excluded.title, content=excluded.content,
                keywords=excluded.keywords, source=excluded.source, updated_at=excluded.updated_at
            """,
            (external_id, title, content, keywords, source, now, now),
        )
        row = self._connection.execute(
            "SELECT id FROM documents WHERE external_id IS ?", (external_id,)
        ).fetchone()
        return int(row["id"]) if row else 0

    def _rebuild_fts_locked(self) -> None:
        if not self._fts_available:
            return
        self._connection.execute("DELETE FROM documents_fts")
        self._connection.execute(
            "INSERT INTO documents_fts(rowid, title, content, keywords) SELECT id, title, content, keywords FROM documents"
        )

    def _enforce_limit_locked(self) -> None:
        self._connection.execute(
            """
            DELETE FROM documents WHERE id IN (
                SELECT id FROM documents ORDER BY CASE WHEN source='builtin' THEN 1 ELSE 0 END,
                    updated_at DESC LIMIT -1 OFFSET ?
            )
            """,
            (self.max_documents,),
        )

    def add_document(
        self,
        title: str,
        content: str,
        keywords: list[str] | None = None,
        source: str = "user",
        external_id: str | None = None,
    ) -> int:
        clean_title = str(title).strip()[:200]
        clean_content = str(content).strip()[:250_000]
        if not clean_title or not clean_content:
            raise KnowledgeError("Knowledge title and content are required")
        with self._lock, self._connection:
            document_id = self._upsert_locked(
                external_id,
                clean_title,
                clean_content,
                " ".join(keywords or []),
                source[:40],
            )
            self._enforce_limit_locked()
            self._rebuild_fts_locked()
            return document_id

    def _candidate_rows(self, query_tokens: list[str]) -> list[sqlite3.Row]:
        if self._fts_available and query_tokens:
            safe_tokens = [token.replace('"', "") for token in query_tokens if len(token) > 1]
            if safe_tokens:
                expression = " OR ".join(f'"{token}"' for token in safe_tokens[:12])
                try:
                    rows = self._connection.execute(
                        """
                        SELECT d.* FROM documents_fts f
                        JOIN documents d ON d.id=f.rowid
                        WHERE documents_fts MATCH ?
                        ORDER BY bm25(documents_fts, 3.0, 1.0, 2.2)
                        LIMIT 120
                        """,
                        (expression,),
                    ).fetchall()
                    if rows:
                        return rows
                except sqlite3.OperationalError:
                    pass
        return self._connection.execute(
            "SELECT * FROM documents ORDER BY updated_at DESC LIMIT 120"
        ).fetchall()

    def search(self, query: str, limit: int = 3) -> list[KnowledgeHit]:
        punctuation = "؟?!.،,:;؛()[]{}\"“”'"
        query_tokens = [
            clean for token in tokenize(query)
            if (clean := token.strip(punctuation))
        ]
        if not query_tokens:
            return []
        with self._lock:
            rows = self._candidate_rows(query_tokens)
        if not rows:
            return []

        document_tokens: list[list[str]] = []
        for row in rows:
            weighted = (
                f"{row['title']} {row['title']} {row['keywords']} {row['keywords']} "
                f"{row['content']}"
            )
            document_tokens.append([
                clean for token in tokenize(weighted)
                if (clean := token.strip(punctuation))
            ])
        document_frequency: Counter[str] = Counter()
        for tokens in document_tokens:
            document_frequency.update(set(tokens))
        count = len(rows)
        query_counts = Counter(query_tokens)
        query_weights = {
            token: (1.0 + math.log(frequency))
            * (math.log((count + 1) / (document_frequency.get(token, 0) + 1)) + 1.0)
            for token, frequency in query_counts.items()
        }
        query_norm = math.sqrt(sum(value * value for value in query_weights.values())) or 1.0

        hits: list[KnowledgeHit] = []
        normalized_query = normalize_text(query)
        for row, tokens in zip(rows, document_tokens):
            counts = Counter(tokens)
            dot = 0.0
            doc_norm_sq = 0.0
            for token, frequency in counts.items():
                weight = (1.0 + math.log(frequency)) * (
                    math.log((count + 1) / (document_frequency.get(token, 0) + 1)) + 1.0
                )
                doc_norm_sq += weight * weight
                dot += weight * query_weights.get(token, 0.0)
            cosine = dot / (query_norm * (math.sqrt(doc_norm_sq) or 1.0))
            normalized_content = normalize_text(str(row["content"]))
            normalized_metadata = normalize_text(f"{row['title']} {row['keywords']}")
            phrase_bonus = 0.12 if normalized_query in normalized_content else 0.0
            metadata_bonus = 0.25 if normalized_query in normalized_metadata else 0.0
            question_words = {
                "چیست", "چیه", "یعنی", "را", "رو", "لطفا", "what", "is", "a",
                "an", "the", "does", "do", "mean", "explain", "please", "how",
                "work", "works", "working", "can", "چیکار", "کار", "میکنه",
                "می‌کنه", "میکند", "می‌کند", "می کند", "میشه", "می‌شود",
                "میشود", "چطور", "چگونه", "ساده",
                "توضیح", "بده", "بگو", "چه", "فرقی", "فرق", "تفاوت",
                "حتما", "حتماً", "necessarily", "always", "really",
                "کاری", "کار", "انجام", "میدهد", "می‌دهد", "میده", "می‌دهد؟",
                "می", "دهد", "ده", "کنه", "کند",
            }
            distinctive = [token for token in query_tokens if token not in question_words]
            metadata_tokens = set(tokenize(normalized_metadata))
            coverage = (
                sum(token in set(tokens) for token in distinctive) / len(distinctive)
                if distinctive else 0.0
            )
            coverage_bonus = 0.18 if coverage == 1.0 else 0.08 * coverage
            # Short definition questions (for example ``RAM چیست؟``) contain
            # very little lexical material.  Give an explicit, bounded boost
            # when every meaningful query token is present in the curated
            # title/keywords instead of forcing the request onto the network.
            distinctive_metadata_bonus = (
                0.18
                if distinctive and all(token in metadata_tokens for token in distinctive)
                else 0.0
            )
            score = min(
                1.0,
                cosine * 1.8 + phrase_bonus + metadata_bonus + coverage_bonus
                + distinctive_metadata_bonus,
            )
            if score > 0:
                hits.append(
                    KnowledgeHit(
                        int(row["id"]), str(row["title"]), str(row["content"]),
                        str(row["source"]), score
                    )
                )
        hits.sort(key=lambda hit: hit.score, reverse=True)
        return hits[: max(1, int(limit))]

    def count(self) -> int:
        with self._lock:
            return int(self._connection.execute("SELECT COUNT(*) FROM documents").fetchone()[0])

    def integrity_check(self) -> bool:
        with self._lock:
            row = self._connection.execute("PRAGMA quick_check").fetchone()
            return bool(row and str(row[0]).casefold() == "ok")

    def close(self) -> None:
        with self._lock:
            if not self._closed:
                self._connection.close()
                self._closed = True
