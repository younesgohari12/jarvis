from __future__ import annotations

import json
import sqlite3
import threading
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any


class MemoryStoreError(RuntimeError):
    pass


@dataclass(frozen=True, slots=True)
class Message:
    id: int
    session_id: int
    role: str
    content: str
    created_at: str
    metadata: dict[str, Any]


@dataclass(frozen=True, slots=True)
class AttachmentRecord:
    id: int
    session_id: int
    name: str
    path: str
    kind: str
    size: int
    created_at: str


@dataclass(frozen=True, slots=True)
class CorrectionRule:
    id: int
    trigger: str
    trigger_normalized: str
    intent: str
    arguments: dict[str, Any]
    source: str
    use_count: int


@dataclass(frozen=True, slots=True)
class ActionRecord:
    id: int
    session_id: int
    tool: str
    arguments: dict[str, Any]
    inverse_tool: str
    inverse_arguments: dict[str, Any]
    created_at: str
    undone_at: str = ""


@dataclass(frozen=True, slots=True)
class EpisodicMemory:
    id: int
    session_id: int | None
    summary: str
    salience: float
    created_at: str


class MemoryStore:
    """Bounded session and long-term memory over one small SQLite database."""

    SCHEMA_VERSION = 4

    def __init__(
        self,
        path: Path,
        max_history_rows: int = 5000,
        max_sessions: int = 100,
        retention_days: int = 90,
    ) -> None:
        self.path = path
        self.max_history_rows = max(100, int(max_history_rows))
        self.max_sessions = max(5, int(max_sessions))
        self.retention_days = max(7, int(retention_days))
        self._lock = threading.RLock()
        self._closed = False
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            self._connection = sqlite3.connect(
                path,
                timeout=3.0,
                check_same_thread=False,
                isolation_level=None,
            )
            self._connection.row_factory = sqlite3.Row
            self._configure()
            self._migrate()
        except sqlite3.Error as exc:
            raise MemoryStoreError(f"Cannot initialize SQLite memory: {exc}") from exc

    @staticmethod
    def _now() -> str:
        return datetime.now(UTC).isoformat(timespec="seconds")

    def _ensure_open(self) -> None:
        if self._closed:
            raise MemoryStoreError("MemoryStore is closed")

    def _configure(self) -> None:
        self._connection.execute("PRAGMA foreign_keys = ON")
        self._connection.execute("PRAGMA busy_timeout = 3000")
        self._connection.execute("PRAGMA synchronous = NORMAL")
        try:
            self._connection.execute("PRAGMA journal_mode = WAL")
        except sqlite3.DatabaseError:
            self._connection.execute("PRAGMA journal_mode = DELETE")

    def _migrate(self) -> None:
        with self._lock, self._connection:
            version = int(self._connection.execute("PRAGMA user_version").fetchone()[0])
            if version > self.SCHEMA_VERSION:
                raise MemoryStoreError(
                    f"Database schema {version} is newer than supported {self.SCHEMA_VERSION}"
                )
            self._connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS sessions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    started_at TEXT NOT NULL,
                    last_active TEXT NOT NULL,
                    closed_at TEXT
                );
                CREATE TABLE IF NOT EXISTS messages (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id INTEGER NOT NULL,
                    role TEXT NOT NULL CHECK(role IN ('user', 'assistant', 'system')),
                    content TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    metadata TEXT NOT NULL DEFAULT '{}',
                    FOREIGN KEY(session_id) REFERENCES sessions(id) ON DELETE CASCADE
                );
                CREATE INDEX IF NOT EXISTS idx_messages_session_id
                    ON messages(session_id, id);
                CREATE TABLE IF NOT EXISTS settings (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS facts (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL,
                    confidence REAL NOT NULL DEFAULT 1.0,
                    updated_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS session_context (
                    session_id INTEGER PRIMARY KEY,
                    topic TEXT NOT NULL DEFAULT '',
                    state_json TEXT NOT NULL DEFAULT '{}',
                    updated_at TEXT NOT NULL,
                    FOREIGN KEY(session_id) REFERENCES sessions(id) ON DELETE CASCADE
                );
                CREATE TABLE IF NOT EXISTS attachments (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id INTEGER NOT NULL,
                    name TEXT NOT NULL,
                    path TEXT NOT NULL,
                    kind TEXT NOT NULL,
                    size INTEGER NOT NULL,
                    created_at TEXT NOT NULL,
                    FOREIGN KEY(session_id) REFERENCES sessions(id) ON DELETE CASCADE
                );
                CREATE INDEX IF NOT EXISTS idx_attachments_session
                    ON attachments(session_id, id);
                CREATE TABLE IF NOT EXISTS corrections (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    trigger TEXT NOT NULL,
                    trigger_normalized TEXT NOT NULL UNIQUE,
                    intent TEXT NOT NULL,
                    arguments_json TEXT NOT NULL DEFAULT '{}',
                    source TEXT NOT NULL DEFAULT 'correction',
                    use_count INTEGER NOT NULL DEFAULT 0,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_corrections_trigger
                    ON corrections(trigger_normalized);
                CREATE TABLE IF NOT EXISTS feedback (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id INTEGER,
                    message_id INTEGER,
                    rating INTEGER NOT NULL CHECK(rating IN (-1, 1)),
                    note TEXT NOT NULL DEFAULT '',
                    created_at TEXT NOT NULL,
                    FOREIGN KEY(session_id) REFERENCES sessions(id) ON DELETE SET NULL,
                    FOREIGN KEY(message_id) REFERENCES messages(id) ON DELETE SET NULL
                );
                CREATE INDEX IF NOT EXISTS idx_feedback_created ON feedback(id);
                CREATE TABLE IF NOT EXISTS decision_log (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id INTEGER,
                    input_excerpt TEXT NOT NULL,
                    intent TEXT NOT NULL,
                    entity TEXT NOT NULL DEFAULT '',
                    confidence REAL NOT NULL,
                    tool TEXT NOT NULL DEFAULT '',
                    confirmation INTEGER NOT NULL DEFAULT 0,
                    source TEXT NOT NULL DEFAULT '',
                    created_at TEXT NOT NULL,
                    FOREIGN KEY(session_id) REFERENCES sessions(id) ON DELETE SET NULL
                );
                CREATE TABLE IF NOT EXISTS episodic_memory (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id INTEGER,
                    summary TEXT NOT NULL,
                    normalized_key TEXT NOT NULL,
                    salience REAL NOT NULL DEFAULT 0.5,
                    created_at TEXT NOT NULL,
                    last_accessed TEXT NOT NULL,
                    access_count INTEGER NOT NULL DEFAULT 0,
                    UNIQUE(session_id, normalized_key),
                    FOREIGN KEY(session_id) REFERENCES sessions(id) ON DELETE SET NULL
                );
                CREATE INDEX IF NOT EXISTS idx_episodic_salience
                    ON episodic_memory(salience DESC, last_accessed DESC);
                CREATE TABLE IF NOT EXISTS semantic_memory (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL,
                    normalized_value TEXT NOT NULL,
                    confidence REAL NOT NULL DEFAULT 1.0,
                    source TEXT NOT NULL DEFAULT 'conversation',
                    conflict_count INTEGER NOT NULL DEFAULT 0,
                    updated_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS action_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id INTEGER NOT NULL,
                    tool TEXT NOT NULL,
                    arguments_json TEXT NOT NULL,
                    inverse_tool TEXT NOT NULL,
                    inverse_arguments_json TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    undone_at TEXT NOT NULL DEFAULT '',
                    FOREIGN KEY(session_id) REFERENCES sessions(id) ON DELETE CASCADE
                );
                CREATE INDEX IF NOT EXISTS idx_action_history_pending
                    ON action_history(session_id, undone_at, id DESC);
                """
            )
            self._connection.execute(f"PRAGMA user_version = {self.SCHEMA_VERSION}")

    def get_setting(self, key: str, default: str | None = None) -> str | None:
        with self._lock:
            self._ensure_open()
            row = self._connection.execute(
                "SELECT value FROM settings WHERE key = ?", (key,)
            ).fetchone()
            return str(row["value"]) if row else default

    def get_bool_setting(self, key: str, default: bool = False) -> bool:
        value = self.get_setting(key)
        if value is None:
            return default
        return value.casefold() in {"1", "true", "yes", "on"}

    def set_setting(self, key: str, value: str | bool | int | float) -> None:
        encoded = str(value).lower() if isinstance(value, bool) else str(value)
        with self._lock, self._connection:
            self._ensure_open()
            self._connection.execute(
                """
                INSERT INTO settings(key, value, updated_at) VALUES (?, ?, ?)
                ON CONFLICT(key) DO UPDATE SET value = excluded.value,
                                                   updated_at = excluded.updated_at
                """,
                (key, encoded, self._now()),
            )

    def start_session(self, resume_hours: int = 24) -> tuple[int, bool]:
        with self._lock, self._connection:
            self._ensure_open()
            previous = self.get_setting("last_session_id")
            if previous and previous.isdigit():
                row = self._connection.execute(
                    "SELECT id, last_active FROM sessions WHERE id = ?", (int(previous),)
                ).fetchone()
                if row:
                    try:
                        last_active = datetime.fromisoformat(str(row["last_active"]))
                        if last_active >= datetime.now(UTC) - timedelta(hours=max(1, resume_hours)):
                            now = self._now()
                            self._connection.execute(
                                "UPDATE sessions SET last_active = ?, closed_at = NULL WHERE id = ?",
                                (now, int(row["id"])),
                            )
                            return int(row["id"]), True
                    except ValueError:
                        pass
            return self._create_session_locked(), False

    def _create_session_locked(self) -> int:
        now = self._now()
        cursor = self._connection.execute(
            "INSERT INTO sessions(started_at, last_active) VALUES (?, ?)", (now, now)
        )
        session_id = int(cursor.lastrowid)
        self.set_setting("last_session_id", session_id)
        self._connection.execute(
            "INSERT OR REPLACE INTO session_context(session_id, topic, state_json, updated_at) VALUES (?, '', '{}', ?)",
            (session_id, now),
        )
        return session_id

    def create_new_session(self) -> int:
        with self._lock, self._connection:
            self._ensure_open()
            return self._create_session_locked()

    def add_message(
        self,
        session_id: int,
        role: str,
        content: str,
        metadata: dict[str, Any] | None = None,
    ) -> int:
        if role not in {"user", "assistant", "system"}:
            raise ValueError(f"Unsupported message role: {role}")
        now = self._now()
        encoded = json.dumps(metadata or {}, ensure_ascii=False, separators=(",", ":"))
        with self._lock, self._connection:
            self._ensure_open()
            cursor = self._connection.execute(
                "INSERT INTO messages(session_id, role, content, created_at, metadata) VALUES (?, ?, ?, ?, ?)",
                (session_id, role, str(content), now, encoded),
            )
            self._connection.execute(
                "UPDATE sessions SET last_active = ? WHERE id = ?", (now, session_id)
            )
            message_id = int(cursor.lastrowid)
            if message_id % 100 == 0:
                self._trim_messages_locked()
            return message_id

    def _trim_messages_locked(self) -> None:
        self._connection.execute(
            "DELETE FROM messages WHERE id IN (SELECT id FROM messages ORDER BY id DESC LIMIT -1 OFFSET ?)",
            (self.max_history_rows,),
        )

    def recent_messages(self, session_id: int, limit: int = 20) -> list[Message]:
        with self._lock:
            self._ensure_open()
            rows = self._connection.execute(
                """
                SELECT id, session_id, role, content, created_at, metadata
                FROM messages WHERE session_id = ? ORDER BY id DESC LIMIT ?
                """,
                (session_id, max(1, int(limit))),
            ).fetchall()
        result: list[Message] = []
        for row in reversed(rows):
            try:
                metadata = json.loads(str(row["metadata"]))
            except json.JSONDecodeError:
                metadata = {}
            result.append(
                Message(
                    int(row["id"]),
                    int(row["session_id"]),
                    str(row["role"]),
                    str(row["content"]),
                    str(row["created_at"]),
                    metadata if isinstance(metadata, dict) else {},
                )
            )
        return result

    def turn_count(self, session_id: int) -> int:
        with self._lock:
            row = self._connection.execute(
                "SELECT COUNT(*) AS count FROM messages WHERE session_id = ? AND role = 'user'",
                (session_id,),
            ).fetchone()
            return int(row["count"])

    def set_fact(self, key: str, value: str, confidence: float = 1.0) -> None:
        with self._lock, self._connection:
            self._ensure_open()
            self._connection.execute(
                """
                INSERT INTO facts(key, value, confidence, updated_at) VALUES (?, ?, ?, ?)
                ON CONFLICT(key) DO UPDATE SET value = excluded.value,
                                               confidence = excluded.confidence,
                                               updated_at = excluded.updated_at
                """,
                (key, value, max(0.0, min(1.0, confidence)), self._now()),
            )
            existing = self._connection.execute(
                "SELECT normalized_value FROM semantic_memory WHERE key = ?", (key,)
            ).fetchone()
            normalized_value = " ".join(str(value).casefold().split())
            conflict = int(bool(existing and str(existing["normalized_value"]) != normalized_value))
            self._connection.execute(
                """
                INSERT INTO semantic_memory(
                    key, value, normalized_value, confidence, source,
                    conflict_count, updated_at
                ) VALUES (?, ?, ?, ?, 'fact', ?, ?)
                ON CONFLICT(key) DO UPDATE SET
                    value=excluded.value,
                    normalized_value=excluded.normalized_value,
                    confidence=excluded.confidence,
                    conflict_count=semantic_memory.conflict_count + excluded.conflict_count,
                    updated_at=excluded.updated_at
                """,
                (
                    key, str(value), normalized_value,
                    max(0.0, min(1.0, confidence)), conflict, self._now(),
                ),
            )

    def get_fact(self, key: str) -> str | None:
        with self._lock:
            row = self._connection.execute(
                "SELECT value FROM facts WHERE key = ?", (key,)
            ).fetchone()
            return str(row["value"]) if row else None

    def facts(self, limit: int = 50) -> dict[str, str]:
        with self._lock:
            rows = self._connection.execute(
                "SELECT key, value FROM facts ORDER BY updated_at DESC LIMIT ?", (max(1, limit),)
            ).fetchall()
            return {str(row["key"]): str(row["value"]) for row in rows}

    def remember_semantic(
        self,
        key: str,
        value: str,
        confidence: float = 1.0,
        source: str = "conversation",
    ) -> None:
        clean_key = str(key).strip()[:120]
        clean_value = str(value).strip()[:2_000]
        if not clean_key or not clean_value:
            raise ValueError("Semantic memory key and value are required")
        normalized = " ".join(clean_value.casefold().split())
        with self._lock, self._connection:
            previous = self._connection.execute(
                "SELECT normalized_value, confidence FROM semantic_memory WHERE key = ?",
                (clean_key,),
            ).fetchone()
            conflict = int(bool(previous and str(previous["normalized_value"]) != normalized))
            previous_confidence = float(previous["confidence"]) if previous else -1.0
            should_replace = previous is None or float(confidence) >= previous_confidence
            if should_replace:
                self._connection.execute(
                    """
                    INSERT INTO semantic_memory(
                        key, value, normalized_value, confidence, source,
                        conflict_count, updated_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(key) DO UPDATE SET
                        value=excluded.value,
                        normalized_value=excluded.normalized_value,
                        confidence=excluded.confidence,
                        source=excluded.source,
                        conflict_count=semantic_memory.conflict_count + excluded.conflict_count,
                        updated_at=excluded.updated_at
                    """,
                    (
                        clean_key, clean_value, normalized,
                        max(0.0, min(1.0, float(confidence))), source[:40],
                        conflict, self._now(),
                    ),
                )
            elif conflict:
                self._connection.execute(
                    "UPDATE semantic_memory SET conflict_count=conflict_count+1 WHERE key=?",
                    (clean_key,),
                )

    def semantic_memories(self, limit: int = 50) -> dict[str, str]:
        with self._lock:
            rows = self._connection.execute(
                "SELECT key, value FROM semantic_memory ORDER BY updated_at DESC LIMIT ?",
                (max(1, min(500, int(limit))),),
            ).fetchall()
        return {str(row["key"]): str(row["value"]) for row in rows}

    def record_episode(
        self, session_id: int | None, summary: str, salience: float = 0.5
    ) -> int:
        clean = " ".join(str(summary).split())[:2_000]
        if not clean:
            raise ValueError("Episode summary is required")
        key = clean.casefold()[:500]
        now = self._now()
        with self._lock, self._connection:
            self._connection.execute(
                """
                INSERT INTO episodic_memory(
                    session_id, summary, normalized_key, salience,
                    created_at, last_accessed
                ) VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT(session_id, normalized_key) DO UPDATE SET
                    salience=MAX(episodic_memory.salience, excluded.salience),
                    last_accessed=excluded.last_accessed
                """,
                (session_id, clean, key, max(0.0, min(1.0, salience)), now, now),
            )
            row = self._connection.execute(
                "SELECT id FROM episodic_memory WHERE session_id IS ? AND normalized_key = ?",
                (session_id, key),
            ).fetchone()
            return int(row["id"]) if row else 0

    def episodes(self, limit: int = 20) -> tuple[EpisodicMemory, ...]:
        with self._lock, self._connection:
            rows = self._connection.execute(
                """
                SELECT id, session_id, summary, salience, created_at
                FROM episodic_memory
                ORDER BY salience DESC, last_accessed DESC LIMIT ?
                """,
                (max(1, min(200, int(limit))),),
            ).fetchall()
            ids = [int(row["id"]) for row in rows]
            if ids:
                placeholders = ",".join("?" for _ in ids)
                self._connection.execute(
                    f"UPDATE episodic_memory SET access_count=access_count+1, last_accessed=? WHERE id IN ({placeholders})",
                    (self._now(), *ids),
                )
        return tuple(
            EpisodicMemory(
                int(row["id"]),
                int(row["session_id"]) if row["session_id"] is not None else None,
                str(row["summary"]), float(row["salience"]), str(row["created_at"]),
            )
            for row in rows
        )

    def get_context(self, session_id: int) -> dict[str, Any]:
        with self._lock:
            row = self._connection.execute(
                "SELECT topic, state_json FROM session_context WHERE session_id = ?", (session_id,)
            ).fetchone()
        if not row:
            return {"topic": ""}
        try:
            state = json.loads(str(row["state_json"]))
        except json.JSONDecodeError:
            state = {}
        if not isinstance(state, dict):
            state = {}
        state["topic"] = str(row["topic"])
        return state

    def set_context(self, session_id: int, topic: str, state: dict[str, Any]) -> None:
        payload = dict(state)
        payload.pop("topic", None)
        encoded = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
        with self._lock, self._connection:
            self._ensure_open()
            self._connection.execute(
                """
                INSERT INTO session_context(session_id, topic, state_json, updated_at)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(session_id) DO UPDATE SET topic = excluded.topic,
                    state_json = excluded.state_json, updated_at = excluded.updated_at
                """,
                (session_id, topic[:80], encoded, self._now()),
            )

    def remember_attachment(
        self, session_id: int, name: str, path: str, kind: str, size: int
    ) -> int:
        with self._lock, self._connection:
            cursor = self._connection.execute(
                "INSERT INTO attachments(session_id, name, path, kind, size, created_at) VALUES (?, ?, ?, ?, ?, ?)",
                (session_id, name, path, kind, int(size), self._now()),
            )
            self.set_fact("last_file", name)
            return int(cursor.lastrowid)

    def last_attachment(self, session_id: int) -> AttachmentRecord | None:
        with self._lock:
            row = self._connection.execute(
                "SELECT * FROM attachments WHERE session_id = ? ORDER BY id DESC LIMIT 1",
                (session_id,),
            ).fetchone()
        if not row:
            return None
        return AttachmentRecord(
            int(row["id"]), int(row["session_id"]), str(row["name"]), str(row["path"]),
            str(row["kind"]), int(row["size"]), str(row["created_at"])
        )

    def clear_session(self, session_id: int) -> None:
        with self._lock, self._connection:
            self._connection.execute("DELETE FROM messages WHERE session_id = ?", (session_id,))
            self._connection.execute("DELETE FROM attachments WHERE session_id = ?", (session_id,))
            self._connection.execute(
                "UPDATE session_context SET topic = '', state_json = '{}', updated_at = ? WHERE session_id = ?",
                (self._now(), session_id),
            )

    def clear_long_term_memory(self) -> None:
        with self._lock, self._connection:
            self._connection.execute("DELETE FROM facts")
            self._connection.execute("DELETE FROM semantic_memory")
            self._connection.execute("DELETE FROM episodic_memory")
            self._connection.execute("DELETE FROM session_context")
            self._connection.execute("DELETE FROM corrections")
            self._connection.execute("DELETE FROM feedback")

    def store_correction(
        self,
        trigger: str,
        trigger_normalized: str,
        intent: str,
        arguments: dict[str, Any],
        source: str = "correction",
        maximum_rows: int = 500,
    ) -> int:
        clean_trigger = str(trigger).strip()[:240]
        normalized = str(trigger_normalized).strip()[:240]
        if not clean_trigger or not normalized or not intent:
            raise ValueError("Correction trigger and intent are required")
        encoded = json.dumps(arguments, ensure_ascii=False, separators=(",", ":"))
        now = self._now()
        with self._lock, self._connection:
            self._connection.execute(
                """
                INSERT INTO corrections(
                    trigger, trigger_normalized, intent, arguments_json,
                    source, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(trigger_normalized) DO UPDATE SET
                    trigger=excluded.trigger,
                    intent=excluded.intent,
                    arguments_json=excluded.arguments_json,
                    source=excluded.source,
                    updated_at=excluded.updated_at
                """,
                (clean_trigger, normalized, intent, encoded, source[:32], now, now),
            )
            self._connection.execute(
                """
                DELETE FROM corrections WHERE id IN (
                    SELECT id FROM corrections ORDER BY updated_at DESC, id DESC
                    LIMIT -1 OFFSET ?
                )
                """,
                (max(10, int(maximum_rows)),),
            )
            row = self._connection.execute(
                "SELECT id FROM corrections WHERE trigger_normalized = ?", (normalized,)
            ).fetchone()
            return int(row["id"]) if row else 0

    @staticmethod
    def _correction_from_row(row: sqlite3.Row) -> CorrectionRule:
        try:
            arguments = json.loads(str(row["arguments_json"]))
        except json.JSONDecodeError:
            arguments = {}
        return CorrectionRule(
            int(row["id"]), str(row["trigger"]), str(row["trigger_normalized"]),
            str(row["intent"]), arguments if isinstance(arguments, dict) else {},
            str(row["source"]), int(row["use_count"])
        )

    def find_correction(self, normalized_text: str) -> CorrectionRule | None:
        normalized = str(normalized_text).strip()
        if not normalized:
            return None
        with self._lock, self._connection:
            row = self._connection.execute(
                "SELECT * FROM corrections WHERE trigger_normalized = ?", (normalized,)
            ).fetchone()
            if row is None:
                candidates = self._connection.execute(
                    "SELECT * FROM corrections ORDER BY LENGTH(trigger_normalized) DESC LIMIT 100"
                ).fetchall()
                row = next(
                    (
                        candidate for candidate in candidates
                        if len(str(candidate["trigger_normalized"])) >= 4
                        and str(candidate["trigger_normalized"]) in normalized
                    ),
                    None,
                )
            if row is None:
                return None
            self._connection.execute(
                "UPDATE corrections SET use_count = use_count + 1, updated_at = ? WHERE id = ?",
                (self._now(), int(row["id"])),
            )
            return self._correction_from_row(row)

    def correction_count(self) -> int:
        with self._lock:
            return int(self._connection.execute("SELECT COUNT(*) FROM corrections").fetchone()[0])

    def add_feedback(
        self,
        session_id: int,
        message_id: int | None,
        positive: bool,
        note: str = "",
        maximum_rows: int = 2000,
    ) -> int:
        with self._lock, self._connection:
            cursor = self._connection.execute(
                "INSERT INTO feedback(session_id, message_id, rating, note, created_at) VALUES (?, ?, ?, ?, ?)",
                (session_id, message_id, 1 if positive else -1, str(note)[:500], self._now()),
            )
            self._connection.execute(
                "DELETE FROM feedback WHERE id IN (SELECT id FROM feedback ORDER BY id DESC LIMIT -1 OFFSET ?)",
                (max(50, int(maximum_rows)),),
            )
            return int(cursor.lastrowid)

    def feedback_count(self) -> int:
        with self._lock:
            return int(self._connection.execute("SELECT COUNT(*) FROM feedback").fetchone()[0])

    def add_decision_log(
        self,
        session_id: int,
        input_excerpt: str,
        intent: str,
        entity: str,
        confidence: float,
        tool: str,
        confirmation: bool,
        source: str,
    ) -> None:
        with self._lock, self._connection:
            self._connection.execute(
                """
                INSERT INTO decision_log(
                    session_id, input_excerpt, intent, entity, confidence,
                    tool, confirmation, source, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    session_id, str(input_excerpt)[:240], intent[:80], entity[:80],
                    max(0.0, min(1.0, float(confidence))), tool[:80],
                    int(confirmation), source[:80], self._now(),
                ),
            )
            self._connection.execute(
                "DELETE FROM decision_log WHERE id IN (SELECT id FROM decision_log ORDER BY id DESC LIMIT -1 OFFSET 1000)"
            )

    def record_reversible_action(
        self,
        session_id: int,
        tool: str,
        arguments: dict[str, Any],
        inverse_tool: str,
        inverse_arguments: dict[str, Any],
    ) -> int:
        if not inverse_tool:
            raise ValueError("An inverse tool is required")
        with self._lock, self._connection:
            cursor = self._connection.execute(
                """
                INSERT INTO action_history(
                    session_id, tool, arguments_json, inverse_tool,
                    inverse_arguments_json, created_at
                ) VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    session_id, tool[:80],
                    json.dumps(arguments, ensure_ascii=False, default=str),
                    inverse_tool[:80],
                    json.dumps(inverse_arguments, ensure_ascii=False, default=str),
                    self._now(),
                ),
            )
            self._connection.execute(
                """
                DELETE FROM action_history WHERE id IN (
                    SELECT id FROM action_history ORDER BY id DESC LIMIT -1 OFFSET 100
                )
                """
            )
            return int(cursor.lastrowid)

    @staticmethod
    def _action_from_row(row: sqlite3.Row) -> ActionRecord:
        def decode(field: str) -> dict[str, Any]:
            try:
                value = json.loads(str(row[field]))
            except json.JSONDecodeError:
                value = {}
            return value if isinstance(value, dict) else {}

        return ActionRecord(
            int(row["id"]), int(row["session_id"]), str(row["tool"]),
            decode("arguments_json"), str(row["inverse_tool"]),
            decode("inverse_arguments_json"), str(row["created_at"]),
            str(row["undone_at"]),
        )

    def last_reversible_action(self, session_id: int) -> ActionRecord | None:
        with self._lock:
            row = self._connection.execute(
                """
                SELECT * FROM action_history
                WHERE session_id=? AND undone_at=''
                ORDER BY id DESC LIMIT 1
                """,
                (session_id,),
            ).fetchone()
        return self._action_from_row(row) if row else None

    def mark_action_undone(self, action_id: int) -> None:
        with self._lock, self._connection:
            self._connection.execute(
                "UPDATE action_history SET undone_at=? WHERE id=? AND undone_at=''",
                (self._now(), int(action_id)),
            )

    def forget(self, scope: str, key: str = "") -> int:
        """Explicit forgetting; never deletes settings or files."""
        normalized_scope = str(scope).casefold().strip()
        clean_key = str(key).strip()
        with self._lock, self._connection:
            if normalized_scope in {"fact", "semantic"} and clean_key:
                before = self._connection.total_changes
                self._connection.execute("DELETE FROM facts WHERE key=?", (clean_key,))
                self._connection.execute("DELETE FROM semantic_memory WHERE key=?", (clean_key,))
                return self._connection.total_changes - before
            if normalized_scope in {"episode", "episodic"} and clean_key:
                cursor = self._connection.execute(
                    "DELETE FROM episodic_memory WHERE normalized_key LIKE ?",
                    (f"%{clean_key.casefold()}%",),
                )
                return max(0, int(cursor.rowcount))
            if normalized_scope == "conversation":
                cursor = self._connection.execute("DELETE FROM messages")
                return max(0, int(cursor.rowcount))
            if normalized_scope == "all_memory":
                before = self._connection.total_changes
                for table in ("facts", "semantic_memory", "episodic_memory", "session_context"):
                    self._connection.execute(f"DELETE FROM {table}")
                return self._connection.total_changes - before
        return 0

    def consolidate_memory(self, maximum_episodes: int = 500) -> dict[str, int]:
        """Deduplicate and age low-salience episodes while preserving facts."""
        with self._lock, self._connection:
            before = int(self._connection.execute("SELECT COUNT(*) FROM episodic_memory").fetchone()[0])
            self._connection.execute(
                """
                DELETE FROM episodic_memory WHERE id IN (
                    SELECT id FROM episodic_memory
                    ORDER BY salience DESC, access_count DESC, last_accessed DESC
                    LIMIT -1 OFFSET ?
                )
                """,
                (max(20, int(maximum_episodes)),),
            )
            after = int(self._connection.execute("SELECT COUNT(*) FROM episodic_memory").fetchone()[0])
            conflicts = int(
                self._connection.execute(
                    "SELECT COALESCE(SUM(conflict_count), 0) FROM semantic_memory"
                ).fetchone()[0]
            )
        return {"episodes_removed": before - after, "episodes_remaining": after, "semantic_conflicts": conflicts}

    def cleanup(self) -> dict[str, int]:
        cutoff = (datetime.now(UTC) - timedelta(days=self.retention_days)).isoformat(timespec="seconds")
        with self._lock, self._connection:
            before_messages = int(self._connection.execute("SELECT COUNT(*) FROM messages").fetchone()[0])
            self._trim_messages_locked()
            self._connection.execute(
                """
                DELETE FROM sessions WHERE last_active < ?
                """,
                (cutoff,),
            )
            self._connection.execute(
                """
                DELETE FROM sessions WHERE id IN (
                    SELECT id FROM sessions ORDER BY last_active DESC, id DESC
                    LIMIT -1 OFFSET ?
                )
                """,
                (self.max_sessions,),
            )
            self._connection.execute(
                "DELETE FROM attachments WHERE created_at < ?", (cutoff,)
            )
            self._connection.execute(
                "DELETE FROM action_history WHERE created_at < ? OR undone_at != ''", (cutoff,)
            )
            after_messages = int(self._connection.execute("SELECT COUNT(*) FROM messages").fetchone()[0])
            session_count = int(self._connection.execute("SELECT COUNT(*) FROM sessions").fetchone()[0])
        return {"messages_removed": before_messages - after_messages, "sessions_remaining": session_count}

    def integrity_check(self) -> bool:
        with self._lock:
            row = self._connection.execute("PRAGMA quick_check").fetchone()
            return bool(row and str(row[0]).casefold() == "ok")

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            try:
                last = self.get_setting("last_session_id")
                if last and last.isdigit():
                    self._connection.execute(
                        "UPDATE sessions SET closed_at = ? WHERE id = ?", (self._now(), int(last))
                    )
            except sqlite3.Error:
                pass
            self._connection.close()
            self._closed = True

    def __enter__(self) -> "MemoryStore":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()


# Compatibility alias for callers that imported the v0.1 exception name.
MemoryError = MemoryStoreError
