from __future__ import annotations

import json
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from jarvis.learning.failures import redact_secrets


class LearningQueue:
    def __init__(self, path: Path, maximum_rows: int = 2000) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.maximum_rows = max(100, min(20_000, int(maximum_rows)))
        self._lock = threading.RLock()

    def add(
        self,
        *,
        original_input: str,
        jarvis_response: str,
        correct_response: str,
        correct_action: str = "",
        correct_arguments: str | dict[str, Any] = "",
        notes: str = "",
    ) -> dict[str, Any]:
        arguments: Any = correct_arguments
        if isinstance(correct_arguments, str) and correct_arguments.strip():
            try:
                arguments = json.loads(correct_arguments)
            except json.JSONDecodeError:
                arguments = {"raw": correct_arguments.strip()}
        row = {
            "id": uuid.uuid4().hex,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "status": "pending",
            "original_input": redact_secrets(str(original_input).strip()[:4000]),
            "jarvis_response": redact_secrets(str(jarvis_response).strip()[:4000]),
            "correct_response": redact_secrets(str(correct_response).strip()[:4000]),
            "correct_action": str(correct_action).strip()[:100],
            "correct_arguments": redact_secrets(arguments),
            "notes": redact_secrets(str(notes).strip()[:1000]),
        }
        if not row["original_input"] or not (row["correct_response"] or row["correct_action"]):
            raise ValueError("Original input and a corrected response or action are required")
        with self._lock:
            rows = self.rows()
            rows.append(row)
            self._write(rows[-self.maximum_rows :])
        return row

    def rows(self, status: str | None = None) -> list[dict[str, Any]]:
        if not self.path.is_file():
            return []
        result: list[dict[str, Any]] = []
        with self._lock:
            for line in self.path.read_text(encoding="utf-8").splitlines():
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if status is None or row.get("status") == status:
                    result.append(row)
        return result

    def set_status(self, row_id: str, status: str) -> bool:
        if status not in {"pending", "approved", "rejected"}:
            raise ValueError("Unsupported queue status")
        rows = self.rows()
        changed = False
        for row in rows:
            if row.get("id") == row_id:
                row["status"] = status
                row["reviewed_at"] = datetime.now(timezone.utc).isoformat()
                changed = True
        if changed:
            with self._lock:
                self._write(rows[-self.maximum_rows :])
        return changed

    def _write(self, rows: list[dict[str, Any]]) -> None:
        temporary = self.path.with_suffix(self.path.suffix + ".tmp")
        temporary.write_text(
            "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
            encoding="utf-8",
        )
        temporary.replace(self.path)
