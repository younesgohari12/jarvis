from __future__ import annotations

import json
import re
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


_SECRET = re.compile(
    r"(?i)(token|password|passwd|secret|api[_-]?key|authorization)(\s*[=:]\s*)([^\s,;]+)"
)


def redact_secrets(value: Any) -> Any:
    if isinstance(value, str):
        return _SECRET.sub(r"\1\2[REDACTED]", value)
    if isinstance(value, dict):
        return {
            str(key): "[REDACTED]"
            if any(marker in str(key).casefold() for marker in ("token", "password", "secret", "api_key", "authorization"))
            else redact_secrets(item)
            for key, item in value.items()
        }
    if isinstance(value, (list, tuple)):
        return [redact_secrets(item) for item in value]
    return value


class FailureCollector:
    """Review-only failure queue used for hard-example mining."""

    STATUSES = {"pending", "approved", "rejected", "resolved"}

    def __init__(self, path: Path, maximum_rows: int = 5000) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.maximum_rows = max(100, min(50_000, int(maximum_rows)))
        self._lock = threading.RLock()

    def add(
        self,
        *,
        user_input: str,
        predicted_intent: str,
        predicted_action: str = "",
        entity: str = "",
        arguments: dict[str, Any] | None = None,
        tool: str = "",
        failure_code: str = "",
        verification: str = "",
        context: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        row = {
            "id": uuid.uuid4().hex,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "status": "pending",
            "input": redact_secrets(str(user_input).strip()[:4000]),
            "prediction": {
                "intent": str(predicted_intent)[:120],
                "action": str(predicted_action)[:120],
                "entity": str(entity)[:500],
                "arguments": redact_secrets(arguments or {}),
                "tool": str(tool)[:120],
            },
            "failure_code": str(failure_code)[:200],
            "verification": str(verification)[:500],
            "context": redact_secrets(context or {}),
            "correct_action": "",
            "correct_arguments": {},
            "correct_response": "",
            "notes": "",
        }
        with self._lock:
            rows = self.rows()
            rows.append(row)
            rows = rows[-self.maximum_rows :]
            self._write(rows)
        return row

    def rows(self, status: str | None = None) -> list[dict[str, Any]]:
        if not self.path.is_file():
            return []
        result: list[dict[str, Any]] = []
        with self._lock:
            try:
                lines = self.path.read_text(encoding="utf-8").splitlines()
            except OSError:
                return []
            for line in lines:
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if isinstance(row, dict) and (status is None or row.get("status") == status):
                    result.append(row)
        return result

    def review(
        self,
        row_id: str,
        status: str,
        *,
        correct_action: str = "",
        correct_arguments: dict[str, Any] | None = None,
        correct_response: str = "",
        notes: str = "",
    ) -> bool:
        if status not in self.STATUSES:
            raise ValueError("Unsupported failure status")
        changed = False
        with self._lock:
            rows = self.rows()
            for row in rows:
                if row.get("id") != row_id:
                    continue
                row.update(
                    {
                        "status": status,
                        "correct_action": str(correct_action).strip()[:120],
                        "correct_arguments": redact_secrets(correct_arguments or {}),
                        "correct_response": str(correct_response).strip()[:4000],
                        "notes": str(notes).strip()[:1000],
                        "reviewed_at": datetime.now(timezone.utc).isoformat(),
                    }
                )
                changed = True
                break
            if changed:
                self._write(rows)
        return changed

    def approved_training_rows(self) -> list[dict[str, Any]]:
        output: list[dict[str, Any]] = []
        for row in self.rows("approved"):
            action = str(row.get("correct_action", ""))
            response = str(row.get("correct_response", ""))
            if action:
                arguments = row.get("correct_arguments", {})
                response = "<tool>" + action + " " + " ".join(
                    f"{key}={value}" for key, value in sorted(arguments.items())
                )
            if row.get("input") and response:
                output.append(
                    {
                        "input": row["input"],
                        "output": response.strip(),
                        "category": "approved_failure_correction",
                        "hard_example_weight": 2.0,
                        "source_failure_id": row["id"],
                    }
                )
        return output

    def _write(self, rows: list[dict[str, Any]]) -> None:
        temporary = self.path.with_suffix(self.path.suffix + ".tmp")
        temporary.write_text(
            "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
            encoding="utf-8",
        )
        temporary.replace(self.path)
