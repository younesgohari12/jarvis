from __future__ import annotations

import json
import re
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


_SECRET_TEXT = re.compile(
    r"(?i)(token|password|passwd|secret|api[_-]?key|authorization)(\s*[=:]\s*)([^\s,;]+)"
)


def _redact(value: Any) -> Any:
    if isinstance(value, str):
        return _SECRET_TEXT.sub(r"\1\2[REDACTED]", value)
    if isinstance(value, dict):
        return {
            str(key): "[REDACTED]"
            if any(marker in str(key).casefold() for marker in ("token", "password", "secret", "api_key", "authorization"))
            else _redact(item)
            for key, item in value.items()
        }
    if isinstance(value, (list, tuple)):
        return [_redact(item) for item in value]
    return value


class JSONLEventLog:
    """Thread-safe bounded JSONL writer used by diagnostics and Lab log viewer."""

    def __init__(self, path: Path, maximum_bytes: int = 2 * 1024 * 1024) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path
        self.maximum_bytes = maximum_bytes
        self._lock = threading.Lock()

    def write(self, event: str, **values: Any) -> None:
        payload = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "event": event,
            **_redact(values),
        }
        line = json.dumps(payload, ensure_ascii=False, default=str) + "\n"
        with self._lock:
            try:
                if self.path.exists() and self.path.stat().st_size > self.maximum_bytes:
                    backup = self.path.with_suffix(self.path.suffix + ".1")
                    if backup.exists():
                        backup.unlink()
                    self.path.replace(backup)
                with self.path.open("a", encoding="utf-8") as handle:
                    handle.write(line)
            except OSError:
                return


class StructuredLogs:
    def __init__(self, directory: Path) -> None:
        self.chat = JSONLEventLog(directory / "chat.jsonl")
        self.agent = JSONLEventLog(directory / "agent.jsonl")
        self.tools = JSONLEventLog(directory / "tools.jsonl")
        self.performance = JSONLEventLog(directory / "performance.jsonl")
