from __future__ import annotations

import os
import tempfile
from pathlib import Path

from jarvis.runtime.bootstrap import RuntimeContext, build_runtime


ROOT = Path(__file__).resolve().parents[1]


class TemporaryRuntime:
    def __init__(self) -> None:
        self._temporary = tempfile.TemporaryDirectory(prefix="jarvis-test-")
        self._previous = os.environ.get("JARVIS_DATA_DIR")
        os.environ["JARVIS_DATA_DIR"] = self._temporary.name
        self.runtime: RuntimeContext | None = None

    def __enter__(self) -> RuntimeContext:
        self.runtime = build_runtime(ROOT)
        return self.runtime

    def __exit__(self, *_: object) -> None:
        if self.runtime:
            self.runtime.close()
        if self._previous is None:
            os.environ.pop("JARVIS_DATA_DIR", None)
        else:
            os.environ["JARVIS_DATA_DIR"] = self._previous
        self._temporary.cleanup()

