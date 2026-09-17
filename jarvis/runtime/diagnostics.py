from __future__ import annotations

import sqlite3
from dataclasses import asdict
from pathlib import Path
from typing import Any

from jarvis.memory.store import MemoryStore
from jarvis.runtime.hardware import HardwareManager
from jarvis.tools.registry import ToolRegistry


class DiagnosticManager:
    """On-demand, read-only health checks for Diagnose mode."""

    def __init__(
        self,
        hardware: HardwareManager,
        memory: MemoryStore,
        tools: ToolRegistry,
        model_paths: tuple[Path, ...],
    ) -> None:
        self.hardware = hardware
        self.memory = memory
        self.tools = tools
        self.model_paths = model_paths

    def run(self, scope: str = "jarvis") -> dict[str, Any]:
        checks: list[dict[str, Any]] = []

        def add(name: str, healthy: bool, detail: str, recovery: str = "") -> None:
            checks.append(
                {
                    "name": name,
                    "healthy": bool(healthy),
                    "detail": detail,
                    "recovery": recovery if not healthy else "",
                }
            )

        try:
            database_ok = self.memory.integrity_check()
        except (sqlite3.Error, RuntimeError) as exc:
            database_ok = False
            database_detail = str(exc)
        else:
            database_detail = "SQLite quick_check passed" if database_ok else "SQLite quick_check failed"
        add(
            "memory_database", database_ok, database_detail,
            "Close Jarvis and restore jarvis.db from a known-good backup.",
        )

        missing_models = [path.name for path in self.model_paths if not path.is_file()]
        add(
            "model_assets", not missing_models,
            "All configured model assets are present" if not missing_models else f"Missing: {', '.join(missing_models)}",
            "Re-extract the release ZIP; normal startup will keep the fallback brain active.",
        )
        tool_count = len(self.tools.names())
        add(
            "tool_registry", tool_count >= 10, f"{tool_count} tools registered",
            "Restart Jarvis and inspect jarvis.log for registration errors.",
        )
        sample = self.hardware.sample()
        add(
            "runtime_resources", True,
            f"Process RAM {sample.app_ram_mb:.1f} MB; sampled CPU {sample.cpu_percent:.1f}%",
        )
        healthy = all(bool(check["healthy"]) for check in checks)
        return {
            "status": "healthy" if healthy else "degraded",
            "scope": str(scope),
            "summary": "No health issue detected" if healthy else "One or more checks need attention",
            "checks": checks,
            "hardware": asdict(self.hardware.info),
        }

