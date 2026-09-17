from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
import shutil

from jarvis.runtime.hardware import HardwareManager


class SystemInfoTool:
    """Read-only system information. No shell, shutdown, or restart methods exist."""

    def __init__(self, hardware: HardwareManager) -> None:
        self.hardware = hardware

    def get_info(self) -> dict[str, object]:
        result = asdict(self.hardware.info)
        try:
            anchor = Path.home().anchor or "/"
            usage = shutil.disk_usage(anchor)
            result.update(
                {
                    "disk_total_mb": round(usage.total / (1024 * 1024)),
                    "disk_free_mb": round(usage.free / (1024 * 1024)),
                    "is_64_bit": "64" in self.hardware.info.architecture,
                }
            )
        except OSError:
            result.update({"disk_total_mb": 0, "disk_free_mb": 0, "is_64_bit": False})
        return result

    @staticmethod
    def disk_info(raw_path: str = "") -> dict[str, object]:
        path = Path(raw_path).expanduser().resolve() if raw_path else Path.home()
        anchor = Path(path.anchor) if path.anchor else path
        try:
            usage = shutil.disk_usage(anchor)
        except OSError as exc:
            return {"path": str(anchor), "available": False, "error": str(exc)}
        return {
            "path": str(anchor),
            "available": True,
            "total_bytes": usage.total,
            "used_bytes": usage.used,
            "free_bytes": usage.free,
            "percent_used": round(usage.used / max(1, usage.total) * 100.0, 2),
        }
