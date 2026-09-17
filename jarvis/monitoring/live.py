from __future__ import annotations

import ctypes
import os
import platform
import shutil
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path

from jarvis.runtime.hardware import HardwareManager


@dataclass(frozen=True, slots=True)
class SystemSnapshot:
    timestamp: float
    cpu_percent: float
    app_cpu_percent: float
    ram_percent: float
    ram_used_mb: float
    ram_total_mb: float
    app_ram_mb: float
    gpu_available: bool
    gpu_percent: float | None
    vram_used_mb: float | None
    vram_total_mb: float | None
    gpu_temperature_c: float | None
    disk_percent: float
    disk_free_gb: float
    network_kbps: float


class _MemoryStatusEx(ctypes.Structure):
    _fields_ = [
        ("dwLength", ctypes.c_ulong),
        ("dwMemoryLoad", ctypes.c_ulong),
        ("ullTotalPhys", ctypes.c_ulonglong),
        ("ullAvailPhys", ctypes.c_ulonglong),
        ("ullTotalPageFile", ctypes.c_ulonglong),
        ("ullAvailPageFile", ctypes.c_ulonglong),
        ("ullTotalVirtual", ctypes.c_ulonglong),
        ("ullAvailVirtual", ctypes.c_ulonglong),
        ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
    ]


class LiveMonitor:
    """On-demand sampler with no resident thread and a slow NVIDIA probe."""

    def __init__(self, hardware: HardwareManager) -> None:
        self.hardware = hardware
        self._system = platform.system()
        self._last_wall = time.monotonic()
        self._last_process = time.process_time()
        self._last_network = self._network_bytes()
        self._last_gpu_probe = 0.0
        self._gpu_cache: tuple[float | None, float | None, float | None, float | None] = (
            None, None, None, None
        )

    def _memory(self) -> tuple[float, float, float]:
        if self._system == "Windows":
            status = _MemoryStatusEx()
            status.dwLength = ctypes.sizeof(status)
            try:
                if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
                    total = status.ullTotalPhys / (1024 * 1024)
                    used = (status.ullTotalPhys - status.ullAvailPhys) / (1024 * 1024)
                    return float(status.dwMemoryLoad), used, total
            except (AttributeError, OSError):
                pass
        path = Path("/proc/meminfo")
        try:
            values: dict[str, float] = {}
            for line in path.read_text(encoding="ascii").splitlines():
                key, raw = line.split(":", 1)
                values[key] = float(raw.split()[0]) / 1024.0
            total = values["MemTotal"]
            available = values.get("MemAvailable", values.get("MemFree", 0.0))
            used = max(0.0, total - available)
            return used / max(total, 1.0) * 100.0, used, total
        except (OSError, ValueError, KeyError):
            total = float(self.hardware.info.ram_total_mb)
            return 0.0, 0.0, total

    def _network_bytes(self) -> int:
        if self._system != "Linux":
            return 0
        try:
            total = 0
            for line in Path("/proc/net/dev").read_text(encoding="ascii").splitlines()[2:]:
                _name, payload = line.split(":", 1)
                fields = payload.split()
                total += int(fields[0]) + int(fields[8])
            return total
        except (OSError, ValueError, IndexError):
            return 0

    def _gpu(self, now: float) -> tuple[float | None, float | None, float | None, float | None]:
        if now - self._last_gpu_probe < 5.0:
            return self._gpu_cache
        self._last_gpu_probe = now
        executable = shutil.which("nvidia-smi")
        if not executable:
            self._gpu_cache = (None, None, None, None)
            return self._gpu_cache
        try:
            completed = subprocess.run(
                [
                    executable,
                    "--query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu",
                    "--format=csv,noheader,nounits",
                ],
                capture_output=True,
                text=True,
                timeout=1.2,
                check=False,
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
            )
            row = completed.stdout.splitlines()[0]
            values = tuple(float(item.strip()) for item in row.split(","))
            self._gpu_cache = values  # type: ignore[assignment]
        except (OSError, ValueError, IndexError, subprocess.TimeoutExpired):
            self._gpu_cache = (None, None, None, None)
        return self._gpu_cache

    def sample(self, disk_path: str | Path | None = None) -> SystemSnapshot:
        now = time.monotonic()
        wall_delta = max(now - self._last_wall, 1e-6)
        process_now = time.process_time()
        app_cpu = (process_now - self._last_process) / wall_delta * 100.0
        app_cpu /= max(1, self.hardware.info.logical_cores)
        self._last_wall, self._last_process = now, process_now

        base = self.hardware.sample()
        ram_percent, ram_used, ram_total = self._memory()
        try:
            disk = shutil.disk_usage(Path(disk_path or Path.cwd().anchor or Path.cwd()))
            disk_percent = (disk.used / max(disk.total, 1)) * 100.0
            disk_free = disk.free / (1024 ** 3)
        except OSError:
            disk_percent, disk_free = 0.0, 0.0
        network = self._network_bytes()
        prior_network = self._last_network
        self._last_network = network
        network_kbps = max(0.0, network - prior_network) / wall_delta / 1024.0
        gpu, vram_used, vram_total, temperature = self._gpu(now)
        gpu_available = gpu is not None or self.hardware.info.gpu_available
        return SystemSnapshot(
            timestamp=time.time(),
            cpu_percent=base.cpu_percent,
            app_cpu_percent=max(0.0, min(100.0, app_cpu)),
            ram_percent=max(0.0, min(100.0, ram_percent)),
            ram_used_mb=ram_used,
            ram_total_mb=ram_total,
            app_ram_mb=base.app_ram_mb,
            gpu_available=gpu_available,
            gpu_percent=gpu,
            vram_used_mb=vram_used,
            vram_total_mb=vram_total,
            gpu_temperature_c=temperature,
            disk_percent=max(0.0, min(100.0, disk_percent)),
            disk_free_gb=max(0.0, disk_free),
            network_kbps=network_kbps,
        )
