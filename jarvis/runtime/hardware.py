from __future__ import annotations

import ctypes
import glob
import os
import platform
import re
import threading
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class HardwareInfo:
    platform: str
    release: str
    architecture: str
    cpu: str
    physical_cores: int
    logical_cores: int
    ram_total_mb: int
    gpu_available: bool
    gpu: str


@dataclass(frozen=True, slots=True)
class HardwareSample:
    cpu_percent: float
    app_ram_mb: float


@dataclass(frozen=True, slots=True)
class PerformanceProfile:
    name: str
    active_fps: int
    idle_fps: int
    particles: int
    glow_layers: int


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


class _ProcessMemoryCounters(ctypes.Structure):
    _fields_ = [
        ("cb", ctypes.c_ulong),
        ("PageFaultCount", ctypes.c_ulong),
        ("PeakWorkingSetSize", ctypes.c_size_t),
        ("WorkingSetSize", ctypes.c_size_t),
        ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
        ("QuotaPagedPoolUsage", ctypes.c_size_t),
        ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
        ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
        ("PagefileUsage", ctypes.c_size_t),
        ("PeakPagefileUsage", ctypes.c_size_t),
    ]


class HardwareManager:
    """Performs only cheap OS queries; no ML framework or heavyweight process probe."""

    def __init__(self) -> None:
        self._system = platform.system()
        self._lock = threading.Lock()
        self.info = self._detect()
        self._previous_cpu = self._read_cpu_times()

    @staticmethod
    def _cpu_name() -> str:
        name = platform.processor().strip()
        if name:
            return name
        cpuinfo = Path("/proc/cpuinfo")
        if cpuinfo.is_file():
            try:
                for line in cpuinfo.read_text(encoding="utf-8", errors="ignore").splitlines():
                    if line.casefold().startswith("model name"):
                        return line.split(":", 1)[-1].strip()
            except OSError:
                pass
        return platform.machine() or "Unknown CPU"

    @staticmethod
    def _physical_cores(logical: int) -> int:
        cpuinfo = Path("/proc/cpuinfo")
        if cpuinfo.is_file():
            try:
                physical_id = "0"
                core_id = ""
                pairs: set[tuple[str, str]] = set()
                for line in cpuinfo.read_text(encoding="utf-8", errors="ignore").splitlines():
                    if line.startswith("physical id"):
                        physical_id = line.split(":", 1)[-1].strip()
                    elif line.startswith("core id"):
                        core_id = line.split(":", 1)[-1].strip()
                    elif not line.strip() and core_id:
                        pairs.add((physical_id, core_id))
                        core_id = ""
                if core_id:
                    pairs.add((physical_id, core_id))
                if pairs:
                    return len(pairs)
            except OSError:
                pass
        return max(1, logical // 2) if logical > 1 else 1

    def _total_ram_bytes(self) -> int:
        if self._system == "Windows":
            status = _MemoryStatusEx()
            status.dwLength = ctypes.sizeof(status)
            try:
                if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
                    return int(status.ullTotalPhys)
            except (AttributeError, OSError):
                return 0
        try:
            return int(os.sysconf("SC_PHYS_PAGES")) * int(os.sysconf("SC_PAGE_SIZE"))
        except (AttributeError, OSError, ValueError):
            return 0

    @staticmethod
    def _windows_gpu() -> str:
        try:
            import winreg

            with winreg.OpenKey(
                winreg.HKEY_LOCAL_MACHINE, r"HARDWARE\DEVICEMAP\VIDEO"
            ) as video_key:
                registry_path, _ = winreg.QueryValueEx(video_key, r"\Device\Video0")
            prefix = "\\Registry\\Machine\\"
            if not str(registry_path).casefold().startswith(prefix.casefold()):
                return ""
            subkey = str(registry_path)[len(prefix) :]
            with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, subkey) as adapter_key:
                for value_name in ("DriverDesc", "HardwareInformation.AdapterString"):
                    try:
                        value, _ = winreg.QueryValueEx(adapter_key, value_name)
                        if isinstance(value, bytes):
                            value = value.decode("utf-16-le", errors="ignore").rstrip("\x00")
                        if str(value).strip():
                            return str(value).strip()
                    except OSError:
                        continue
        except (ImportError, OSError):
            pass
        return ""

    @staticmethod
    def _linux_gpu() -> str:
        models: list[str] = []
        for information in glob.glob("/proc/driver/nvidia/gpus/*/information"):
            try:
                content = Path(information).read_text(encoding="utf-8", errors="ignore")
                match = re.search(r"^Model:\s*(.+)$", content, re.MULTILINE)
                if match:
                    models.append(match.group(1).strip())
            except OSError:
                continue
        if models:
            return ", ".join(dict.fromkeys(models))

        vendors = {"0x10de": "NVIDIA GPU", "0x1002": "AMD GPU", "0x8086": "Intel GPU"}
        for vendor_path in glob.glob("/sys/class/drm/card[0-9]*/device/vendor"):
            try:
                vendor = Path(vendor_path).read_text(encoding="ascii").strip().casefold()
                if vendor in vendors:
                    models.append(vendors[vendor])
            except OSError:
                continue
        return ", ".join(dict.fromkeys(models))

    def _gpu_name(self) -> str:
        if self._system == "Windows":
            return self._windows_gpu()
        if self._system == "Linux":
            return self._linux_gpu()
        visible = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
        if visible and visible != "-1":
            return "CUDA-compatible GPU"
        return ""

    def _detect(self) -> HardwareInfo:
        logical = max(1, os.cpu_count() or 1)
        gpu = self._gpu_name()
        return HardwareInfo(
            platform=self._system or "Unknown",
            release=platform.release(),
            architecture=platform.machine() or "Unknown",
            cpu=self._cpu_name(),
            physical_cores=self._physical_cores(logical),
            logical_cores=logical,
            ram_total_mb=round(self._total_ram_bytes() / (1024 * 1024)),
            gpu_available=bool(gpu),
            gpu=gpu or "Not detected (CPU fallback active)",
        )

    def _read_cpu_times(self) -> tuple[int, int] | None:
        if self._system == "Windows":
            idle = ctypes.c_ulonglong()
            kernel = ctypes.c_ulonglong()
            user = ctypes.c_ulonglong()
            try:
                ok = ctypes.windll.kernel32.GetSystemTimes(
                    ctypes.byref(idle), ctypes.byref(kernel), ctypes.byref(user)
                )
            except (AttributeError, OSError):
                return None
            if ok:
                return int(idle.value), int(kernel.value + user.value)
            return None

        stat_path = Path("/proc/stat")
        if stat_path.is_file():
            try:
                parts = stat_path.read_text(encoding="ascii").splitlines()[0].split()[1:]
                values = [int(value) for value in parts]
                idle = values[3] + (values[4] if len(values) > 4 else 0)
                return idle, sum(values)
            except (OSError, ValueError, IndexError):
                return None
        return None

    def _process_ram_bytes(self) -> int:
        if self._system == "Windows":
            counters = _ProcessMemoryCounters()
            counters.cb = ctypes.sizeof(counters)
            try:
                process = ctypes.windll.kernel32.GetCurrentProcess()
                if ctypes.windll.psapi.GetProcessMemoryInfo(
                    process, ctypes.byref(counters), counters.cb
                ):
                    return int(counters.WorkingSetSize)
            except (AttributeError, OSError):
                return 0

        statm = Path("/proc/self/statm")
        if statm.is_file():
            try:
                resident_pages = int(statm.read_text(encoding="ascii").split()[1])
                return resident_pages * int(os.sysconf("SC_PAGE_SIZE"))
            except (OSError, ValueError, IndexError):
                pass
        try:
            import resource

            usage = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
            return usage if self._system == "Darwin" else usage * 1024
        except (ImportError, ValueError):
            return 0

    def sample(self) -> HardwareSample:
        with self._lock:
            current = self._read_cpu_times()
            cpu_percent = 0.0
            if current is not None and self._previous_cpu is not None:
                idle_delta = current[0] - self._previous_cpu[0]
                total_delta = current[1] - self._previous_cpu[1]
                if total_delta > 0:
                    cpu_percent = 100.0 * (1.0 - idle_delta / total_delta)
            self._previous_cpu = current
            ram_mb = self._process_ram_bytes() / (1024 * 1024)
        return HardwareSample(
            cpu_percent=max(0.0, min(100.0, cpu_percent)),
            app_ram_mb=max(0.0, ram_mb),
        )

    def recommended_performance(self) -> str:
        ram = self.info.ram_total_mb
        cores = self.info.logical_cores
        if (ram and ram < 4096) or cores <= 2:
            return "LOW"
        if self.info.gpu_available and ram >= 8192 and cores >= 6:
            return "ULTRA"
        return "BALANCED"

    @staticmethod
    def performance_profile(name: str) -> PerformanceProfile:
        profiles = {
            "LOW": PerformanceProfile("LOW", 20, 3, 6, 2),
            "BALANCED": PerformanceProfile("BALANCED", 30, 6, 12, 3),
            "ULTRA": PerformanceProfile("ULTRA", 45, 10, 22, 5),
        }
        return profiles.get(name.upper(), profiles["BALANCED"])
