from __future__ import annotations

import ctypes
import platform
import socket
import subprocess
import time
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class SystemActionResult:
    success: bool
    action: str
    verified: bool
    code: str
    detail: str = ""


class _SystemPowerStatus(ctypes.Structure):
    _fields_ = (
        ("ac_line_status", ctypes.c_byte),
        ("battery_flag", ctypes.c_byte),
        ("battery_life_percent", ctypes.c_byte),
        ("reserved", ctypes.c_byte),
        ("battery_life_time", ctypes.c_ulong),
        ("battery_full_life_time", ctypes.c_ulong),
    )


class SystemController:
    VK_VOLUME_MUTE = 0xAD
    VK_VOLUME_DOWN = 0xAE
    VK_VOLUME_UP = 0xAF
    KEYEVENTF_KEYUP = 0x0002

    def __init__(self) -> None:
        self.is_windows = platform.system() == "Windows"
        self._user32 = ctypes.windll.user32 if self.is_windows else None
        self._winmm = ctypes.windll.winmm if self.is_windows else None
        self._previous_volume = 50

    def _unsupported(self, action: str) -> SystemActionResult:
        return SystemActionResult(
            False, action, True, "unsupported_platform",
            "This system action requires Windows.",
        )

    def _media_key(self, virtual_key: int, presses: int = 1) -> bool:
        if not self._user32:
            return False
        for _ in range(max(1, min(100, presses))):
            self._user32.keybd_event(virtual_key, 0, 0, 0)
            self._user32.keybd_event(virtual_key, 0, self.KEYEVENTF_KEYUP, 0)
        return True

    def volume_up(self, steps: int = 1) -> SystemActionResult:
        if not self.is_windows:
            return self._unsupported("volume_up")
        current = self._wave_volume()
        if current is not None:
            result = self.set_volume(current + max(1, min(20, int(steps))) * 2)
            return SystemActionResult(result.success, "volume_up", result.verified, result.code, result.detail)
        success = self._media_key(self.VK_VOLUME_UP, steps)
        return SystemActionResult(success, "volume_up", False, "key_sent" if success else "failed")

    def volume_down(self, steps: int = 1) -> SystemActionResult:
        if not self.is_windows:
            return self._unsupported("volume_down")
        current = self._wave_volume()
        if current is not None:
            result = self.set_volume(current - max(1, min(20, int(steps))) * 2)
            return SystemActionResult(result.success, "volume_down", result.verified, result.code, result.detail)
        success = self._media_key(self.VK_VOLUME_DOWN, steps)
        return SystemActionResult(success, "volume_down", False, "key_sent" if success else "failed")

    def mute(self) -> SystemActionResult:
        if not self.is_windows:
            return self._unsupported("mute")
        current = self._wave_volume()
        if current is not None:
            if current > 0:
                self._previous_volume = current
            result = self.set_volume(0)
            return SystemActionResult(result.success, "mute", result.verified, "muted" if result.success else result.code)
        success = self._media_key(self.VK_VOLUME_MUTE)
        return SystemActionResult(success, "mute", False, "toggle_sent" if success else "failed")

    def unmute(self) -> SystemActionResult:
        if not self.is_windows:
            return self._unsupported("unmute")
        current = self._wave_volume()
        if current is not None:
            target = current if current > 0 else max(1, self._previous_volume)
            result = self.set_volume(target)
            return SystemActionResult(result.success, "unmute", result.verified, "unmuted" if result.success else result.code)
        success = self._media_key(self.VK_VOLUME_MUTE)
        return SystemActionResult(success, "unmute", False, "toggle_sent" if success else "failed")

    def _wave_volume(self) -> int | None:
        if not self._winmm:
            return None
        value = ctypes.c_ulong()
        try:
            if self._winmm.waveOutGetVolume(0xFFFFFFFF, ctypes.byref(value)) != 0:
                return None
        except (AttributeError, OSError):
            return None
        left = value.value & 0xFFFF
        right = (value.value >> 16) & 0xFFFF
        return round(((left + right) / 2) / 65535 * 100)

    def set_volume(self, percent: int) -> SystemActionResult:
        if not self.is_windows:
            return self._unsupported("set_volume")
        target = max(0, min(100, int(percent)))
        success = False
        verified = False
        if self._winmm:
            raw = round(target / 100 * 65535)
            stereo = raw | (raw << 16)
            try:
                success = self._winmm.waveOutSetVolume(0xFFFFFFFF, stereo) == 0
            except (AttributeError, OSError):
                success = False
            measured = self._wave_volume()
            verified = bool(success and measured is not None and abs(measured - target) <= 2)
        if not success:
            lowered = self._media_key(self.VK_VOLUME_DOWN, 50)
            raised = self._media_key(self.VK_VOLUME_UP, round(target / 2)) if target else True
            success = lowered and raised
        return SystemActionResult(
            success, "set_volume", verified,
            "volume_set" if verified else "approximate_level_set" if success else "failed",
            f"target_percent={target}",
        )

    def set_brightness(self, percent: int) -> SystemActionResult:
        if not self.is_windows:
            return self._unsupported("set_brightness")
        target = max(0, min(100, int(percent)))
        script = (
            "$m=(Get-WmiObject -Namespace root/WMI -Class WmiMonitorBrightnessMethods);"
            f"if($m){{$m.WmiSetBrightness(1,{target})|Out-Null;exit 0}}else{{exit 3}}"
        )
        try:
            completed = subprocess.run(
                ["powershell.exe", "-NoProfile", "-NonInteractive", "-Command", script],
                capture_output=True, timeout=5.0, check=False,
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
            )
        except (OSError, subprocess.SubprocessError) as exc:
            return SystemActionResult(False, "set_brightness", True, "brightness_failed", str(exc)[:300])
        success = completed.returncode == 0
        return SystemActionResult(
            success, "set_brightness", success,
            "brightness_set" if success else "brightness_not_supported",
            f"target_percent={target}",
        )

    def battery_info(self) -> dict[str, Any]:
        if not self.is_windows:
            return {"available": False, "platform": platform.system()}
        status = _SystemPowerStatus()
        try:
            success = bool(ctypes.windll.kernel32.GetSystemPowerStatus(ctypes.byref(status)))
        except (AttributeError, OSError):
            success = False
        if not success:
            return {"available": False, "platform": "Windows"}
        percent = int(status.battery_life_percent)
        return {
            "available": status.battery_flag != -128 and percent != 255,
            "percent": None if percent == 255 else percent,
            "plugged_in": status.ac_line_status == 1,
            "charging": bool(status.battery_flag & 8),
            "seconds_remaining": None if status.battery_life_time == 0xFFFFFFFF else int(status.battery_life_time),
        }

    def network_info(self) -> dict[str, Any]:
        result: dict[str, Any] = {
            "hostname": socket.gethostname(),
            "platform": platform.system(),
            "addresses": [],
        }
        try:
            result["addresses"] = sorted(
                {value[4][0] for value in socket.getaddrinfo(socket.gethostname(), None)}
            )[:20]
        except OSError:
            pass
        if self.is_windows:
            try:
                completed = subprocess.run(
                    ["ipconfig", "/all"], capture_output=True, text=True,
                    encoding="utf-8", errors="replace", timeout=5.0, check=False,
                    creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
                )
                result["ipconfig"] = completed.stdout[:24_000]
            except (OSError, subprocess.SubprocessError):
                result["ipconfig"] = ""
        return result

    def shutdown(self, delay_seconds: int = 15) -> SystemActionResult:
        return self._power_command("shutdown", ["shutdown.exe", "/s", "/t", str(max(0, min(300, int(delay_seconds))))])

    def restart(self, delay_seconds: int = 15) -> SystemActionResult:
        return self._power_command("restart", ["shutdown.exe", "/r", "/t", str(max(0, min(300, int(delay_seconds))))])

    def logoff(self) -> SystemActionResult:
        return self._power_command("logoff", ["shutdown.exe", "/l"])

    def sleep(self) -> SystemActionResult:
        if not self.is_windows:
            return self._unsupported("sleep")
        try:
            accepted = bool(ctypes.windll.powrprof.SetSuspendState(False, False, False))
        except (AttributeError, OSError) as exc:
            return SystemActionResult(False, "sleep", True, "sleep_failed", str(exc)[:300])
        return SystemActionResult(accepted, "sleep", False, "sleep_requested" if accepted else "sleep_failed")

    def _power_command(self, action: str, arguments: list[str]) -> SystemActionResult:
        if not self.is_windows:
            return self._unsupported(action)
        started = time.perf_counter()
        try:
            completed = subprocess.run(
                arguments, capture_output=True, timeout=4.0, check=False,
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
            )
        except (OSError, subprocess.SubprocessError) as exc:
            return SystemActionResult(False, action, True, "power_command_failed", str(exc)[:300])
        success = completed.returncode == 0
        return SystemActionResult(
            success, action, False,
            "power_command_accepted" if success else "power_command_failed",
            f"latency_ms={(time.perf_counter() - started) * 1000.0:.3f}",
        )
