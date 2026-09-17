from __future__ import annotations

import ctypes
import platform
import time
from dataclasses import dataclass
from difflib import SequenceMatcher
from typing import Iterable

from jarvis.utils.text import normalize_text


@dataclass(frozen=True, slots=True)
class WindowInfo:
    handle: int
    title: str
    pid: int
    left: int
    top: int
    width: int
    height: int
    minimized: bool
    maximized: bool


@dataclass(frozen=True, slots=True)
class WindowActionResult:
    success: bool
    action: str
    window: WindowInfo | None
    verified: bool
    code: str
    detail: str = ""


class _Rect(ctypes.Structure):
    _fields_ = (
        ("left", ctypes.c_long),
        ("top", ctypes.c_long),
        ("right", ctypes.c_long),
        ("bottom", ctypes.c_long),
    )


class WindowManager:
    """Bounded Win32 top-level-window control with post-action verification."""

    SW_HIDE = 0
    SW_MINIMIZE = 6
    SW_MAXIMIZE = 3
    SW_RESTORE = 9
    WM_CLOSE = 0x0010

    def __init__(self, maximum_windows: int = 250) -> None:
        self.is_windows = platform.system() == "Windows"
        self.maximum_windows = max(20, min(500, int(maximum_windows)))
        self._user32 = ctypes.windll.user32 if self.is_windows else None

    def _unsupported(self, action: str) -> WindowActionResult:
        return WindowActionResult(
            False,
            action,
            None,
            True,
            "unsupported_platform",
            "Window control is available only on a Windows desktop session.",
        )

    def _window_info(self, handle: int) -> WindowInfo | None:
        if not self._user32 or not self._user32.IsWindow(handle):
            return None
        length = int(self._user32.GetWindowTextLengthW(handle))
        buffer = ctypes.create_unicode_buffer(max(2, length + 1))
        self._user32.GetWindowTextW(handle, buffer, len(buffer))
        title = buffer.value.strip()
        pid = ctypes.c_ulong()
        self._user32.GetWindowThreadProcessId(handle, ctypes.byref(pid))
        rectangle = _Rect()
        if not self._user32.GetWindowRect(handle, ctypes.byref(rectangle)):
            rectangle = _Rect(0, 0, 0, 0)
        return WindowInfo(
            handle=int(handle),
            title=title,
            pid=int(pid.value),
            left=int(rectangle.left),
            top=int(rectangle.top),
            width=max(0, int(rectangle.right - rectangle.left)),
            height=max(0, int(rectangle.bottom - rectangle.top)),
            minimized=bool(self._user32.IsIconic(handle)),
            maximized=bool(self._user32.IsZoomed(handle)),
        )

    def enumerate_windows(self) -> tuple[WindowInfo, ...]:
        if not self._user32:
            return ()
        values: list[WindowInfo] = []
        callback_type = ctypes.WINFUNCTYPE(ctypes.c_bool, ctypes.c_void_p, ctypes.c_void_p)

        @callback_type
        def callback(handle: int, _lparam: int) -> bool:
            if len(values) >= self.maximum_windows:
                return False
            if not self._user32.IsWindowVisible(handle):
                return True
            info = self._window_info(int(handle))
            if info and info.title:
                values.append(info)
            return True

        self._user32.EnumWindows(callback, 0)
        return tuple(values)

    @staticmethod
    def _score(query: str, window: WindowInfo) -> float:
        title = normalize_text(window.title)
        if not query:
            return 0.0
        if query == title:
            return 1.0
        if query in title:
            return min(0.98, 0.78 + len(query) / max(1, len(title)) * 0.2)
        return SequenceMatcher(None, query, title).ratio()

    def find_window(
        self,
        query: str = "",
        *,
        pids: Iterable[int] = (),
    ) -> tuple[WindowInfo, ...]:
        normalized = normalize_text(query).strip()
        pid_set = {int(value) for value in pids if int(value) > 0}
        candidates = [
            value for value in self.enumerate_windows()
            if not pid_set or value.pid in pid_set
        ]
        if normalized:
            ranked = sorted(
                ((self._score(normalized, value), value) for value in candidates),
                key=lambda item: (item[0], len(item[1].title)),
                reverse=True,
            )
            return tuple(value for score, value in ranked if score >= 0.48)[:20]
        return tuple(candidates[:20])

    def _resolve(self, query: str, pids: Iterable[int]) -> WindowInfo | None:
        matches = self.find_window(query, pids=pids)
        return matches[0] if matches else None

    def _show_action(
        self,
        action: str,
        command: int,
        query: str,
        pids: Iterable[int] = (),
    ) -> WindowActionResult:
        if not self._user32:
            return self._unsupported(action)
        window = self._resolve(query, pids)
        if window is None:
            return WindowActionResult(False, action, None, True, "window_not_found")
        self._user32.ShowWindow(window.handle, command)
        if action in {"focus", "restore"}:
            self._user32.SetForegroundWindow(window.handle)
        time.sleep(0.04)
        updated = self._window_info(window.handle)
        checks = {
            "minimize": bool(updated and updated.minimized),
            "maximize": bool(updated and updated.maximized),
            "restore": bool(updated and not updated.minimized and not updated.maximized),
            "focus": bool(updated and self._user32.GetForegroundWindow() == window.handle),
        }
        verified = checks.get(action, updated is not None)
        return WindowActionResult(
            bool(verified), action, updated or window, True,
            f"{action}d" if verified else f"{action}_verification_failed",
        )

    def focus(self, query: str = "", pids: Iterable[int] = ()) -> WindowActionResult:
        return self._show_action("focus", self.SW_RESTORE, query, pids)

    def minimize(self, query: str = "", pids: Iterable[int] = ()) -> WindowActionResult:
        return self._show_action("minimize", self.SW_MINIMIZE, query, pids)

    def maximize(self, query: str = "", pids: Iterable[int] = ()) -> WindowActionResult:
        return self._show_action("maximize", self.SW_MAXIMIZE, query, pids)

    def restore(self, query: str = "", pids: Iterable[int] = ()) -> WindowActionResult:
        return self._show_action("restore", self.SW_RESTORE, query, pids)

    def close(self, query: str = "", pids: Iterable[int] = ()) -> WindowActionResult:
        if not self._user32:
            return self._unsupported("close")
        window = self._resolve(query, pids)
        if window is None:
            return WindowActionResult(False, "close", None, True, "window_not_found")
        posted = bool(self._user32.PostMessageW(window.handle, self.WM_CLOSE, 0, 0))
        for _ in range(10):
            time.sleep(0.05)
            if not self._user32.IsWindow(window.handle):
                return WindowActionResult(True, "close", window, True, "closed")
        return WindowActionResult(
            False, "close", self._window_info(window.handle) or window, True,
            "still_open" if posted else "close_failed",
        )

    def move(
        self,
        query: str,
        x: int,
        y: int,
        pids: Iterable[int] = (),
    ) -> WindowActionResult:
        return self._move_resize("move", query, x=x, y=y, width=None, height=None, pids=pids)

    def resize(
        self,
        query: str,
        width: int,
        height: int,
        pids: Iterable[int] = (),
    ) -> WindowActionResult:
        return self._move_resize(
            "resize", query, x=None, y=None, width=width, height=height, pids=pids
        )

    def _move_resize(
        self,
        action: str,
        query: str,
        *,
        x: int | None,
        y: int | None,
        width: int | None,
        height: int | None,
        pids: Iterable[int],
    ) -> WindowActionResult:
        if not self._user32:
            return self._unsupported(action)
        window = self._resolve(query, pids)
        if window is None:
            return WindowActionResult(False, action, None, True, "window_not_found")
        target_x = window.left if x is None else max(-32768, min(32767, int(x)))
        target_y = window.top if y is None else max(-32768, min(32767, int(y)))
        target_width = window.width if width is None else max(120, min(16384, int(width)))
        target_height = window.height if height is None else max(80, min(16384, int(height)))
        changed = bool(
            self._user32.MoveWindow(
                window.handle, target_x, target_y, target_width, target_height, True
            )
        )
        updated = self._window_info(window.handle)
        verified = bool(
            changed
            and updated
            and updated.left == target_x
            and updated.top == target_y
            and abs(updated.width - target_width) <= 2
            and abs(updated.height - target_height) <= 2
        )
        return WindowActionResult(
            verified, action, updated or window, True,
            f"{action}d" if verified else f"{action}_verification_failed",
        )
