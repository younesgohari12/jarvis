from __future__ import annotations

import ctypes
import platform
import time
from ctypes import wintypes
from dataclasses import dataclass
from typing import Iterable


@dataclass(frozen=True, slots=True)
class InputActionResult:
    success: bool
    action: str
    verified: bool
    code: str
    detail: str = ""


_ULONG_PTR = ctypes.c_ulonglong if ctypes.sizeof(ctypes.c_void_p) == 8 else ctypes.c_ulong


class _KeyboardInput(ctypes.Structure):
    _fields_ = (
        ("virtual_key", ctypes.c_ushort),
        ("scan_code", ctypes.c_ushort),
        ("flags", ctypes.c_ulong),
        ("time", ctypes.c_ulong),
        ("extra_info", _ULONG_PTR),
    )


class _MouseInput(ctypes.Structure):
    _fields_ = (
        ("dx", ctypes.c_long),
        ("dy", ctypes.c_long),
        ("mouse_data", ctypes.c_ulong),
        ("flags", ctypes.c_ulong),
        ("time", ctypes.c_ulong),
        ("extra_info", _ULONG_PTR),
    )


class _InputUnion(ctypes.Union):
    _fields_ = (("keyboard", _KeyboardInput), ("mouse", _MouseInput))


class _Input(ctypes.Structure):
    _anonymous_ = ("value",)
    _fields_ = (("type", ctypes.c_ulong), ("value", _InputUnion))


class InputController:
    """Win32 input fallback. UI Automation or browser DOM should be preferred."""

    INPUT_MOUSE = 0
    INPUT_KEYBOARD = 1
    KEYEVENTF_KEYUP = 0x0002
    KEYEVENTF_UNICODE = 0x0004
    MOUSEEVENTF_MOVE = 0x0001
    MOUSEEVENTF_LEFTDOWN = 0x0002
    MOUSEEVENTF_LEFTUP = 0x0004
    MOUSEEVENTF_RIGHTDOWN = 0x0008
    MOUSEEVENTF_RIGHTUP = 0x0010
    MOUSEEVENTF_WHEEL = 0x0800
    WHEEL_DELTA = 120

    KEYS = {
        "backspace": 0x08, "tab": 0x09, "enter": 0x0D, "shift": 0x10,
        "ctrl": 0x11, "control": 0x11, "alt": 0x12, "pause": 0x13,
        "capslock": 0x14, "escape": 0x1B, "esc": 0x1B, "space": 0x20,
        "pageup": 0x21, "pagedown": 0x22, "end": 0x23, "home": 0x24,
        "left": 0x25, "up": 0x26, "right": 0x27, "down": 0x28,
        "insert": 0x2D, "delete": 0x2E, "win": 0x5B, "windows": 0x5B,
        **{f"f{index}": 0x6F + index for index in range(1, 13)},
    }

    def __init__(self) -> None:
        self.is_windows = platform.system() == "Windows"
        self._user32 = ctypes.windll.user32 if self.is_windows else None

    @staticmethod
    def _result(
        success: bool, action: str, code: str, detail: str = "", verified: bool = False,
    ) -> InputActionResult:
        return InputActionResult(success, action, verified, code, detail)

    def _unsupported(self, action: str) -> InputActionResult:
        return self._result(
            False, action, "unsupported_platform",
            "Keyboard and mouse control requires an interactive Windows desktop.", True,
        )

    def _send(self, inputs: list[_Input]) -> bool:
        if not self._user32 or not inputs:
            return False
        array_type = _Input * len(inputs)
        sent = int(self._user32.SendInput(len(inputs), array_type(*inputs), ctypes.sizeof(_Input)))
        return sent == len(inputs)

    @staticmethod
    def _keyboard(virtual_key: int, scan_code: int = 0, flags: int = 0) -> _Input:
        return _Input(
            type=InputController.INPUT_KEYBOARD,
            value=_InputUnion(
                keyboard=_KeyboardInput(virtual_key, scan_code, flags, 0, 0)
            ),
        )

    @staticmethod
    def _mouse(flags: int, data: int = 0) -> _Input:
        return _Input(
            type=InputController.INPUT_MOUSE,
            value=_InputUnion(mouse=_MouseInput(0, 0, data, flags, 0, 0)),
        )

    @classmethod
    def _key_code(cls, value: str) -> int:
        key = str(value).strip().casefold()
        if key in cls.KEYS:
            return cls.KEYS[key]
        if len(key) == 1 and key.isascii():
            if "a" <= key <= "z":
                return ord(key.upper())
            if "0" <= key <= "9":
                return ord(key)
        raise ValueError(f"Unsupported key: {value}")

    def type_text(self, text: str) -> InputActionResult:
        if not self._user32:
            return self._unsupported("type_text")
        clean = str(text)
        if not clean or len(clean) > 4000:
            return self._result(False, "type_text", "invalid_text", verified=True)
        units = clean.encode("utf-16-le")
        events: list[_Input] = []
        for index in range(0, len(units), 2):
            code_unit = int.from_bytes(units[index : index + 2], "little")
            events.append(self._keyboard(0, code_unit, self.KEYEVENTF_UNICODE))
            events.append(
                self._keyboard(
                    0, code_unit, self.KEYEVENTF_UNICODE | self.KEYEVENTF_KEYUP
                )
            )
        success = self._send(events)
        return self._result(
            success, "type_text", "typed" if success else "send_input_failed",
            f"characters={len(clean)}", False,
        )

    def press_key(self, key: str, presses: int = 1) -> InputActionResult:
        if not self._user32:
            return self._unsupported("press_key")
        count = max(1, min(50, int(presses)))
        code = self._key_code(key)
        events: list[_Input] = []
        for _ in range(count):
            events.extend((self._keyboard(code), self._keyboard(code, flags=self.KEYEVENTF_KEYUP)))
        success = self._send(events)
        return self._result(
            success, "press_key", "pressed" if success else "send_input_failed",
            f"key={key}; presses={count}", False,
        )

    def hotkey(self, keys: Iterable[str]) -> InputActionResult:
        if not self._user32:
            return self._unsupported("hotkey")
        values = tuple(str(value) for value in keys)
        if not 2 <= len(values) <= 5:
            return self._result(False, "hotkey", "invalid_hotkey", verified=True)
        codes = [self._key_code(value) for value in values]
        events = [self._keyboard(code) for code in codes]
        events.extend(self._keyboard(code, flags=self.KEYEVENTF_KEYUP) for code in reversed(codes))
        success = self._send(events)
        return self._result(
            success, "hotkey", "pressed" if success else "send_input_failed",
            "+".join(values), False,
        )

    def move_mouse(self, x: int, y: int) -> InputActionResult:
        if not self._user32:
            return self._unsupported("move_mouse")
        width = int(self._user32.GetSystemMetrics(0))
        height = int(self._user32.GetSystemMetrics(1))
        target_x, target_y = int(x), int(y)
        if not 0 <= target_x < width or not 0 <= target_y < height:
            return self._result(False, "move_mouse", "coordinates_out_of_bounds", verified=True)
        success = bool(self._user32.SetCursorPos(target_x, target_y))
        point = wintypes.POINT()
        verified = bool(success)
        if self._user32.GetCursorPos(ctypes.byref(point)):
            verified = point.x == target_x and point.y == target_y
        return self._result(success and verified, "move_mouse", "moved" if verified else "move_failed", verified=True)

    def click(self, button: str = "left", count: int = 1) -> InputActionResult:
        if not self._user32:
            return self._unsupported("click")
        normalized = str(button).casefold().strip()
        flags = {
            "left": (self.MOUSEEVENTF_LEFTDOWN, self.MOUSEEVENTF_LEFTUP),
            "right": (self.MOUSEEVENTF_RIGHTDOWN, self.MOUSEEVENTF_RIGHTUP),
        }.get(normalized)
        if flags is None:
            return self._result(False, "click", "unsupported_button", verified=True)
        clicks = max(1, min(3, int(count)))
        success = True
        for index in range(clicks):
            success = self._send([self._mouse(flags[0]), self._mouse(flags[1])]) and success
            if index + 1 < clicks:
                time.sleep(0.06)
        return self._result(success, "click", "clicked" if success else "send_input_failed", verified=False)

    def scroll(self, amount: int) -> InputActionResult:
        if not self._user32:
            return self._unsupported("scroll")
        notches = max(-100, min(100, int(amount)))
        if notches == 0:
            return self._result(False, "scroll", "zero_scroll", verified=True)
        data = ctypes.c_ulong(notches * self.WHEEL_DELTA).value
        success = self._send([self._mouse(self.MOUSEEVENTF_WHEEL, data)])
        return self._result(success, "scroll", "scrolled" if success else "send_input_failed", verified=False)
