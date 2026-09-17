from __future__ import annotations

import platform


class ClipboardError(RuntimeError):
    pass


class ClipboardTool:
    """Small text-only clipboard adapter; no command shell is involved."""

    def __init__(self) -> None:
        self.system = platform.system().casefold()

    def read_text(self) -> str:
        if self.system == "windows":
            return self._windows_read()
        return self._tk_read()

    def write_text(self, text: str) -> dict[str, object]:
        value = str(text)
        if len(value.encode("utf-8")) > 1024 * 1024:
            raise ClipboardError("Clipboard text exceeds the 1 MB safety limit")
        if self.system == "windows":
            self._windows_write(value)
        else:
            self._tk_write(value)
        return {"success": True, "characters": len(value)}

    def clear(self) -> dict[str, object]:
        self.write_text("")
        return {"success": True, "characters": 0, "cleared": True}

    @staticmethod
    def _windows_read() -> str:
        try:
            import ctypes

            user32 = ctypes.windll.user32
            kernel32 = ctypes.windll.kernel32
            user32.GetClipboardData.restype = ctypes.c_void_p
            kernel32.GlobalLock.restype = ctypes.c_void_p
            if not user32.OpenClipboard(None):
                raise ClipboardError("Clipboard is busy")
            try:
                handle = user32.GetClipboardData(13)  # CF_UNICODETEXT
                if not handle:
                    return ""
                pointer = kernel32.GlobalLock(handle)
                if not pointer:
                    raise ClipboardError("Cannot lock clipboard data")
                try:
                    return ctypes.wstring_at(pointer)
                finally:
                    kernel32.GlobalUnlock(handle)
            finally:
                user32.CloseClipboard()
        except (AttributeError, OSError) as exc:
            raise ClipboardError(f"Cannot read clipboard: {exc}") from exc

    @staticmethod
    def _windows_write(text: str) -> None:
        try:
            import ctypes

            user32 = ctypes.windll.user32
            kernel32 = ctypes.windll.kernel32
            kernel32.GlobalAlloc.restype = ctypes.c_void_p
            kernel32.GlobalLock.restype = ctypes.c_void_p
            user32.SetClipboardData.restype = ctypes.c_void_p
            data = (text + "\0").encode("utf-16-le")
            handle = kernel32.GlobalAlloc(0x0002, len(data))  # GMEM_MOVEABLE
            if not handle:
                raise ClipboardError("Cannot allocate clipboard memory")
            pointer = kernel32.GlobalLock(handle)
            if not pointer:
                kernel32.GlobalFree(handle)
                raise ClipboardError("Cannot lock clipboard memory")
            ctypes.memmove(pointer, data, len(data))
            kernel32.GlobalUnlock(handle)
            if not user32.OpenClipboard(None):
                kernel32.GlobalFree(handle)
                raise ClipboardError("Clipboard is busy")
            try:
                user32.EmptyClipboard()
                if not user32.SetClipboardData(13, handle):
                    kernel32.GlobalFree(handle)
                    raise ClipboardError("Cannot set clipboard text")
                handle = None
            finally:
                user32.CloseClipboard()
        except (AttributeError, OSError) as exc:
            raise ClipboardError(f"Cannot write clipboard: {exc}") from exc

    @staticmethod
    def _tk_read() -> str:
        try:
            import tkinter as tk

            root = tk.Tk()
            root.withdraw()
            try:
                return root.clipboard_get()
            finally:
                root.destroy()
        except Exception as exc:
            raise ClipboardError(f"Cannot read clipboard: {exc}") from exc

    @staticmethod
    def _tk_write(text: str) -> None:
        try:
            import tkinter as tk

            root = tk.Tk()
            root.withdraw()
            root.clipboard_clear()
            root.clipboard_append(text)
            root.update()
            root.destroy()
        except Exception as exc:
            raise ClipboardError(f"Cannot write clipboard: {exc}") from exc
