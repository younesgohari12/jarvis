from __future__ import annotations

import csv
import io
import os
import platform
import signal
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path

from jarvis.runtime.app_index import AppIndex, IndexedApp
from jarvis.tools.windows import WindowManager


@dataclass(frozen=True, slots=True)
class RunningProcess:
    pid: int
    name: str


@dataclass(frozen=True, slots=True)
class ProcessResult:
    success: bool
    app_id: str
    label: str
    action: str
    running: bool
    verified: bool
    code: str = ""
    detail: str = ""


class ProcessManager:
    """Allowlisted process operations; never accepts arbitrary shell commands."""

    def __init__(self, app_index: AppIndex, windows: WindowManager | None = None) -> None:
        self.app_index = app_index
        self.system = platform.system().casefold()
        self.windows = windows or WindowManager()

    def _processes(self) -> tuple[RunningProcess, ...]:
        try:
            if self.system == "windows":
                completed = subprocess.run(
                    ["tasklist", "/FO", "CSV", "/NH"], capture_output=True,
                    text=True, timeout=4.0, check=False, creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
                )
                rows = csv.reader(io.StringIO(completed.stdout))
                return tuple(
                    RunningProcess(int(row[1]), row[0])
                    for row in rows if len(row) >= 2 and row[1].replace(",", "").isdigit()
                )
            completed = subprocess.run(
                ["ps", "-eo", "pid=,comm="], capture_output=True,
                text=True, timeout=4.0, check=False,
            )
            result: list[RunningProcess] = []
            for line in completed.stdout.splitlines():
                parts = line.strip().split(None, 1)
                if len(parts) == 2 and parts[0].isdigit():
                    result.append(RunningProcess(int(parts[0]), Path(parts[1]).name))
            return tuple(result)
        except (OSError, subprocess.SubprocessError, ValueError):
            return ()

    @staticmethod
    def _matches(app: IndexedApp, process: RunningProcess) -> bool:
        name = process.name.casefold()
        candidates = {Path(value).name.casefold() for value in app.process_names}
        candidates.update(Path(value).name.casefold() for value in app.launchers)
        return name in candidates or Path(name).stem in {Path(value).stem for value in candidates}

    def is_app_running(self, app_query: str) -> bool:
        app = self.app_index.resolve(app_query, refresh_if_missing=False)
        return bool(app and any(self._matches(app, process) for process in self._processes()))

    def find_running_app(self, app_query: str) -> tuple[RunningProcess, ...]:
        app = self.app_index.resolve(app_query, refresh_if_missing=False)
        if app is None:
            return ()
        return tuple(process for process in self._processes() if self._matches(app, process))

    def list_running_apps(self) -> tuple[RunningProcess, ...]:
        return self._processes()

    def find_installed_app(self, query: str) -> tuple[IndexedApp, ...]:
        return self.app_index.find(query)

    def list_installed_apps(self, refresh: bool = False) -> tuple[IndexedApp, ...]:
        return self.app_index.all_apps(refresh=refresh)

    def default_browser_id(self) -> str:
        """Resolve the Windows HTTPS default without launching a browser."""
        if self.system == "windows":
            try:
                import winreg

                path = (
                    r"Software\Microsoft\Windows\Shell\Associations\UrlAssociations"
                    r"\https\UserChoice"
                )
                with winreg.OpenKey(winreg.HKEY_CURRENT_USER, path) as key:
                    prog_id = str(winreg.QueryValueEx(key, "ProgId")[0]).casefold()
                if "chrome" in prog_id:
                    return "chrome"
                if "msedge" in prog_id or prog_id.startswith("edge"):
                    return "edge"
                if "firefox" in prog_id:
                    return "firefox"
            except (OSError, AttributeError, ImportError):
                pass
        configured = Path(os.environ.get("BROWSER", "")).name.casefold()
        if "chrome" in configured:
            return "chrome"
        if "edge" in configured:
            return "edge"
        if "firefox" in configured:
            return "firefox"
        return ""

    def _launch(self, executable: str, extra_args: tuple[str, ...]) -> bool:
        try:
            if self.system == "windows" and executable.casefold().endswith(".lnk") and not extra_args:
                os.startfile(executable)  # type: ignore[attr-defined]
            else:
                subprocess.Popen(
                    [executable, *extra_args], stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL, close_fds=self.system != "windows",
                )
            return True
        except OSError:
            return False

    def open_app(self, app_query: str, extra_args: tuple[str, ...] = ()) -> ProcessResult:
        app = self.app_index.resolve(app_query)
        label = app.name if app else app_query.strip() or "application"
        app_id = app.app_id if app else app_query.strip().casefold()
        if app is None:
            return ProcessResult(False, app_id, label, "open", False, True, "not_found")
        launched = False
        for executable in self.app_index.executable_candidates(app):
            if self._launch(executable, extra_args):
                launched = True
                break
        if not launched:
            return ProcessResult(False, app.app_id, app.name, "open", False, True, "not_found")
        running = False
        for _ in range(5):
            time.sleep(0.08)
            if self.is_app_running(app.app_id):
                running = True
                break
        return ProcessResult(True, app.app_id, app.name, "open", running, running, "opened")

    def close_app(self, app_query: str) -> ProcessResult:
        app = self.app_index.resolve(app_query)
        label = app.name if app else app_query.strip() or "application"
        app_id = app.app_id if app else app_query.strip().casefold()
        if app is None:
            return ProcessResult(False, app_id, label, "close", False, True, "not_found")
        matches = self.find_running_app(app.app_id)
        if not matches:
            return ProcessResult(True, app.app_id, app.name, "close", False, True, "already_stopped")
        if self.system == "windows" and app.app_id == "explorer":
            closed_windows = self._close_visible_windows({process.pid for process in matches})
            return ProcessResult(
                closed_windows > 0,
                app.app_id,
                app.name,
                "close",
                True,
                closed_windows > 0,
                "windows_closed" if closed_windows else "no_window_found",
                f"closed_windows={closed_windows}",
            )
        if self.system == "windows":
            pids = {process.pid for process in matches}
            visible_before = self._visible_window_count(pids)
            if visible_before == 0 and app.app_id in {"chrome", "edge", "firefox"}:
                return ProcessResult(
                    True, app.app_id, app.name, "close", True, True,
                    "background_only", "No visible browser window remained",
                )
            if visible_before > 0:
                self._close_visible_windows(pids)
                for _ in range(12):
                    time.sleep(0.08)
                    visible_after = self._visible_window_count(pids)
                    running_after = self.is_app_running(app.app_id)
                    if not running_after:
                        return ProcessResult(
                            True, app.app_id, app.name, "close", False, True, "closed"
                        )
                    if visible_after == 0 and app.app_id in {"chrome", "edge", "firefox"}:
                        return ProcessResult(
                            True, app.app_id, app.name, "close", True, True,
                            "windows_closed", "Visible browser windows closed; background process remains",
                        )
        try:
            if self.system == "windows":
                process_names = tuple(dict.fromkeys(process.name for process in matches))
                self._taskkill_windows(process_names, force=False)
            else:
                for process in matches:
                    if process.pid != os.getpid():
                        os.kill(process.pid, signal.SIGTERM)
        except (OSError, subprocess.SubprocessError):
            return ProcessResult(False, app.app_id, app.name, "close", True, True, "close_failed")
        for _ in range(10):
            time.sleep(0.08)
            if not self.is_app_running(app.app_id):
                return ProcessResult(True, app.app_id, app.name, "close", False, True, "closed")
        if self.system == "windows":
            try:
                self._taskkill_windows(process_names, force=True)
            except (OSError, subprocess.SubprocessError):
                return ProcessResult(False, app.app_id, app.name, "close", True, True, "close_failed")
            for _ in range(10):
                time.sleep(0.08)
                if not self.is_app_running(app.app_id):
                    return ProcessResult(
                        True, app.app_id, app.name, "close", False, True, "closed_forced"
                    )
        return ProcessResult(False, app.app_id, app.name, "close", True, True, "still_running")

    @staticmethod
    def _taskkill_windows(process_names: tuple[str, ...], force: bool) -> None:
        for process_name in process_names:
            command = ["taskkill", "/IM", process_name, "/T"]
            if force:
                command.append("/F")
            subprocess.run(
                command, capture_output=True, timeout=4.0, check=False,
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
            )

    @staticmethod
    def _visible_window_count(pids: set[int]) -> int:
        try:
            import ctypes

            user32 = ctypes.windll.user32
            visible = 0

            @ctypes.WINFUNCTYPE(ctypes.c_bool, ctypes.c_void_p, ctypes.c_void_p)
            def callback(hwnd: int, _lparam: int) -> bool:
                nonlocal visible
                pid = ctypes.c_ulong()
                user32.GetWindowThreadProcessId(hwnd, ctypes.byref(pid))
                if pid.value in pids and user32.IsWindowVisible(hwnd):
                    visible += 1
                return True

            user32.EnumWindows(callback, 0)
            return visible
        except (AttributeError, OSError):
            return -1

    @staticmethod
    def _close_visible_windows(pids: set[int]) -> int:
        """Close Explorer windows with WM_CLOSE without terminating Windows Shell."""
        try:
            import ctypes

            user32 = ctypes.windll.user32
            closed = 0

            @ctypes.WINFUNCTYPE(ctypes.c_bool, ctypes.c_void_p, ctypes.c_void_p)
            def callback(hwnd: int, _lparam: int) -> bool:
                nonlocal closed
                pid = ctypes.c_ulong()
                user32.GetWindowThreadProcessId(hwnd, ctypes.byref(pid))
                if pid.value in pids and user32.IsWindowVisible(hwnd):
                    if user32.PostMessageW(hwnd, 0x0010, 0, 0):  # WM_CLOSE
                        closed += 1
                return True

            user32.EnumWindows(callback, 0)
            return closed
        except (AttributeError, OSError):
            return 0

    def close_apps(self, app_queries: tuple[str, ...]) -> tuple[ProcessResult, ...]:
        unique = tuple(dict.fromkeys(value.strip() for value in app_queries if value.strip()))
        return tuple(self.close_app(value) for value in unique)

    def close_all_browsers(self) -> tuple[ProcessResult, ...]:
        browsers = ("chrome", "edge", "firefox")
        return tuple(self.close_app(browser) for browser in browsers)

    def close_default_browser(self) -> ProcessResult:
        browser = self.default_browser_id()
        if not browser:
            return ProcessResult(
                False, "default_browser", "مرورگر پیش‌فرض", "close",
                False, True, "default_browser_unknown",
            )
        return self.close_app(browser)

    def restart_app(self, app_query: str) -> ProcessResult:
        closed = self.close_app(app_query)
        if not closed.success and closed.code not in {"already_stopped"}:
            return closed
        return self.open_app(app_query)

    def focus_app(self, app_query: str) -> ProcessResult:
        app = self.app_index.resolve(app_query, refresh_if_missing=False)
        label = app.name if app else app_query.strip() or "application"
        app_id = app.app_id if app else app_query.strip().casefold()
        if app is None:
            return ProcessResult(False, app_id, label, "focus", False, True, "not_found")
        running = self.is_app_running(app.app_id)
        if not running:
            return ProcessResult(False, app.app_id, app.name, "focus", False, True, "not_running")
        if self.system != "windows":
            return ProcessResult(True, app.app_id, app.name, "focus", True, False, "focus_unverified")
        pids = {process.pid for process in self.find_running_app(app.app_id)}
        window_result = self.windows.focus(app.name, pids)
        if window_result.success:
            return ProcessResult(True, app.app_id, app.name, "focus", True, True, "focused")
        try:
            import ctypes
            user32 = ctypes.windll.user32
            process_names = {Path(value).stem.casefold() for value in app.process_names}
            pids = {process.pid for process in self._processes() if Path(process.name).stem.casefold() in process_names}
            focused = False

            @ctypes.WINFUNCTYPE(ctypes.c_bool, ctypes.c_void_p, ctypes.c_void_p)
            def callback(hwnd: int, _lparam: int) -> bool:
                nonlocal focused
                pid = ctypes.c_ulong()
                user32.GetWindowThreadProcessId(hwnd, ctypes.byref(pid))
                if pid.value in pids and user32.IsWindowVisible(hwnd):
                    user32.ShowWindow(hwnd, 9)
                    focused = bool(user32.SetForegroundWindow(hwnd))
                    return False
                return True

            user32.EnumWindows(callback, 0)
            return ProcessResult(focused, app.app_id, app.name, "focus", True, True, "focused" if focused else "focus_failed")
        except (AttributeError, OSError):
            return ProcessResult(False, app.app_id, app.name, "focus", True, False, "focus_failed")

    def _window_state_action(self, app_query: str, action: str) -> ProcessResult:
        app = self.app_index.resolve(app_query, refresh_if_missing=False)
        label = app.name if app else app_query.strip() or "application"
        app_id = app.app_id if app else app_query.strip().casefold()
        if app is None:
            return ProcessResult(False, app_id, label, action, False, True, "not_found")
        matches = self.find_running_app(app.app_id)
        if not matches:
            return ProcessResult(False, app.app_id, app.name, action, False, True, "not_running")
        pids = {process.pid for process in matches}
        method = getattr(self.windows, action)
        outcome = method(app.name, pids)
        return ProcessResult(
            outcome.success, app.app_id, app.name, action, True,
            outcome.verified, outcome.code, outcome.detail,
        )

    def minimize_app(self, app_query: str) -> ProcessResult:
        return self._window_state_action(app_query, "minimize")

    def maximize_app(self, app_query: str) -> ProcessResult:
        return self._window_state_action(app_query, "maximize")

    def restore_app(self, app_query: str) -> ProcessResult:
        return self._window_state_action(app_query, "restore")
