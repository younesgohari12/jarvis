from __future__ import annotations

import os
import platform
import shutil
import subprocess
import webbrowser
from pathlib import Path

from jarvis.entities.resolver import AppEntity


class DesktopToolError(RuntimeError):
    pass


class DesktopTool:
    """Allowlisted desktop launch operations; never accepts arbitrary commands."""

    def __init__(self) -> None:
        self.system = platform.system().casefold()

    @staticmethod
    def _known_windows_paths(app_id: str) -> tuple[Path, ...]:
        local = os.environ.get("LOCALAPPDATA", "")
        program_files = os.environ.get("ProgramFiles", "")
        program_files_x86 = os.environ.get("ProgramFiles(x86)", "")
        values: dict[str, tuple[tuple[str, str], ...]] = {
            "chrome": (
                (program_files, "Google/Chrome/Application/chrome.exe"),
                (program_files_x86, "Google/Chrome/Application/chrome.exe"),
                (local, "Google/Chrome/Application/chrome.exe"),
            ),
            "vscode": (
                (local, "Programs/Microsoft VS Code/Code.exe"),
                (program_files, "Microsoft VS Code/Code.exe"),
            ),
        }
        return tuple(
            Path(base) / relative for base, relative in values.get(app_id, ()) if base
        )

    def open_app(self, app: AppEntity) -> str:
        platform_key = "windows" if self.system == "windows" else "darwin" if self.system == "darwin" else "linux"
        candidates = app.commands.get(platform_key, ())
        if not candidates:
            raise DesktopToolError(f"No launcher is configured for {app.name} on this platform")
        if platform_key == "darwin":
            subprocess.Popen(
                ["open", "-a", candidates[0]],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            return app.name
        if platform_key == "windows":
            for known_path in self._known_windows_paths(app.entity_id):
                if known_path.is_file():
                    subprocess.Popen(
                        [str(known_path)], stdout=subprocess.DEVNULL,
                        stderr=subprocess.DEVNULL,
                    )
                    return app.name
        for candidate in candidates:
            executable = shutil.which(candidate)
            if executable:
                subprocess.Popen(
                    [executable], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
                )
                return app.name
            if self.system == "windows" and candidate.casefold().endswith(".exe"):
                try:
                    subprocess.Popen(
                        [candidate], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
                    )
                    return app.name
                except OSError:
                    continue
        raise DesktopToolError(f"{app.name} is not installed or is not available in PATH")

    @staticmethod
    def open_default_browser() -> bool:
        return bool(webbrowser.open("about:blank", new=2))

    def open_folder(self, raw_path: str | Path) -> str:
        if self.system == "windows" and str(raw_path) == "shell:MyComputerFolder":
            subprocess.Popen(
                ["explorer.exe", "shell:MyComputerFolder"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            return "This PC"
        path = Path(raw_path).expanduser().resolve()
        if not path.is_dir():
            raise DesktopToolError(f"Folder does not exist: {path}")
        if self.system == "windows":
            os.startfile(str(path))  # type: ignore[attr-defined]
        elif self.system == "darwin":
            subprocess.Popen(["open", str(path)], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        else:
            opener = shutil.which("xdg-open")
            if not opener:
                raise DesktopToolError("No desktop folder opener was found")
            subprocess.Popen([opener, str(path)], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return str(path)

    def open_file(self, raw_path: str | Path) -> str:
        path = Path(raw_path).expanduser().resolve()
        if not path.is_file():
            raise DesktopToolError(f"File does not exist: {path}")
        if self.system == "windows":
            os.startfile(str(path))  # type: ignore[attr-defined]
        elif self.system == "darwin":
            subprocess.Popen(["open", str(path)], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        else:
            opener = shutil.which("xdg-open")
            if not opener:
                raise DesktopToolError("No desktop file opener was found")
            subprocess.Popen([opener, str(path)], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return str(path)
