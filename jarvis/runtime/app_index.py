from __future__ import annotations

import json
import os
import platform
import shutil
import time
from dataclasses import dataclass
from difflib import SequenceMatcher
from pathlib import Path
from typing import Iterable

from jarvis.entities.resolver import AppEntity
from jarvis.utils.text import normalize_text


@dataclass(frozen=True, slots=True)
class IndexedApp:
    app_id: str
    name: str
    aliases: tuple[str, ...]
    launchers: tuple[str, ...]
    process_names: tuple[str, ...]
    source: str = "registry"


class AppIndex:
    """Lazy, bounded Windows app discovery with a persistent JSON cache."""

    CACHE_VERSION = 1
    CACHE_TTL_SECONDS = 7 * 24 * 60 * 60
    _PROCESSES: dict[str, tuple[str, ...]] = {
        "chrome": ("chrome.exe", "chrome"),
        "edge": ("msedge.exe", "msedge"),
        "firefox": ("firefox.exe", "firefox"),
        "steam": ("steam.exe", "steam"),
        "discord": ("discord.exe", "Discord"),
        "telegram": ("telegram.exe", "Telegram"),
        "spotify": ("spotify.exe", "Spotify"),
        "vscode": ("code.exe", "code"),
        "notepad": ("notepad.exe", "notepad"),
        "calculator": ("calculatorapp.exe", "calc.exe", "gnome-calculator"),
        "explorer": ("explorer.exe", "explorer"),
        "powershell": ("powershell.exe", "pwsh.exe", "pwsh"),
        "cmd": ("cmd.exe", "cmd"),
        "paint": ("mspaint.exe", "mspaint"),
    }
    _KNOWN_WINDOWS: dict[str, tuple[str, ...]] = {
        "chrome": (
            r"%ProgramFiles%\Google\Chrome\Application\chrome.exe",
            r"%ProgramFiles(x86)%\Google\Chrome\Application\chrome.exe",
            r"%LOCALAPPDATA%\Google\Chrome\Application\chrome.exe",
        ),
        "edge": (
            r"%ProgramFiles(x86)%\Microsoft\Edge\Application\msedge.exe",
            r"%ProgramFiles%\Microsoft\Edge\Application\msedge.exe",
        ),
        "firefox": (
            r"%ProgramFiles%\Mozilla Firefox\firefox.exe",
            r"%ProgramFiles(x86)%\Mozilla Firefox\firefox.exe",
        ),
        "steam": (
            r"%ProgramFiles(x86)%\Steam\steam.exe",
            r"%ProgramFiles%\Steam\steam.exe",
        ),
        "discord": (r"%LOCALAPPDATA%\Discord\Update.exe",),
        "telegram": (
            r"%APPDATA%\Telegram Desktop\Telegram.exe",
            r"%LOCALAPPDATA%\Programs\Telegram Desktop\Telegram.exe",
        ),
        "spotify": (r"%APPDATA%\Spotify\Spotify.exe",),
        "vscode": (
            r"%LOCALAPPDATA%\Programs\Microsoft VS Code\Code.exe",
            r"%ProgramFiles%\Microsoft VS Code\Code.exe",
        ),
    }

    def __init__(self, cache_path: Path, entities: Iterable[AppEntity]) -> None:
        self.cache_path = cache_path
        self.system = platform.system().casefold()
        self._static = tuple(self._from_entity(entity) for entity in entities)
        self._dynamic: tuple[IndexedApp, ...] | None = None

    def _from_entity(self, entity: AppEntity) -> IndexedApp:
        platform_key = "windows" if self.system == "windows" else "darwin" if self.system == "darwin" else "linux"
        launchers = list(entity.commands.get(platform_key, ()))
        if self.system == "windows":
            for raw in self._KNOWN_WINDOWS.get(entity.entity_id, ()):
                expanded = os.path.expandvars(raw)
                if expanded and expanded not in launchers:
                    launchers.insert(0, expanded)
        processes = self._PROCESSES.get(entity.entity_id, tuple(Path(value).name for value in launchers))
        return IndexedApp(
            entity.entity_id,
            entity.name,
            entity.aliases,
            tuple(dict.fromkeys(launchers)),
            tuple(dict.fromkeys(processes)),
            "configured",
        )

    @staticmethod
    def _score(query: str, app: IndexedApp) -> float:
        values = (app.app_id, normalize_text(app.name), *app.aliases)
        if query in values:
            return 1.0
        if any(query in value or value in query for value in values if len(value) >= 3):
            return 0.91
        return max(SequenceMatcher(None, query, value).ratio() for value in values)

    def _load_cache(self) -> tuple[IndexedApp, ...]:
        if self._dynamic is not None:
            return self._dynamic
        try:
            payload = json.loads(self.cache_path.read_text(encoding="utf-8"))
            age = time.time() - float(payload.get("created_at", 0.0))
            if int(payload.get("version", 0)) != self.CACHE_VERSION or age > self.CACHE_TTL_SECONDS:
                raise ValueError("stale app cache")
            self._dynamic = tuple(
                IndexedApp(
                    str(item["id"]), str(item["name"]),
                    tuple(str(value) for value in item.get("aliases", [])),
                    tuple(str(value) for value in item.get("launchers", [])),
                    tuple(str(value) for value in item.get("processes", [])),
                    str(item.get("source", "cache")),
                )
                for item in payload.get("apps", [])
            )
        except (OSError, ValueError, TypeError, KeyError, json.JSONDecodeError):
            self._dynamic = ()
        return self._dynamic

    @staticmethod
    def _clean_display_icon(value: str) -> str:
        return value.strip().strip('"').split(",", 1)[0].strip().strip('"')

    def _registry_apps(self) -> list[IndexedApp]:
        if self.system != "windows":
            return []
        try:
            import winreg  # type: ignore[import-not-found]
        except ImportError:
            return []
        found: list[IndexedApp] = []
        roots = (winreg.HKEY_CURRENT_USER, winreg.HKEY_LOCAL_MACHINE)
        locations = (
            r"SOFTWARE\Microsoft\Windows\CurrentVersion\App Paths",
            r"SOFTWARE\Microsoft\Windows\CurrentVersion\Uninstall",
            r"SOFTWARE\WOW6432Node\Microsoft\Windows\CurrentVersion\Uninstall",
        )
        for root in roots:
            for location in locations:
                try:
                    parent = winreg.OpenKey(root, location)
                except OSError:
                    continue
                try:
                    count = min(2500, winreg.QueryInfoKey(parent)[0])
                    for index in range(count):
                        try:
                            child_name = winreg.EnumKey(parent, index)
                            child = winreg.OpenKey(parent, child_name)
                            try:
                                if location.endswith("App Paths"):
                                    launcher = str(winreg.QueryValue(child, None))
                                    name = Path(child_name).stem
                                else:
                                    name = str(winreg.QueryValueEx(child, "DisplayName")[0])
                                    launcher = self._clean_display_icon(str(winreg.QueryValueEx(child, "DisplayIcon")[0]))
                            finally:
                                winreg.CloseKey(child)
                            if name and launcher and Path(launcher).suffix.casefold() in {".exe", ".lnk"}:
                                app_id = normalize_text(name).replace(" ", "-")[:80]
                                found.append(
                                    IndexedApp(
                                        app_id, name, (normalize_text(name),), (launcher,),
                                        (Path(launcher).name,), "windows_registry",
                                    )
                                )
                        except OSError:
                            continue
                finally:
                    winreg.CloseKey(parent)
        return found

    def _shortcut_apps(self) -> list[IndexedApp]:
        if self.system != "windows":
            return []
        roots = tuple(
            Path(os.path.expandvars(value))
            for value in (
                r"%APPDATA%\Microsoft\Windows\Start Menu\Programs",
                r"%ProgramData%\Microsoft\Windows\Start Menu\Programs",
            )
            if os.path.expandvars(value) != value
        )
        found: list[IndexedApp] = []
        for root in roots:
            if not root.is_dir():
                continue
            for index, path in enumerate(root.rglob("*.lnk")):
                if index >= 2500:
                    break
                name = path.stem
                found.append(
                    IndexedApp(
                        normalize_text(name).replace(" ", "-")[:80], name,
                        (normalize_text(name),), (str(path),), (), "start_menu",
                    )
                )
        return found

    def _filesystem_apps(self) -> list[IndexedApp]:
        if self.system != "windows":
            return []
        roots = tuple(
            Path(value)
            for value in (
                os.environ.get("ProgramFiles", ""),
                os.environ.get("ProgramFiles(x86)", ""),
                str(Path(os.environ.get("LOCALAPPDATA", "")) / "Programs")
                if os.environ.get("LOCALAPPDATA") else "",
            )
            if value and Path(value).is_dir()
        )
        found: list[IndexedApp] = []
        seen = 0
        for root in roots:
            for pattern in ("*/*.exe", "*/*/*.exe"):
                for path in root.glob(pattern):
                    seen += 1
                    if seen > 3000:
                        return found
                    name = path.stem
                    if name.casefold() in {"uninstall", "unins000", "setup", "update", "updater"}:
                        continue
                    found.append(
                        IndexedApp(
                            normalize_text(name).replace(" ", "-")[:80], name,
                            (normalize_text(name), normalize_text(path.parent.name)),
                            (str(path),), (path.name,), "program_files",
                        )
                    )
        return found

    def refresh(self) -> tuple[IndexedApp, ...]:
        dynamic = self._registry_apps() + self._shortcut_apps() + self._filesystem_apps()
        unique: dict[tuple[str, str], IndexedApp] = {}
        for app in dynamic:
            key = (normalize_text(app.name), app.launchers[0] if app.launchers else "")
            unique[key] = app
        self._dynamic = tuple(unique.values())
        payload = {
            "version": self.CACHE_VERSION,
            "created_at": time.time(),
            "apps": [
                {
                    "id": app.app_id, "name": app.name, "aliases": list(app.aliases),
                    "launchers": list(app.launchers), "processes": list(app.process_names),
                    "source": app.source,
                }
                for app in self._dynamic
            ],
        }
        try:
            self.cache_path.parent.mkdir(parents=True, exist_ok=True)
            self.cache_path.write_text(
                json.dumps(payload, ensure_ascii=False, separators=(",", ":")), encoding="utf-8"
            )
        except OSError:
            pass
        return self._dynamic

    def resolve(self, query: str, *, refresh_if_missing: bool = True) -> IndexedApp | None:
        normalized = normalize_text(query).strip()
        if not normalized:
            return None
        candidates = list(self._static) + list(self._load_cache())
        ranked = sorted(
            ((self._score(normalized, app), app) for app in candidates),
            key=lambda item: item[0], reverse=True,
        )
        if ranked and ranked[0][0] >= 0.72:
            return ranked[0][1]
        if refresh_if_missing and self.system == "windows":
            refreshed = self.refresh()
            ranked = sorted(
                ((self._score(normalized, app), app) for app in (*self._static, *refreshed)),
                key=lambda item: item[0], reverse=True,
            )
            if ranked and ranked[0][0] >= 0.72:
                return ranked[0][1]
        return None

    def find(self, query: str, maximum_results: int = 10) -> tuple[IndexedApp, ...]:
        normalized = normalize_text(query).strip()
        if not normalized:
            return ()
        candidates = list(self._static) + list(self._load_cache())
        if self.system == "windows" and not self._load_cache():
            candidates.extend(self.refresh())
        unique: dict[tuple[str, tuple[str, ...]], IndexedApp] = {
            (normalize_text(app.name), app.launchers): app for app in candidates
        }
        ranked = sorted(
            ((self._score(normalized, app), app) for app in unique.values()),
            key=lambda item: (item[0], item[1].source == "configured"),
            reverse=True,
        )
        return tuple(
            app for score, app in ranked[: max(1, min(50, int(maximum_results)))]
            if score >= 0.42
        )

    def all_apps(self, *, refresh: bool = False) -> tuple[IndexedApp, ...]:
        dynamic = self.refresh() if refresh else self._load_cache()
        unique: dict[tuple[str, tuple[str, ...]], IndexedApp] = {}
        for app in (*self._static, *dynamic):
            unique[(normalize_text(app.name), app.launchers)] = app
        return tuple(sorted(unique.values(), key=lambda app: app.name.casefold()))

    @staticmethod
    def executable_candidates(app: IndexedApp) -> tuple[str, ...]:
        candidates: list[str] = []
        for value in app.launchers:
            expanded = os.path.expandvars(value)
            if Path(expanded).is_file():
                candidates.append(expanded)
                continue
            located = shutil.which(expanded)
            if located:
                candidates.append(located)
            elif platform.system() == "Windows" and expanded.casefold().endswith(".exe"):
                candidates.append(expanded)
        return tuple(dict.fromkeys(candidates))
