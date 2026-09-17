from __future__ import annotations

import os
import re
import tempfile
from dataclasses import dataclass
from pathlib import Path, PureWindowsPath

from jarvis.utils.text import normalize_text


@dataclass(frozen=True, slots=True)
class FolderResolution:
    entity_type: str
    entity_id: str
    path: str
    label: str


class DriveResolver:
    _PERSIAN = {
        "سی": "C", "دی": "D", "ای": "E", "اف": "F", "جی": "G",
        "اچ": "H", "آی": "I", "جِی": "J", "کی": "K",
    }
    _LETTER = re.compile(
        r"(?:درایو|drive)\s*([a-z])\b|\b([a-z])\s*(?:درایو|drive)\b|"
        r"(?:داخل|توی|تو|در|in)\s+([a-z])(?::)?\b|^([a-z])(?:\s|$)",
        re.I,
    )

    @classmethod
    def resolve(cls, text: str) -> FolderResolution | None:
        normalized = normalize_text(text)
        if re.search(r"(?:این\s+کامپیوتر|this\s+pc|my\s+computer)", normalized, re.I):
            return FolderResolution("folder", "this_pc", "shell:MyComputerFolder", "This PC")
        match = cls._LETTER.search(normalized)
        letter = next((value for value in match.groups() if value), "") if match else ""
        if not letter and normalized in cls._PERSIAN:
            letter = cls._PERSIAN[normalized]
        if not letter and "درایو" in normalized:
            for word, candidate in cls._PERSIAN.items():
                if re.search(rf"(?:درایو\s*{re.escape(word)}|{re.escape(word)}\s*درایو)", normalized):
                    letter = candidate
                    break
        if not letter:
            return None
        drive = letter.upper()
        return FolderResolution("drive", drive, f"{drive}:\\", f"Drive {drive}")


class KnownFolderResolver:
    _ALIASES: dict[str, tuple[str, ...]] = {
        "desktop": ("desktop", "دسکتاپ", "صفحه کار"),
        "downloads": ("downloads", "download", "دانلودها", "دانلود"),
        "documents": ("documents", "document", "داکیومنت", "اسناد"),
        "pictures": ("pictures", "picture", "عکس ها", "تصاویر"),
        "videos": ("videos", "video", "ویدیوها", "فیلم ها"),
        "music": ("music", "موزیک", "آهنگ ها"),
        "home": ("home", "user folder", "پوشه کاربر", "خانه"),
        "appdata": ("appdata", "اپ دیتا", "اپدیتا"),
        "temp": ("temp", "temporary", "موقت", "تمپ"),
    }

    @staticmethod
    def _paths() -> dict[str, Path]:
        home = Path.home()
        return {
            "desktop": home / "Desktop",
            "downloads": home / "Downloads",
            "documents": home / "Documents",
            "pictures": home / "Pictures",
            "videos": home / "Videos",
            "music": home / "Music",
            "home": home,
            "appdata": Path(os.environ.get("APPDATA") or os.environ.get("LOCALAPPDATA") or home),
            "temp": Path(tempfile.gettempdir()),
        }

    @classmethod
    def resolve(cls, text: str) -> FolderResolution | None:
        normalized = normalize_text(text)
        paths = cls._paths()
        for folder_id, aliases in cls._ALIASES.items():
            if any(alias in normalized for alias in aliases):
                return FolderResolution("folder", folder_id, str(paths[folder_id]), folder_id.title())
        return None

    @classmethod
    def named_child(cls, text: str, base: str = "") -> FolderResolution | None:
        source = str(text).strip().replace("ي", "ی").replace("ك", "ک")
        match = re.search(
            r"(?:پوشه|فولدر|folder|directory)\s+([\w.‌ -]{1,80}?)(?=\s+(?:رو|را|باز|بخون|پیدا|داخل)|$)",
            source,
            re.I,
        )
        if not match:
            return None
        name = match.group(1).strip(" .")
        if not name or normalize_text(name) in {
            "یه", "یک", "a", "the", "باز", "باز کن", "بیار", "اجرا کن",
            "open", "launch", "start", "run",
        }:
            return None
        if re.match(r"^[A-Za-z]:[\\/]", base):
            path = str(PureWindowsPath(base) / name)
        else:
            parent = Path(base) if base else Path.home()
            path = str(parent / name)
        return FolderResolution("folder", name, path, name)
