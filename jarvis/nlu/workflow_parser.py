from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

from jarvis.nlu.action_parser import Action, ActionParser
from jarvis.utils.text import normalize_text


@dataclass(frozen=True, slots=True)
class FileWorkflowSpec:
    root: str
    extension: str
    modified: str
    order_by: str
    descending: bool
    limit: int
    final_action: str
    destination: str

    def to_dict(self) -> dict[str, object]:
        return {
            "root": self.root,
            "extension": self.extension,
            "modified": self.modified,
            "order_by": self.order_by,
            "descending": self.descending,
            "limit": self.limit,
            "final_action": self.final_action,
            "destination": self.destination,
        }


class FileWorkflowParser:
    """Parses search -> select -> act file workflows from composable constraints."""

    _FOLDERS = {
        "desktop": ("desktop", "دسکتاپ", "صفحه کار"),
        "downloads": ("downloads", "download", "دانلودها", "دانلود"),
        "documents": ("documents", "document", "اسناد"),
        "pictures": ("pictures", "تصاویر", "عکس ها"),
        "videos": ("videos", "ویدیوها", "فیلم ها"),
        "music": ("music", "موزیک", "آهنگ ها"),
    }
    _EXTENSIONS = {
        "pdf": ".pdf", "پی دی اف": ".pdf", "پی‌دی‌اف": ".pdf",
        "text": ".txt", "txt": ".txt", "متنی": ".txt",
        "python": ".py", "پایتون": ".py", "json": ".json",
        "zip": ".zip", "زیپ": ".zip", "markdown": ".md",
        "md": ".md", "csv": ".csv", "html": ".html",
    }
    _SEARCH = re.compile(r"(?:پیدا|بگرد|جستجو|find|locate|search)", re.I)
    _FINAL = re.compile(r"(?:منتقل|جابجا|ببر(?:ش)?|کپی|باز|حذف|move|copy|open|delete|remove)", re.I)
    _LARGEST = re.compile(r"(?:بزرگترین|بزرگ.?ترین|حجیم.?ترین|largest|biggest)", re.I)
    _SMALLEST = re.compile(r"(?:کوچکترین|کوچک.?ترین|کم.?حجم.?ترین|smallest)", re.I)
    _NEWEST = re.compile(r"(?:جدیدترین|تازه.?ترین|newest|latest)", re.I)
    _OLDEST = re.compile(r"(?:قدیمی.?ترین|oldest)", re.I)
    _TODAY = re.compile(r"(?:امروز|today|today's|modified\s+today)", re.I)
    _YESTERDAY = re.compile(r"(?:دیروز|yesterday)", re.I)

    @classmethod
    def _folder_mentions(cls, normalized: str) -> list[tuple[int, str, str]]:
        home = Path.home()
        output: list[tuple[int, str, str]] = []
        for folder_id, aliases in cls._FOLDERS.items():
            for alias in aliases:
                match = re.search(rf"(?<!\w){re.escape(alias)}(?!\w)", normalized, re.I)
                if match:
                    output.append((match.start(), folder_id, str(home / folder_id.title())))
                    break
        return sorted(output)

    @classmethod
    def parse(cls, text: str) -> FileWorkflowSpec | None:
        normalized = normalize_text(text)
        if not re.search(r"(?:فایل|file|pdf|txt|json|پایتون|python)", normalized, re.I):
            return None
        if not (cls._SEARCH.search(normalized) and cls._FINAL.search(normalized)):
            return None
        folders = cls._folder_mentions(normalized)
        if not folders:
            return None
        action_matches = list(cls._FINAL.finditer(normalized))
        final_match = action_matches[-1]
        search_matches = list(cls._SEARCH.finditer(normalized))
        if not search_matches or final_match.start() <= search_matches[-1].start():
            return None
        final_token = final_match.group(0)
        action = ActionParser.parse(final_token).action
        final_action = {
            Action.MOVE: "move_file", Action.COPY: "copy_file",
            Action.OPEN: "open_file", Action.DELETE: "delete_file",
        }.get(action, "")
        if not final_action:
            return None

        # Persian normally puts the destination before the final verb while
        # English puts it after it.  Mention order is stable in both forms:
        # search root first, destination last.
        source = folders[0]
        destination = (
            folders[-1][2]
            if len(folders) >= 2 and final_action in {"move_file", "copy_file"}
            else ""
        )
        if final_action in {"move_file", "copy_file"} and not destination:
            return None

        extension = ""
        for alias, candidate in cls._EXTENSIONS.items():
            if re.search(rf"(?<!\w){re.escape(alias)}(?:ها|s)?(?!\w)", normalized, re.I):
                extension = candidate
                break
        modified = "today" if cls._TODAY.search(normalized) else "yesterday" if cls._YESTERDAY.search(normalized) else "any"
        if cls._LARGEST.search(normalized):
            order_by, descending = "size", True
        elif cls._SMALLEST.search(normalized):
            order_by, descending = "size", False
        elif cls._NEWEST.search(normalized):
            order_by, descending = "modified", True
        elif cls._OLDEST.search(normalized):
            order_by, descending = "modified", False
        else:
            order_by, descending = "modified", True
        return FileWorkflowSpec(
            root=source[2], extension=extension, modified=modified,
            order_by=order_by, descending=descending, limit=1,
            final_action=final_action, destination=destination,
        )
