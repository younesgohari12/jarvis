from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path, PureWindowsPath
from typing import Any

from jarvis.tools.drives import DriveResolver, KnownFolderResolver
from jarvis.utils.text import normalize_text


_PERSIAN_DIGITS = str.maketrans("۰۱۲۳۴۵۶۷۸۹٠١٢٣٤٥٦٧٨٩", "01234567890123456789")


@dataclass(frozen=True, slots=True)
class StructuredEntity:
    kind: str
    value: str
    normalized: str
    confidence: float
    span: tuple[int, int] = (0, 0)

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "value": self.value,
            "normalized": self.normalized,
            "confidence": round(self.confidence, 4),
            "span": list(self.span),
        }


@dataclass(frozen=True, slots=True)
class FileRequest:
    entity_type: str
    name: str
    path: str
    base: str
    extension: str
    content: str
    drive: str = ""
    referenced: bool = False

    @property
    def complete(self) -> bool:
        return bool(self.path)


class PersianNumberParser:
    """Small deterministic number normalizer for NLU slots, not response logic."""

    _ONES = {
        "صفر": 0, "یک": 1, "يه": 1, "یه": 1, "دو": 2, "سه": 3,
        "چهار": 4, "پنج": 5, "شش": 6, "شیش": 6, "هفت": 7,
        "هشت": 8, "نه": 9, "ده": 10, "یازده": 11, "دوازده": 12,
        "سیزده": 13, "چهارده": 14, "پانزده": 15, "شانزده": 16,
        "هفده": 17, "هجده": 18, "نوزده": 19,
    }
    _TENS = {
        "بیست": 20, "سی": 30, "چهل": 40, "پنجاه": 50,
        "شصت": 60, "هفتاد": 70, "هشتاد": 80, "نود": 90,
    }
    _SCALES = {"صد": 100, "هزار": 1_000, "میلیون": 1_000_000}

    @classmethod
    def parse(cls, value: str) -> int | float | None:
        normalized = normalize_text(value).translate(_PERSIAN_DIGITS)
        numeric = re.fullmatch(r"[-+]?\d+(?:[.,]\d+)?", normalized)
        if numeric:
            number = float(normalized.replace(",", "."))
            return int(number) if number.is_integer() else number
        words = [word for word in re.split(r"[\s‌-]+", normalized) if word != "و"]
        if not words:
            return None
        total = current = 0
        consumed = False
        for word in words:
            if word in cls._ONES:
                current += cls._ONES[word]
                consumed = True
            elif word in cls._TENS:
                current += cls._TENS[word]
                consumed = True
            elif word == "صد":
                current = max(1, current) * 100
                consumed = True
            elif word in {"هزار", "میلیون"}:
                total += max(1, current) * cls._SCALES[word]
                current = 0
                consumed = True
            else:
                return None
        return total + current if consumed else None

    @classmethod
    def find(cls, text: str) -> int | float | None:
        normalized = normalize_text(text).translate(_PERSIAN_DIGITS)
        numeric = re.search(r"(?<!\w)[-+]?\d+(?:[.,]\d+)?(?!\w)", normalized)
        if numeric:
            return cls.parse(numeric.group(0))
        words = re.findall(r"[؀-ۿ]+", normalized)
        for width in range(min(5, len(words)), 0, -1):
            for start in range(0, len(words) - width + 1):
                result = cls.parse(" ".join(words[start : start + width]))
                if result is not None:
                    return result
        return None


class EntityExtractorV7:
    """Persian/English entity extraction for desktop requests.

    The extractor returns typed data only. Routing and tool policy remain separate,
    which keeps it useful for new tools and avoids query-specific command rules.
    """

    _WINDOWS_PATH = re.compile(
        r"(?P<path>[A-Za-z]:[\\/](?:[^\n\r\"'<>|?*]|[?*](?=[^\\/]*$))*)"
    )
    _POSIX_PATH = re.compile(r"(?P<path>(?:~|\.{1,2})?/(?:[^\n\r\"']+))")
    _URL = re.compile(r"\b(?:https?://|www\.)[^\s<>\"']+", re.I)
    _EXTENSION = re.compile(r"(?<!\w)(\.[A-Za-z0-9]{1,12})\b")
    _PERCENT = re.compile(r"([۰-۹٠-٩\d]{1,3})\s*(?:%|درصد|percent)\b", re.I)
    _EDUCATIONAL = re.compile(
        r"(?:چطور|چگونه|چه طور|روش|آموزش|مثال|نمونه کد|در پایتون|در جاوا|"
        r"how\s+(?:do|can|to)|show\s+me\s+how|example|tutorial|in\s+python)" , re.I,
    )
    _QUESTION_END = re.compile(r"(?:[؟?]|چیه|چیست|میشه|می شود|does it work|mean)\s*$", re.I)
    _DRY_RUN = re.compile(
        r"(?:اجرا\s+نکن|انجام\s+نده|فقط\s+(?:برنامه|مراحل|پیش.?نمایش)|"
        r"بدون\s+اجرا|حالت\s+آزمایشی|dry[ -]?run|do\s+not\s+execute|preview\s+only)", re.I,
    )
    _FILE_WORD = re.compile(r"(?:فایل(?:ی|ها|های)?|file(?:s)?)", re.I)
    _FOLDER_WORD = re.compile(r"(?:پوشه(?:ای|ها|های)?|فولدر(?:ی|ها|های)?|folder(?:s)?|director(?:y|ies))", re.I)
    _ACTION_SUBJECT = re.compile(
        r"(?:فایل|file|پوشه|فولدر|folder|directory|حذف|پاک.?کردن|delete|remove|"
        r"خاموش|ریستارت|shutdown|restart|command|دستور|terminal|ترمینال)", re.I,
    )
    _CREATE_NAME_PATTERNS = (
        re.compile(
            r"(?:فایل(?:ی)?|پوشه(?:ای)?|فولدر(?:ی)?|file|folder|directory)\s+"
            r"(?:به\s+(?:نام|اسم)|با\s+(?:نام|اسم)|موسوم\s+به|named|called)\s+[\"«']?"
            r"(?P<name>[^\"»'،,]+?)[\"»']?(?=\s+(?:بساز|ایجاد|درست|create|make|داخل|در|تو|توی|on|in)|$)",
            re.I,
        ),
        re.compile(
            r"(?:به\s+(?:نام|اسم)|موسوم\s+به|named|called)\s+[\"«']?(?P<name>[^\"»'،,]+?)"
            r"[\"»']?(?=\s+(?:بساز|ایجاد|درست|create|make)|$)", re.I,
        ),
        re.compile(
            r"(?:create|make)\s+(?:a|an|the)?\s*(?:new\s+)?(?:file|folder|directory)\s+"
            r"[\"']?(?P<name>[\w .+_-]{1,100})[\"']?(?=\s+(?:in|on|inside|at)|$)", re.I,
        ),
    )
    _TRAILING = re.compile(
        r"\s+(?:رو|را|داخل|در|تو|توی|روی|on|in|بساز|ایجاد\s+کن|درست\s+کن|create|make).*$",
        re.I,
    )

    @classmethod
    def is_educational(cls, text: str) -> bool:
        normalized = normalize_text(text)
        if not (cls._EDUCATIONAL.search(normalized) and cls._ACTION_SUBJECT.search(normalized)):
            return False
        # Dataset and natural-language questions often append an example number
        # after the question mark (or omit punctuation entirely).  Explicit
        # how-to/code wording still has interrogative polarity and must never be
        # interpreted as permission to execute the mentioned destructive verb.
        return bool(
            cls._QUESTION_END.search(normalized)
            or re.search(
                r"(?:چطور|چگونه|چه طور|روش|آموزش|مثال|نمونه کد|در پایتون|"
                r"how\s+(?:do|can|to)|show\s+me\s+how|example|tutorial|in\s+python)",
                normalized,
                re.I,
            )
        )

    @classmethod
    def is_dry_run(cls, text: str) -> bool:
        return bool(cls._DRY_RUN.search(normalize_text(text)))

    @classmethod
    def _explicit_path(cls, text: str) -> str:
        match = cls._WINDOWS_PATH.search(text) or cls._POSIX_PATH.search(text)
        if not match:
            return ""
        return match.group("path").rstrip(" .،")

    @classmethod
    def _name(cls, text: str, entity_type: str) -> str:
        for pattern in cls._CREATE_NAME_PATTERNS:
            match = pattern.search(text)
            if match:
                candidate = cls._TRAILING.sub("", match.group("name")).strip(" .،:؛")
                if candidate:
                    return Path(candidate).name
        noun = r"(?:فایل(?:ی)?|file)" if entity_type == "file" else r"(?:پوشه(?:ای)?|فولدر(?:ی)?|folder|directory)"
        match = re.search(
            rf"{noun}\s+[\"«']?(?P<name>[\w.+‌-]{{1,100}})[\"»']?"
            rf"(?=\s+(?:رو|را)?\s*(?:بساز|ایجاد|درست|create|make)|$)",
            text,
            re.I,
        )
        if not match:
            return ""
        candidate = match.group("name").strip(" .،")
        ignored = {"یک", "یه", "جدید", "new", "a", "an", "the"}
        return "" if normalize_text(candidate) in ignored else Path(candidate).name

    @staticmethod
    def _join(base: str, name: str) -> str:
        if re.match(r"^[A-Za-z]:[\\/]", base):
            return str(PureWindowsPath(base) / name)
        return str(Path(base) / name)

    @classmethod
    def file_request(
        cls,
        text: str,
        context: dict[str, Any] | None = None,
        *,
        default_text_extension: str = ".txt",
    ) -> FileRequest | None:
        normalized = normalize_text(text)
        has_file = bool(cls._FILE_WORD.search(normalized))
        has_folder = bool(cls._FOLDER_WORD.search(normalized))
        if not (has_file or has_folder):
            return None
        entity_type = "folder" if has_folder and not has_file else "file"
        context = context or {}
        explicit_path = cls._explicit_path(text)
        drive = DriveResolver.resolve(text)
        known = KnownFolderResolver.resolve(text)
        base = drive.path if drive else known.path if known else ""
        if not base:
            base = str(context.get("last_folder", ""))
        name = cls._name(text, entity_type)
        extension = Path(name).suffix.casefold() if name else ""
        if entity_type == "file" and name and not extension:
            extension = default_text_extension
            name += extension
        path = explicit_path or (cls._join(base, name) if base and name else "")
        if explicit_path and not name:
            name = PureWindowsPath(explicit_path).name if re.match(r"^[A-Za-z]:", explicit_path) else Path(explicit_path).name
            extension = Path(name).suffix.casefold()
        content_match = re.search(
            r"(?:با\s+محتوای|محتوا(?:ش)?\s*[:=]|with\s+(?:the\s+)?content\s*[:=]?)\s*(.+)$",
            text,
            re.I,
        )
        return FileRequest(
            entity_type=entity_type,
            name=name,
            path=path,
            base=base,
            extension=extension,
            content=content_match.group(1).strip() if content_match else "",
            drive=drive.entity_id if drive else "",
            referenced=not bool(explicit_path or drive or known) and bool(context.get("last_folder")),
        )

    @classmethod
    def extract(cls, text: str, context: dict[str, Any] | None = None) -> tuple[StructuredEntity, ...]:
        normalized = normalize_text(text)
        entities: list[StructuredEntity] = []
        for kind, pattern in (("url", cls._URL), ("path", cls._WINDOWS_PATH), ("path", cls._POSIX_PATH)):
            for match in pattern.finditer(text):
                value = match.group(0).rstrip(".،")
                entities.append(StructuredEntity(kind, value, normalize_text(value), 0.99, match.span()))
        drive = DriveResolver.resolve(text)
        if drive:
            entities.append(StructuredEntity("drive", drive.entity_id, drive.path, 0.99))
        known = KnownFolderResolver.resolve(text)
        if known:
            entities.append(StructuredEntity("folder", known.entity_id, known.path, 0.98))
        request = cls.file_request(text, context)
        if request:
            if request.name:
                entities.append(StructuredEntity(request.entity_type, request.name, request.path or request.name, 0.96))
            if request.extension:
                entities.append(StructuredEntity("extension", request.extension, request.extension, 0.99))
        for match in cls._PERCENT.finditer(normalized.translate(_PERSIAN_DIGITS)):
            entities.append(StructuredEntity("percentage", match.group(1), match.group(1), 0.99, match.span()))
        for match in re.finditer(r"(?<!\w)[۰-۹٠-٩\d]+(?:[.,][۰-۹٠-٩\d]+)?(?!\w)", text):
            value = match.group(0).translate(_PERSIAN_DIGITS)
            entities.append(StructuredEntity("number", value, value.replace(",", "."), 0.98, match.span()))
        if cls.is_dry_run(normalized):
            entities.append(StructuredEntity("execution_mode", "dry_run", "dry_run", 1.0))
        return tuple(entities)
