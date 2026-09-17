from __future__ import annotations

import re
from dataclasses import dataclass
from enum import StrEnum

from jarvis.utils.text import normalize_text
from jarvis.nlu.typo import TypoNormalizer


class Action(StrEnum):
    UNKNOWN = "unknown"
    OPEN = "open"
    CLOSE = "close"
    FOCUS = "focus"
    MINIMIZE = "minimize"
    MAXIMIZE = "maximize"
    RESTORE = "restore"
    RESTART = "restart"
    IS_RUNNING = "is_running"
    SEARCH = "search"
    READ = "read"
    FIND = "find"
    LIST = "list"
    DELETE = "delete"
    COPY = "copy"
    RENAME = "rename"
    MOVE = "move"
    WRITE = "write"
    APPEND = "append"
    CREATE = "create"
    RUN = "run"
    CLICK = "click"
    TYPE = "type"
    CHANGE = "change"
    RESEARCH = "research"
    ASK = "ask"


@dataclass(frozen=True, slots=True)
class ActionMatch:
    action: Action
    confidence: float
    evidence: str = ""


class ActionParser:
    """Verb-first parser. Destructive/closing polarity always wins over entities."""

    _PATTERNS: tuple[tuple[Action, re.Pattern[str]], ...] = (
        (
            Action.DELETE,
            re.compile(
                r"(?:حذف\s*(?:کن|کنید|کنش)?|پاک\s*(?:کن|کنید|کنش)?|"
                r"\b(?:delete|remove|erase|format)\b|فرمت\s*(?:کن)?)",
                re.I,
            ),
        ),
        (
            Action.RESTART,
            re.compile(
                r"(?:دوباره\s+(?:اجرا|باز)\s*(?:کن|کنش)?|ریستارت\s*(?:کن|کنش)?|"
                r"\b(?:restart|relaunch)\b)",
                re.I,
            ),
        ),
        (
            Action.CLOSE,
            re.compile(
                r"(?:ببند(?:ش|ید)?|بنداز(?:ش)?|خارج\s+شو|بسته\s*(?:کن|کنید)?|"
                r"\b(?:close|quit|terminate|kill|exit)\b)",
                re.I,
            ),
        ),
        (
            Action.FOCUS,
            re.compile(
                r"(?:بیار(?:ش)?\s+(?:جلو|بالا)|ببرش\s+جلو|روش\s+برو|برو\s+(?:روی|روش)|فعال\s*(?:کن|کنش)|"
                r"\b(?:focus|activate|switch\s+to)\b)",
                re.I,
            ),
        ),
        (
            Action.MINIMIZE,
            re.compile(
                r"(?:کوچیک(?:ش)?\s*(?:کن)?|کوچکش\s+کن|مینیمایز\s*(?:کن)?|"
                r"\bminimi[sz]e\b)", re.I,
            ),
        ),
        (
            Action.MAXIMIZE,
            re.compile(
                r"(?:بزرگ(?:ش)?\s*(?:کن)?|تمام\s+صفحه(?:ش)?\s*(?:کن)?|ماکسیمایز\s*(?:کن)?|"
                r"\bmaximi[sz]e\b)", re.I,
            ),
        ),
        (
            Action.RESTORE,
            re.compile(
                r"(?:برگردون(?:ش)?\s+به\s+حالت\s+عادی|حالت\s+عادی(?:ش)?\s*(?:کن)?|"
                r"\brestore(?:\s+window)?\b)", re.I,
            ),
        ),
        (
            Action.IS_RUNNING,
            re.compile(
                r"(?:باز(?:ه|\s+(?:هست|است))\??|فعال(?:ه|\s+(?:هست|است))\??|"
                r"در\s+حال\s+اجرا(?:ست|ه|\s+(?:هست|است))\??|"
                r"اجرا\s+میشه|\b(?:is|check)\b.*\b(?:running|open)\b)",
                re.I,
            ),
        ),
        (
            Action.RESEARCH,
            re.compile(
                r"(?:تحقیق\s*(?:کن|کنید)?|بررسی\s+عمیق\s*(?:کن)?|\bresearch\b)", re.I,
            ),
        ),
        (
            Action.SEARCH,
            re.compile(
                r"(?:سرچ\s*(?:کن|بزن|کنید)?|جستجو\s*(?:کن|کنید)?|گوگل\s*کن|"
                r"بگرد|\bbegard\b|\b(?:search|look\s+up)\b|^google(?=\s+\S+))",
                re.I,
            ),
        ),
        (
            Action.READ,
            re.compile(
                r"(?:بخون|بخوان|بخوانید|مطالعه\s*(?:کن)?|محتوا(?:ش|ی)?\s*(?:رو)?\s*نشون\s*بده|"
                r"\b(?:read|inspect|show\s+(?:me\s+)?(?:the\s+)?contents?)\b)",
                re.I,
            ),
        ),
        (
            Action.FIND,
            re.compile(
                r"(?:پیدا\s*(?:کن|کنید|کنش)?|دنبالش?\s+بگرد|"
                r"\b(?:find|locate)\b)",
                re.I,
            ),
        ),
        (
            Action.LIST,
            re.compile(
                r"(?:(?:لیست|فهرست)\s*(?:کن|بده|نشون\s+بده)?|"
                r"چه\s+(?:فایل|برنامه|چیز)(?:ایی|هایی)?|\blist\b)",
                re.I,
            ),
        ),
        (
            Action.CREATE,
            re.compile(
                r"(?:بساز(?:ش)?|ایجاد\s*(?:کن|کنید)?|درست(?:ش)?\s*(?:کن|کنید)?|"
                r"\bcreate\b|\bmake\s+(?:a|an|the)?\s*(?:file|folder|directory))",
                re.I,
            ),
        ),
        (
            Action.COPY,
            re.compile(r"(?:کپی\s*(?:کن|کنش)?|رونوشت\s*(?:بگیر)?|\bcopy\b)", re.I),
        ),
        (
            Action.RENAME,
            re.compile(r"(?:تغییر\s*نام\s*(?:بده|کن)?|اسمش\s*رو\s*بذار|\brename\b)", re.I),
        ),
        (
            Action.MOVE,
            re.compile(r"(?:منتقل\s*(?:کن)?|انتقال\s*(?:بده|کن)?|جابجا\s*(?:کن)?|ببر(?:ش)?(?:\s+به)?|\bmove\b)", re.I),
        ),
        (
            Action.APPEND,
            re.compile(r"(?:ته(?:ش)?\s+اضافه\s*(?:کن)?|به\s+آخر(?:ش)?\s+اضافه\s*(?:کن)?|\bappend\b)", re.I),
        ),
        (
            Action.TYPE,
            re.compile(r"(?:تایپ\s*(?:کن|کنید)?|وارد\s*(?:کن|کنید)?|\btype\b)", re.I),
        ),
        (
            Action.CLICK,
            re.compile(r"(?:کلیک\s*(?:کن|کنید)?|دابل\s*کلیک|راست\s*کلیک|\bclick\b)", re.I),
        ),
        (
            Action.CHANGE,
            re.compile(r"(?:تغییر\s*(?:بده|کن)?|عوض\s*(?:کن)?|\bchange\b|\bset\b)", re.I),
        ),
        (
            Action.RUN,
            re.compile(
                r"(?:(?:دستور|command|terminal|ترمینال|cmd|powershell).*(?:اجرا|run)|"
                r"\b(?:run|execute)\s+(?:the\s+)?command\b)", re.I,
            ),
        ),
        (
            Action.WRITE,
            re.compile(r"(?:بنویس|ذخیره\s*(?:کن)?|\b(?:write|save)\b)", re.I),
        ),
        (
            Action.OPEN,
            re.compile(
                r"(?:باز\s*(?:کن|کنید|کنش)?|بازش\s+کن|بیار(?:ش|ید)?|"
                r"اجرا\s*(?:کن|کنید|کنش)?|برو(?:\s+تو|\s+داخل)?|"
                r"\b(?:open|launch|start|run|bring\s+up|go\s+to)\b)",
                re.I,
            ),
        ),
    )

    @classmethod
    def parse(cls, text: str) -> ActionMatch:
        normalized = normalize_text(TypoNormalizer.correct(text))
        if re.search(r"\b(?:baz|vaaz)\s+kon\b|\bejra\s+kon\b", normalized, re.I):
            return ActionMatch(Action.OPEN, 0.94, "finglish_open")
        for action, pattern in cls._PATTERNS:
            match = pattern.search(normalized)
            if match:
                return ActionMatch(action, 0.99, match.group(0))
        return ActionMatch(Action.UNKNOWN, 0.0)

    @classmethod
    def has_action(cls, text: str) -> bool:
        return cls.parse(text).action is not Action.UNKNOWN
