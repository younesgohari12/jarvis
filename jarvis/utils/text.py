from __future__ import annotations

import math
import re
import unicodedata
from collections import Counter
from collections.abc import Mapping


_ARABIC_TO_PERSIAN = str.maketrans(
    {
        "ي": "ی",
        "ى": "ی",
        "ك": "ک",
        "ؤ": "و",
        "ئ": "ی",
        "ۀ": "ه",
        "ة": "ه",
        "إ": "ا",
        "أ": "ا",
        "ٱ": "ا",
        "ـ": "",
    }
)
_DIACRITICS = re.compile(r"[\u064B-\u065F\u0670\u06D6-\u06ED]")
_SPACE = re.compile(r"\s+")
_WORD = re.compile(r"[\w\u0600-\u06FF']+", re.UNICODE)
_PERSIAN = re.compile(r"[\u0600-\u06FF]")
_LATIN = re.compile(r"[A-Za-z]")

# These language-neutral concepts are input features, not response rules. The
# neural model learns how much each concept matters for an intent.
_SEMANTIC_GROUPS: dict[str, frozenset[str]] = {
    "greeting": frozenset({"سلام", "درود", "صبح", "عصر", "hello", "hi", "hey", "morning", "evening", "howdy"}),
    "assistant": frozenset({"جارویس", "jarvis", "assistant", "دستیار"}),
    "wellbeing": frozenset({"حالت", "خوبی", "چطوری", "اوضاعت", "حال", "feel", "feeling", "doing", "going", "okay", "good"}),
    "low_mood": frozenset({"حوصله", "ناراحت", "خسته", "انرژی", "گرفته", "غمگین", "mood", "down", "tired", "sad", "energy", "great"}),
    "name": frozenset({"اسم", "اسمم", "اسمت", "نام", "نامم", "نامت", "name", "called", "call"}),
    "remember": frozenset({"یادت", "به خاطر", "remember", "saved", "ذخیره"}),
    "capability": frozenset({"قابلیت", "امکانات", "ابزار", "بلدی", "کمکی", "توانی", "can", "capabilities", "features", "tools", "able", "useful"}),
    "self_intro": frozenset({"معرفی", "خودت", "هستی", "درباره", "introduce", "yourself", "describe", "exactly"}),
    "thanks": frozenset({"ممنون", "مرسی", "سپاس", "تشکر", "thanks", "thank", "appreciate", "cheers"}),
    "farewell": frozenset({"خداحافظ", "فعلا", "بای", "بعد", "خدانگهدار", "goodbye", "bye", "later", "leaving", "night"}),
    "help": frozenset({"راهنما", "راهنمایی", "کمک", "دستور", "استفاده", "help", "guide", "instructions", "commands", "use"}),
    "time": frozenset({"ساعت", "زمان", "وقت", "time", "clock"}),
    "date": frozenset({"تاریخ", "امروز", "روز", "چندمه", "date", "today", "day"}),
    "project": frozenset({"پروژه", "برنامه", "سیستم", "اپ", "نرم", "project", "application", "program", "system", "coding"}),
    "bot": frozenset({"ربات", "بات", "bot", "telegram"}),
    "file": frozenset({"فایل", "پیوست", "متن", "کد", "file", "attachment", "contents", "preview", "code"}),
    "archive": frozenset({"زیپ", "آرشیو", "فشرده", "zip", "archive", "compressed", "folders"}),
    "open": frozenset({"باز", "اجرا", "برو", "open", "launch", "start", "run"}),
    "browser": frozenset({"مرورگر", "کروم", "گوگل", "وب", "browser", "chrome", "google", "web"}),
    "system_info": frozenset({"سخت", "سیستم", "رم", "پردازنده", "هسته", "گرافیک", "hardware", "ram", "cpu", "cores", "gpu", "specifications"}),
    "settings": frozenset({"تنظیمات", "آپشن", "شخصیت", "settings", "options", "personality", "panel"}),
    "clear": frozenset({"پاک", "خالی", "جدید", "تمیز", "ببند", "clear", "wipe", "reset", "new", "close"}),
    "conversation": frozenset({"گفتگو", "چت", "مکالمه", "جلسه", "conversation", "chat", "session"}),
    "internet": frozenset({"اینترنت", "آنلاین", "وب", "گوگل", "internet", "online", "web", "google"}),
    "search": frozenset({"جستجو", "سرچ", "پیدا", "بگرد", "search", "find", "look"}),
    "privacy": frozenset({"آفلاین", "خصوصی", "داده", "محلی", "اینترنت", "offline", "private", "privacy", "data", "local", "cloud", "api"}),
    "affirm": frozenset({"بله", "آره", "حتما", "باشه", "قبوله", "تایید", "yes", "yeah", "sure", "okay", "confirm", "confirmed", "ahead"}),
    "reject": frozenset({"نه", "خیر", "لغو", "نمیخوام", "نمی", "no", "nope", "cancel", "never", "don't", "reject"}),
    "question": frozenset({"چرا", "چیست", "چیه", "کجاست", "چطور", "what", "why", "where", "how", "does", "mean"}),
}


def normalize_text(text: str) -> str:
    value = unicodedata.normalize("NFKC", str(text)).translate(_ARABIC_TO_PERSIAN)
    value = _DIACRITICS.sub("", value).casefold()
    value = _SPACE.sub(" ", value).strip()
    return value


def detect_language(text: str) -> str:
    normalized = normalize_text(text)
    persian_count = len(_PERSIAN.findall(normalized))
    latin_count = len(_LATIN.findall(normalized))
    persian_words = re.findall(r"[\u0621-\u063A\u0641-\u064A\u0671-\u06D3‌]+", normalized)
    persian_grammar = {
        "فرق", "تفاوت", "چیست", "چیه", "چرا", "چطور", "چگونه", "هست",
        "نیست", "میکنه", "می‌کند", "کن", "بگو", "بده", "رو", "را", "با",
        "برای", "درباره", "یعنی", "کدام", "آیا",
    }
    if persian_count >= 3 and set(persian_words).intersection(persian_grammar):
        return "fa"
    return "fa" if persian_count >= max(1, latin_count) else "en"


def tokenize(text: str) -> list[str]:
    return _WORD.findall(normalize_text(text))


def stable_hash(value: str) -> int:
    result = 2_166_136_261
    for byte in value.encode("utf-8", errors="ignore"):
        result ^= byte
        result = (result * 16_777_619) & 0xFFFFFFFF
    return result


def hashed_features(text: str, size: int) -> dict[int, float]:
    """Create a normalized sparse Unicode n-gram vector with stable hashing."""
    normalized = normalize_text(text)
    if not normalized:
        return {}

    counts: Counter[int] = Counter()
    compact = f"^{normalized}$"
    for width in (2, 3, 4):
        for index in range(max(0, len(compact) - width + 1)):
            gram = compact[index : index + width]
            counts[stable_hash(f"c{width}:{gram}") % size] += 1

    words = tokenize(normalized)
    for word in words:
        counts[stable_hash(f"w:{word}") % size] += 2
        if word.isascii() and len(word) > 5:
            for suffix in ("ing", "ed", "es", "s"):
                if word.endswith(suffix) and len(word) - len(suffix) >= 3:
                    counts[stable_hash(f"stem:{word[:-len(suffix)]}") % size] += 1
                    break
    for first, second in zip(words, words[1:]):
        counts[stable_hash(f"b:{first}|{second}") % size] += 2

    word_set = set(words)
    for concept, terms in _SEMANTIC_GROUPS.items():
        if word_set.intersection(terms) or any(" " in term and term in normalized for term in terms):
            counts[stable_hash(f"semantic:{concept}") % size] += 5

    weighted = {index: 1.0 + math.log(count) for index, count in counts.items()}
    norm = math.sqrt(sum(value * value for value in weighted.values())) or 1.0
    return {index: value / norm for index, value in weighted.items()}


def cosine_sparse(first: Mapping[int, float], second: Mapping[int, float]) -> float:
    if len(first) > len(second):
        first, second = second, first
    return sum(value * second.get(index, 0.0) for index, value in first.items())


def compact_excerpt(text: str, limit: int = 64) -> str:
    value = _SPACE.sub(" ", text).strip()
    if len(value) <= limit:
        return value
    return value[: max(1, limit - 1)].rstrip() + "…"
