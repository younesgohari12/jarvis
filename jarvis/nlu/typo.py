from __future__ import annotations

import re
import unicodedata


def _damerau_distance(left: str, right: str) -> int:
    if left == right:
        return 0
    rows = len(left) + 1
    columns = len(right) + 1
    matrix = [[0] * columns for _ in range(rows)]
    for index in range(rows):
        matrix[index][0] = index
    for index in range(columns):
        matrix[0][index] = index
    for i in range(1, rows):
        for j in range(1, columns):
            cost = int(left[i - 1] != right[j - 1])
            matrix[i][j] = min(
                matrix[i - 1][j] + 1,
                matrix[i][j - 1] + 1,
                matrix[i - 1][j - 1] + cost,
            )
            if i > 1 and j > 1 and left[i - 1] == right[j - 2] and left[i - 2] == right[j - 1]:
                matrix[i][j] = min(matrix[i][j], matrix[i - 2][j - 2] + cost)
    return matrix[-1][-1]


class TypoNormalizer:
    """Conservative edit-distance correction for action/entity vocabulary."""

    _WORDS = (
        "انتقال", "جابجا", "فایل", "پوشه", "فولدر", "کروم", "تلگرام",
        "ویندوز", "دسکتاپ", "دانلود", "اجرا", "باز", "ببند", "حذف",
        "کپی", "ذخیره", "روشنایی", "درایو", "بساز", "ایجاد", "درست",
    )
    _PHRASES = ("باز کن", "اجرا کن", "منتقل کن", "جابجا کن", "حذف کن", "کپی کن")
    _PERSIAN_WORD = re.compile(r"[؀-ۿ‌]{2,20}")
    _TWO_WORDS = re.compile(r"(?P<left>[؀-ۿ‌]{2,12})\s+(?P<right>[؀-ۿ‌]{2,12})")
    _ARABIC_TO_PERSIAN = str.maketrans(
        {"ي": "ی", "ى": "ی", "ك": "ک", "ة": "ه", "ۀ": "ه", "ـ": ""}
    )

    @classmethod
    def correct(cls, text: str) -> str:
        if re.search(r"(?:https?://|[A-Za-z]:[\\/]|\{[^\n]*\})", text):
            # Paths/URLs/JSON are preserved byte-for-byte. Other words around
            # them are already handled by deterministic intent/entity parsing.
            return text
        # Do not case-fold the complete request: filenames and JSON values are
        # entities, so changing ``Atlas_01`` to ``atlas_01`` is data corruption.
        normalized = unicodedata.normalize("NFKC", str(text)).translate(cls._ARABIC_TO_PERSIAN)
        normalized = re.sub(r"\s+", " ", normalized).strip()

        def phrase(match: re.Match[str]) -> str:
            value = match.group(0)
            compact = value.replace(" ", "")
            candidates = [
                candidate for candidate in cls._PHRASES
                if candidate.replace(" ", "")[0] == compact[0]
                and _damerau_distance(compact, candidate.replace(" ", "")) <= 1
            ]
            return candidates[0] if len(candidates) == 1 else value

        normalized = cls._TWO_WORDS.sub(phrase, normalized)

        def word(match: re.Match[str]) -> str:
            value = match.group(0)
            if value in cls._WORDS or len(value) < 4:
                return value
            object_suffix = value.endswith("و") and len(value) >= 5
            pronoun_suffix = value.endswith("ش") and len(value) >= 5
            base = value[:-1] if object_suffix or pronoun_suffix else value
            candidates = [
                candidate for candidate in cls._WORDS
                if candidate[0] == base[0]
                and abs(len(candidate) - len(base)) <= 1
                and _damerau_distance(base, candidate) <= 1
            ]
            if len(candidates) != 1:
                return value
            suffix = " رو" if object_suffix else "ش" if pronoun_suffix else ""
            return candidates[0] + suffix

        return cls._PERSIAN_WORD.sub(word, normalized)
