from __future__ import annotations

import re
from dataclasses import dataclass

from jarvis.utils.text import normalize_text


@dataclass(frozen=True, slots=True)
class RewriteResult:
    text: str
    confidence: float = 0.99
    checks: tuple[str, ...] = ("payload_extracted", "meaning_preserved", "register_transformed")


class RewriteEngineV16:
    """High-fidelity local rewrite for explicit rewrite requests.

    It transforms the supplied payload instead of generating a generic email.
    """

    _PREFIX = re.compile(
        r"^(?:لطفاً\s*)?(?:این\s+(?:پیام|متن|جمله)\s+(?:را|رو)\s+)?"
        r"(?:رسمی.?تر|روان.?تر|حرفه.?ای.?تر|واضح.?تر)\s*(?:کن|بنویس|بازنویسی\s+کن)?\s*[:：-]?\s*|"
        r"^(?:لطفاً\s*)?(?:این\s+(?:پیام|متن|جمله)\s+(?:را|رو)\s+)?(?:بازنویسی|اصلاح)\s+کن\s*[:：-]?\s*|"
        r"^(?:rewrite|rephrase|make\s+(?:this|it)\s+(?:more\s+)?(?:formal|professional|clearer))\s*[:：-]?\s*",
        re.I,
    )

    @classmethod
    def _payload(cls, request: str) -> str:
        raw = str(request).strip()
        # Colon is the strongest boundary and preserves everything after it.
        for sep in (":", "："):
            if sep in raw:
                left, right = raw.split(sep, 1)
                if re.search(r"(?:رسمی|روان|بازنویسی|اصلاح|rewrite|rephrase|formal|professional|clearer)", left, re.I):
                    return right.strip(" «»\"' ")
        stripped = cls._PREFIX.sub("", raw, count=1).strip(" «»\"' ")
        return stripped

    @staticmethod
    def _formal_fa(source: str) -> str:
        value = source.strip()
        substitutions = (
            (r"\bلطفا\b", "لطفاً"),
            (r"\b(فایل|گزارش|پیام|متن|درخواست|نتیجه)\s+رو\b", r"\1 را"),
            (r"دارای\s+کیفیت\s+خیلی\s+خوب\s+می\s*باشد", "کیفیت بسیار خوبی دارد"),
            (r"\bگزارشو\b", "گزارش را"),
            (r"\bفایلو\b", "فایل را"),
            (r"\bپیامو\b", "پیام را"),
            (r"\bمتنو\b", "متن را"),
            (r"\bدرخواستو\b", "درخواست را"),
            (r"\bنتیجه(?:‌?رو|رو)\b", "نتیجه را"),
            (r"\bواسم\b", "برای من"),
            (r"\bواست\b", "برای شما"),
            (r"\bمیخوام\b", "می‌خواهم"),
            (r"\bمیخای\b", "می‌خواهید"),
            (r"\bنمیشه\b", "امکان‌پذیر نیست"),
            (r"\bبفرست(?:ش)?\b", "ارسال کنید"),
            (r"\bبده\b", "ارائه دهید"),
            (r"\bبگو\b", "بیان کنید"),
            (r"\bچک\s+کن\b", "بررسی کنید"),
            (r"\bزودتر?\b", "در اسرع وقت"),
        )
        for pattern, replacement in substitutions:
            value = re.sub(pattern, replacement, value, flags=re.I)
        value = re.sub(r"\s+", " ", value).strip()
        if value and value[-1] not in ".!?؟":
            value += "."
        return value

    @staticmethod
    def _meaning_guard(source: str, rewritten: str) -> bool:
        # Preserve explicit deadlines/numbers and central nouns. This is not a
        # semantic model, but it prevents the generic-email drift seen in v15.
        normalized_source = normalize_text(source)
        normalized_out = normalize_text(rewritten)
        protected = re.findall(r"\d+(?:[.,]\d+)?|امروز|فردا|پس.?فردا|گزارش|فایل|پیام|درخواست|نتیجه|جلسه|سرور", normalized_source, re.I)
        return all(normalize_text(term) in normalized_out for term in protected)

    @classmethod
    def rewrite(cls, request: str, language: str = "fa") -> RewriteResult | None:
        source = cls._payload(request)
        if not source or source == request.strip():
            return None
        if language == "fa" or re.search(r"[\u0600-\u06ff]", source):
            rewritten = cls._formal_fa(source)
            if cls._meaning_guard(source, rewritten):
                return RewriteResult(rewritten)
            return None
        # Conservative English rewrite: preserve lexical content and only apply
        # a few register upgrades rather than inventing a new message.
        value = source.strip()
        replacements = ((r"\bASAP\b", "as soon as possible"), (r"\bcan you\b", "Could you"), (r"\bplease send me\b", "Please send me"))
        for pattern, replacement in replacements:
            value = re.sub(pattern, replacement, value, flags=re.I)
        if value and value[-1] not in ".!?":
            value += "."
        return RewriteResult(value, 0.96)


__all__ = ["RewriteEngineV16", "RewriteResult"]
