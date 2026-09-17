from __future__ import annotations

import re
from dataclasses import dataclass

from jarvis.utils.text import normalize_text, tokenize


@dataclass(frozen=True, slots=True)
class DecisionAssessment:
    needs_fresh_information: bool
    is_complex: bool
    confidence: float
    reasons: tuple[str, ...]


class DecisionEngine:
    """Cheap, deterministic gate before neural inference and expensive tools."""

    _FRESH = re.compile(
        r"(?:\b(?:latest|newest|current|today|tonight|now|live|breaking|news|weather|"
        r"price|score|version|release|recent|updated)\b|"
        r"آخرین|جدیدترین|امروز|امشب|الان|لحظه.?ای|اخبار|خبر|هوا|آب.?و.?هوا|"
        r"قیمت|نرخ|نتیجه|چند\s+چنده|نسخه\s+(?:فعلی|جدید)|به.?روز)",
        re.I,
    )
    _VOLATILE = re.compile(
        r"(?:bitcoin|btc|crypto|stock|exchange\s+rate|weather|forecast|score|"
        r"بیت.?کوین|ارز|بورس|دلار|طلا|هوا|فوتبال|نتیجه\s+بازی)", re.I
    )
    _COMPLEX = re.compile(
        r"(?:\b(?:compare|versus|\bvs\b|trade.?offs?|pros? and cons?|architecture|"
        r"debug|diagnose|root cause|design|strategy|explain how|error.*fix|python.*error|error.*python)\b|"
        r"مقایسه|بهتره\s+یا|کدام.*بهتر|which.*better|مزایا\s+و\s+معایب|علت\s+ریشه|ریشه\s+خطا|"
        r"معماری|طراحی|استراتژی|خطا.*حل|ارور.*حل|دیباگ|تحلیل)",
        re.I,
    )

    @classmethod
    def needs_fresh_information(cls, text: str) -> bool:
        normalized = normalize_text(text)
        if normalized in {"چه خبر", "چه خبر؟", "what's up", "whats up"}:
            return False
        if cls._FRESH.search(normalized):
            return True
        return bool(
            cls._VOLATILE.search(normalized)
            and re.search(r"(?:چنده|what|how much|price|rate)", normalized)
        )

    @classmethod
    def is_complex(cls, text: str) -> bool:
        normalized = normalize_text(text)
        tokens = tokenize(normalized)
        if cls._COMPLEX.search(normalized):
            return True
        connectors = len(re.findall(r"(?:\band\b|\bor\b|\bthen\b|و|یا|بعد)", normalized))
        return len(tokens) >= 24 or (len(tokens) >= 14 and connectors >= 2)

    @classmethod
    def assess(cls, text: str) -> DecisionAssessment:
        fresh = cls.needs_fresh_information(text)
        complex_question = cls.is_complex(text)
        reasons: list[str] = []
        if fresh:
            reasons.append("fresh_information")
        if complex_question:
            reasons.append("multi_pass_reasoning")
        confidence = 0.94 if reasons else 0.82
        return DecisionAssessment(fresh, complex_question, confidence, tuple(reasons))
