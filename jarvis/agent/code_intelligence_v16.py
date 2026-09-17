from __future__ import annotations

import re

from jarvis.agent.code_intelligence_v15 import CodeIntelligenceV15, CodeTraceAnswer
from jarvis.utils.text import normalize_text


class CodeIntelligenceV16(CodeIntelligenceV15):
    """v16 code-trace guard.

    In addition to explicit "what is the output" prompts, a self-contained pasted
    Python snippet with ``print(...)`` is treated as a trace request. This guard
    runs before conversation/memory resolution so code cannot be reinterpreted as
    a remembered fact or an algebra equation.
    """

    _AUTHORING = re.compile(
        r"(?:بنویس|بساز|پیاده.?سازی|تابع|اسکریپت|الگوریتم|اصلاح\s+کن|"
        r"\b(?:write|create|implement|build|function|script|fix|debug)\b)",
        re.I,
    )
    _BARE_PRINT = re.compile(r"\bprint\s*\(", re.I)
    _BARE_STATE = re.compile(
        r"(?:^|[;\n])\s*[A-Za-z_]\w*\s*=\s*[^=]|\bfor\s+[A-Za-z_]\w*\s+in\s+range\s*\(",
        re.I,
    )

    @classmethod
    def matches(cls, text: str) -> bool:
        if super().matches(text):
            return True
        raw = str(text).strip()
        normalized = normalize_text(raw)
        if cls._AUTHORING.search(normalized):
            return False
        return bool(cls._BARE_PRINT.search(raw) and cls._BARE_STATE.search(raw))


__all__ = ["CodeIntelligenceV16", "CodeTraceAnswer"]
