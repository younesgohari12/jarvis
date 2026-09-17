from __future__ import annotations

import re

from jarvis.nlu.action_parser import ActionParser


class TaskSegmenter:
    """Segments explicit task chains without splitting every Persian/English 'and'."""

    _STRONG = re.compile(
        r"\s+(?:بعد(?:ش)?|سپس|و\s+بعد|حالا\s+بعد|then|and\s+then|after\s+that)\s+",
        re.I,
    )
    _WEAK = re.compile(r"\s+(?:و|and)\s+", re.I)

    @classmethod
    def split(cls, text: str) -> tuple[str, ...]:
        # A colon-delimited translation/rewrite/code payload is one atomic task.
        # Words such as «بعد» may legitimately occur *inside* that payload (e.g.
        # «جلسه به هفته بعد منتقل شد») and must never be interpreted as the
        # task-chain separator "then".
        if re.search(
            r"(?:ترجمه|به\s+(?:انگلیسی|فارسی)\s+(?:بگو|کن)|(?:انگلیسی|فارسی)(?:ش|اش|‌اش)?\s*(?:کن|بگو)|"
            r"بازنویسی|رسمی.?تر|روان.?تر|rewrite|rephrase|translate|code\s+trace|خروجی\s+این\s+کد).{0,80}[:：]",
            text, re.I | re.S,
        ):
            return (text.strip(),)
        # Numeric story problems often use «بعد / سپس» to describe events, not
        # separate user commands (e.g. "45 pages and then 38 pages"). Keep the
        # whole problem intact so the reasoning engine can solve all steps.
        if (
            re.search(r"\d", text)
            and re.search(
                r"(?:صفحه|کتاب|قیمت|تومان|سن|ساله|کارگر|ساعت|روز|سرعت|مسافت|مخزن|"
                r"pages?|book|price|cost|years?\s+old|worker|hours?|speed|distance|tank)",
                text, re.I,
            )
            and re.search(r"(?:چند|چقدر|باقی|مانده|می.?شود|how\s+(?:many|much|long)|remaining|left|find|calculate)", text, re.I)
        ):
            return (text.strip(),)
        strong_parts = [part.strip(" ،,") for part in cls._STRONG.split(text) if part.strip(" ،,")]
        output: list[str] = []
        for part in strong_parts:
            weak_parts = [value.strip(" ،,") for value in cls._WEAK.split(part)]
            if len(weak_parts) == 1:
                output.append(part)
                continue
            buffer = weak_parts[0]
            for candidate in weak_parts[1:]:
                left_action = ActionParser.has_action(buffer)
                right_action = ActionParser.has_action(candidate)
                if left_action and right_action:
                    output.append(buffer)
                    buffer = candidate
                else:
                    buffer = f"{buffer} و {candidate}"
            output.append(buffer)
        return tuple(value for value in output if value)
