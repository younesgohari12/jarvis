from __future__ import annotations

import re
from dataclasses import dataclass

from jarvis.entities.resolver import EntityResolver
from jarvis.utils.text import normalize_text


@dataclass(frozen=True, slots=True)
class LearningInstruction:
    trigger: str
    action_text: str
    source: str


class LearningParser:
    """Extracts explicit user corrections and teach commands without retraining."""

    _WHEN_FA = re.compile(r"وقتی\s+(?:میگم|می\s*گم|گفتم)\s+(.+)", re.I)
    _WHEN_EN = re.compile(r"when\s+i\s+say\s+(.+)", re.I)

    def __init__(self, entities: EntityResolver) -> None:
        self.entities = entities

    @staticmethod
    def _clean(value: str) -> str:
        return value.strip(" \t\r\n،,:؛.!؟?\"'«»")

    def parse(self, text: str) -> LearningInstruction | None:
        normalized = normalize_text(text)
        match = self._WHEN_FA.search(normalized) or self._WHEN_EN.search(normalized)
        is_teach = bool(
            re.search(r"(?:یاد\s*بگیر|به\s*خاطر\s*بسپار|learn\s+that|remember\s+that)", normalized)
        ) or bool(match)
        is_correction = bool(
            re.search(r"^(?:نه|اشتباهه|منظورم اینه|no[, ]|that's wrong)", normalized)
            and ("وقتی" in normalized or "when i say" in normalized)
        )
        if not (is_teach or is_correction):
            return None
        if not match:
            return None
        tail = self._clean(match.group(1))
        if " باید " in f" {tail} ":
            trigger, action = tail.split(" باید ", 1)
            return LearningInstruction(self._clean(trigger), self._clean(action), "correction" if is_correction else "teach")
        english_split = re.split(r"\s+(?:then|you should|please)\s+", tail, maxsplit=1)
        if len(english_split) == 2:
            return LearningInstruction(
                self._clean(english_split[0]), self._clean(english_split[1]),
                "correction" if is_correction else "teach"
            )

        positions: list[int] = []
        for website in self.entities.websites:
            for alias in website.aliases:
                index = tail.find(alias)
                if index > 0:
                    positions.append(index)
        for app in self.entities.apps:
            for alias in app.aliases:
                index = tail.find(alias)
                if index > 0:
                    positions.append(index)
        project_index = tail.find("پروژه ")
        if project_index > 0:
            positions.append(project_index)
        if positions:
            split_at = min(positions)
            trigger = self._clean(tail[:split_at])
            action = self._clean(tail[split_at:])
            if trigger and action:
                return LearningInstruction(
                    trigger, action, "correction" if is_correction else "teach"
                )
        return None
