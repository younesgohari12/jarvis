"""JARVIS v19 model-backed semantic slot binding.

This module intentionally does not own routing. It binds roles only after a task has
semantic evidence, so generic numbers/percentages cannot steal system actions,
probability questions, or sequences.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from jarvis.agent.semantic_models_v19 import NumericRoleTaggerV19, SemanticFrameClassifierV19
from jarvis.utils.text import normalize_text

_DIGITS = str.maketrans("۰۱۲۳۴۵۶۷۸۹٠١٢٣٤٥٦٧٨٩", "01234567890123456789")


@dataclass(frozen=True, slots=True)
class SlotFrame:
    task: str
    slots: dict[str, Any]
    confidence: float = 0.0
    evidence: tuple[str, ...] = ()


class SemanticSlotBinderV19:
    """Bind numeric mentions by learned local/global context, not list position."""

    def __init__(self, root: Path | None = None) -> None:
        root = root or Path(__file__).resolve().parents[2]
        self.numeric = NumericRoleTaggerV19(root / "models" / "numeric_role_v19.npz")
        self.frame = SemanticFrameClassifierV19(root / "models" / "semantic_frame_v19.npz")

    @staticmethod
    def _norm(text: str) -> str:
        return normalize_text(text).translate(_DIGITS).replace("٫", ".").replace("٪", "%")

    @staticmethod
    def _mentions(text: str):
        for m in re.finditer(r"(?<![\w.])\d+(?:[.,]\d+)?", text):
            yield m, float(m.group().replace(",", "."))

    def role_values(self, text: str, *, min_confidence: float = .42) -> dict[str, float]:
        value = self._norm(text)
        best: dict[str, tuple[float, float]] = {}
        if not self.numeric.ready:
            return {}
        for m, number in self._mentions(value):
            pred = self.numeric.predict_number(value, m.start(), m.end())
            if pred is None or pred.confidence < min_confidence:
                continue
            previous = best.get(pred.label)
            if previous is None or pred.confidence > previous[1]:
                best[pred.label] = (number, pred.confidence)
        return {role: value_conf[0] for role, value_conf in best.items()}

    @staticmethod
    def _strict_ratio_split(text: str) -> bool:
        if not re.search(r"(?:نسبت|ratio|\d\s*:\s*\d)", text, re.I):
            return False
        if re.search(r"(?:دنباله|تصاعد|جمله\s+\d|sequence|geometric|\bgp\b|term)", text, re.I):
            return False
        return bool(re.search(r"(?:تقسیم|پخش|سهم|split|divide|distribute|allocate)", text, re.I))

    def bind(self, text: str) -> SlotFrame:
        value = self._norm(text)
        frame = self.frame.predict(value) if self.frame.ready else None
        # Collision guard: system actions, probability and sequence are not generic
        # percentage/ratio tasks and are handled by their dedicated paths.
        if re.search(r"(?:صدا|ولوم|volume|روشنایی|نور\s+صفحه|brightness)", value, re.I):
            return SlotFrame("unknown", {}, 1.0, ("system_action_guard",))
        if re.search(r"(?:احتمال|شانس|probability|chance)", value, re.I):
            roles = self.role_values(value)
            if {"probability", "n", "k"} <= roles.keys():
                return SlotFrame("probability", roles, frame.confidence if frame else .8, ("trained_numeric_roles",))
            return SlotFrame("unknown", roles, frame.confidence if frame else .0, ("probability_guard",))
        if re.search(r"(?:دنباله|تصاعد|sequence|geometric|\bgp\b|term)", value, re.I):
            return SlotFrame("unknown", {}, frame.confidence if frame else .0, ("sequence_guard",))

        roles = self.role_values(value)
        if self._strict_ratio_split(value):
            # Structural ratio grammar resolves a/b identity; learned roles bind total
            # even when total appears after the ratio.
            rm = re.search(r"(?:نسبت|ratio)\s*(\d+(?:\.\d+)?)\s*(?:به|to|:)\s*(\d+(?:\.\d+)?)", value, re.I)
            if not rm:
                rm = re.search(r"(\d+(?:\.\d+)?)\s*:\s*(\d+(?:\.\d+)?)", value)
            if rm:
                a, b = map(float, rm.groups())
                total = roles.get("total")
                if total is None:
                    span = rm.span()
                    candidates = [n for m, n in self._mentions(value) if m.end() <= span[0] or m.start() >= span[1]]
                    if len(candidates) == 1:
                        total = candidates[0]
                if total is not None and a > 0 and b > 0:
                    return SlotFrame(
                        "ratio_split", {"total": total, "ratio_a": a, "ratio_b": b, "operation": "split"},
                        max(.9, frame.confidence if frame else 0.0), ("trained_numeric_roles", "strict_split_semantics"),
                    )

        if re.search(r"(?:کارگر|workers?)", value, re.I) and re.search(r"(?:قطعه|کالا|واحد|items?|units?|pieces?)", value, re.I):
            if {"workers", "hours", "output"} <= roles.keys():
                return SlotFrame("work_rate_observation", {k: roles[k] for k in ("workers", "hours", "output")}, max(.85, frame.confidence if frame else 0.0), ("trained_numeric_roles",))
        return SlotFrame("unknown", roles, frame.confidence if frame else .0, ("trained_numeric_roles",) if roles else ())


__all__ = ["SlotFrame", "SemanticSlotBinderV19"]
