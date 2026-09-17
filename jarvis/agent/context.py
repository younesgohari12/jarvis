from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from jarvis.utils.text import normalize_text


@dataclass(frozen=True, slots=True)
class ContextDecision:
    response_intent: str = ""
    topic: str = ""
    state: dict[str, Any] = field(default_factory=dict)
    facts: dict[str, str] = field(default_factory=dict)
    slots: dict[str, str] = field(default_factory=dict)

    @property
    def handled(self) -> bool:
        return bool(self.response_intent)


class ContextEngine:
    """Maintains lightweight dialogue state without a generative model."""

    def __init__(self, rules_path: Path) -> None:
        try:
            with rules_path.open("r", encoding="utf-8") as handle:
                payload = json.load(handle)
        except (OSError, json.JSONDecodeError) as exc:
            raise RuntimeError(f"Cannot load context rules: {exc}") from exc
        self._project_terms = {
            str(key): tuple(normalize_text(str(value)) for value in values)
            for key, values in payload.get("project_terms", {}).items()
        }
        self._pending = dict(payload.get("pending_states", {}))

    def _project_label(self, text: str) -> str:
        normalized = normalize_text(text)
        for label, terms in self._project_terms.items():
            if any(term in normalized for term in terms):
                return label
        return ""

    def analyze(
        self, text: str, language: str, routed_intent: str, current: dict[str, Any]
    ) -> ContextDecision:
        state = dict(current)
        topic = str(state.pop("topic", ""))
        pending = str(state.get("pending", ""))
        normalized = normalize_text(text)
        if pending == "project_kind":
            configuration = self._pending.get("project_kind", {})
            maximum = int(configuration.get("maximum_reply_characters", 100))
            if len(normalized) <= maximum and routed_intent not in {
                "greeting", "goodbye", "help", "rejection", "ask_time", "ask_date"
            }:
                label = self._project_label(text) or normalized[:60]
                state.pop("pending", None)
                state["project_kind"] = label
                return ContextDecision(
                    str(configuration.get("response_intent", "context_project_kind")),
                    "project", state,
                    {"last_project": label, "current_topic": "project"},
                    {"project_kind": label},
                )
        if routed_intent == "user_low_mood":
            state["mood"] = "low"
            return ContextDecision(
                "context_low_mood", "wellbeing", state, {"current_topic": "wellbeing"}
            )
        if routed_intent == "project_context":
            technology = self._project_label(text)
            state["pending"] = "project_kind"
            if technology:
                state["technology"] = technology
            return ContextDecision(
                "context_project_start", "project", state,
                {"current_topic": "project"}, {"technology": technology}
            )
        if routed_intent == "project_kind" and topic == "project":
            label = self._project_label(text) or normalized[:60]
            state["project_kind"] = label
            state.pop("pending", None)
            return ContextDecision(
                "context_project_kind", "project", state,
                {"last_project": label, "current_topic": "project"},
                {"project_kind": label},
            )
        return ContextDecision(topic=topic, state=state)

