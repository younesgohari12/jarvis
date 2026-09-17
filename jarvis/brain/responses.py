from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from jarvis.personalities.engine import PersonalityEngine


class ResponseDataError(RuntimeError):
    pass


class _SafeSlots(dict[str, Any]):
    def __missing__(self, key: str) -> str:
        return ""


class ResponseEngine:
    """Context-aware template realization; semantic choice happens before this layer."""

    def __init__(self, responses_path: Path, personalities: PersonalityEngine) -> None:
        try:
            with responses_path.open("r", encoding="utf-8") as handle:
                payload = json.load(handle)
        except (OSError, json.JSONDecodeError) as exc:
            raise ResponseDataError(f"Cannot load {responses_path.name}: {exc}") from exc
        self._responses = payload.get("intents", {}) if isinstance(payload, dict) else {}
        self._personalities = personalities

    @staticmethod
    def _pick(options: list[str], seed: str) -> str:
        if not options:
            return ""
        digest = hashlib.blake2s(seed.encode("utf-8"), digest_size=4).digest()
        return options[int.from_bytes(digest, "big") % len(options)]

    def render(
        self,
        intent: str,
        language: str,
        message: str,
        turn_index: int,
        slots: dict[str, Any] | None = None,
    ) -> str:
        slots = slots or {}
        override = self._personalities.response_override(intent, language, message)
        if override:
            template = override
        else:
            intent_data = self._responses.get(intent, {})
            options = intent_data.get(language) or intent_data.get("en") or []
            template = self._pick(
                [str(value) for value in options] if isinstance(options, list) else [],
                f"{message}|{intent}|{turn_index}",
            )
        if not template and intent != "unknown_information":
            return self.render("unknown_information", language, message, turn_index, slots)
        try:
            base = template.format_map(_SafeSlots(slots)).strip()
        except (ValueError, AttributeError):
            base = template.strip()
        return self._personalities.apply_style(base, intent, language, message)

