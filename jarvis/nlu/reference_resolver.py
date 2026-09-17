from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from jarvis.utils.text import normalize_text


@dataclass(frozen=True, slots=True)
class ResolvedReference:
    entity_type: str
    entity_id: str
    label: str = ""
    source: str = "context"


class ReferenceResolver:
    _REFERENCE = re.compile(
        r"(?:ببندش|بازش\s+کن|دوباره\s+بازش\s+کن|همون(?:و|ش|\s+قبلیه)?|"
        r"اون(?:و|ش|\s+برنامه|\s+مرورگر|\s+فایل|\s+پوشه)?|داخلش|توش|"
        r"\b(?:it|that|same\s+one|previous\s+one|in\s+it)\b)",
        re.I,
    )
    _PREVIOUS = re.compile(r"(?:قبلی|previous)", re.I)
    _BROWSER = re.compile(r"(?:مرورگر|browser|داخلش|توش|in\s+it)", re.I)

    @classmethod
    def has_reference(cls, text: str) -> bool:
        return bool(cls._REFERENCE.search(normalize_text(text)))

    @staticmethod
    def _coerce(value: Any, default_type: str = "") -> ResolvedReference | None:
        if isinstance(value, dict):
            entity_id = str(value.get("id", "")).strip()
            entity_type = str(value.get("type", default_type)).strip()
            if entity_id and entity_type:
                return ResolvedReference(entity_type, entity_id, str(value.get("label", "")))
        if isinstance(value, str) and value.strip() and default_type:
            return ResolvedReference(default_type, value.strip())
        return None

    @classmethod
    def resolve(
        cls,
        text: str,
        context: dict[str, Any],
        expected_types: tuple[str, ...] = (),
    ) -> ResolvedReference | None:
        normalized = normalize_text(text)
        if not cls.has_reference(normalized):
            return None
        if cls._BROWSER.search(normalized):
            browser = cls._coerce(context.get("last_browser"), "app")
            if browser:
                return browser
        recent = context.get("recent_entities", [])
        if isinstance(recent, list) and recent:
            start = 1 if cls._PREVIOUS.search(normalized) and len(recent) > 1 else 0
            for value in recent[start:]:
                resolved = cls._coerce(value)
                if resolved and (not expected_types or resolved.entity_type in expected_types):
                    return resolved
        last = cls._coerce(context.get("last_entity"))
        if last and (not expected_types or last.entity_type in expected_types):
            return last
        key_by_type = {
            "app": "last_opened_app",
            "file": "last_file",
            "folder": "last_folder",
            "drive": "last_drive",
            "url": "last_opened_url",
            "website": "last_opened_url",
        }
        for entity_type in expected_types:
            value = cls._coerce(context.get(key_by_type.get(entity_type, "")), entity_type)
            if value:
                return value
        return None
