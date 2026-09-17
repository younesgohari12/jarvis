from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


class PersonalityError(RuntimeError):
    pass


@dataclass(frozen=True, slots=True)
class PersonalityProfile:
    name: str
    display_name: str
    description: str
    style_probability: float
    prefixes: dict[str, tuple[str, ...]]
    suffixes: dict[str, tuple[str, ...]]
    overrides: dict[str, dict[str, tuple[str, ...]]]


class PersonalityEngine:
    def __init__(self, directory: Path, selected: str = "Normal") -> None:
        self._profiles: dict[str, PersonalityProfile] = {}
        for path in sorted(directory.glob("*.json")):
            profile = self._load(path)
            self._profiles[profile.name.casefold()] = profile
        if not self._profiles:
            raise PersonalityError("No personality profiles were found")
        self._current = self._resolve(selected) or next(iter(self._profiles.values()))

    @staticmethod
    def _tuple_map(value: Any) -> dict[str, tuple[str, ...]]:
        if not isinstance(value, dict):
            return {}
        result: dict[str, tuple[str, ...]] = {}
        for key, items in value.items():
            if isinstance(items, list):
                result[str(key)] = tuple(str(item) for item in items if str(item))
        return result

    @classmethod
    def _load(cls, path: Path) -> PersonalityProfile:
        try:
            with path.open("r", encoding="utf-8") as handle:
                raw = json.load(handle)
        except (OSError, json.JSONDecodeError) as exc:
            raise PersonalityError(f"Cannot load personality {path.name}: {exc}") from exc
        overrides: dict[str, dict[str, tuple[str, ...]]] = {}
        for intent, languages in raw.get("overrides", {}).items():
            overrides[str(intent)] = cls._tuple_map(languages)
        probability = min(1.0, max(0.0, float(raw.get("style_probability", 0.0))))
        return PersonalityProfile(
            name=str(raw["name"]),
            display_name=str(raw.get("display_name", raw["name"])),
            description=str(raw.get("description", "")),
            style_probability=probability,
            prefixes=cls._tuple_map(raw.get("prefixes", {})),
            suffixes=cls._tuple_map(raw.get("suffixes", {})),
            overrides=overrides,
        )

    def _resolve(self, name: str) -> PersonalityProfile | None:
        return self._profiles.get(str(name).casefold())

    @property
    def current(self) -> PersonalityProfile:
        return self._current

    def available_names(self) -> tuple[str, ...]:
        preferred = ("Normal", "Kind", "Angry", "Loti", "Gang", "Professional", "Funny")
        existing = {profile.name: profile for profile in self._profiles.values()}
        ordered = [name for name in preferred if name in existing]
        ordered.extend(sorted(name for name in existing if name not in ordered))
        return tuple(ordered)

    def select(self, name: str) -> PersonalityProfile:
        profile = self._resolve(name)
        if profile is None:
            raise PersonalityError(f"Unknown personality: {name}")
        self._current = profile
        return profile

    @staticmethod
    def _pick(items: tuple[str, ...], seed: str) -> str:
        if not items:
            return ""
        digest = hashlib.blake2b(seed.encode("utf-8"), digest_size=4).digest()
        return items[int.from_bytes(digest, "big") % len(items)]

    def response_override(self, intent: str, language: str, message: str) -> str:
        languages = self._current.overrides.get(intent, {})
        items = languages.get(language) or languages.get("en") or ()
        return self._pick(items, f"override|{self._current.name}|{message}")

    def apply_style(self, text: str, intent: str, language: str, message: str) -> str:
        if not text or intent in self._current.overrides:
            return text
        digest = hashlib.sha256(
            f"style|{self._current.name}|{intent}|{message}".encode("utf-8")
        ).digest()
        chance = int.from_bytes(digest[:2], "big") / 65535.0
        if chance > self._current.style_probability:
            return text
        prefix = self._pick(
            self._current.prefixes.get(language, ()), f"prefix|{message}|{intent}"
        )
        suffix = self._pick(
            self._current.suffixes.get(language, ()), f"suffix|{message}|{intent}"
        )
        return f"{prefix}{text}{suffix}".strip()
