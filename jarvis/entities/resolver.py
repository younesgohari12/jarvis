from __future__ import annotations

import json
import re
import urllib.parse
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from jarvis.utils.text import normalize_text


class EntityDataError(RuntimeError):
    pass


@dataclass(frozen=True, slots=True)
class WebsiteEntity:
    entity_id: str
    name: str
    url: str
    search_url: str
    aliases: tuple[str, ...]

    def search_address(self, query: str) -> str:
        if not self.search_url:
            return self.url
        return self.search_url.format(query=urllib.parse.quote_plus(query.strip()))


@dataclass(frozen=True, slots=True)
class AppEntity:
    entity_id: str
    name: str
    aliases: tuple[str, ...]
    commands: dict[str, tuple[str, ...]]


@dataclass(frozen=True, slots=True)
class EntityMatch:
    kind: str
    entity_id: str
    name: str
    alias: str
    score: float
    website: WebsiteEntity | None = None
    app: AppEntity | None = None


def _read_object(path: Path) -> dict[str, Any]:
    try:
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except (OSError, json.JSONDecodeError) as exc:
        raise EntityDataError(f"Cannot load entity registry {path.name}: {exc}") from exc
    if not isinstance(payload, dict):
        raise EntityDataError(f"Entity registry root must be an object: {path.name}")
    return payload


class EntityResolver:
    """Resolves aliases with longest-match priority and explicit app/site separation."""

    _OPEN_WORDS = re.compile(
        r"(?:باز\s*(?:کن|کنی|کنید)?|بیار|بیارید|اجرا\s*(?:کن|کنی|کنید)?|"
        r"برو(?:\s+تو|\s+داخل)?|بزن|open|launch|run|start|bring\s+up|take\s+me\s+to)",
        re.I,
    )
    _SEARCH_WORDS = re.compile(
        r"(?:سرچ\s*(?:کن|بزن)?|جستجو\s*(?:کن)?|بگرد|پیدا\s*(?:کن)?|"
        r"گوگل\s*(?:کن)?|search(?:\s+for)?|look\s+up|google)",
        re.I,
    )
    _DEFAULT_BROWSER = re.compile(
        r"(?:مرورگر(?:\s+پیش\s*فرض)?|default\s+browser|web\s+browser|browser)", re.I
    )
    _URL = re.compile(r"https?://[^\s<>\"']+", re.I)

    def __init__(self, websites_path: Path, apps_path: Path) -> None:
        website_data = _read_object(websites_path)
        app_data = _read_object(apps_path)
        websites: list[WebsiteEntity] = []
        for item in website_data.get("websites", []):
            aliases = self._aliases(item.get("aliases", []))
            websites.append(
                WebsiteEntity(
                    str(item["id"]), str(item["name"]), str(item["url"]),
                    str(item.get("search_url", "")), aliases
                )
            )
        apps: list[AppEntity] = []
        for item in app_data.get("apps", []):
            aliases = self._aliases(item.get("aliases", []))
            commands = {
                key: tuple(str(value) for value in values)
                for key, values in item.items()
                if key in {"windows", "linux", "darwin"} and isinstance(values, list)
            }
            apps.append(AppEntity(str(item["id"]), str(item["name"]), aliases, commands))
        if not websites or not apps:
            raise EntityDataError("Website and app registries must not be empty")
        self.websites = tuple(websites)
        self.apps = tuple(apps)
        self._website_by_id = {item.entity_id: item for item in websites}
        self._app_by_id = {item.entity_id: item for item in apps}

    @staticmethod
    def _aliases(values: Any) -> tuple[str, ...]:
        aliases = {normalize_text(str(value)) for value in values if str(value).strip()}
        return tuple(sorted(aliases, key=lambda value: (-len(value), value)))

    @staticmethod
    def _alias_present(text: str, alias: str) -> bool:
        start = 0
        while True:
            index = text.find(alias, start)
            if index < 0:
                return False
            before = text[index - 1] if index else " "
            end = index + len(alias)
            after = text[end] if end < len(text) else " "
            before_ok = not (before.isalnum() or before == "_")
            after_ok = not (after.isalnum() or after == "_")
            persian_object_suffix = after == "و" and (
                end + 1 == len(text)
                or not (text[end + 1].isalnum() or text[end + 1] == "_")
            )
            if before_ok and (after_ok or persian_object_suffix):
                return True
            start = index + 1

    def _best_website(self, normalized: str) -> EntityMatch | None:
        matches: list[EntityMatch] = []
        for entity in self.websites:
            for alias in entity.aliases:
                if self._alias_present(normalized, alias):
                    matches.append(
                        EntityMatch(
                            "website", entity.entity_id, entity.name, alias,
                            min(0.99, 0.78 + len(alias) / 100), website=entity
                        )
                    )
                    break
        return max(matches, key=lambda item: (len(item.alias), item.score), default=None)

    def _best_app(self, normalized: str) -> EntityMatch | None:
        matches: list[EntityMatch] = []
        for entity in self.apps:
            for alias in entity.aliases:
                if self._alias_present(normalized, alias):
                    matches.append(
                        EntityMatch(
                            "app", entity.entity_id, entity.name, alias,
                            min(0.99, 0.8 + len(alias) / 100), app=entity
                        )
                    )
                    break
        return max(matches, key=lambda item: (len(item.alias), item.score), default=None)

    def website(self, entity_id: str) -> WebsiteEntity | None:
        return self._website_by_id.get(str(entity_id).casefold())

    def app(self, entity_id: str) -> AppEntity | None:
        return self._app_by_id.get(str(entity_id).casefold())

    def resolve_website(self, text: str) -> EntityMatch | None:
        return self._best_website(normalize_text(text))

    def resolve_app(self, text: str) -> EntityMatch | None:
        return self._best_app(normalize_text(text))

    def resolve_open(self, text: str) -> EntityMatch | None:
        normalized = normalize_text(text)
        if not self._OPEN_WORDS.search(normalized):
            return None
        app = self._best_app(normalized)
        website = self._best_website(normalized)
        if app and (app.entity_id == "chrome" or not website):
            return app
        if website:
            return website
        return app

    def is_default_browser_request(self, text: str) -> bool:
        normalized = normalize_text(text)
        return bool(self._OPEN_WORDS.search(normalized) and self._DEFAULT_BROWSER.search(normalized))

    def explicit_url(self, text: str) -> str:
        match = self._URL.search(text)
        return match.group(0).rstrip(".,،؛") if match else ""

    def extract_search(self, text: str) -> tuple[str, EntityMatch | None]:
        normalized = normalize_text(text)
        if not self._SEARCH_WORDS.search(normalized):
            return "", None
        website = self._best_website(normalized)
        # "گوگل کن X" is a web-search verb, while "توی گوگل X رو سرچ کن"
        # still carries a Google entity. Both intentionally become web search.
        if website and website.entity_id == "google":
            normalized = re.sub(r"گوگل\s*(?:کن)?|google", " ", normalized, count=1)
        query = normalized
        if website:
            query = query.replace(website.alias, " ", 1)
        query = self._SEARCH_WORDS.sub(" ", query, count=1)
        query = query.strip()
        query = re.sub(
            r"^(?:تو|توی|داخل|در|درباره|راجع به|برای|on|in|about|for)\s+", "", query
        )
        query = re.sub(
            r"^(?:وب|اینترنت|the\s+web|web)(?:\s+(?:for|درباره|برای))?\s+", "", query
        )
        query = re.sub(r"\s+", " ", query).strip(" :،؟?")
        return query, website
