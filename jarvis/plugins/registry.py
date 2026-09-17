from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from jarvis.skills.registry import SkillRegistry
from jarvis.tools.registry import ToolRegistry


_ID = re.compile(r"^[a-z][a-z0-9_-]{1,63}$")
_VERSION = re.compile(r"^\d+\.\d+\.\d+$")


@dataclass(frozen=True, slots=True)
class PluginManifest:
    plugin_id: str
    name: str
    version: str
    description: str
    required_skills: tuple[str, ...]
    required_tools: tuple[str, ...]
    source: Path
    available: bool
    missing: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.plugin_id, "name": self.name, "version": self.version,
            "description": self.description,
            "required_skills": list(self.required_skills),
            "required_tools": list(self.required_tools),
            "available": self.available, "missing": list(self.missing),
        }


class PluginRegistry:
    """Validates capability packs while keeping executable code deny-by-default."""

    def __init__(
        self, root: Path, tools: ToolRegistry, skills: SkillRegistry,
    ) -> None:
        self.root = root
        self.tools = tools
        self.skills = skills
        self._plugins: dict[str, PluginManifest] = {}
        self._errors: list[str] = []

    def discover(self) -> tuple[PluginManifest, ...]:
        self._plugins.clear()
        self._errors.clear()
        if not self.root.is_dir():
            return ()
        for path in sorted(self.root.glob("*/plugin.json")):
            try:
                manifest = self._load(path)
                if manifest.plugin_id in self._plugins:
                    raise ValueError(f"duplicate plugin id {manifest.plugin_id}")
                self._plugins[manifest.plugin_id] = manifest
            except (OSError, ValueError, TypeError, json.JSONDecodeError) as exc:
                self._errors.append(f"{path.parent.name}: {exc}")
        return self.list()

    def _load(self, path: Path) -> PluginManifest:
        raw = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(raw, dict) or raw.get("format") != "jarvis-plugin-v1":
            raise ValueError("unsupported plugin manifest format")
        plugin_id = str(raw.get("id", ""))
        version = str(raw.get("version", ""))
        if not _ID.fullmatch(plugin_id) or not _VERSION.fullmatch(version):
            raise ValueError("invalid plugin id or version")
        required_skills = tuple(dict.fromkeys(str(value) for value in raw.get("required_skills", [])))
        required_tools = tuple(dict.fromkeys(str(value) for value in raw.get("required_tools", [])))
        missing_skills = [f"skill:{name}" for name in required_skills if not (self.skills.get(name) and self.skills.get(name).available)]
        missing_tools = [f"tool:{name}" for name in required_tools if self.tools.spec(name) is None]
        missing = tuple(missing_skills + missing_tools)
        if raw.get("entrypoint"):
            raise ValueError("executable entrypoints require a future signed-plugin loader")
        return PluginManifest(
            plugin_id, str(raw.get("name", plugin_id)), version,
            str(raw.get("description", "")), required_skills, required_tools,
            path, not missing, missing,
        )

    def list(self) -> tuple[PluginManifest, ...]:
        return tuple(self._plugins[key] for key in sorted(self._plugins))

    def report(self) -> dict[str, Any]:
        plugins = self.list()
        return {
            "status": "ready" if not self._errors else "degraded",
            "discovered": len(plugins),
            "available": sum(plugin.available for plugin in plugins),
            "plugins": [plugin.to_dict() for plugin in plugins],
            "errors": list(self._errors),
            "code_loading": "disabled-by-default",
        }
