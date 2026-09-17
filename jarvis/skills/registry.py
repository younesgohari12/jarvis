from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from jarvis.agent.permissions import PermissionLayer, PermissionLevel
from jarvis.tools.registry import ToolRegistry


_ID = re.compile(r"^[a-z][a-z0-9_-]{1,63}$")
_VERSION = re.compile(r"^\d+\.\d+\.\d+$")


@dataclass(frozen=True, slots=True)
class SkillManifest:
    skill_id: str
    name: str
    version: str
    description: str
    required_tools: tuple[str, ...]
    maximum_permission: str
    languages: tuple[str, ...]
    source: Path
    available: bool
    missing_tools: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.skill_id,
            "name": self.name,
            "version": self.version,
            "description": self.description,
            "required_tools": list(self.required_tools),
            "maximum_permission": self.maximum_permission,
            "languages": list(self.languages),
            "available": self.available,
            "missing_tools": list(self.missing_tools),
        }


class SkillRegistry:
    """Discovers declarative skills without importing untrusted Python code.

    A skill is an audited capability bundle backed by already registered tools.
    Tool invocation still passes through the central L0-L4 permission layer.
    """

    def __init__(self, root: Path, tools: ToolRegistry) -> None:
        self.root = root
        self.tools = tools
        self._skills: dict[str, SkillManifest] = {}
        self._errors: list[str] = []

    @property
    def errors(self) -> tuple[str, ...]:
        return tuple(self._errors)

    def discover(self) -> tuple[SkillManifest, ...]:
        self._skills.clear()
        self._errors.clear()
        if not self.root.is_dir():
            return ()
        for path in sorted(self.root.glob("*/skill.json")):
            try:
                manifest = self._load(path)
                if manifest.skill_id in self._skills:
                    raise ValueError(f"duplicate skill id {manifest.skill_id}")
                self._skills[manifest.skill_id] = manifest
            except (OSError, ValueError, TypeError, json.JSONDecodeError) as exc:
                self._errors.append(f"{path.parent.name}: {exc}")
        return self.list()

    def _load(self, path: Path) -> SkillManifest:
        raw = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(raw, dict) or raw.get("format") != "jarvis-skill-v1":
            raise ValueError("unsupported skill manifest format")
        skill_id = str(raw.get("id", ""))
        version = str(raw.get("version", ""))
        if not _ID.fullmatch(skill_id):
            raise ValueError("invalid skill id")
        if not _VERSION.fullmatch(version):
            raise ValueError("invalid semantic version")
        required = tuple(dict.fromkeys(str(value) for value in raw.get("required_tools", [])))
        maximum = str(raw.get("maximum_permission", "L0")).upper()
        if maximum not in {level.code for level in PermissionLevel}:
            raise ValueError("maximum_permission must be L0-L4")
        missing = tuple(name for name in required if self.tools.spec(name) is None)
        declared_level = PermissionLevel(int(maximum[1:]))
        excessive: list[str] = []
        for name in required:
            tool = self.tools.spec(name)
            if tool is None:
                continue
            actual = PermissionLayer.level_for(tool.effective_permission_category({}))
            if actual is None or actual > declared_level:
                excessive.append(name)
        if excessive:
            raise ValueError(
                f"required tools exceed declared maximum_permission {maximum}: {', '.join(excessive)}"
            )
        return SkillManifest(
            skill_id=skill_id,
            name=str(raw.get("name", skill_id)),
            version=version,
            description=str(raw.get("description", "")),
            required_tools=required,
            maximum_permission=maximum,
            languages=tuple(str(value) for value in raw.get("languages", ["fa", "en"])),
            source=path,
            available=not missing,
            missing_tools=missing,
        )

    def list(self) -> tuple[SkillManifest, ...]:
        return tuple(self._skills[key] for key in sorted(self._skills))

    def get(self, skill_id: str) -> SkillManifest | None:
        return self._skills.get(skill_id.casefold().strip())

    def report(self) -> dict[str, Any]:
        skills = self.list()
        return {
            "status": "ready" if not self._errors else "degraded",
            "discovered": len(skills),
            "available": sum(skill.available for skill in skills),
            "skills": [skill.to_dict() for skill in skills],
            "errors": list(self._errors),
            "execution_policy": "registered-tools-only; central L0-L4 checks remain authoritative",
        }
