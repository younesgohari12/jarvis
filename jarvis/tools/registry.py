from __future__ import annotations

import threading
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from jarvis.agent.permissions import PermissionLayer, PermissionLevel


class ToolError(RuntimeError):
    pass


class ArgumentValidationError(ToolError):
    def __init__(self, tool: str, missing: tuple[str, ...]) -> None:
        self.tool = tool
        self.missing = missing
        super().__init__(f"Tool arguments are incomplete for {tool}: {', '.join(missing)}")


ToolHandler = Callable[[dict[str, Any]], Any]
RiskResolver = Callable[[dict[str, Any]], str]


@dataclass(frozen=True, slots=True)
class ToolSpec:
    name: str
    description: str
    category: str
    handler: ToolHandler
    required: tuple[str, ...] = ()
    optional: tuple[str, ...] = ()
    risk: str = "safe"
    semantic_actions: tuple[str, ...] = ()
    entity_types: tuple[str, ...] = ()
    risk_resolver: RiskResolver | None = None

    def effective_permission_category(self, arguments: dict[str, Any] | None = None) -> str:
        """Resolve dynamic risk and enforce the declarative risk floor centrally."""
        category = self.risk_resolver(arguments or {}) if self.risk_resolver else self.category
        level = PermissionLayer.level_for(category)
        risk = self.risk.casefold().strip()
        if risk == "dangerous" and (level is None or level < PermissionLevel.DESTRUCTIVE):
            return "destructive"
        if risk == "confirm" and (level is None or level < PermissionLevel.MODIFY):
            return "sensitive_action"
        return category

    def schema(self) -> dict[str, Any]:
        effective = self.effective_permission_category({})
        level = PermissionLayer.level_for(effective)
        return {
            "name": self.name,
            "required": list(self.required),
            "optional": list(self.optional),
            "risk": self.risk,
            "actions": list(self.semantic_actions),
            "entity_types": list(self.entity_types),
            "permission": effective,
            "permission_level": level.code if level is not None else "BLOCKED",
            "dynamic_risk": self.risk_resolver is not None,
        }


class ToolRegistry:
    def __init__(self, permissions: PermissionLayer | None = None) -> None:
        self._permissions = permissions or PermissionLayer()
        self._tools: dict[str, ToolSpec] = {}
        self._lock = threading.RLock()

    def register(self, tool: ToolSpec) -> None:
        key = tool.name.casefold().strip()
        if not key or any(character.isspace() for character in key):
            raise ToolError("Tool name must be a non-empty identifier without spaces")
        with self._lock:
            if key in self._tools:
                raise ToolError(f"Tool is already registered: {tool.name}")
            self._tools[key] = tool

    def names(self) -> tuple[str, ...]:
        with self._lock:
            return tuple(sorted(tool.name for tool in self._tools.values()))

    def describe(self) -> tuple[tuple[str, str, str], ...]:
        with self._lock:
            return tuple(
                (tool.name, tool.description, tool.category)
                for tool in sorted(self._tools.values(), key=lambda item: item.name)
            )

    def capabilities(self) -> tuple[dict[str, Any], ...]:
        with self._lock:
            return tuple(
                tool.schema() for tool in sorted(self._tools.values(), key=lambda item: item.name)
            )

    def spec(self, name: str) -> ToolSpec | None:
        with self._lock:
            return self._tools.get(name.casefold().strip())

    def select(self, action: str, entity_type: str) -> ToolSpec | None:
        with self._lock:
            candidates = [
                tool for tool in self._tools.values()
                if action in tool.semantic_actions
                and (not tool.entity_types or entity_type in tool.entity_types)
            ]
        if not candidates:
            return None
        return sorted(
            candidates,
            key=lambda item: (entity_type not in item.entity_types, len(item.required), item.name),
        )[0]

    def validate(self, name: str, arguments: dict[str, Any] | None = None) -> tuple[str, ...]:
        tool = self.spec(name)
        if tool is None:
            raise ToolError(f"Unknown tool: {name}")
        supplied = arguments or {}
        missing = tuple(
            key for key in tool.required
            if key not in supplied
            or supplied[key] is None
            or (isinstance(supplied[key], str) and not supplied[key].strip())
            or (isinstance(supplied[key], (list, tuple, dict, set)) and not supplied[key])
        )
        return missing

    def invoke(
        self,
        name: str,
        arguments: dict[str, Any] | None = None,
        *,
        confirmed: bool = False,
    ) -> Any:
        with self._lock:
            tool = self._tools.get(name.casefold().strip())
        if tool is None:
            raise ToolError(f"Unknown tool: {name}")
        missing = self.validate(tool.name, arguments)
        if missing:
            raise ArgumentValidationError(tool.name, missing)
        category = tool.effective_permission_category(arguments)
        decision = self._permissions.check(category, confirmed)
        if not decision.allowed:
            raise ToolError(decision.reason)
        try:
            return tool.handler(arguments or {})
        except ToolError:
            raise
        except Exception as exc:
            raise ToolError(f"Tool {tool.name} failed: {exc}") from exc
