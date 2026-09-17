from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum


class PermissionLevel(IntEnum):
    READ = 0
    SAFE = 1
    MODIFY = 2
    DESTRUCTIVE = 3
    CRITICAL = 4

    @property
    def code(self) -> str:
        return f"L{int(self)}"


@dataclass(frozen=True, slots=True)
class PermissionDecision:
    allowed: bool
    requires_confirmation: bool
    reason: str = ""
    level: str = "SAFE"


class PermissionLayer:
    READ = frozenset({"read_only", "information", "network", "filesystem_read"})
    SAFE = frozenset({"external_app", "caution"})
    MODIFY = frozenset(
        {
            "write_file", "rename", "move", "sensitive_action", "ui_control",
            "browser_interaction", "clipboard_access", "process_control",
        }
    )
    DESTRUCTIVE = frozenset(
        {"destructive", "delete", "system_control", "process_terminate", "clipboard_destructive"}
    )
    CRITICAL = frozenset({"power_control", "shell", "privileged"})

    @classmethod
    def level_for(cls, category: str) -> PermissionLevel | None:
        normalized = category.casefold().strip()
        for level, categories in (
            (PermissionLevel.READ, cls.READ),
            (PermissionLevel.SAFE, cls.SAFE),
            (PermissionLevel.MODIFY, cls.MODIFY),
            (PermissionLevel.DESTRUCTIVE, cls.DESTRUCTIVE),
            (PermissionLevel.CRITICAL, cls.CRITICAL),
        ):
            if normalized in categories:
                return level
        return None

    def check(self, category: str, confirmed: bool = False) -> PermissionDecision:
        normalized = category.casefold().strip()
        level = self.level_for(normalized)
        if level is PermissionLevel.READ:
            return PermissionDecision(True, False, level=level.code)
        if level is PermissionLevel.SAFE:
            return PermissionDecision(True, False, "Safe action is logged", level.code)
        if level is PermissionLevel.MODIFY:
            if confirmed:
                return PermissionDecision(True, True, level=level.code)
            return PermissionDecision(
                False, True, "A modifying action needs explicit user confirmation", level.code
            )
        if level in {PermissionLevel.DESTRUCTIVE, PermissionLevel.CRITICAL}:
            if confirmed:
                return PermissionDecision(True, True, level=level.code)
            return PermissionDecision(
                False, True,
                f"{level.code} operation blocked until explicit confirmation",
                level.code,
            )
        if normalized == "denied":
            return PermissionDecision(False, False, "Operation is blocked by policy", "BLOCKED")
        return PermissionDecision(False, False, "Unknown permission category", "BLOCKED")
