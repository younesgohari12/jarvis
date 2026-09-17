"""Extensible, permission-aware tool system with lazy exports."""

from typing import Any

__all__ = ["ToolRegistry", "ToolSpec"]


def __getattr__(name: str) -> Any:
    if name in __all__:
        from .registry import ToolRegistry, ToolSpec

        return {"ToolRegistry": ToolRegistry, "ToolSpec": ToolSpec}[name]
    raise AttributeError(name)
