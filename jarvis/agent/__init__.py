"""Agent orchestration layer with lazy public exports to prevent import cycles."""

from typing import Any

__all__ = ["AgentReply", "JarvisAgent"]


def __getattr__(name: str) -> Any:
    if name in __all__:
        from .core import AgentReply, JarvisAgent

        return {"AgentReply": AgentReply, "JarvisAgent": JarvisAgent}[name]
    raise AttributeError(name)
