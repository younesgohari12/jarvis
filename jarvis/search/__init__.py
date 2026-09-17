"""Lightweight public web search with lazy public exports."""

from typing import Any

__all__ = ["SearchEngine", "SearchEvidence", "SearchReport"]


def __getattr__(name: str) -> Any:
    if name in __all__:
        from .engine import SearchEngine, SearchEvidence, SearchReport

        return {
            "SearchEngine": SearchEngine,
            "SearchEvidence": SearchEvidence,
            "SearchReport": SearchReport,
        }[name]
    raise AttributeError(name)
