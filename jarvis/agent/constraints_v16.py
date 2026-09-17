from __future__ import annotations

import re

from jarvis.agent.constraints_v15 import (
    ConstraintAwareComposer as _ConstraintAwareComposerV15,
    ConstraintExtractor as _ConstraintExtractorV15,
    ResponseConstraints,
    ResponseVerifier,
    VerificationResult,
)


class ConstraintExtractor(_ConstraintExtractorV15):
    """v16 semantic variants for must-include constraints."""

    _INCLUDE = _ConstraintExtractorV15._INCLUDE + (
        re.compile(
            r"(?:حتماً|حتما|باید)\s*(?:کلمه|واژه|عبارت)?\s*[«\"']([^»\"']{1,60})[»\"']\s*(?:را\s*)?"
            r"(?:داشته\s+باشد|در\s+(?:متن|پاسخ)\s+باشد|بیاور|ذکر\s+کن|استفاده\s+کن|قرار\s+بده)",
            re.I,
        ),
        re.compile(
            r"(?:کلمه|واژه|عبارت)\s*[«\"']([^»\"']{1,60})[»\"'].{0,24}?(?:حتماً|حتما|باید).{0,16}?(?:باشد|بیاید|ذکر\s+شود|استفاده\s+شود)",
            re.I,
        ),
        re.compile(
            r"\b(?:must\s+(?:include|contain|have)|make\s+sure\s+(?:it\s+)?(?:includes?|contains?))\s+(?:the\s+)?(?:word|phrase)?\s*[\"']([^\"']{1,60})[\"']",
            re.I,
        ),
    )


class ConstraintAwareComposer(_ConstraintAwareComposerV15):
    pass


__all__ = ["ConstraintExtractor", "ResponseConstraints", "ResponseVerifier", "VerificationResult", "ConstraintAwareComposer"]
