"""JARVIS v22 — Reflection Loop V2.

Inherits the v21 bounded repair loop (max 2 retries, targeted slot repair,
source-constrained reparse) and adds typed-repair traceability: every typed
failure carries the original IR, the failed check and a repair hint. Dimension
failures are NEVER auto-repaired into a number — they abstain (spec §8).
"""
from __future__ import annotations

from jarvis.agent.reflection_v21 import ReflectionLoopV21
from jarvis.agent.verifier_v22 import TYPED_FAILURES, REPAIR_HINTS


class ReflectionLoopV22(ReflectionLoopV21):
    def __init__(self, executor, verifier, parser, max_retries=2):
        super().__init__(executor, verifier, parser, max_retries)

    def solve(self, ir):
        candidate, verdict, attempts, world = super().solve(ir)
        if not verdict.passed and attempts:
            typed = [c for c in verdict.failed_checks if c in TYPED_FAILURES]
            if typed:
                check = typed[0]
                attempts[-1]['v22_targeted_repair'] = {
                    'original_ir': attempts[0].get('ir'),
                    'failed_check': check,
                    'repair_hint': REPAIR_HINTS.get(check, 'Clarify the problem.'),
                    'strategy': 'abstain_or_clarify',
                    'auto_numeric_repair': False,
                }
        return candidate, verdict, attempts, world
