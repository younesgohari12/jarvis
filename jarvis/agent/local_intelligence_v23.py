"""JARVIS v23 intelligence engine — VERIFIED NUMERIC SERVICES over v22.4.2.

V23_LANGUAGE_BRAIN_PLAN.md (مرحله 1 — no GPU, deterministic architecture):

  * units_service_v23   — standalone unit conversions (کارتابل واحد)
  * work_rate_v23       — work-rate algebra (scale / inversion / combined /
                          per-worker sums / mid-task crew changes)
  * narrative_math_v23  — narrative arithmetic chains measured 0/8+0/8 on the
                          v22.4.2 diagnostic

Ordering guarantees (byte-compatibility by construction):

  1. The COMPLETE v22.4.2 pipeline runs first and verbatim. Any verified
     v22 answer is returned untouched — v23 never re-interprets it.
  2. Structured dimension refusals (v22_dimension_block) are deliberate
     fail-closed verdicts and are NEVER overridden by v23 services.
  3. Only when v22 produced NO verified answer do the v23 services run, each
     with its own witness re-derivation + numeric_guard before speaking.

Nothing from v22.4.2 is removed, reinterpreted, or re-tuned.
"""
from __future__ import annotations

from jarvis.agent.local_intelligence_v22 import LocalIntelligenceV22
from jarvis.agent import units_service_v23 as units_svc
from jarvis.agent import work_rate_v23 as work_svc
from jarvis.agent import narrative_math_v23 as narr_svc


class LocalIntelligenceV23(LocalIntelligenceV22):
    VERSION = '6.0.0-v23.0'

    # ------------------------------------------------------------------
    @classmethod
    def matches(cls, text):
        return LocalIntelligenceV22.matches(text)

    # ------------------------------------------------------------------
    def _solve_inner(self, text, language, LocalIntelligenceAnswer):
        answer = super()._solve_inner(text, language, LocalIntelligenceAnswer)

        # 1) a verified v22 answer is FINAL — never re-interpreted by v23.
        if answer is not None and self.last_trace \
                and self.last_trace.get('verification', {}).get('passed'):
            return answer

        # 2) structured dimension refusals are deliberate verdicts.
        if self.last_trace and self.last_trace.get('v22_dimension_block'):
            return answer

        # 3) v23 verified numeric services (only on v22 abstain/refuse/None)
        service_answer = self._v23_services(text, language,
                                            LocalIntelligenceAnswer)
        if service_answer is not None:
            return service_answer
        return answer

    # ------------------------------------------------------------------
    def _v23_services(self, text, language, LocalIntelligenceAnswer):
        try:
            result = units_svc.solve_conversion(text, language)
            if result is not None:
                return self._accept_v23(result, units_svc.render,
                                        ('v23_units_service',
                                         'unit_conversion_service',
                                         'witness_verified'),
                                        text, language, LocalIntelligenceAnswer)
            result = work_svc.solve_work_rate(text, language)
            if result is not None:
                return self._accept_v23(result, work_svc.render,
                                        ('v23_work_rate_service',
                                         'work_rate_algebra',
                                         'witness_verified'),
                                        text, language, LocalIntelligenceAnswer)
            result = narr_svc.solve_narrative(text, language)
            if result is not None:
                return self._accept_v23(result, narr_svc.render,
                                        ('v23_narrative_service',
                                         'narrative_arithmetic',
                                         'witness_verified'),
                                        text, language, LocalIntelligenceAnswer)
        except Exception:
            return None   # a v23 failure must never break the v22 fallback
        return None

    # ------------------------------------------------------------------
    def _accept_v23(self, result, renderer, checks, text, language,
                    LocalIntelligenceAnswer):
        """Render + register a verified v23 result (trace mirrors the v24
        semantic-model contract so downstream audits stay uniform)."""
        value = float(result['value'])
        if self.last_trace is None:
            self.last_trace = {'verification': {'passed': True,
                                                'checks': list(checks),
                                                'failed_checks': []}}
        self.last_trace['verification'] = {
            'passed': True, 'checks': list(checks), 'failed_checks': []}
        self.last_trace['v23_service'] = {
            'kind': result.get('kind'), 'value': value,
            'verified': True, 'witness': 'rederivation_agrees'}
        rendered = renderer(result, language)
        return LocalIntelligenceAnswer(rendered, 'reasoned_answer', .88, checks)
