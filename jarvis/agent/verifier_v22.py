"""JARVIS v22 — Universal Verifier V2.

The independent challenger. v21's verifier repeated the parser's semantic
reading (correlated failure). V2 adds witnesses that are independent of the
parser interpretation:

  * typed-dimension audit    (quantity_v22 — USD never adds to ITEM)
  * expressive numeric-role audit (numeric_roles_v22 — slot swaps caught)
  * temporal witness         (temporal_v22 — 23:00 + 2h must be 01:00, not 25)

v21 checks are inherited verbatim and can never be weakened: V2 only ADDS
failures. A typed failure always ends in abstention/clarification — never a
silently "fixed" answer.
"""
from __future__ import annotations

from contextvars import ContextVar

from jarvis.agent.verification_v21 import UniversalVerifier, AUTHORITATIVE_SOURCE
from jarvis.agent.verification_v20 import Verdict, close
from jarvis.agent.quantity_v22 import (
    Quantity, UnitIncompatibilityError, extract_typed_quantities,
    validate_operation_chain, dimension_verdict, compatible,
)
from jarvis.agent.numeric_roles_v22 import (
    classify_source_numbers_v2, role_slot_consistency, ROLE_CHECK, SLOT_CHECK,
)
from jarvis.agent import temporal_v22
from jarvis.agent.language_brain_v22 import detect_age_difference

__all__ = ['UniversalVerifierV2', 'dimension_verdict', 'UnitIncompatibilityError',
           'ORIGINAL_SOURCE']

# v22.1 — the immutable user source. Every v22 witness reads the ORIGINAL
# user text through this context; NLU-normalized text is parse-assistance
# only and can never become the verification source (P0 fix: a paraphrase
# must not be verified against itself).
ORIGINAL_SOURCE = ContextVar('jarvis_v22_original_source', default=None)

# Failure names introduced by the V2 audit layer.
TYPED_FAILURES = {
    'currency_dimension_mismatch',
    'source_operation_dimension_consistency',
    'source_currency_consistency',
    'typed_audit_internal_error',
    ROLE_CHECK,
    SLOT_CHECK,
}

REPAIR_HINTS = {
    'currency_dimension_mismatch': 'Do not add item counts to a currency balance; convert or separate the chains.',
    'source_operation_dimension_consistency': 'Operation values must share the dimension of the initial quantity.',
    'source_currency_consistency': 'Two different currencies never add without an explicit exchange rate.',
    ROLE_CHECK: 'Slots must honour source order: first workers count is workers_initial, second is workers_target.',
    SLOT_CHECK: 'Every numeric slot must be traceable to a span in the immutable user source.',
    'typed_audit_internal_error': 'The typed audit layer failed internally; the result cannot be verified and is discarded (fail-closed).',
}

# Tasks where the slot-traceability witness is enforced. Other tasks keep the
# full v21 invariant set (word-derived slots like 'پنجم'->n or unit-converted
# time slots stay untouched).
TRACEABLE_TASKS = {'finance', 'inventory', 'work_rate'}

CHAIN_OPS = ('add', 'subtract', 'inventory_add', 'inventory_remove', 'credit', 'debit')

import re as _re
COUNT_LANGUAGE = _re.compile(
    r'چند\s*گروه|به\s*چند\s*روش|چند\s*نفره|ways\s*to\s*choose|how\s*many\s+(?:groups|ways)', _re.I)
DIFFERENCE_LANGUAGE = _re.compile(
    r'چند\s*سال\s*(?:بزرگ|کوچک)|اختلاف[^؟?;؛]{0,15}چند|چند[^؟?;؛]{0,15}اختلاف'
    r'|how\s+many\s+years\s+(?:older|younger)|what\s+is\s+the\s+age\s+difference', _re.I)

# Add/subtract chain language that must never span two different known dims.
CHAIN_LANGUAGE = _re.compile(
    r'جمع\s*کن|جمع\s*می?شود|جمع\s*می\u200cشود|اضافه\s*کن|کم\s*کن|کم\s*می?شود|'
    r'کم\s*می\u200cشود|\badd\b|\bplus\b|\bsubtract\b|\bsum\b|\btotal\b', _re.I)
_CHAIN_SAFE_DIMS = {'dimensionless', 'identifier', 'clock_time', 'percentage',
                    'probability', 'date'}


def cross_dimension_chain_failure(quantities, source_text: str) -> list:
    """v22.1: an explicit add/subtract request over two KNOWN different
    dimensions (hours + km, kg + liters, USD + items, ...) is refused.
    Untyped (dimensionless) numbers never trigger this guard."""
    if not source_text or not CHAIN_LANGUAGE.search(source_text):
        return []
    known = {q.dimension for q in quantities if q.dimension not in _CHAIN_SAFE_DIMS}
    if len(known) < 2:
        return []
    # rate-style contexts (per hour/day) legitimately mix time with counts or
    # money; every other known-dimension pair inside one add/subtract request
    # is refused (hours+km, kg+liters, USD+items, ...).
    if known <= {'time', 'count'} or known <= {'time', 'currency'}:
        return []
    return ['source_operation_dimension_consistency']


class UniversalVerifierV2(UniversalVerifier):
    """v21 verifier + independent typed witnesses. Additive only."""

    # ------------------------------------------------------------------
    def verify(self, ir, candidate):
        verdict = super().verify(ir, candidate)
        try:
            self._typed_audit(ir, verdict)
        except Exception as exc:  # v22.1: FAIL-CLOSED, never fail-open.
            # A safety verifier that crashes must not fall back to the
            # unchallenged v21 verdict: an internal audit error means the
            # result CANNOT be verified, so the engine abstains/clarifies.
            verdict.checks.append('typed_audit_internal_error')
            verdict.failed_checks.append('typed_audit_internal_error')
            verdict.repair_stage = 'abstain'
            verdict.repair_hint = (
                REPAIR_HINTS['typed_audit_internal_error']
                + f' ({type(exc).__name__})')
        if verdict.failed_checks:
            verdict.failed_checks = list(dict.fromkeys(verdict.failed_checks))
            verdict.checks = list(dict.fromkeys(verdict.checks))
            verdict.passed = False
            if any(f in TYPED_FAILURES for f in verdict.failed_checks):
                verdict.repair_stage = 'abstain'
                verdict.repair_hint = next(
                    (REPAIR_HINTS[f] for f in verdict.failed_checks if f in REPAIR_HINTS),
                    'Clarify dimensions before computing.')
        return verdict

    # ------------------------------------------------------------------
    def _typed_audit(self, ir, verdict: Verdict):
        # v22.1: the typed audit ALWAYS reads the immutable original user
        # source — never a normalized/paraphrased rewrite of it.
        source_text = ORIGINAL_SOURCE.get() or AUTHORITATIVE_SOURCE.get() \
            or ir.source_text or ''
        if not source_text:
            return
        quantities = extract_typed_quantities(source_text)
        verdict.checks += ['typed_quantity_audit']

        # -- 1. whole-source dimension audit (currencies vs counts) --------
        dv = dimension_verdict(quantities)
        for name in dv['failed_checks']:
            verdict.failed_checks.append(name)

        # -- 1b. cross-dimension addition guard (v22.1) ---------------------
        # '2 ساعت و 120 کیلومتر را جمع کن' must never compute: an explicit
        # add/subtract request over two KNOWN different dimensions fails.
        for name in cross_dimension_chain_failure(quantities, source_text):
            verdict.failed_checks.append(name)

        # -- 2. operation-chain dimension audit ----------------------------
        slots = getattr(ir, 'slots', None) or {}
        operations = getattr(ir, 'operations', None) or []
        initial = slots.get('initial')
        if initial is not None and operations:
            iq = self._quantity_for(quantities, initial, prefer=('currency', 'count'))
            if iq is not None and iq.dimension in ('currency', 'count'):
                verdict.checks.append('source_operation_dimension_consistency')
                for name in validate_operation_chain(iq, operations, quantities):
                    verdict.failed_checks.append(name)

        # -- 3. expressive numeric-role audit ------------------------------
        roles = classify_source_numbers_v2(source_text)
        verdict.checks.append('numeric_role_audit')
        ir_dict = ir.to_dict() if hasattr(ir, 'to_dict') else dict(ir)
        rc = role_slot_consistency(ir_dict, roles)
        # slot-traceability only for the tasks listed above (false-positive guard)
        rc['failed_checks'] = [f for f in rc['failed_checks']
                               if f == ROLE_CHECK or ir_dict.get('task') in TRACEABLE_TASKS]
        # counting-language guard: 'چند گروه X نفره' must never be read as a
        # money/graph chain — it is a counting question and needs n/k slots.
        if COUNT_LANGUAGE.search(source_text) and ir_dict.get('task') in (
                'graph', 'finance', 'inventory', 'speed', 'age'):
            rc['failed_checks'].append(ROLE_CHECK)
        # difference-language guard: an age question asking the DIFFERENCE
        # ('چند سال بزرگتر از', 'age difference') must never be answered with
        # a sum/relative interpretation.
        if DIFFERENCE_LANGUAGE.search(source_text) and ir_dict.get('task') == 'age' \
                and (ir_dict.get('slots', {}) or {}).get('query') != 'difference':
            rc['failed_checks'].append(ROLE_CHECK)
        for name in rc['failed_checks']:
            verdict.failed_checks.append(name)

    # ------------------------------------------------------------------
    def verify_age_difference(self, source_text, detection, value) -> Verdict:
        """Independent entity-bound witness for the v22 age-difference
        operation: re-read the IMMUTABLE source, re-detect the two named
        ages, and confirm answer == abs(age_a - age_b)."""
        import re
        try:
            source_text = ORIGINAL_SOURCE.get() or source_text
            t = (source_text or '').translate(
                str.maketrans('۰۱۲۳۴۵۶۷۸۹٠١٢٣٤٥٦٧٨٩', '01234567890123456789'))
            if not DIFFERENCE_LANGUAGE.search(t):
                return Verdict(False, ['age_difference_language_missing'],
                               'No age-difference question in the immutable source.',
                               'none', ['age_difference'])
            fresh = detect_age_difference(t)
            if not fresh:
                return Verdict(False, ['age_difference_entities_missing'],
                               'Two named ages are required.', 'none',
                               ['age_difference'])
            (name_a, age_a), (name_b, age_b) = fresh
            expected = abs(float(age_a) - float(age_b))
            passed = (close(float(value), expected) and 0 <= expected <= 150
                      and str(name_a).strip() != str(name_b).strip())
            return Verdict(passed,
                           [] if passed else ['age_difference_consistency'],
                           '' if passed else 'Answer must equal the absolute age difference of the two bound entities.',
                           'none' if passed else 'abstain',
                           ['age_difference', 'entity_binding'])
        except Exception:
            return Verdict(False, ['age_difference_verification_error'],
                           'Age-difference witness failed internally.',
                           'abstain', ['age_difference'])

    # ------------------------------------------------------------------
    @staticmethod
    def _quantity_for(quantities, value, prefer=()):
        matches = [q for q in quantities if abs(float(q.value) - float(value)) < 1e-9]
        for dim in prefer:
            for q in matches:
                if q.dimension == dim:
                    return q
        return matches[0] if matches else None

    # ------------------------------------------------------------------
    def verify_temporal(self, source_text, clock_result, language='fa') -> Verdict:
        """Independent temporal witness: recompute start+Σdurations from the
        raw source and compare with the world-model result (calendar aware)."""
        import re
        source_text = ORIGINAL_SOURCE.get() or source_text  # immutable source
        try:
            t = (source_text or '').translate(str.maketrans('۰۱۲۳۴۵۶۷۸۹', '0123456789'))
            start = None
            m = re.search(r'(?:ساعت|at)\s*(\d{1,2})(?::(\d{2}))?', t, re.I)
            if m:
                start = f"{int(m.group(1)):02d}:{int(m.group(2) or 0):02d}"
            durations = []
            for dm in re.finditer(
                    r'(?:مدت(?:\s*کار)?|duration|دیرش)\D{0,12}?(\d+(?:\.\d+)?)\s*(ساعت|دقیقه|hours?|minutes?|hr|min)'
                    r'|\b(\d+(?:\.\d+)?)\s*(hours?|minutes?)\s*(?:of\s*work)?'
                    r'|\b(\d+(?:\.\d+)?)\s*(ساعت|دقیقه)\s*(?:طول|تمام|کشید)'
                    r'|\b(?:lasts?)\s+(\d+(?:\.\d+)?)\s*(hours?|minutes?)', t, re.I):
                value = dm.group(1) or dm.group(3) or dm.group(5) or dm.group(7)
                unit = dm.group(2) or dm.group(4) or dm.group(6) or dm.group(8)
                unit = str(unit).lower()
                if unit.startswith(('ساعت', 'hour', 'hr', 'h')):
                    durations.append((float(value), 'hour'))
                else:
                    durations.append((float(value), 'minute'))
            if start is None or not durations:
                return Verdict(False, ['temporal_source_incomplete'],
                               'Start time or duration missing', 'none',
                               ['temporal_world_model'])
            clock = temporal_v22.parse_clock_time(start)
            expected = temporal_v22.add_duration(
                clock, *[temporal_v22.Duration.of(v, u) for v, u in durations])
            observed = clock_result if isinstance(clock_result, temporal_v22.ClockTime) \
                else temporal_v22.parse_clock_time(clock_result)
            passed = (expected.iso() == observed.iso()
                      and expected.day_offset == observed.day_offset
                      and 0 <= observed.hour < 24)
            return Verdict(passed,
                           [] if passed else ['temporal_calendar_wrap', 'temporal_consistency'],
                           '' if passed else 'Clock arithmetic must wrap at 24h with day_offset.',
                           'none' if passed else 'response',
                           ['temporal_world_model', 'calendar_wrap'])
        except Exception:
            return Verdict(False, ['temporal_verification_error'], 'Temporal witness failed',
                           'none', ['temporal_world_model'])
