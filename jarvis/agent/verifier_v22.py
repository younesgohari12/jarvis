"""JARVIS v22 — Universal Verifier V2 (v22.4 TRUE ROOT-CAUSE HARDENED).

The independent challenger. v21's verifier repeated the parser's semantic
reading (correlated failure). V2 adds witnesses that are independent of the
parser interpretation:

  * typed-dimension audit    (quantity_v22 — USD never adds to ITEM)
  * operation-aware dimension algebra (semantics_v22_4 — the SAME dimension
    pair may be legal for one operator and illegal for another: USD/hour x
    hour is legal, hour + USD is not; the v22.3 blanket {time,count} /
    {time,currency} exemptions are REMOVED at the root)
  * expressive numeric-role audit (numeric_roles_v22 — slot swaps caught)
  * temporal witness         (temporal_v22 — 23:00 + 2h must be 01:00, not 25)

v21 checks are inherited verbatim and can never be weakened: V2 only ADDS
failures. A typed failure always ends in abstention/clarification — never a
silently "fixed" answer.

v22.4 additions:
  * structured DimensionFailure metadata on every dimension rejection
    (operator, left/right dimension + unit) so the renderer never guesses
    (spec §22/§23);
  * the immutable ORIGINAL_SOURCE stays authoritative (spec §43);
  * fail-closed behaviour is preserved (spec §45).
"""
from __future__ import annotations

from contextvars import ContextVar

from jarvis.agent.verification_v21 import UniversalVerifier, AUTHORITATIVE_SOURCE
from jarvis.agent.verification_v20 import Verdict, close
from jarvis.agent.quantity_v22 import (
    Quantity, UnitIncompatibilityError, extract_typed_quantities,
    validate_operation_chain, dimension_verdict, compatible,
)
from jarvis.agent.semantics_v22_4 import (
    DimensionFailure, cross_dimension_chain_failure_v4,
    numeric_guard, probability_guard,
)
from jarvis.agent.numeric_roles_v22 import (
    classify_source_numbers_v2, role_slot_consistency, ROLE_CHECK, SLOT_CHECK,
)
from jarvis.agent import temporal_v22
from jarvis.agent.language_brain_v22 import detect_age_difference

__all__ = ['UniversalVerifierV2', 'dimension_verdict', 'UnitIncompatibilityError',
           'ORIGINAL_SOURCE', 'LAST_DIMENSION_FAILURE']

# v22.1 — the immutable user source. Every v22 witness reads the ORIGINAL
# user text through this context; NLU-normalized text is parse-assistance
# only and can never become the verification source (P0 fix: a paraphrase
# must not be verified against itself).
ORIGINAL_SOURCE = ContextVar('jarvis_v22_original_source', default=None)

# v22.4 — structured failure channel: the verifier records the EXACT
# DimensionFailure of the last audit so the renderer can speak the real
# dimension pair (never a guessed 'currency_item').
LAST_DIMENSION_FAILURE = ContextVar('jarvis_v22_dimension_failure', default=None)

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
    r'|how\s+many\s+years\s+(?:older|younger)|what\s+is\s+the\s+age\s+difference'
    r'|how\s+many\s+years\s+apart|فاصله\s*سن', _re.I)

# Add/subtract chain language that must never span two different known dims.
CHAIN_LANGUAGE = _re.compile(
    r'جمع\s*کن|جمع\s*می?شود|جمع\s*می\u200cشود|اضافه\s*کن|کم\s*کن|کم\s*می?شود|'
    r'کم\s*می\u200cشود|\badd\b|\bplus\b|\bsubtract\b|\bsum\b|\btotal\b', _re.I)


def cross_dimension_chain_failure(quantities, source_text: str) -> list:
    """v22.4: operation-aware replacement for the v22.3 blanket exemptions.

    Every explicit add/subtract request is validated PAIRWISE with
    validate_binary_operation — there is no globally whitelisted dimension
    pair any more. 'هر روز 5 کالا اضافه می شود؛ بعد از 3 روز' stays legal
    because the 3 روز is a temporal qualifier, not an addition operand;
    '2 hours + 5 USD' is now refused (spec §5).

    Returns a list of check names (backward compatible); the structured
    DimensionFailure objects are available through
    cross_dimension_chain_failure_structured.
    """
    failures = cross_dimension_chain_failure_v4(source_text or '', quantities)
    if not failures:
        return []
    return ['source_operation_dimension_consistency']


def cross_dimension_chain_failure_structured(quantities, source_text: str) -> list[DimensionFailure]:
    """Structured variant used by the verifier to attach exact metadata."""
    return cross_dimension_chain_failure_v4(source_text or '', quantities)


class UniversalVerifierV2(UniversalVerifier):
    """v21 verifier + independent typed witnesses. Additive only."""

    # ------------------------------------------------------------------
    def verify(self, ir, candidate):
        # v22.4: numeric answer invariants — NaN/Infinity never verify (§50).
        if candidate is not None and not isinstance(candidate, (list, tuple, dict)):
            guard = numeric_guard(candidate)
            if guard:
                verdict = Verdict(False, [guard],
                                  'The computed answer is not a finite number; it cannot be verified.',
                                  'abstain', ['numeric_answer_guard'])
                return self._finalize(verdict)
            if isinstance(candidate, float) or isinstance(candidate, int):
                # probability-typed answers must stay in [0,1]
                try:
                    ir_answer_type = getattr(ir, 'answer_type', '')
                except Exception:
                    ir_answer_type = ''
                if ir_answer_type == 'probability':
                    pguard = probability_guard(candidate)
                    if pguard:
                        return self._finalize(Verdict(
                            False, [pguard],
                            'Probability answers must remain within [0,1].',
                            'abstain', ['probability_invariant']))
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
        return self._finalize(verdict)

    # ------------------------------------------------------------------
    def _finalize(self, verdict: Verdict) -> Verdict:
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

        # -- 1b. operation-aware cross-dimension addition guard (v22.4) ----
        # '2 ساعت و 120 کیلومتر را جمع کن' must never compute; '2 hours +
        # 5 USD' is refused too; temporal qualifiers ('بعد از 3 روز') are
        # correctly excluded instead of whitelisting dimension pairs.
        structured = cross_dimension_chain_failure_structured(quantities, source_text)
        for name in cross_dimension_chain_failure(quantities, source_text):
            verdict.failed_checks.append(name)
        if structured:
            failure = structured[0]
            LAST_DIMENSION_FAILURE.set(failure)
            verdict.checks.append('structured_dimension_failure')

        # -- 2. operation-chain dimension audit ----------------------------
        slots = getattr(ir, 'slots', None) or {}
        operations = getattr(ir, 'operations', None) or []
        initial = slots.get('initial')
        if initial is not None and operations:
            iq = self._quantity_for(quantities, initial, prefer=('currency', 'count'))
            # v23 ROOT-CAUSE FIX: an UNTYPED initial (e.g. a parsed 0) no
            # longer skips the chain audit — the algebra adopts the first
            # typed operand and keeps auditing ('150 دلار + 20 کالا' from a
            # zero start is still a dimension violation).
            if iq is None or iq.dimension in ('currency', 'count'):
                verdict.checks.append('source_operation_dimension_consistency')
                from jarvis.agent.quantity_v22 import Quantity as _Q
                chain_initial = iq if iq is not None else _Q(0.0, 'dimensionless', '')
                for name in validate_operation_chain(chain_initial, operations, quantities):
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
            elif re.search(r'\b\d{1,2}:\d{2}\b|\b\d{1,2}\s*o\u2019?clock\b', t, re.I):
                # v22.2: a bare clock token ("It is 23:00 now.") also names the
                # start instant even without a ساعت/at cue — accept it as the
                # start witness ONLY when the cued form is absent.
                cm = re.search(r'\b(\d{1,2}):(\d{2})\b|\b(\d{1,2})\s*o\u2019?clock\b', t, re.I)
                hh = cm.group(1) or cm.group(3)
                mm = cm.group(2) or 0
                start = f"{int(hh):02d}:{int(mm):02d}"
            durations = []
            consumed: list[tuple[int, int]] = []
            for dm in re.finditer(
                    r'(?:مدت(?:\s*کار)?|duration|دیرش)\D{0,12}?(\d+(?:\.\d+)?)\s*(ساعت|دقیقه|hours?|minutes?|hr|min)'
                    r'|\b(\d+(?:\.\d+)?)\s*(hours?|minutes?)\s*(?:of\s*work)?'
                    r'|\b(\d+(?:\.\d+)?)\s*(ساعت|دقیقه)\s*(?:طول|تمام|کشید)'
                    r'|\b(?:lasts?|takes?)\s+(\d+(?:\.\d+)?)\s*(hours?|minutes?)'
                    # v22.4: bare FA compound durations ('3 ساعت و 30 دقیقه')
                    r'|\b(\d+(?:\.\d+)?)\s*(ساعت|دقیقه|ثانیه)\b', t, re.I):
                value = dm.group(1) or dm.group(3) or dm.group(5) or dm.group(7) \
                    or dm.group(9)
                unit = dm.group(2) or dm.group(4) or dm.group(6) or dm.group(8) \
                    or dm.group(10)
                # a span may only be counted once (the cued alternative wins)
                if any(dm.start() < e and s < dm.end() for s, e in consumed):
                    continue
                consumed.append((dm.start(), dm.end()))
                unit = str(unit).lower()
                if unit.startswith(('ساعت', 'hour', 'hr', 'h')):
                    durations.append((float(value), 'hour'))
                elif unit.startswith(('ثانیه', 'sec')):
                    durations.append((float(value) / 3600.0, 'hour'))
                else:
                    durations.append((float(value), 'minute'))
            # v22.4 (spec §37): an END frame ('ends at 22:00. It lasted 2h.')
            # computes the START: expected = end − Σdurations.
            end_frame = bool(re.search(
                r'(?:ends?|finishes?)\s*(?:ساعت|at)?\s*\d'
                r'|تمام\s*می[\s\u200c]*شود|به\s*پایان\s*می[\s\u200c]*رسد', t, re.I))
            if end_frame:
                em = re.search(r'(?:ends?|finishes?)\s*(?:ساعت|at)?\s*(\d{1,2})(?::(\d{2}))?'
                               r'|تمام\s*می[\s\u200c]*شود\s*(?:ساعت)?\s*(\d{1,2})(?::(\d{2}))?', t, re.I)
                if em:
                    hh = em.group(1) or em.group(3)
                    mm = em.group(2) or em.group(4) or 0
                    end_clock = temporal_v22.parse_clock_time(f'{int(hh):02d}:{int(mm):02d}')
                    expected = temporal_v22.subtract_duration(
                        end_clock, *[temporal_v22.Duration.of(v, u) for v, u in durations])
                    observed = clock_result if isinstance(clock_result, temporal_v22.ClockTime) \
                        else temporal_v22.parse_clock_time(clock_result)
                    passed = (expected.iso() == observed.iso()
                              and expected.day_offset == observed.day_offset)
                    return Verdict(passed,
                                   [] if passed else ['temporal_calendar_wrap', 'temporal_consistency'],
                                   '' if passed else 'End − duration must equal the start (calendar-aware).',
                                   'none' if passed else 'response',
                                   ['temporal_world_model', 'calendar_wrap', 'end_frame'])
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
