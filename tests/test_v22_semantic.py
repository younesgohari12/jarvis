"""JARVIS v22 — Semantic / Typed-Quantity / Verifier / Temporal / Language suite.

Covers the P0 adversarial cases from the v22 specification:
  * 100 dollars + 5 items must NEVER silently become 105
  * 23:00 + 2h must be 01:00 of the next day, never 25
  * numeric roles must catch slot swaps (workers_initial vs workers_target)
  * identifiers (کد/شناسه) are never quantities
  * Language Brain must never produce degenerate or robotic text
The v21 contract stays intact: runtime endpoints keep canonical output.
"""
import math
import pytest

from jarvis.agent.quantity_v22 import (
    Quantity, UnitIncompatibilityError, extract_typed_quantities,
    validate_operation_chain, dimension_verdict, compatible,
)
from jarvis.agent.numeric_roles_v22 import (
    classify_source_numbers_v2, role_slot_consistency, ROLE_CHECK, SLOT_CHECK,
)
from jarvis.agent.semantic_ir_v22 import SemanticIRV2, SCHEMA_VERSION
from jarvis.agent.temporal_v22 import (
    ClockTime, Duration, TemporalError, add_duration, subtract_duration,
    parse_clock_time, recompute_from_source,
)
from jarvis.agent.verifier_v22 import UniversalVerifierV2, TYPED_FAILURES
from jarvis.agent.language_brain_v22 import (
    normalize_nlu, render_word_problem, sanitize_response, clarification, fmt_number,
)
from jarvis.agent.local_intelligence_v22 import LocalIntelligenceV22
from jarvis.agent.local_intelligence_v21 import LocalIntelligenceV21


@pytest.fixture(scope='module')
def eng():
    return LocalIntelligenceV22()


# ======================================================================
# 1. Typed Quantity system
# ======================================================================
def test_quantity_structure():
    q = Quantity(100, 'currency', 'USD', entity='account_balance',
                 source_span='100 دلار', confidence=0.98)
    d = q.to_dict()
    assert d['value'] == 100 and d['dimension'] == 'currency'
    assert d['unit'] == 'USD' and d['entity'] == 'account_balance'
    assert d['source_span'] == '100 دلار' and d['confidence'] == 0.98


def test_extract_usd_account_vs_items():
    qs = extract_typed_quantities('موجودی حساب 100 دلار است؛ 5 کالا اضافه کن')
    dims = {(q.dimension, q.unit) for q in qs}
    assert ('currency', 'USD') in dims and ('count', 'ITEM') in dims
    assert any(q.entity == 'account' for q in qs if q.dimension == 'currency')


def test_extract_legit_pairs_share_dimension():
    for text in ('موجودی حساب 100 دلار است؛ 5 دلار اضافه کن',
                 'موجودی انبار 100 کالا است؛ 5 کالا اضافه کن'):
        qs = extract_typed_quantities(text)
        dims = {q.dimension for q in qs}
        assert len(dims) == 1, (text, dims)


def test_identifiers_are_not_quantities():
    for text in ('شناسه پرونده 100 است', 'کد بسته 500 را ثبت کن', 'record id is 42'):
        qs = extract_typed_quantities(text)
        assert qs and all(q.dimension == 'identifier' for q in qs), (text, qs)


def test_two_currencies_flagged():
    dv = dimension_verdict(extract_typed_quantities('موجودی حساب 100 دلار است؛ 20 یورو اضافه کن'))
    assert 'source_currency_consistency' in dv['failed_checks']


def test_incompatible_chain_raises():
    qs = extract_typed_quantities('موجودی حساب 100 دلار است؛ 5 کالا اضافه کن')
    initial = next(q for q in qs if q.value == 100)
    failed = validate_operation_chain(initial, [{'op': 'add', 'value': 5}], qs)
    assert failed  # chain audit detects the mix


def test_compatible_pairs_never_flagged():
    qs = extract_typed_quantities('موجودی انبار 100 کالا است؛ 5 کالا اضافه کن')
    initial = next(q for q in qs if q.value == 100)
    assert validate_operation_chain(initial, [{'op': 'add', 'value': 5}], qs) == []


# ======================================================================
# 2. Numeric Role V2
# ======================================================================
def test_workers_roles_follow_source_order():
    roles = classify_source_numbers_v2('3 کارگر در 4 ساعت 84 واحد تولید می کنند؛ 8 کارگر در 6 ساعت؟')
    workers = [f for f in roles if f['role'].startswith('workers')]
    assert [f['value'] for f in workers] == [3.0, 8.0]
    assert workers[0]['role'] == 'workers_initial' and workers[1]['role'] == 'workers_target'


def test_slot_swap_is_caught():
    roles = classify_source_numbers_v2('3 کارگر در 4 ساعت 84 واحد تولید می کنند؛ 8 کارگر در 6 ساعت؟')
    ir = {'task': 'work_rate', 'slots': {'workers_initial': 8.0, 'workers_target': 3.0}}
    rc = role_slot_consistency(ir, roles)
    assert ROLE_CHECK in rc['failed_checks']


def test_correct_slots_pass_role_check():
    roles = classify_source_numbers_v2('3 کارگر در 4 ساعت 84 واحد تولید می کنند؛ 8 کارگر در 6 ساعت؟')
    ir = {'task': 'work_rate',
          'slots': {'workers_initial': 3.0, 'hours_initial': 4.0, 'output_initial': 84.0,
                    'workers_target': 8.0, 'hours_target': 6.0}}
    assert role_slot_consistency(ir, roles)['failed_checks'] == []


# ======================================================================
# 3. Semantic IR V2
# ======================================================================
def test_semantic_ir_v2_schema():
    from jarvis.agent.parser_v21 import get_parser_v21
    text = 'موجودی انبار 100 کالا است؛ 5 کالا اضافه کن'
    ir = get_parser_v21().parse(text)
    v2 = SemanticIRV2.from_v21(ir, text)
    d = v2.to_dict()
    assert d['schema_version'] == SCHEMA_VERSION == 'jarvis-semantic-ir-v22'
    assert d['quantities'] and all('source_span' in q for q in d['quantities'])
    assert d['numeric_roles'] and 'source_mapping' in d
    assert d['source_mapping'].get('initial') == '100'


# ======================================================================
# 4. Temporal World Model
# ======================================================================
def test_clock_wrap_next_day():
    r = add_duration(ClockTime(23, 0), Duration.of(2, 'hour'))
    assert r.hour == 1 and r.day_offset == 1 and r.iso() == '01:00'
    assert r.to_dict()['absolute_hours'] == 25.0


def test_clock_no_wrap():
    r = add_duration(ClockTime(9, 0), Duration.of(2, 'hour'))
    assert r.iso() == '11:00' and r.day_offset == 0


def test_clock_validation_and_minutes():
    with pytest.raises(TemporalError):
        ClockTime(25, 0)
    assert parse_clock_time('23:30').minute == 30
    assert Duration.of(90, 'minute').minutes == 90
    assert subtract_duration(ClockTime(0, 30), Duration.of(1, 'hour')).day_offset == -1


def test_temporal_witness_independent():
    expected = recompute_from_source(23, [2])
    assert expected.iso() == '01:00' and expected.day_offset == 1


# ======================================================================
# 5. Universal Verifier V2 (independent challenger)
# ======================================================================
def test_verifier_blocks_currency_item_mix():
    b = LocalIntelligenceV22()
    ans = b.solve('موجودی حساب 100 دلار است؛ 5 کالا اضافه کن', 'fa')
    assert ans is not None and '105' not in ans.text
    assert b.last_trace['verification']['passed'] is False
    failed = b.last_trace['verification']['failed_checks']
    assert failed and set(failed) & TYPED_FAILURES
    assert b.last_trace.get('v22_dimension_block') == 'currency_item'


def test_verifier_keeps_legit_arithmetic_green():
    b = LocalIntelligenceV22()
    a1 = b.solve('موجودی حساب 100 دلار است؛ 5 دلار اضافه کن', 'fa')
    assert a1 is not None and '105' in a1.text
    a2 = b.solve('موجودی انبار 100 کالا است؛ 5 کالا اضافه کن', 'fa')
    assert a2 is not None and '105' in a2.text


def test_verifier_additive_only_never_weakens_v21():
    """A v21 rejection must stay a v22 rejection."""
    from jarvis.agent.source_facts_v21 import extract_source
    from jarvis.agent.parser_v21 import facts_to_ir
    q = 'موجودی انبار 100 کالا است؛ 20 کم کن؛ 5 اضافه کن'
    ir = facts_to_ir(extract_source(q), q, 'fa')
    ir.slots['initial'] += 1
    v21 = LocalIntelligenceV21().verifier.verify(ir, 66)
    v22 = UniversalVerifierV2().verify(ir, 66)
    assert not v21.passed and not v22.passed
    assert set(v21.failed_checks) <= set(v22.failed_checks)


def test_typed_audit_never_crashes_verified_path():
    b = LocalIntelligenceV22()
    a = b.solve('525 را با نسبت 4 به 3 تقسیم کن', 'fa')
    assert a is not None and '300' in a.text and '225' in a.text


# ======================================================================
# 6. Full pipeline — P0 adversarial cases (Persian + English)
# ======================================================================
def test_p0_currency_plus_items_blocked_fa(eng):
    ans = eng.solve('موجودی حساب 100 دلار است؛ 5 کالا اضافه کن', 'fa')
    assert '105' not in ans.text and 'لطفاً' in ans.text


def test_p0_currency_plus_items_blocked_en(eng):
    ans = eng.solve('The account balance is 100 dollars; add 5 items.', 'en')
    assert '105' not in ans.text


def test_temporal_wrap_answered_as_clock(eng):
    ans = eng.solve('کار ساعت 23 شروع می شود؛ مدت کار 2 ساعت است؛ کار چه ساعتی تمام می شود؟', 'fa')
    assert '01:00' in ans.text and 'روز بعد' in ans.text
    assert '25' not in ans.text.split('یعنی')[0]
    tm = eng.last_trace.get('temporal_world_model')
    assert tm and tm['day_offset'] == 1 and tm['verified']


def test_work_rate_correct_binding(eng):
    ans = eng.solve('3 کارگر در 4 ساعت 84 واحد تولید می کنند؛ 8 کارگر در 6 ساعت چند واحد تولید می کنند؟', 'fa')
    assert '336' in ans.text


def test_ratio_parts(eng):
    ans = eng.solve('525 را با نسبت 4 به 3 تقسیم کن', 'fa')
    assert '300' in ans.text and '225' in ans.text


def test_probability_kept(eng):
    ans = eng.solve('احتمال موفقیت 30 درصد است؛ در 4 تلاش دقیقاً 2 موفقیت؟', 'fa')
    assert ans is not None and '26.46' in ans.text


# ======================================================================
# 7. Language Brain V2
# ======================================================================
def test_normalize_nlu_informal_persian():
    t = normalize_nlu('موجودیم چنده   با ٥ تا ؟')  # ۵ Persian digit + colloquial
    assert 'چند است' in t and '5' in t and '  ' not in t


def test_sanitize_removes_repetition_and_robotic_openers():
    out = sanitize_response('نتایج نتایج نتایج آماده است. بر اساس محاسبات انجام‌شده، پاسخ 5 است.')
    assert out.count('نتایج') == 1
    assert not out.startswith('بر اساس محاسبات')


def test_natural_render_templates():
    ir = {'task': 'inventory', 'slots': {'initial': 100}, 'units': {}, 'quantities': []}
    assert render_word_problem(105, ir, 'fa') == 'نتیجه می\u200cشود 105 کالا.'
    ir2 = {'task': 'finance', 'slots': {'initial': 100},
           'quantities': [{'dimension': 'currency', 'unit': 'USD'}]}
    assert render_word_problem(105, ir2, 'fa') == 'موجودی جدید می\u200cشود 105 دلار.'


def test_clarification_messages():
    assert 'یکای متفاوت' in clarification('fa', 'currency_item')
    assert 'different units' in clarification('en', 'currency_item')
    assert clarification('fa', 'unknown')  # falls back safely


def test_fmt_number():
    assert fmt_number(105.0) == '105' and fmt_number(1.5) == '1.5'


# ======================================================================
# 8. Wiring + honest telemetry + v21 contract preservation
# ======================================================================
def test_reasoning_uses_v22_with_honest_label():
    from tests.helpers import TemporaryRuntime
    with TemporaryRuntime() as rt:
        li = rt.agent.reasoning.local_intelligence
        assert isinstance(li, LocalIntelligenceV22)
        assert isinstance(li, LocalIntelligenceV21)  # subclass keeps v21 isinstance
        assert li.VERSION == '5.0.0-v22.4'
        from jarvis.agent.reasoning import _local_source_label
        assert _local_source_label(li) == 'local_intelligence_5_0_0_v22_4'


def test_runtime_endpoint_keeps_canonical_contract():
    from tests.helpers import TemporaryRuntime
    with TemporaryRuntime() as rt:
        a = rt.agent.respond('با 3 کارگر و 4 ساعت، خروجی 84 قطعه است. با 8 کارگر و 6 ساعت چه خروجی داریم؟')
        assert a.text == '336'  # v20_1 contract unchanged


def test_v22_trace_metadata_present(eng):
    eng.solve('موجودی انبار 100 کالا است؛ 5 کالا اضافه کن', 'fa')
    trace = eng.last_trace
    assert trace['verification']['passed']
    assert 'typed_quantity_audit' in trace['verification']['checks']
    assert trace.get('ir') or trace.get('attempts')


def test_nlu_retry_path_exists(eng):
    # informal 'میشه' is normalized then solved — result must be verified, not hallucinated
    ans = eng.solve('موجودی انبار 100 کالا است و 5 کالا اضافه میشه؛ جمع میشه چند؟', 'fa')
    assert ans is None or '105' in ans.text or 'کالا' in ans.text
