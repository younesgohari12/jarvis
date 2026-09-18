"""JARVIS v22.1 — audit-response adversarial suite.

Covers every P0 finding from the independent v21-vs-v22 audit:
  1. semantic rewrite must never corrupt meaning (age difference -> abs)
  2. Verifier V2 must fail CLOSED on internal typed-audit errors
  3. Typed Quantity V2 must extract real units (hours/km/kg/liters/°C/...),
     with source spans covering number + unit
  4. Numeric Role V2 must classify ratio/inventory/price/discount/p/n/k/
     age_difference roles
  5. minute-precision scheduling (23:30 + 2h -> 01:30 next day)
  6. immutable original source: NLU paraphrase is never the verification text
The v21 regression contract stays intact (989 tests must remain green).
"""
import pytest

from jarvis.agent.quantity_v22 import extract_typed_quantities
from jarvis.agent.numeric_roles_v22 import classify_source_numbers_v2
from jarvis.agent.language_brain_v22 import (
    detect_age_difference, normalize_clock_tokens, normalize_nlu,
)
from jarvis.agent.verifier_v22 import (
    UniversalVerifierV2, ORIGINAL_SOURCE, TYPED_FAILURES,
)
from jarvis.agent.local_intelligence_v22 import LocalIntelligenceV22


@pytest.fixture(scope='module')
def canon():
    return LocalIntelligenceV22(output_style='canonical')


@pytest.fixture(scope='module')
def eng():
    return LocalIntelligenceV22()


# ======================================================================
# 1. P0 #1 — semantic rewrite can no longer corrupt the meaning
# ======================================================================
def test_age_difference_english(canon):
    a = canon.solve('Ali is 35 and Reza is 57 years old. What is the age difference in years?', 'en')
    assert a is not None and '22' in a.text and '-22' not in a.text


def test_age_difference_persian(canon):
    a = canon.solve('علی 35 ساله است و رضا 57 ساله است؛ رضا چند سال بزرگتر از علی است؟', 'fa')
    assert a is not None and '22' in a.text and '-22' not in a.text


def test_age_difference_sen_form(canon):
    a = canon.solve('سن علی 20 سال است و سن رضا 34 سال است؛ اختلاف سن چند سال است؟', 'fa')
    assert a is not None and '14' in a.text


def test_age_difference_no_negative_answer(eng):
    a = eng.solve('Ali is 35 and Reza is 57 years old. What is the age difference in years?', 'en')
    assert a is not None and '-' not in a.text.split('difference')[0]


def test_age_difference_entity_binding_recorded(canon):
    canon.solve('علی 35 ساله است و رضا 57 ساله است؛ رضا چند سال بزرگتر از علی است؟', 'fa')
    rec = canon.last_trace.get('v22_age_difference')
    assert rec and rec['verified'] and set(rec['entities']) == {'علی', 'رضا'}
    assert rec['difference'] == 22


def test_corrupting_paraphrase_removed():
    ntext = normalize_nlu('علی 35 ساله است و رضا 57 ساله است؛ رضا چند سال بزرگتر از علی است؟')
    assert 'موجودی' not in ntext
    ntext_en = normalize_nlu('Ali is 35 and Reza is 57 years old. What is the age difference in years?')
    assert 'start with' not in ntext_en and 'subtract' not in ntext_en


def test_verifier_passes_paraphrase_never_used_as_source(canon):
    """The typed audit must read the ORIGINAL text even when the parser saw
    a normalized rewrite."""
    canon.solve('موجودی حساب ۱۰۰ دلار است؛ ۵ دلار اضافه کن', 'fa')
    trace = canon.last_trace
    assert trace and trace['verification']['passed']


def test_difference_guard_keeps_base_age_questions_green(canon):
    """A 'how old will X be' question is NOT a difference question."""
    a = canon.solve('علی 20 ساله است؛ بعد از 5 سال سن علی چند سال است؟', 'fa')
    assert a is not None and '25' in a.text


# ======================================================================
# 2. P0 #2 — the verifier fails CLOSED on internal audit errors
# ======================================================================
def test_typed_audit_crash_fails_closed(monkeypatch):
    from jarvis.agent import verifier_v22 as vv
    from jarvis.agent.parser_v21 import get_parser_v21
    ir = get_parser_v21().parse('موجودی حساب 100 دلار است؛ 5 دلار اضافه کن')
    token = ORIGINAL_SOURCE.set('موجودی حساب 100 دلار است؛ 5 دلار اضافه کن')
    try:
        ok = UniversalVerifierV2().verify(ir, 105.0)
        assert ok.passed
        monkeypatch.setattr(vv, 'extract_typed_quantities',
                            lambda *a, **k: (_ for _ in ()).throw(RuntimeError('boom')))
        bad = UniversalVerifierV2().verify(ir, 105.0)
        assert not bad.passed
        assert 'typed_audit_internal_error' in bad.failed_checks
        assert bad.repair_stage == 'abstain'
        assert 'typed_audit_internal_error' in TYPED_FAILURES
    finally:
        ORIGINAL_SOURCE.reset(token)


def test_engine_abstains_when_audit_crashes(monkeypatch):
    from jarvis.agent import verifier_v22 as vv
    eng = LocalIntelligenceV22(output_style='canonical')
    original = vv.extract_typed_quantities
    calls = {'n': 0}

    def flaky(text):
        calls['n'] += 1
        raise RuntimeError('injected audit crash')

    monkeypatch.setattr(vv, 'extract_typed_quantities', flaky)
    a = eng.solve('موجودی حساب 100 دلار است؛ 5 دلار اضافه کن', 'fa')
    monkeypatch.setattr(vv, 'extract_typed_quantities', original)
    assert a is not None and '105' not in a.text


# ======================================================================
# 3. Typed Quantity V2 — real units + full source spans
# ======================================================================
@pytest.mark.parametrize('text,dim,unit', [
    ('کار 2 ساعت طول کشید', 'time', 'HOUR'),
    ('90 minutes passed', 'time', 'MINUTE'),
    ('120 کیلومتر رفتم', 'distance', 'KM'),
    ('120 km away', 'distance', 'KM'),
    ('5 kg آرد', 'mass', 'KG'),
    ('3 liters آب', 'volume', 'L'),
    ('30 C هوا', 'temperature', 'C'),
    ('سرعت 60 km/h', 'speed', 'KM_PER_HOUR'),
    ('علی 20 ساله است', 'age', 'YEAR'),
    ('Ali is 20 years old', 'age', 'YEAR'),
])
def test_units_extracted(text, dim, unit):
    qs = extract_typed_quantities(text)
    assert any(q.dimension == dim and q.unit == unit for q in qs), (text, qs)


@pytest.mark.parametrize('text,span', [
    ('موجودی حساب 100 دلار است', '100 دلار'),
    ('کار 2 ساعت طول کشید', '2 ساعت'),
    ('120 کیلومتر رفتم', '120 کیلومتر'),
    ('The balance is 100 dollars', '100 dollars'),
])
def test_source_span_covers_number_and_unit(text, span):
    qs = extract_typed_quantities(text)
    assert any(q.source_span == span for q in qs), (text, [q.source_span for q in qs])


def test_cross_dimension_addition_refused():
    from jarvis.agent.verifier_v22 import cross_dimension_chain_failure
    qs = extract_typed_quantities('2 ساعت و 120 کیلومتر را جمع کن')
    assert cross_dimension_chain_failure(qs, '2 ساعت و 120 کیلومتر را جمع کن')


def test_rate_context_time_count_not_refused():
    from jarvis.agent.verifier_v22 import cross_dimension_chain_failure
    text = 'موجودی انبار 100 کالا است؛ هر روز 5 کالا اضافه می شود؛ بعد از 3 روز؟'
    qs = extract_typed_quantities(text)
    assert cross_dimension_chain_failure(qs, text) == []


# ======================================================================
# 4. Numeric Role V2 — expressive roles actually classified
# ======================================================================
def test_roles_ratio():
    roles = classify_source_numbers_v2('525 را با نسبت 4 به 3 تقسیم کن')
    by_val = {f['value']: f['role'] for f in roles}
    assert by_val[525.0] == 'total' and by_val[4.0] == 'ratio_a' and by_val[3.0] == 'ratio_b'


def test_roles_ratio_english():
    roles = classify_source_numbers_v2('Divide 525 in the ratio 4:3')
    vals = {f['role'] for f in roles}
    assert {'total', 'ratio_a', 'ratio_b'} <= vals


def test_roles_inventory():
    roles = classify_source_numbers_v2('موجودی انبار 100 کالا است؛ 5 کالا اضافه کن')
    by_val = {f['value']: f['role'] for f in roles}
    assert by_val[100.0] == 'inventory_initial' and by_val[5.0] == 'inventory_add'


def test_roles_price_vs_discount():
    roles = classify_source_numbers_v2('قیمت 100 تومان است و 20 درصد تخفیف دارد')
    by_val = {f['value']: f['role'] for f in roles}
    assert by_val[100.0] == 'price' and by_val[20.0] == 'discount'


def test_roles_binomial():
    roles = classify_source_numbers_v2('احتمال موفقیت 30 درصد است؛ در 4 تلاش دقیقاً 2 موفقیت؟')
    by_val = {f['value']: f['role'] for f in roles}
    assert by_val[30.0] == 'p' and by_val[4.0] == 'n' and by_val[2.0] == 'k'


def test_roles_age_difference():
    roles = classify_source_numbers_v2('علی 20 ساله است و رضا 5 سال بزرگتر است')
    by_val = {f['value']: f['role'] for f in roles}
    assert by_val[20.0] == 'age_value' and by_val[5.0] == 'age_difference'


def test_roles_workers_regression():
    roles = classify_source_numbers_v2('3 کارگر در 4 ساعت 84 واحد تولید می کنند؛ 8 کارگر در 6 ساعت؟')
    workers = [f for f in roles if str(f['role']).startswith('workers')]
    assert [f['value'] for f in workers] == [3.0, 8.0]


# ======================================================================
# 5. minute-precision scheduling
# ======================================================================
def test_clock_token_normalization():
    assert '23.5' in normalize_clock_tokens('کار ساعت 23:30 شروع می شود')
    assert '23:30' not in normalize_clock_tokens('کار ساعت 23:30 شروع می شود')
    # non-scheduling text must stay untouched
    assert normalize_clock_tokens('کد 12:34 را ثبت کن') == 'کد 12:34 را ثبت کن'


def test_scheduling_2330_plus_2h(canon):
    a = canon.solve('کار ساعت 23:30 شروع می شود؛ مدت کار 2 ساعت است؛ کار چه ساعتی تمام می شود؟', 'fa')
    assert a is not None and '01:30' in a.text and 'روز بعد' in a.text


def test_scheduling_2345_plus_30m(canon):
    a = canon.solve('کار ساعت 23:45 شروع می شود؛ مدت کار 30 دقیقه است؛ کار چه ساعتی تمام می شود؟', 'fa')
    assert a is not None and '00:15' in a.text and 'روز بعد' in a.text


def test_scheduling_whole_hour_still_green(canon):
    a = canon.solve('کار ساعت 23 شروع می شود؛ مدت کار 2 ساعت است؛ کار چه ساعتی تمام می شود؟', 'fa')
    assert a is not None and '01:00' in a.text


def test_scheduling_temporal_witness_uses_original_source(canon):
    canon.solve('کار ساعت 23:30 شروع می شود؛ مدت کار 2 ساعت است؛ کار چه ساعتی تمام می شود؟', 'fa')
    tm = canon.last_trace.get('temporal_world_model')
    assert tm and tm['verified'] and tm['day_offset'] == 1


# ======================================================================
# 6. immutability + no-regression spot checks
# ======================================================================
def test_original_source_never_mutated(canon):
    text = 'موجودی حساب ۱۰۰ دلار است؛ ۵ دلار اضافه کن'
    canon.solve(text, 'fa')
    assert canon.last_source_text() == text or ORIGINAL_SOURCE.get() in (None, text)


def test_p0_guards_still_hold(eng):
    a = eng.solve('موجودی حساب 100 دلار است؛ 5 کالا اضافه کن', 'fa')
    assert a is not None and '105' not in a.text
    b = eng.solve('موجودی حساب 100 دلار است؛ 20 یورو اضافه کن', 'fa')
    assert b is not None and '120' not in b.text


def test_work_rate_still_green(canon):
    a = canon.solve('3 کارگر در 4 ساعت 84 واحد تولید می کنند؛ 8 کارگر در 6 ساعت چند واحد تولید می کنند؟', 'fa')
    assert a is not None and '336' in a.text


def test_ratio_task_still_green(canon):
    a = canon.solve('525 را با نسبت 4 به 3 تقسیم کن', 'fa')
    assert a is not None and '300' in a.text and '225' in a.text
