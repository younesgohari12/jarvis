"""JARVIS v22.4 — TRUE ROOT-CAUSE HARDENING test suite.

Covers the P0 matrix of the v22.4 specification:
  §5/§7   operation-aware dimension algebra (FA + EN)
  §9/§10  speed unit mapping (m/s / km/h / mph / unknown)
  §11     explicit unit conversion separate from extraction
  §13/§15 exact SourceSpan invariants
  §16-§21 generalized entity-bound age semantics
  §25-§31 ownership event model (transfers, query binding, currency safety)
  §32-§35 inventory typed events + product binding
  §36-§40 generalized temporal frame + calendar invariants
  §41     price_delta role separation
  §23/§24 structured dimension-failure metadata + messages
  §47-§52 identifiers, negation, decimals, non-finite, division, probability
  §53     fact-locked NLG (unit can never be substituted)
"""
import math
import pytest

from jarvis.agent.quantity_v22 import extract_typed_quantities
from jarvis.agent.verifier_v22 import (
    UniversalVerifierV2, cross_dimension_chain_failure,
    cross_dimension_chain_failure_structured, ORIGINAL_SOURCE,
    LAST_DIMENSION_FAILURE,
)
from jarvis.agent.semantics_v22_4 import (
    SourceSpan, make_span, SpanIntegrityError, IndexMap,
    Dimension, DimensionExpr, dimension_expr,
    validate_binary_operation, validate_rate_pair_addition,
    convert_unit, DimensionFailure,
    extract_age_facts, bind_age_query, solve_age_query,
    extract_ownership_model, execute_ownership,
    extract_inventory_model, execute_inventory_model,
    extract_temporal_frame, solve_rate, numeric_guard, probability_guard,
)
from jarvis.agent.numeric_roles_v22 import classify_source_numbers_v2
from jarvis.agent.language_brain_v22 import (
    clarification, structured_clarification, render_word_problem,
)
from jarvis.agent import temporal_v22
from jarvis.agent.local_intelligence_v22 import LocalIntelligenceV22


@pytest.fixture(scope='module')
def eng():
    return LocalIntelligenceV22()


# ======================================================================
# §5/§7 — P0 dimension algebra matrix (both languages)
# ======================================================================
@pytest.mark.parametrize('text', [
    '2 hours + 5 USD',
    '5 USD + 2 hours',
    '2 ساعت + 5 دلار',
    '2 hours + 5 dollars',
    '2 ساعت و 5 دلار را با هم جمع کن',
    '10 items + 3 hours',
    '3 hours + 10 items را جمع کن',
])
def test_p0_time_currency_count_addition_rejected(eng, text):
    ans = eng.solve(text, 'fa' if any('\u0600' <= c <= '\u06FF' for c in text) else 'en')
    assert ans is not None
    for bad in ('7', '25', '13', '15'):
        assert bad not in ans.text or 'نمی' in ans.text or 'cannot' in ans.text.lower()
    assert eng.last_trace['verification']['passed'] is False


def test_p0_time_distance_addition_rejected():
    qs = extract_typed_quantities('2 ساعت و 120 کیلومتر را جمع کن')
    assert cross_dimension_chain_failure(qs, '2 ساعت و 120 کیلومتر را جمع کن')


@pytest.mark.parametrize('text', [
    '10 USD/hour + 3 ITEM/hour',
    '10 دلار در ساعت و 3 کالا در ساعت جمع می شود',
])
def test_p0_rate_family_mismatch_rejected(text):
    qs = extract_typed_quantities(text)
    failures = cross_dimension_chain_failure_structured(qs, text)
    assert failures, text


def test_mass_volume_addition_rejected_structured():
    text = '2 کیلوگرم و 3 لیتر را جمع کن'
    qs = extract_typed_quantities(text)
    failures = cross_dimension_chain_failure_structured(qs, text)
    assert failures and failures[0].key == 'mass_volume'


# -- the same pairs under a DIFFERENT operator or context stay legal -----
@pytest.mark.parametrize('text', [
    'موجودی انبار 100 کالا است؛ هر روز 5 کالا اضافه می شود؛ بعد از 3 روز؟',
    'موجودی حساب 100 دلار است؛ 5 دلار اضافه کن',
    '10 USD/hour + 3 USD/hour',
    'سرعت 60 کیلومتر بر ساعت است؛ در 2 ساعت چند کیلومتر می رود؟',
])
def test_rate_contexts_stay_legal(text):
    qs = extract_typed_quantities(text)
    assert cross_dimension_chain_failure(qs, text) == []


def test_validate_binary_operation_matrix():
    ok = validate_binary_operation
    assert ok('add', ('currency', 'USD'), ('currency', 'USD')) is None
    assert ok('subtract', ('currency', 'USD'), ('currency', 'USD')) is None
    assert ok('add', ('currency', 'USD'), ('currency', 'EUR')) is not None
    assert ok('add', ('currency', 'USD'), ('count', 'ITEM')) is not None
    assert ok('add', ('currency', 'USD'), ('time', 'HOUR')) is not None
    assert ok('add', ('time', 'HOUR'), ('distance', 'KM')) is not None
    assert ok('add', ('mass', 'KG'), ('volume', 'L')) is not None
    assert ok('add', ('distance', 'KM'), ('distance', 'KM')) is None
    assert ok('add', ('time', 'HOUR'), ('time', 'HOUR')) is None
    # division: DISTANCE/TIME -> SPEED etc.
    assert ok('divide', ('distance', 'KM'), ('time', 'HOUR')) is None
    assert ok('divide', ('currency', 'USD'), ('time', 'HOUR')) is None
    assert ok('divide', ('count', 'ITEM'), ('time', 'HOUR')) is None
    # multiplication: rate × base -> numerator
    assert ok('multiply', ('speed', 'KM_PER_HOUR'), ('time', 'HOUR')) is None
    assert ok('multiply', ('currency_rate', 'USD_PER_HOUR'), ('time', 'HOUR')) is None
    assert ok('multiply', ('count_rate', 'ITEM_PER_HOUR'), ('time', 'HOUR')) is None
    # dimensionless never contaminates
    assert ok('add', ('dimensionless', 'NONE'), ('currency', 'USD')) is None


def test_rate_addition_same_family_valid():
    assert validate_rate_pair_addition('USD_PER_HOUR', 'USD_PER_HOUR') is None
    f = validate_rate_pair_addition('USD_PER_HOUR', 'ITEM_PER_HOUR')
    assert f is not None and f.key == 'rate_mismatch'


# ======================================================================
# §9/§10 — speed unit mapping
# ======================================================================
@pytest.mark.parametrize('text,unit', [
    ('10 m/s', 'M_PER_SECOND'),
    ('10 meters per second', 'M_PER_SECOND'),
    ('10 metres per second', 'M_PER_SECOND'),
    ('سرعت 10 متر بر ثانیه', 'M_PER_SECOND'),
    ('36 km/h', 'KM_PER_HOUR'),
    ('36 kmph', 'KM_PER_HOUR'),
    ('36 kilometers per hour', 'KM_PER_HOUR'),
    ('36 kilometres per hour', 'KM_PER_HOUR'),
    ('سرعت 36 کیلومتر بر ساعت', 'KM_PER_HOUR'),
    ('5 mph', 'MILE_PER_HOUR'),
    ('5 miles per hour', 'MILE_PER_HOUR'),
    ('سرعت 5 مایل بر ساعت', 'MILE_PER_HOUR'),
])
def test_speed_unit_mapping(text, unit):
    qs = extract_typed_quantities(text)
    assert any(q.dimension == 'speed' and q.unit == unit for q in qs), (text, qs)


def test_unknown_speed_unit_never_substituted():
    qs = extract_typed_quantities('سرعت 60')
    assert all(q.unit != 'KM_PER_HOUR' for q in qs)
    speed_qs = [q for q in qs if q.dimension == 'speed']
    assert all(q.unit == 'UNKNOWN_UNIT' for q in speed_qs)


def test_quantity_preserves_raw_and_normalized_unit():
    qs = extract_typed_quantities('10 m/s')
    q = qs[0]
    assert q.raw_unit == 'm/s'
    assert q.normalized_unit == 'M_PER_SECOND'
    assert q.dimension == 'speed'


# ======================================================================
# §11 — explicit conversion separate from extraction
# ======================================================================
def test_extraction_does_not_convert():
    qs = extract_typed_quantities('10 m/s')
    assert qs[0].value == 10.0 and qs[0].unit == 'M_PER_SECOND'


def test_explicit_conversion_metadata():
    conv = convert_unit(10, 'M_PER_SECOND', 'KM_PER_HOUR')
    assert conv == {'source_value': 10.0, 'source_unit': 'M_PER_SECOND',
                    'result_value': 36.0, 'result_unit': 'KM_PER_HOUR',
                    'factor': 3.6}


def test_conversion_refuses_unknown_pair():
    assert convert_unit(1, 'USD', 'KM') is None


# ======================================================================
# §13/§15 — SourceSpan exactness
# ======================================================================
def test_source_span_is_frozen_type():
    s = make_span('hello world', 0, 5)
    assert isinstance(s, SourceSpan) and s.text == 'hello'
    with pytest.raises(Exception):
        s.start = 2  # frozen


def test_span_invariant_enforced():
    with pytest.raises(SpanIntegrityError):
        SourceSpan(0, 4, 'hellx').validate('hello world')   # wrong text
    with pytest.raises(SpanIntegrityError):
        SourceSpan(0, 99, 'x').validate('short')            # out of range
    # a genuinely correct span validates cleanly
    assert SourceSpan(0, 4, 'hell').validate('hello world') is not None


@pytest.mark.parametrize('text', ['50 کالا', '3 workers', '8 نفر',
                                  'موجودی حساب 100 دلار است',
                                  'کار 2 ساعت طول کشید', 'سرعت 10 متر بر ثانیه',
                                  '120 کیلومتر رفتم', '5 kg آرد'])
def test_exact_spans_on_original_text(text):
    for q in extract_typed_quantities(text):
        assert 0 <= q.start <= q.end <= len(text)
        assert text[q.start:q.end] == q.source_span, (text, q.source_span)


def test_index_map_roundtrip():
    im = IndexMap.identity(5)
    assert im.to_original(3) == 3
    # transformed text with 2 chars removed at position 1
    im2 = IndexMap([0, 3, 4], 5)
    assert im2.to_original(0) == 0 and im2.to_original(1) == 3


# ======================================================================
# §16-§21 — entity-bound age semantics
# ======================================================================
@pytest.mark.parametrize('text,a,va,b,vb', [
    ('Ali is 35 and Reza is 57 years old. What is the age difference between Ali and Reza?',
     'Ali', 35.0, 'Reza', 57.0),
    ('Ali is 35 years old and Reza is 57 years old. What is the age difference?',
     'Ali', 35.0, 'Reza', 57.0),
    ('Ali, aged 35, and Reza, aged 57. What is the age difference between Ali and Reza?',
     'Ali', 35.0, 'Reza', 57.0),
    ("Ali's age is 35 and Reza's age is 57. How many years apart are Ali and Reza?",
     'Ali', 35.0, 'Reza', 57.0),
    ('علی 35 ساله است و رضا 57 ساله است. اختلاف سن علی و رضا چند سال است؟',
     'علی', 35.0, 'رضا', 57.0),
    ('علی 35 سال دارد و رضا 57 سال دارد. اختلاف سن علی و رضا چند سال است؟',
     'علی', 35.0, 'رضا', 57.0),
    ('سن علی 35 و سن رضا 57 است. اختلاف سن چند سال است؟', 'علی', 35.0, 'رضا', 57.0),
    ('سن علی 35 سال است و سن رضا 57 سال است. اختلاف سن علی و رضا چند سال است؟',
     'علی', 35.0, 'رضا', 57.0),
])
def test_age_entity_binding_generalizes(text, a, va, b, vb):
    solved = solve_age_query(text)
    assert solved is not None, text
    assert solved['ages'][a] == va and solved['ages'][b] == vb
    assert solved['difference'] == abs(va - vb)


@pytest.mark.parametrize('text,expected', [
    ('Ali is 35. Reza is 57. Sara is 10. What is the age difference between Ali and Reza?', 22.0),
    ('Ali is 35. Reza is 57. Sara is 10. What is the age difference between Reza and Sara?', 47.0),
    ('علی 35 ساله است. رضا 57 ساله است. سارا 10 ساله است. اختلاف سن رضا و سارا چند سال است؟', 47.0),
    ('Reza is 22 years older than Ali. How much older is Reza than Ali?', 22.0),
    ('رضا 22 سال از علی بزرگتر است. رضا چند سال از علی بزرگتر است؟', 22.0),
])
def test_age_query_binds_named_entities(eng, text, expected):
    ans = eng.solve(text, 'fa' if any('\u0600' <= c <= '\u06FF' for c in text) else 'en')
    assert ans is not None and f'{expected:g}' in ans.text, (text, ans)


def test_age_difference_invariant_non_negative():
    solved = solve_age_query('علی 35 سال دارد و رضا 57 سال دارد. اختلاف سن رضا و علی چند سال است؟')
    assert solved is not None and solved['difference'] >= 0


def test_age_facts_are_entity_attribute_value():
    facts = extract_age_facts('Ali, aged 35')
    assert facts and facts[0].entity == 'Ali' and facts[0].value == 35.0
    d = facts[0].to_dict()
    assert d['attribute'] == 'age' and d['unit'] == 'YEAR' and d['source_span']


# ======================================================================
# §25-§31 — ownership event model
# ======================================================================
@pytest.mark.parametrize('text,expected', [
    ('Ali has 704 dollars. Sara has 223 dollars. Ali transfers 15 dollars to Sara. '
     "What is Sara's balance?", 238.0),
    ('Ali has 704 dollars. Sara has 223 dollars. Ali pays 15 dollars to Sara. '
     "What is Sara's balance?", 238.0),
    ('Ali has 704 dollars. Sara has 223 dollars. Ali gives Sara 15 dollars. '
     "What is Sara's balance?", 238.0),
    ('Ali has 704 dollars. Sara has 223 dollars. Sara receives 15 dollars from Ali. '
     "What is Sara's balance?", 238.0),
    ('Ali has 704 dollars. Sara has 223 dollars. 15 dollars is sent from Ali to Sara. '
     "What is Sara's balance?", 238.0),
    ('علی 704 دلار دارد. سارا 223 دلار دارد. علی 15 دلار به سارا می‌دهد. موجودی سارا چند است؟',
     238.0),
    ('علی 704 دلار دارد. سارا 223 دلار دارد. علی 15 دلار برای سارا می‌فرستد. موجودی سارا چند است؟',
     238.0),
    ('علی 704 دلار دارد. سارا 223 دلار دارد. سارا 15 دلار از علی دریافت می‌کند. موجودی سارا چند است؟',
     238.0),
])
def test_ownership_direction_resolves(text, expected):
    model = extract_ownership_model(text)
    assert model is not None and not model.get('conflict'), text
    result = execute_ownership(model)
    assert result['ok'] and abs(result['value'] - expected) < 1e-9, result


def test_ownership_sender_query_binding():
    text = ('Ali has 704 dollars. Sara has 223 dollars. Ali transfers 15 dollars to Sara. '
            "What is Ali's balance?")
    model = extract_ownership_model(text)
    result = execute_ownership(model)
    assert result['ok'] and result['value'] == 689.0   # sender, not receiver


def test_ownership_combined_and_difference_queries():
    base = 'Ali has 100 dollars. Bob has 50 dollars. Ali transfers 20 dollars to Bob. '
    combined = extract_ownership_model(base + 'What is the combined total?')
    assert execute_ownership(combined)['value'] == 150.0
    difference = extract_ownership_model(
        base + 'What is the difference in balance between Ali and Bob?')
    assert execute_ownership(difference)['value'] == 10.0


def test_multi_transfer_sequential_state():
    text = ('Ali has 100 dollars. Sara has 50 dollars. Reza has 20 dollars. '
            'Ali gives Sara 10. Sara gives Reza 5. Reza gives Ali 2. '
            "What is Reza's balance?")
    model = extract_ownership_model(text)
    result = execute_ownership(model)
    assert result['ok'] and result['value'] == 23.0   # 20 +5 -2


def test_transfer_conserves_total_funds():
    text = ('Ali has 100 dollars. Sara has 50 dollars. Ali transfers 30 dollars to Sara. '
            "What is Sara's balance?")
    model = extract_ownership_model(text)
    result = execute_ownership(model)
    assert math.isclose(sum(result['balances'].values()), 150.0)


def test_ownership_currency_safety():
    text = ("Ali has 100 USD. Sara has 50 EUR. Ali transfers 20 to Sara. "
            "What is Sara's balance?")
    model = extract_ownership_model(text)
    assert model is not None and model.get('conflict') == 'ownership_unit_conflict'


def test_impossible_transfer_refused():
    text = ('Ali has 10 dollars. Sara has 50 dollars. Ali transfers 30 dollars to Sara. '
            "What is Sara's balance?")
    model = extract_ownership_model(text)
    result = execute_ownership(model)
    assert not result['ok'] and result['reason'] == 'impossible_transfer'


def test_ownership_event_structure():
    text = ('Ali has 100 dollars. Sara has 50 dollars. Ali transfers 30 dollars to Sara. '
            "What is Sara's balance?")
    model = extract_ownership_model(text)
    e = model['events'][0]
    assert e['type'] == 'transfer' and e['source'] == 'Ali' and e['target'] == 'Sara'
    assert e['amount'] == 30.0 and e['source_span']


# ======================================================================
# §32-§35 — inventory typed events + product binding
# ======================================================================
def test_inventory_event_types_classified(eng):
    ans = eng.solve('موجودی انبار 100 کالا است؛ 7 کالا فروخته شد و 3 کالا آسیب دید و '
                    '2 کالا مرجوع شد. موجودی چند است؟', 'fa')
    assert ans is not None and '92' in ans.text, ans
    ir = eng.last_trace.get('ir') or {}
    types = {e.get('type') for e in ir.get('operations', [])}
    assert {'sale', 'damage', 'return'} <= types, types


@pytest.mark.parametrize('text,expected', [
    ('موجودی انبار 100 کالا است؛ 20 کالا رسید؛ 7 کالا فروخته شد؛ 3 کالا آسیب دید؛ '
     '2 کالا مرجوع شد. موجودی چند است؟', 112.0),
])
def test_inventory_event_order_preserved(eng, text, expected):
    ans = eng.solve(text, 'fa')
    assert ans is not None and f'{expected:g}' in ans.text


def test_multi_product_inventory_binding(eng):
    text = ('Product A stock = 10. Product B stock = 20. 5 of Product A sold. '
            'What is the stock of Product A?')
    ans = eng.solve(text, 'en')
    assert ans is not None and '5' in ans.text, ans
    model = eng.last_trace['v22_4_model']
    assert model['products']['Product B'] == 20.0   # untouched
    assert model['products']['Product A'] == 5.0


def test_multi_product_fa_binding(eng):
    text = ('موجودی محصول الف 10 کالا است و موجودی محصول ب 20 کالا است. '
            '5 کالا از محصول الف فروخته شد. موجودی محصول الف چند است؟')
    ans = eng.solve(text, 'fa')
    assert ans is not None and '5' in ans.text, ans
    assert eng.last_trace['v22_4_model']['products']['محصول ب'] == 20.0


def test_inventory_order_is_semantic():
    """Event ORDER is semantic: reordering the same events changes the
    verdict because an intermediate stock goes negative (spec §34)."""
    events = [
        {'event_id': 'e0', 'type': 'sale', 'direction': -1, 'quantity': 60.0,
         'unit': 'ITEM', 'product': 'X', 'source_span': ''},
        {'event_id': 'e1', 'type': 'sale', 'direction': -1, 'quantity': 60.0,
         'unit': 'ITEM', 'product': 'X', 'source_span': ''},
        {'event_id': 'e2', 'type': 'return', 'direction': 1, 'quantity': 30.0,
         'unit': 'ITEM', 'product': 'X', 'source_span': ''},
    ]
    model = {'products': {'X': 100.0}, 'events': events, 'text': 'X?'}
    forward = execute_inventory_model(model)          # 100-60=40, 40-60 -> impossible
    assert not forward['ok'] and forward['reason'] == 'impossible_inventory'
    model2 = {'products': {'X': 100.0}, 'events': list(reversed(events)), 'text': 'X?'}
    reordered = execute_inventory_model(model2)       # 130, 70, 10 -> ok
    assert reordered['ok'] and reordered['value'] == 10.0


def test_inventory_event_structure_fields():
    text = ('Product A stock = 10. Product B stock = 20. 5 of Product A sold. '
            'What is the stock of Product A?')
    model = extract_inventory_model(text)
    assert model is not None
    e = model['events'][0]
    assert set(e) >= {'event_id', 'type', 'direction', 'quantity', 'unit', 'source_span'}


# ======================================================================
# §36-§40 — generalized temporal frame
# ======================================================================
@pytest.mark.parametrize('text,expected_iso,day', [
    ('A train departs at 23:30. The trip takes 45 minutes. When does it arrive?', '00:15', 1),
    ('The shop opens at 09:30. It stays open for 3 hours. When does it close?', '12:30', 0),
    ('A meeting begins at 22:10. It lasts 2 hours. When does it end?', '00:10', 1),
    ('کار ساعت 23:30 شروع می شود؛ مدت کار 90 دقیقه است؛ چه ساعتی تمام می شود؟', '01:00', 1),
])
def test_generalized_scheduling_frames(eng, text, expected_iso, day):
    language = 'fa' if any('\u0600' <= c <= '\u06FF' for c in text) else 'en'
    ans = eng.solve(text, language)
    assert ans is not None and expected_iso in ans.text, (text, ans)
    tm = eng.last_trace.get('temporal_world_model') or {}
    assert tm.get('day_offset') == day and tm.get('verified')


def test_time_invariants_calendar_wrap():
    r1 = temporal_v22.add_duration(temporal_v22.parse_clock_time('23:59'),
                                   temporal_v22.Duration.of(2, 'minute'))
    assert r1.iso() == '00:01' and r1.day_offset == 1
    r2 = temporal_v22.add_duration(temporal_v22.parse_clock_time('23:30'),
                                   temporal_v22.Duration.of(90, 'minute'))
    assert r2.iso() == '01:00' and r2.day_offset == 1
    r3 = temporal_v22.subtract_duration(temporal_v22.parse_clock_time('00:15'),
                                        temporal_v22.Duration.of(30, 'minute'))
    assert r3.iso() == '23:45' and r3.day_offset == -1


def test_bare_clock_requires_context():
    # a bare clock token with no temporal frame must not compose a schedule
    frame = extract_temporal_frame('The clock shows 23:00.')
    assert frame is None
    frame2 = extract_temporal_frame('ساعت 23:00')
    assert frame2 is None


def test_end_plus_duration_computes_start(eng):
    frame = extract_temporal_frame('The movie ends at 22:00. It lasted 2 hours. When did it start?')
    assert frame is not None and frame.query == 'start'
    ans = eng.solve('The movie ends at 22:00. It lasted 2 hours. When did it start?', 'en')
    assert ans is not None and '20:00' in ans.text, ans


# ======================================================================
# §41 — price delta role
# ======================================================================
@pytest.mark.parametrize('text,val', [
    ('The price was reduced by 20 dollars.', 20.0),
    ('The price increased by 20 dollars.', 20.0),
    ('Cost decreased by 15 dollars.', 15.0),
    ('قیمت 20 دلار کاهش یافت.', 20.0),
])
def test_price_delta_role(text, val):
    roles = classify_source_numbers_v2(text)
    got = next(f['role'] for f in roles if f['value'] == val)
    assert got == 'price_delta'


def test_price_and_discount_roles_separated():
    roles = classify_source_numbers_v2('قیمت کتاب 50 دلار است و 20 درصد تخفیف دارد')
    by_val = {f['value']: f['role'] for f in roles}
    assert by_val[50.0] == 'price' and by_val[20.0] == 'discount_percentage'


# ======================================================================
# §23/§24 — structured dimension failure metadata + messages
# ======================================================================
def test_structured_failure_object_fields(eng):
    eng.solve('2 ساعت و 120 کیلومتر را جمع کن', 'fa')
    failure = eng.last_trace.get('v22_dimension_failure')
    assert failure is not None
    assert set(failure) >= {'failure_type', 'operator', 'left_dimension', 'left_unit',
                            'right_dimension', 'right_unit', 'message_fa', 'message_en'}
    assert failure['operator'] == 'add'
    assert {failure['left_dimension'], failure['right_dimension']} == {'time', 'distance'}


def test_structured_messages_exact():
    assert structured_clarification('fa', DimensionFailure(
        'dimension_mismatch', 'add', 'time', 'HOUR', 'distance', 'KM')) \
        .startswith('زمان و مسافت را نمی‌توان مستقیماً جمع کرد')
    assert structured_clarification('fa', DimensionFailure(
        'dimension_mismatch', 'add', 'mass', 'KG', 'volume', 'L')) \
        .startswith('جرم و حجم دو کمیت متفاوت هستند')
    assert structured_clarification('en', DimensionFailure(
        'dimension_mismatch', 'add', 'currency', 'USD', 'count', 'ITEM')) \
        .startswith('Money and item counts cannot be directly added')
    assert structured_clarification('fa', DimensionFailure(
        'currency_mix', 'add', 'currency', 'USD', 'currency', 'EUR', 'دو ارز',
        'Two currencies')).startswith('دو ارز')
    assert structured_clarification('en', DimensionFailure(
        'currency_mix', 'add', 'currency', 'USD', 'currency', 'EUR', 'دو ارز',
        'Two currencies')).startswith('Two currencies')


def test_time_distance_no_longer_claims_money_items(eng):
    ans = eng.solve('2 ساعت و 120 کیلومتر را جمع کن', 'fa')
    assert 'پول' not in ans.text and 'کالا' not in ans.text
    assert 'زمان و مسافت' in ans.text


# ======================================================================
# §47-§52 — identifiers, negation, decimals, non-finite, division, probability
# ======================================================================
@pytest.mark.parametrize('text', [
    'کد بسته 500 است', 'شناسه پرونده 1001 است', 'record id is 42',
    'Order #500', 'Version 22.4', 'Python 3.13',
])
def test_identifiers_are_not_arithmetic_values(text):
    qs = extract_typed_quantities(text)
    if qs:
        assert all(q.dimension in ('identifier', 'dimensionless') for q in qs)


def test_negation_is_preserved_by_normalization():
    from jarvis.agent.language_brain_v22 import normalize_nlu
    t = normalize_nlu('موجودی 100 دلار است؛ 5 دلار اضافه نکن')
    assert 'اضافه نکن' in t
    t2 = normalize_nlu('do not add 5 dollars; do not subtract 3 dollars')
    assert 'do not add' in t2 and 'do not subtract' in t2


def test_decimal_and_negative_values():
    qs = extract_typed_quantities('2.5 hours passed')
    assert qs[0].value == 2.5
    qs2 = extract_typed_quantities('۲٫۵ ساعت')
    assert any(abs(q.value - 2.5) < 1e-9 for q in qs2)
    qs3 = extract_typed_quantities('-5 C')
    assert any(q.value == -5.0 for q in qs3)


def test_non_finite_never_verified():
    assert numeric_guard(float('nan')) == 'non_finite_nan'
    assert numeric_guard(float('inf')) == 'non_finite_infinity'
    assert numeric_guard(float('-inf')) == 'non_finite_infinity'
    assert numeric_guard(42.0) is None


def test_probability_invariant():
    assert probability_guard(0.5) is None
    assert probability_guard(1.0) is None
    assert probability_guard(-0.1) == 'probability_range'
    assert probability_guard(1.1) == 'probability_range'


def test_division_by_zero_structured():
    from jarvis.agent.execution_v20 import GraphExecutor, ExecutionError
    ex = GraphExecutor()
    with pytest.raises(ExecutionError):
        ex.step(10, {'op': 'divide', 'value': 0})


# ======================================================================
# §53 — fact-locked NLG
# ======================================================================
def test_renderer_never_substitutes_speed_unit():
    ir = {'task': 'speed', 'slots': {}, 'units': {},
          'quantities': [{'dimension': 'speed', 'unit': 'M_PER_SECOND',
                          'normalized_unit': 'M_PER_SECOND'}]}
    out = render_word_problem(10, ir, 'fa')
    assert 'متر بر ثانیه' in out and 'کیلومتر' not in out
    out_en = render_word_problem(10, ir, 'en')
    assert 'm/s' in out_en and 'km/h' not in out_en


def test_km_per_hour_speed_renders_km():
    ir = {'task': 'speed', 'slots': {}, 'units': {},
          'quantities': [{'dimension': 'speed', 'unit': 'KM_PER_HOUR',
                          'normalized_unit': 'KM_PER_HOUR'}]}
    out = render_word_problem(60, ir, 'fa')
    assert 'کیلومتر بر ساعت' in out


# ======================================================================
# rate arithmetic (spec §6/§7 P0 matrix)
# ======================================================================
def test_rate_application_matrix():
    assert solve_rate('5 USD/hour × 2 hours')['value'] == 10.0
    assert solve_rate('5 دلار در ساعت × 2 ساعت')['value'] == 10.0
    assert solve_rate('10 items/hour × 3 hours')['value'] == 30.0
    assert solve_rate('60 km/hour × 2 hours')['value'] == 120.0


def test_rate_addition_and_conflict(eng):
    a = eng.solve('10 USD/hour + 3 USD/hour?', 'en')
    assert a is not None and '13' in a.text
    b = eng.solve('10 USD/hour + 3 ITEM/hour?', 'en')
    assert b is not None and 'cannot' in b.text.lower()


# ======================================================================
# verifier independence + fail-closed (§43/§45)
# ======================================================================
def test_verifier_fail_closed_on_internal_error(monkeypatch):
    import jarvis.agent.verifier_v22 as vv
    from jarvis.agent.parser_v21 import get_parser_v21
    ir = get_parser_v21().parse('موجودی حساب 100 دلار است؛ 5 دلار اضافه کن')
    v = UniversalVerifierV2()

    def boom(*a, **k):
        raise RuntimeError('injected')
    monkeypatch.setattr(vv, 'extract_typed_quantities', boom)
    verdict = v.verify(ir, 105)
    assert not verdict.passed
    assert 'typed_audit_internal_error' in verdict.failed_checks
    assert verdict.repair_stage == 'abstain'


def test_original_source_contextvar_isolation():
    token = ORIGINAL_SOURCE.set('سیستم A')
    try:
        assert ORIGINAL_SOURCE.get() == 'سیستم A'
    finally:
        ORIGINAL_SOURCE.reset(token)
    assert ORIGINAL_SOURCE.get() is None
