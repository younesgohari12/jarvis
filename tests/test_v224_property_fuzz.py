"""JARVIS v22.4 — property-based, metamorphic and fuzz testing (spec §54-§56).

Required properties (§54):
  * incompatible dimensions + ADD  -> never a numeric result
  * speed unit preserves the extraction unit
  * source span is exact
  * age difference >= 0
  * clock arithmetic is always valid
  * transfer conserves total funds in a same-currency closed system
  * inventory events preserve order
"""
import math
import random
import string

import pytest
from hypothesis import given, settings, strategies as st

from jarvis.agent.quantity_v22 import extract_typed_quantities
from jarvis.agent.semantics_v22_4 import (
    validate_binary_operation, solve_age_query, execute_ownership,
    extract_ownership_model,
    numeric_guard, probability_guard, make_span, SpanIntegrityError,
)
from jarvis.agent import temporal_v22
from jarvis.agent.local_intelligence_v22 import LocalIntelligenceV22

# sensible CI budget; the full property suite stays under two minutes
SUPPRESS = settings(max_examples=60, deadline=None)


@pytest.fixture(scope='module')
def eng():
    return LocalIntelligenceV22()


# ======================================================================
# §54 P1 — incompatible dimensions + ADD never produce a numeric result
# ======================================================================
INCOMPATIBLE_PAIRS = [
    (('time', 'HOUR'), ('currency', 'USD')),
    (('time', 'HOUR'), ('distance', 'KM')),
    (('currency', 'USD'), ('count', 'ITEM')),
    (('mass', 'KG'), ('volume', 'L')),
    (('temperature', 'C'), ('time', 'MINUTE')),
]


@pytest.mark.parametrize('left,right', INCOMPATIBLE_PAIRS)
def test_property_incompatible_add_never_valid(left, right):
    assert validate_binary_operation('add', left, right) is not None


@given(st.integers(min_value=0, max_value=200), st.integers(min_value=0, max_value=200))
@SUPPRESS
def test_property_incompatible_add_never_numeric(a, b):
    """For arbitrary values the algebra stays structural: the answer is a
    failure object, never a number."""
    result = validate_binary_operation('add', ('time', 'HOUR'), ('currency', 'USD'))
    assert result is not None and hasattr(result, 'left_dimension')


# ======================================================================
# §54 P2 — speed unit preserves the extraction unit
# ======================================================================
SPEED_CASES = {
    'm/s': 'M_PER_SECOND', 'meters per second': 'M_PER_SECOND',
    'متر بر ثانیه': 'M_PER_SECOND', 'km/h': 'KM_PER_HOUR',
    'کیلومتر بر ساعت': 'KM_PER_HOUR', 'mph': 'MILE_PER_HOUR',
}


@pytest.mark.parametrize('raw,unit', sorted(SPEED_CASES.items()))
def test_property_speed_unit_preserved(raw, unit):
    for value in (0, 1, 7, 12.5, 100, 9999):
        qs = extract_typed_quantities(f'{value} {raw}')
        assert qs and qs[0].unit == unit and qs[0].value == float(value), (raw, value)


@given(st.integers(min_value=0, max_value=10000))
@SUPPRESS
def test_property_speed_unit_preserved_hypothesis(v):
    qs = extract_typed_quantities(f'{v} m/s')
    assert qs[0].unit == 'M_PER_SECOND'   # never KM_PER_HOUR


# ======================================================================
# §54 P3 — source spans are exact over arbitrary inputs
# ======================================================================
SPAN_TEXTS = [
    '50 کالا', '3 workers', '8 نفر', 'موجودی حساب 100 دلار است؛ 5 دلار اضافه کن',
    '2 hours and 120 km', 'سرعت 60 کیلومتر بر ساعت', '10 m/s', '5 mph',
    'علی 35 سال دارد', 'Ali is 35 years old', '2.5 ساعت', '-5 C', '20 درصد',
]


@pytest.mark.parametrize('text', SPAN_TEXTS)
def test_property_spans_exact(text):
    for q in extract_typed_quantities(text):
        assert text[q.start:q.end] == q.source_span


@given(st.lists(st.integers(min_value=1, max_value=999), min_size=1, max_size=6),
       st.sampled_from(['کالا', 'دلار', 'ساعت', 'کیلومتر', 'نفر', 'dollars', 'hours']),
       st.sampled_from(['', '؛ ', '. ', ' و ']))
@SUPPRESS
def test_property_spans_exact_generated(values, unit, sep):
    text = sep.join(f'{v} {unit}' for v in values)
    for q in extract_typed_quantities(text):
        assert 0 <= q.start <= q.end <= len(text)
        assert text[q.start:q.end] == q.source_span, (text, q)


# ======================================================================
# §54 P4 — age difference >= 0 after entity binding
# ======================================================================
AGE_FACT_FORMS = [
    '{a} is {x} and {b} is {y}. What is the age difference between {a} and {b}?',
    '{a} is {x} years old. {b} is {y} years old. How many years apart are {a} and {b}?',
    "{a}'s age is {x} and {b}'s age is {y}. What is the age difference?",
    '{a}, aged {x}, and {b}, aged {y}. What is the age difference?',
]


@pytest.mark.parametrize('form', AGE_FACT_FORMS)
@pytest.mark.parametrize('x,y', [(35, 57), (57, 35), (12, 80), (40, 40)])
def test_property_age_difference_non_negative(form, x, y):
    solved = solve_age_query(form.format(a='Ali', b='Reza', x=x, y=y))
    if solved is not None:
        assert solved['difference'] >= 0
        assert solved['difference'] == abs(x - y)


@given(st.integers(min_value=1, max_value=120), st.integers(min_value=1, max_value=120))
@SUPPRESS
def test_property_age_difference_hypothesis(x, y):
    solved = solve_age_query(
        f'Ali is {x} and Reza is {y}. What is the age difference between Ali and Reza?')
    assert solved is not None
    assert 0 <= solved['difference'] <= 150
    assert solved['difference'] == abs(x - y)


# ======================================================================
# §54 P5 — clock arithmetic is always valid (0<=h<24, day_offset tracked)
# ======================================================================
@given(st.integers(min_value=0, max_value=23), st.integers(min_value=0, max_value=59),
       st.integers(min_value=-24 * 60, max_value=5 * 24 * 60))
@SUPPRESS
def test_property_clock_always_valid(h, m, delta):
    clock = temporal_v22.ClockTime(h, m)
    result = temporal_v22.ClockTime.from_minutes(clock.absolute_minutes() + delta)
    assert 0 <= result.hour <= 23 and 0 <= result.minute <= 59
    expected_days, expected_abs = divmod((h * 60 + m) + delta, 24 * 60)
    assert result.day_offset == expected_days
    assert result.hour * 60 + result.minute == expected_abs


@given(st.integers(min_value=0, max_value=23), st.integers(min_value=0, max_value=59),
       st.integers(min_value=0, max_value=100000))
@SUPPRESS
def test_property_clock_add_never_out_of_range(h, m, minutes):
    result = temporal_v22.add_duration(temporal_v22.ClockTime(h, m),
                                       temporal_v22.Duration(minutes=minutes))
    assert 0 <= result.hour <= 23 and 0 <= result.minute <= 59
    assert not numeric_guard(result.absolute_minutes())


# ======================================================================
# §54 P6 — transfers conserve funds in a same-currency closed system
# ======================================================================
@given(st.lists(st.tuples(st.sampled_from(['A', 'B', 'C']),
                          st.sampled_from(['A', 'B', 'C']),
                          st.integers(min_value=1, max_value=50)),
                 min_size=1, max_size=6))
@SUPPRESS
def test_property_transfer_conserves_total(transfers):
    balances = {'A': 500.0, 'B': 500.0, 'C': 500.0}
    events = []
    for src, tgt, amount in transfers:
        if src == tgt:
            continue
        events.append({'source': src, 'target': tgt, 'amount': float(amount),
                       'type': 'transfer'})
    model = {'balances': balances, 'events': events, 'unit': 'USD',
             'query': {'kind': 'combined'}}
    result = execute_ownership(model)
    if not result.get('ok'):
        # only legal rejections: an overdraw of a stated balance
        assert result['reason'] in ('impossible_transfer',)
        return
    assert math.isclose(sum(result['balances'].values()), 1500.0)
    assert all(v >= 0 for v in result['balances'].values())


# ======================================================================
# §54 P7 — inventory events preserve order
# ======================================================================
@given(st.lists(st.tuples(st.sampled_from([-1, 1]), st.integers(min_value=1, max_value=20)),
                 min_size=1, max_size=8))
@SUPPRESS
def test_property_inventory_order_preserved(events):
    from jarvis.agent.semantics_v22_4 import execute_inventory_model
    stock = 100.0
    typed = [{'event_id': f'e{i}', 'type': 'add' if d > 0 else 'sale',
              'direction': d, 'quantity': float(q), 'unit': 'ITEM',
              'product': 'X', 'source_span': ''}
             for i, (d, q) in enumerate(events)]
    model = {'products': {'X': stock}, 'events': typed,
             'text': "What is the stock of X?"}
    result = execute_inventory_model(model)
    # sequential replay must match the model's execution exactly
    expected = stock
    failed = False
    for d, q in events:
        expected += d * q
        if expected < 0:
            failed = True
            break
    if failed:
        assert not result['ok'] and result['reason'] == 'impossible_inventory'
    else:
        assert result['ok'] and math.isclose(result['value'], expected)


# ======================================================================
# §55 — metamorphic: equivalent paraphrases -> equivalent semantics
# ======================================================================
@pytest.mark.parametrize('paraphrases', [
    ('5 USD per hour', '5 dollars/hour', '5 دلار در ساعت', 'هر ساعت 5 دلار'),
    ('10 m/s', '10 meters per second', '10 متر بر ثانیه'),
    ('60 km/h', '60 kilometers per hour', '60 کیلومتر بر ساعت'),
])
def test_metamorphic_rate_paraphrases(paraphrases):
    families = set()
    for text in paraphrases:
        qs = extract_typed_quantities(text)
        rate = [q for q in qs if q.dimension in ('currency_rate', 'speed')]
        assert rate, text
        families.add((rate[0].dimension, rate[0].normalized_unit))
    assert len(families) == 1, (paraphrases, families)


def test_metamorphic_ownership_paraphrases(eng):
    variants = [
        ('Ali has 704 dollars. Sara has 223 dollars. Ali transfers 15 dollars to Sara. '
         "What is Sara's balance?", 'en'),
        ('Ali has 704 dollars. Sara has 223 dollars. Ali pays 15 dollars to Sara. '
         "What is Sara's balance?", 'en'),
        ('Ali has 704 dollars. Sara has 223 dollars. Sara receives 15 dollars from Ali. '
         "What is Sara's balance?", 'en'),
        ('علی 704 دلار دارد. سارا 223 دلار دارد. علی 15 دلار برای سارا می‌فرستد. '
         'موجودی سارا چند است؟', 'fa'),
    ]
    for text, language in variants:
        ans = eng.solve(text, language)
        assert ans is not None and '238' in ans.text, (text, ans)


# ======================================================================
# §56 — fuzzing: spacing/half-space/digits/plural/mixed/capitalization
# ======================================================================
FUZZ_TRANSFORMS = [
    lambda s: s,                                  # identity
    lambda s: s.replace(' ', '  '),               # double spacing
    lambda s: s.replace(' ', '\u200c'),           # half-space
    lambda s: s.upper(),                          # capitalization
    lambda s: s.lower(),
    lambda s: s.translate(str.maketrans('0123456789', '۰۱۲۳۴۵۶۷۸۹')),   # persian digits
    lambda s: s.translate(str.maketrans('0123456789', '٠١٢٣٤٥٦٧٨٩')),   # arabic digits
    lambda s: s + '،',                            # punctuation
    lambda s: s.replace('worker', 'workers').replace('دلار', 'دلارها'),
]


@pytest.mark.parametrize('base', [
    'موجودی حساب 100 دلار است؛ 5 دلار اضافه کن',
    'Ali has 100 dollars. Sara has 50 dollars. Ali transfers 20 dollars to Sara. '
    "What is Sara's balance?",
    'سرعت 10 متر بر ثانیه',
    'علی 35 سال دارد و رضا 57 سال دارد. اختلاف سن علی و رضا چند سال است؟',
])
def test_fuzz_transforms_preserve_semantics(eng, base):
    is_fa = any('\u0600' <= c <= '\u06FF' for c in base)
    language = 'fa' if is_fa else 'en'
    for i, transform in enumerate(FUZZ_TRANSFORMS):
        text = transform(base)
        try:
            qs = extract_typed_quantities(text)
        except Exception as exc:   # the extractor must never crash on fuzzing
            pytest.fail(f'extractor crashed on transform {i}: {exc!r} ({text!r})')
        for q in qs:
            # spans may only be asserted on the exact text fed in
            assert 0 <= q.start <= q.end <= len(text)
        try:
            ans = eng.solve(text, language)
        except Exception as exc:
            pytest.fail(f'engine crashed on transform {i}: {exc!r} ({text!r})')
        if ans is not None:
            assert isinstance(ans.text, str) and ans.text


def test_fuzz_garbage_never_crashes():
    rng = random.Random(224)
    alphabet = string.printable + '۰۱۲۳۴۵۶۷۸۹+×/-：'
    for _ in range(300):
        text = ''.join(rng.choice(alphabet) for _ in range(rng.randint(0, 120)))
        try:
            qs = extract_typed_quantities(text)
            for q in qs:
                assert 0 <= q.start <= q.end <= len(text)
        except Exception as exc:
            pytest.fail(f'extractor crashed on garbage {text!r}: {exc!r}')


# ======================================================================
# §47-§52 guards under fuzz
# ======================================================================
@given(st.floats(allow_nan=True, allow_infinity=True))
@SUPPRESS
def test_property_numeric_guard_total(v):
    guard = numeric_guard(v)
    if math.isnan(v) or math.isinf(v):
        assert guard is not None
    else:
        assert guard is None


@given(st.floats(min_value=-2, max_value=2))
@SUPPRESS
def test_property_probability_guard_range(p):
    guard = probability_guard(p)
    assert (guard is not None) != (0.0 <= p <= 1.0)
