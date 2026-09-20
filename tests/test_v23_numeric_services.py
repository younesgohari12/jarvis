"""JARVIS v23.0 — verified numeric services: units / work-rate / narrative.

Covers the v23 Language Brain Plan (مرحله 1) deliverables:

  * unit conversion service (کارتابل واحد) — typed, fail-closed, witnessed
  * work-rate algebra service — scale / inverted hours / combined inversion /
    per-worker sums / same-crew / mid-task leave / per-plural / rate nouns
  * narrative arithmetic service — the measured persian_math/english_math gaps
  * engine wiring — v23 runs ONLY on v22 abstain/refuse, never overrides a
    verified v22 answer or a structured dimension refusal
  * routing — the v23 numeric route outranks close-app collisions and
    word-problem labels; translation guards keep precedence
  * hard negatives — unsupported units / insufficient info must abstain
"""
from __future__ import annotations

import pytest

from jarvis.agent import units_service_v23 as u23
from jarvis.agent import work_rate_v23 as w23
from jarvis.agent import narrative_math_v23 as n23
from jarvis.agent.local_intelligence_v22 import LocalIntelligenceV22
from jarvis.agent.local_intelligence_v23 import LocalIntelligenceV23


# ======================================================================
# 1. Unit conversion service (کارتابل واحد)
# ======================================================================
@pytest.mark.parametrize('text,lang,expected', [
    ('How many grams are 2.5 kilograms?', 'en', 2500.0),
    ('How many kilometers are 5000 meters?', 'en', 5.0),
    ('How many minutes are 2.5 hours?', 'en', 150.0),
    ('How many seconds are 3 minutes?', 'en', 180.0),
    ('How many hours are 270 minutes?', 'en', 4.5),
    ('A cable is 1.5 meters long; how many centimeters is that?', 'en', 150.0),
    ('A tank holds 2 liters; how many milliliters is that?', 'en', 2000.0),
    ('A speed of 72 km/h equals how many m/s?', 'en', 20.0),
    ('A speed of 10 m/s equals how many km/h?', 'en', 36.0),
    ('What is 300 grams in kilograms?', 'en', 0.3),
    ('2 tons how many kilograms?', 'en', 2000.0),
    ('یک مسافت ۳ کیلومتر است؛ چند متر است؟', 'fa', 3000.0),
    ('۲.۵ کیلوگرم چند گرم است؟', 'fa', 2500.0),
    ('نیم ساعت چند دقیقه است؟', 'fa', 30.0),
    ('سه ساعت چند دقیقه می‌شود؟', 'fa', 180.0),
    ('۵ تن چند کیلوگرم می‌شود؟', 'fa', 5000.0),
    ('سرعت ۹۰ کیلومتر بر ساعت چند متر بر ثانیه است؟', 'fa', 25.0),
    ('سرعت ۲۵ متر بر ثانیه چند کیلومتر بر ساعت است؟', 'fa', 90.0),
])
def test_units_service_converts(text, lang, expected):
    result = u23.solve_conversion(text, lang)
    assert result is not None
    assert abs(result['value'] - expected) < 1e-9
    # witness: a second derivation agrees (service contract)
    again = u23.solve_conversion(text, lang)
    assert again is not None and abs(again['value'] - result['value']) < 1e-9


@pytest.mark.parametrize('text,lang', [
    ('How many furlongs are 3 kilometers?', 'en'),
    ('How many parsecs are 4 kilometers?', 'en'),
    ('How many Fahrenheit are 30 Celsius?', 'en'),
    ('۳۰ درجه سلسیوس چند فارنهایت است؟', 'fa'),
    ('۴ کیلومتر چند فرسخ است؟', 'fa'),
    ('۱۲ مگاپیکسل چند کیلوبایت است؟', 'fa'),
    ('۲ ساعت و ۵ دلار را جمع کن', 'fa'),
    ('How many dollars are 500 minutes?', 'en'),
    ('چند گرم است؟', 'fa'),
])
def test_units_service_abstains(text, lang):
    """Unsupported/unknown/mixed units are fail-closed refusals, never guesses."""
    assert u23.solve_conversion(text, lang) is None


# ======================================================================
# 2. Work-rate algebra service
# ======================================================================
@pytest.mark.parametrize('text,lang,expected', [
    # forward scale
    ('If 6 workers assemble 96 chairs in 4 hours, how many chairs do 9 workers assemble in 5 hours?', 'en', 180.0),
    ('If 4 machines produce 220 parts in 5 hours, how many parts do 7 machines produce in 3 hours?', 'en', 231.0),
    ('3 workers produce 90 pieces in 60 minutes. How many pieces do 5 workers produce in 40 minutes?', 'en', 100.0),
    ('اگر ۴ کارگر در ۳ ساعت ۶۰ قطعه تولید کنند، ۶ کارگر در ۵ ساعت چند قطعه تولید می‌کنند؟', 'fa', 150.0),
    ('اگر ۸ نفر در ۲ ساعت ۴۰ کالا بسته‌بندی کنند، ۵ نفر در ۶ ساعت چند کالا بسته‌بندی می‌کنند؟', 'fa', 75.0),
    ('A team of 8 developers closes 40 tickets in 2 hours. How many tickets would a team of 10 developers close in 5 hours at the same pace?', 'en', 125.0),
    ('A crew of 4 packs 60 boxes in 3 hours. How many boxes does a crew of 6 pack in 2 hours at the same pace?', 'en', 60.0),
    # inverted hours (same total output)
    ('10 workers produce 200 units in 4 hours. How many hours do 8 workers need to produce 200 units?', 'en', 5.0),
    ('If 5 workers need 6 hours to paint a fence, how many hours do 3 workers need for the same fence?', 'en', 10.0),
    ('۵ کارگر برای رنگ‌آمیزی حصار به ۶ ساعت نیاز دارند؛ ۳ کارگر چند ساعت نیاز دارند؟', 'fa', 10.0),
    # harmonic combined inversion
    ('Ali can finish a report in 6 hours and Sara in 3 hours. Working together, how many hours do they need?', 'en', 2.0),
    ('One tap fills a tank in 12 hours, another in 4 hours. Together, how many hours to fill the tank?', 'en', 3.0),
    ('Ali alone cleans a yard in 8 hours and Reza alone in 8 hours. How many hours do they need together?', 'en', 4.0),
    # per-worker rate sums
    ('Ali builds 4 chairs per hour and Sara builds 3 chairs per hour. Working together for 2 hours, how many chairs do they build?', 'en', 14.0),
    ('A worker installs 6 windows an hour; a second installs 5 windows an hour. Together, how many windows in 4 hours?', 'en', 44.0),
    # mid-task crew change
    ('12 workers load 480 crates in 4 hours. After 2 hours, half of the workers leave. How many crates are loaded in total after 3 hours?', 'en', 300.0),
    # same-crew extrapolation
    ('A shop assembles 15 bicycles in 6 hours with 5 workers. If no workers are added or removed, how many bicycles are assembled in 8 hours?', 'en', 20.0),
    # extended rate nouns
    ('A printer prints 38 pages per minute. How many pages in 2.5 minutes?', 'en', 95.0),
    ('A machine packs 60 boxes every 2 hours. How many boxes in 5 hours?', 'en', 150.0),
    # 3 workers pack 45 boxes in 5 hours; 7 workers in 2 hours
    ('3 workers pack 45 boxes in 5 hours. 7 workers pack how many boxes in 2 hours?', 'en', 42.0),
])
def test_work_rate_service_solves(text, lang, expected):
    result = w23.solve_work_rate(text, lang)
    assert result is not None
    assert abs(result['value'] - expected) < 1e-6
    again = w23.solve_work_rate(text, lang)
    assert again is not None and abs(again['value'] - result['value']) < 1e-9


@pytest.mark.parametrize('text,lang', [
    ('چند کارگر چند قطعه تولید می‌کنند؟', 'fa'),
    ('Some workers produce 40 pieces. How long does it take?', 'en'),
    ('اگر سرعت کار بیشتر شود، چند قطعه تولید می‌شود؟', 'fa'),
    ('A team finishes the job faster than before. How many workers joined?', 'en'),
    ('Workers produce pieces in hours. How many pieces in 5 hours?', 'en'),
])
def test_work_rate_service_abstains_on_insufficient_info(text, lang):
    assert w23.solve_work_rate(text, lang) is None


# ======================================================================
# 3. Narrative arithmetic service
# ======================================================================
@pytest.mark.parametrize('text,lang,expected', [
    ('مجموع ۲۵۰ و ۱۷۵ چند می‌شود؟', 'fa', 425.0),
    ('تفاضل ۹۸ و ۴۵ چند است؟', 'fa', 53.0),
    ('What is the difference between 98 and 45?', 'en', 53.0),
    ('اگر قیمت یک کالا از ۸۰۰ به ۶۵۰ برسد، کاهش چند تومان بوده است؟', 'fa', 150.0),
    ('یک باسکول ۴۳۰ کیلوگرم گندم نشان می‌دهد؛ کیسه‌های خالی ۳۰ کیلوگرم است. گندم خالص چند کیلوگرم است؟', 'fa', 400.0),
    ('نصف یک عدد ۴۵ است؛ آن عدد چند است؟', 'fa', 90.0),
    ('One third of a number is 17. What is the number?', 'en', 51.0),
    ('Half of a number is 45. What is the number?', 'en', 90.0),
    ('اگر هر بسته ۱۲ قلم کالا داشته باشد، ۳۵ بسته چند قلم دارد؟', 'fa', 420.0),
    ('Each box holds 12 items. How many items do 35 boxes hold?', 'en', 420.0),
    ('یک مستطیل با طول ۱۲ و عرض ۷ متر، محیطش چند متر است؟', 'fa', 38.0),
    ('A rectangle is 12 meters long and 7 meters wide. What is its perimeter in meters?', 'en', 38.0),
    ('A recipe needs 450 grams of flour per batch. How many grams of flour for 4 batches?', 'en', 1800.0),
    ('A bus has 57 passengers; 19 get off and 12 get on. How many passengers now?', 'en', 50.0),
    ('A farmer sold 128 apples and has 272 left. How many apples did he start with?', 'en', 400.0),
    ('A rope 9 meters long is cut into equal 0.75-meter pieces. How many pieces?', 'en', 12.0),
    ('The temperature rose 6 degrees, then fell 11 degrees, ending at 9 degrees. What did it start at?', 'en', 14.0),
    ('A printer prints 38 pages per minute. How many pages in 2.5 minutes?', 'en', 95.0),
    ('A warehouse stores 340 crates in the morning and 185 crates in the afternoon. How many crates in total?', 'en', 525.0),
    ('ثمن یک مبلغ ۹۰ هزار تومان و مابقی ۲۱۰ هزار تومان است؛ کل چقدر است؟', 'fa', 300.0),
])
def test_narrative_service_solves(text, lang, expected):
    result = n23.solve_narrative(text, lang)
    assert result is not None
    assert abs(result['value'] - expected) < 1e-6
    again = n23.solve_narrative(text, lang)
    assert again is not None and abs(again['value'] - result['value']) < 1e-9


@pytest.mark.parametrize('text,lang', [
    ('یکی از اعداد ۴۵ است؛ عدد دیگر چند است؟', 'fa'),
    ('A crate lost some apples and now has 30. How many did it lose?', 'en'),
    ('قیمت کالا کاهش یافت؛ چقدر کاهش یافت؟', 'fa'),
    ('The temperature changed twice and ended at 9 degrees. What did it start at?', 'en'),
])
def test_narrative_service_abstains_on_missing_slots(text, lang):
    assert n23.solve_narrative(text, lang) is None


# ======================================================================
# 4. Engine wiring — v23 never overrides v22 verdicts
# ======================================================================
@pytest.fixture(scope='module')
def eng():
    return LocalIntelligenceV23(output_style='canonical')


def test_v23_engine_answers_units(eng):
    a = eng.solve('How many grams are 2.5 kilograms?', 'en')
    assert a is not None and '2500' in a.text
    assert 'v23_units_service' in a.checks


def test_v23_engine_answers_work_rate(eng):
    a = eng.solve('If 6 workers assemble 96 chairs in 4 hours, how many chairs do 9 workers assemble in 5 hours?', 'en')
    assert a is not None and '180' in a.text


def test_v23_engine_answers_narrative(eng):
    a = eng.solve('One third of a number is 17. What is the number?', 'en')
    assert a is not None and '51' in a.text


def test_v23_engine_preserves_v22_answers(eng):
    # a v22-verified canonical answer must be byte-identical (contract)
    a = eng.solve('با 3 کارگر و 4 ساعت، خروجی 84 قطعه است. با 8 کارگر و 6 ساعت چه خروجی داریم؟', 'fa')
    assert a is not None and a.text == '336'


def test_v23_engine_keeps_dimension_refusals(eng):
    a = eng.solve('۲ ساعت و ۵ دلار را جمع کن', 'fa')
    assert a is not None
    assert 'نمی\u200cتوان مستقیماً جمع کرد' in a.text
    trace = eng.last_trace
    assert trace.get('v22_dimension_block')


def test_v23_version_labels():
    assert LocalIntelligenceV23.VERSION == '6.0.0-v23.0'
    assert LocalIntelligenceV22.VERSION == '5.0.0-v22.4'
    assert issubclass(LocalIntelligenceV23, LocalIntelligenceV22)


def test_v23_witness_registered(eng):
    eng.solve('How many minutes are 2.5 hours?', 'en')
    trace = eng.last_trace
    assert trace['verification']['passed']
    assert trace['v23_service']['witness'] == 'rederivation_agrees'
    assert trace['v23_service']['kind'] == 'unit_conversion'


# ======================================================================
# 5. Routing — v23 numeric route precedence
# ======================================================================
def test_router_v23_numeric_route():
    from jarvis.agent.intent_router import IntentRouter
    # packaging math must NOT be stolen by close-app ('بسته' cue)
    assert IntentRouter._v23_numeric_route(
        'اگر ۸ نفر در ۲ ساعت ۴۰ کالا بسته‌بندی کنند، ۵ نفر در ۶ ساعت چند کالا بسته‌بندی می‌کنند؟')
    assert IntentRouter._v23_numeric_route('How many grams are 2.5 kilograms?')
    assert not IntentRouter._v23_numeric_route('چند کارگر چند قطعه تولید می‌کنند؟')
    assert not IntentRouter._v23_numeric_route('سلام')


def test_router_routes_packaging_math_to_local_intelligence():
    from tests.helpers import TemporaryRuntime
    with TemporaryRuntime() as rt:
        route = rt.agent.router.route(
            'اگر ۸ نفر در ۲ ساعت ۴۰ کالا بسته‌بندی کنند، ۵ نفر در ۶ ساعت چند کالا بسته‌بندی می‌کنند؟')
        assert route.intent == 'local_intelligence'
        assert route.source == 'local_intelligence_router_v23'


def test_full_runtime_answers_blind_families():
    from tests.helpers import TemporaryRuntime
    with TemporaryRuntime() as rt:
        rt.memory.set_setting('memory_enabled', False)
        rt.memory.set_setting('internet_enabled', False)
        for question, expected in [
            ('How many kilometers are 5000 meters?', '5'),
            ('اگر ۴ کارگر در ۳ ساعت ۶۰ قطعه تولید کنند، ۶ کارگر در ۵ ساعت چند قطعه تولید می‌کنند؟', '150'),
            ('A bus has 57 passengers; 19 get off and 12 get on. How many passengers now?', '50'),
        ]:
            reply = rt.agent.respond(question)
            assert expected in reply.text, f'{question!r} -> {reply.text!r}'
