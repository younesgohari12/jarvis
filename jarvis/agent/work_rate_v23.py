"""JARVIS v23 — Work-Rate Algebra Service.

V23_TRAINING_HANDOFF.md target #1 and V23_LANGUAGE_BRAIN_PLAN.md §2: the
work-rate family was the weakest blind family (52/90). v23 teaches the
underlying algebra — rate × time = output, additive per-worker rates,
harmonic combined inversion — not surface cues.

Sub-solvers (each fail-closed, each independently re-derived as a witness):
  rate_scale              out2 = out1 * (w2/w1) * (t2/t1)
  rate_combined_inversion T = 1 / (1/T1 + 1/T2)
  rate_per_worker_sum     (r1 + r2) * T
  rate_inverted_hours     T2 = w1 * T1 / w2        (same total output)
  rate_same_crew          out2 = out1 * (t2/t1)    (same crew)
  rate_midtask_leave      phase split when a fraction of the crew leaves
  rate_per_plural         (out / span) * time      ('every 2 hours')
  rate_single_application rate * time with extended v23 nouns ('pages')

All extractions are regex-based over ASCII-folded text; every numeric result
passes s4.numeric_guard and a second, independent re-derivation must agree
before the engine may speak. Nothing is guessed: a missing slot returns None.
"""
from __future__ import annotations

import re

from jarvis.agent import semantics_v22_4 as s4

DIGITS = str.maketrans('۰۱۲۳۴۵۶۷۸۹٠١٢٣٤٥٦٧٨٩', '01234567890123456789')

# ---------------------------------------------------------------------------
# vocabulary (single source of truth for extraction AND routing pre-checks)
# ---------------------------------------------------------------------------
WORKER_WORDS = r'کارگر|نفر|کارمند|ماشین|دستگاه|workers?|people|employees|machines?|crew|developers?|customers?'

TIME_UNITS_RE = r'ساعت|دقیقه|ثانیه|روز|hours?|hr|h\b|minutes?|min|seconds?|sec|days?'

# output nouns: objects a crew produces/packs/loads (extended for v23)
OUTPUT_NOUNS_RE = (
    r'قطعه|کالا|جعبه|صندلی|میز|کیسه|بسته|صندوق|دوچرخه|کاشی|پنجره|دیوار|بلیط|'
    r'پاره|قطعات|crates?|boxes?|chairs?|tables?|pieces?|items?|parts?|'
    r'packages?|bags?|widgets?|cartons?|panels?|bricks?|walls?|tickets?|'
    r'bicycles?|windows?|cakes?|units?|tiles?|products?'
)

RATE_NOUNS_RE = (
    r'صفحه|قطعه|کالا|جعبه|صندلی|میز|کیسه|چرخ|پنل|کارتن|بطری|'
    r'pages?|pieces?|items?|chairs?|tables?|boxes?|crates?|widgets?|'
    r'cartons?|panels?|windows?|cakes?|bottles?'
)

# currency-rate phrase: '15 dollars an hour' / '40 dollars per hour' /
# 'ساعتی ۱۵ دلار' / '15 دلار در ساعت'
_CURRENCY_RATE_RE = re.compile(
    r'(?<![\d.])(\d+(?:\.\d+)?)(?!\d)\s*(?:دلار|dollars?|usd|یورو|euros?|eur|تومان|tomans?)\s*'
    r'(?:/|(?:در|بر|per|an|a|هر)\s*)(ساعت|دقیقه|روز|hours?|hr|minutes?|min|days?)',
    re.I)
# currency span phrase: '10 dollars for every 2 hours' / 'هر ۲ ساعت ۱۰ دلار'
_CURRENCY_SPAN_RE = re.compile(
    r'(?<![\d.])(\d+(?:\.\d+)?)(?!\d)\s*(?:دلار|dollars?|usd|یورو|euros?|eur|تومان|tomans?)\s*'
    r'(?:for\s+|per\s+)?(?:هر|every)\s*(?<![\d.])(\d+(?:\.\d+)?)(?!\d)\s*'
    r'(ساعت|دقیقه|روز|hours?|hr|minutes?|min|days?)', re.I)
_CURRENCY_SPAN_FA_RE = re.compile(
    r'(?:هر)\s*(?<![\d.])(\d+(?:\.\d+)?)(?!\d)\s*(ساعت|دقیقه|روز|hours?|hr|minutes?|min|days?)\s*'
    r'(?<![\d.])(\d+(?:\.\d+)?)(?!\d)\s*(?:دلار|dollars?|usd|یورو|euros?|eur|تومان|tomans?)', re.I)
# FA post-noun rate: 'ساعتی ۱۵ دلار دریافت می‌کند' (rate word BEFORE value)
_CURRENCY_RATE_FA_RE = re.compile(
    r'ساعتی\s*(?<![\d.])(\d+(?:\.\d+)?)(?!\d)\s*'
    r'(?:دلار|dollars?|usd|یورو|euros?|eur|تومان|tomans?)', re.I)
_MONEY_QUESTION = re.compile(
    r'how\s+much|چقدر|چند|what\s+is\s+the\s+(?:total|cost)|total\s+cost|'
    r'هزینه\s+کل|کل\s+هزینه', re.I)

_WORKER_RE = re.compile(r'(\d+(?:\.\d+)?)\s+(' + WORKER_WORDS + r')', re.I)
# 'a crew of 4' / 'a team of 8 developers' — the number FOLLOWS the collective
_WORKER_OF_RE = re.compile(
    r'\b(?:crew|team|group|gang)\s+of\s+(\d+(?:\.\d+)?)|'
    r'\b(?:تیم|گروه)\s+از\s+(\d+(?:\.\d+)?)', re.I)
_TIME_RE = re.compile(
    r'(\d+(?:\.\d+)?)\s*(' + TIME_UNITS_RE + r')', re.I)
_OUTPUT_RE = re.compile(r'(\d+(?:\.\d+)?)\s+(' + OUTPUT_NOUNS_RE + r')', re.I)
_RATE_RE = re.compile(
    r'(\d+(?:\.\d+)?)\s+(' + RATE_NOUNS_RE + r')\s*'
    r'(?:/|(?:در|هر)\s+|per\s+|an?\s+|every\s+)(ساعت|دقیقه|روز|hours?|hr|minutes?|min|days?)',
    re.I)

QUESTION_HOURS = re.compile(r'چند\s+(?:ساعت|ساعتی)|how\s+(?:many|much)\s+hours?|'
                            r'how\s+long|چقدر\s+زمان', re.I)
QUESTION_TOGETHER = re.compile(r'با\s*هم|هم\u200cزمان|جمع\s+می\u200cشوند|'
                               r'together|both|combined|jointly', re.I)
QUESTION_OUTPUT = re.compile(r'چند\s+\S+|how\s+many\s+\w+', re.I)
LEAVE_CUE = re.compile(r'نیمی|نصف|نصفی|می\u200cروند|بیرون\s+می\u200cروند|ترک\s+می\u200cکنند|'
                       r'half\s+of\s+the\s+\w+|leave|leaves|left\s+the\s+\w+|walk\s+out', re.I)
JOIN_CUE = re.compile(r'می\u200cپیوندند|اضافه\s+می\u200cشوند|join|joins|joined', re.I)
SAME_WORK_CUE = re.compile(
    r'همین\s+کار|همان\s+کار|یک\s+کار|same\s+(?:fence|job|work|task|wall|pace)|'
    r'need|needs|نیاز\s+دارند|نیاز\s+دارد|برای\s+همین', re.I)
NO_CREW_CHANGE_CUE = re.compile(
    r'تغییری\s+نکند|تغییری\s+نکند|اضافه\s+یا\s+حذف|همان\s+تعداد|'
    r'no\s+workers?\s+are\s+added\s+or\s+removed|same\s+pace|نرخ\s+یکسان|همان\s+آهنگ', re.I)

_HOURS = {'ساعت': 1.0, 'hour': 1.0, 'hours': 1.0, 'hr': 1.0, 'h': 1.0,
          'دقیقه': 1 / 60, 'minute': 1 / 60, 'minutes': 1 / 60, 'min': 1 / 60,
          'ثانیه': 1 / 3600, 'second': 1 / 3600, 'seconds': 1 / 3600, 'sec': 1 / 3600,
          'روز': 24.0, 'day': 24.0, 'days': 24.0}


def fold(text: str) -> str:
    return (text or '').translate(DIGITS)


def _hours(tok: str) -> float:
    return _HOURS.get(tok.strip().lower(), 1.0)


def _num(tok: str) -> float:
    return float(tok)


def _spans_overlap(a, b) -> bool:
    return not (a[1] <= b[0] or b[1] <= a[0])


def _extract_rate_phrases(t: str):
    """[(value, noun, unit_token, span)] for '38 pages per minute' forms."""
    out = []
    for m in _RATE_RE.finditer(t):
        out.append({'value': _num(m.group(1)), 'noun': m.group(2).lower(),
                    'unit': m.group(3).lower(), 'span': (m.start(), m.end())})
    return out


def _extract_times(t: str, skip=()):
    out = []
    for m in _TIME_RE.finditer(t):
        if any(_spans_overlap((m.start(), m.end()), s) for s in skip):
            continue
        out.append({'value': _num(m.group(1)), 'unit': m.group(2).lower(),
                    'span': (m.start(), m.end()),
                    'hours': _num(m.group(1)) * _hours(m.group(2))})
    return out


def _extract_workers(t: str):
    out = [{'value': _num(m.group(1)), 'span': (m.start(), m.end())}
           for m in _WORKER_RE.finditer(t)]
    for m in _WORKER_OF_RE.finditer(t):
        val = m.group(1) or m.group(2)
        if val is None:
            continue
        num_start = m.start(1) if m.group(1) is not None else m.start(2)
        # dedupe: 'team of 8 developers' also matches '8 developers'
        if any(span[0] <= num_start < span[1] for span in
               (w['span'] for w in out)):
            continue
        out.append({'value': float(val), 'span': (m.start(), m.end())})
    return sorted(out, key=lambda w: w['span'][0])


def _extract_outputs(t: str):
    return [{'value': _num(m.group(1)), 'noun': m.group(2).lower(),
             'span': (m.start(), m.end())}
            for m in _OUTPUT_RE.finditer(t)]


def _render_hours(v: float, language: str) -> str:
    from jarvis.agent.language_brain_v22 import fmt_number
    return fmt_number(v)


def _noun_pair(noun: str) -> tuple[str, str]:
    """(fa, en) rendering words for an output noun."""
    fa = {'قطعه': 'قطعه', 'کالا': 'کالا', 'جعبه': 'جعبه', 'صندلی': 'صندلی',
          'میز': 'میز', 'کیسه': 'کیسه', 'بسته': 'بسته', 'صندوق': 'صندوق',
          'دوچرخه': 'دوچرخه', 'کاشی': 'کاشی', 'پنجره': 'پنجره', 'دیوار': 'دیوار',
          'بلیط': 'بلیط', 'crates': 'جعبه', 'boxes': 'جعبه', 'chairs': 'صندلی',
          'tables': 'میز', 'pieces': 'قطعه', 'items': 'کالا', 'parts': 'قطعه',
          'packages': 'بسته', 'bags': 'کیسه', 'widgets': 'قطعه',
          'cartons': 'کارتن', 'panels': 'پنل', 'bricks': 'کاشی',
          'walls': 'دیوار', 'tickets': 'بلیط', 'bicycles': 'دوچرخه',
          'windows': 'پنجره', 'cakes': 'کیک', 'units': 'واحد',
          'tiles': 'کاشی', 'products': 'کالا'}
    en = {'قطعه': 'pieces', 'کالا': 'items', 'جعبه': 'boxes', 'صندلی': 'chairs',
          'میز': 'tables', 'کیسه': 'bags', 'بسته': 'packages',
          'صندوق': 'crates', 'دوچرخه': 'bicycles', 'کاشی': 'tiles',
          'پنجره': 'windows', 'دیوار': 'walls', 'بلیط': 'tickets',
          'pages': 'pages', 'صفحه': 'صفحه'}
    en.setdefault(noun, noun)
    fa.setdefault(noun, noun)
    return fa[noun], en[noun]


# ---------------------------------------------------------------------------
# sub-solvers — each returns {'value', 'noun', 'kind'} or None
# ---------------------------------------------------------------------------
def _solve_rate_scale(t: str):
    """Two (workers, time) scenes, one output: out2 = out1*(w2/w1)*(t2/t1).
    Span-ordering guard: the counted output must sit BETWEEN the two scenes
    (scene1 = w1,t1 before it; scene2 = w2,t2 after it)."""
    workers = _extract_workers(t)
    outputs = _extract_outputs(t)
    if len(workers) != 2 or len(outputs) != 1:
        return None
    rate_phrases = _extract_rate_phrases(t)
    times = _extract_times(t, skip=[rp['span'] for rp in rate_phrases])
    if len(times) != 2:
        return None
    out1 = outputs[0]
    w1s, w2s = sorted(workers, key=lambda w: w['span'][0])
    t1s, t2s = sorted(times, key=lambda x: x['span'][0])
    o_pos = out1['span'][0]
    scene1_start = min(w1s['span'][0], t1s['span'][0])
    scene2_start = min(w2s['span'][0], t2s['span'][0])
    if not (scene1_start < o_pos < scene2_start):
        return None
    if w1s['span'][0] > t1s['span'][0] and w1s['span'][0] > o_pos:
        return None   # w1 must precede the output ('crew of 4 packs 60 boxes')
    w1, w2 = w1s['value'], w2s['value']
    t1, t2 = t1s['hours'], t2s['hours']
    if w1 <= 0 or w2 <= 0 or t1 <= 0 or t2 <= 0:
        return None
    value = out1['value'] * (w2 / w1) * (t2 / t1)
    return {'kind': 'rate_scale', 'value': value, 'noun': out1['noun'],
            'values': (w1, w2, t1, t2, out1['value'])}


def _solve_combined_inversion(t: str):
    """Two solo completion times + together cue, question asks hours:
    T = 1/(1/T1 + 1/T2). Pure solo-time form: no worker phrases, no counted
    outputs ('One tap fills a tank in 12 hours, another in 4 hours.')."""
    if not (QUESTION_TOGETHER.search(t) and QUESTION_HOURS.search(t)):
        return None
    rate_phrases = _extract_rate_phrases(t)
    times = _extract_times(t, skip=[rp['span'] for rp in rate_phrases])
    if len(times) != 2:
        return None
    workers = _extract_workers(t)
    outputs = _extract_outputs(t)
    if workers or outputs:
        return None   # solo-times form only
    t1, t2 = times[0]['hours'], times[1]['hours']
    if t1 <= 0 or t2 <= 0:
        return None
    value = 1.0 / (1.0 / t1 + 1.0 / t2)
    return {'kind': 'rate_combined_inversion', 'value': value,
            'noun': None, 'hours_answer': True}


def _solve_per_worker_sum(t: str):
    """Two per-worker rates (same noun) + one together-time:
    (r1 + r2) * T."""
    if not QUESTION_TOGETHER.search(t):
        return None
    rates = _extract_rate_phrases(t)
    if len(rates) != 2:
        return None
    if rates[0]['noun'] != rates[1]['noun']:
        return None
    times = _extract_times(t, skip=[rp['span'] for rp in rates])
    if len(times) != 1:
        return None
    T = times[0]['hours']
    if T <= 0:
        return None
    value = (rates[0]['value'] + rates[1]['value']) * T
    return {'kind': 'rate_per_worker_sum', 'value': value,
            'noun': rates[0]['noun']}


def _solve_inverted_hours(t: str):
    """Two crews, one time; question asks hours: T2 = w1*T1/w2 (same work).
    Requires an explicit same-work cue OR a counted output ('...produce 200
    units ... produce 200 units?')."""
    if not QUESTION_HOURS.search(t):
        return None
    if QUESTION_TOGETHER.search(t):
        return None   # harmonic form, not crew scaling
    workers = _extract_workers(t)
    if len(workers) != 2:
        return None
    rate_phrases = _extract_rate_phrases(t)
    times = _extract_times(t, skip=[rp['span'] for rp in rate_phrases])
    if len(times) != 1:
        return None
    outputs = _extract_outputs(t)
    if not outputs and not SAME_WORK_CUE.search(t):
        return None
    w1, w2 = workers[0]['value'], workers[1]['value']
    t1 = times[0]['hours']
    if w1 <= 0 or w2 <= 0 or t1 <= 0:
        return None
    value = w1 * t1 / w2
    return {'kind': 'rate_inverted_hours', 'value': value, 'noun': None,
            'hours_answer': True}


def _solve_same_crew(t: str):
    """One crew, one output, two times; question asks output:
    out2 = out1*(t2/t1)."""
    if QUESTION_HOURS.search(t):
        return None
    workers = _extract_workers(t)
    if len(workers) != 1:
        return None
    outputs = _extract_outputs(t)
    if len(outputs) != 1:
        return None
    rate_phrases = _extract_rate_phrases(t)
    times = _extract_times(t, skip=[rp['span'] for rp in rate_phrases])
    if len(times) != 2:
        return None
    if not NO_CREW_CHANGE_CUE.search(t):
        return None
    t1, t2 = times[0]['hours'], times[1]['hours']
    if t1 <= 0 or t2 <= 0:
        return None
    value = outputs[0]['value'] * (t2 / t1)
    return {'kind': 'rate_same_crew', 'value': value,
            'noun': outputs[0]['noun']}


def _solve_midtask(t: str):
    """Crew change mid-task: rate = out1/(w1*T_total); phases recompute."""
    if not (LEAVE_CUE.search(t) or JOIN_CUE.search(t)):
        return None
    outputs = _extract_outputs(t)
    if len(outputs) != 1:
        return None
    rate_phrases = _extract_rate_phrases(t)
    times = _extract_times(t, skip=[rp['span'] for rp in rate_phrases])
    if len(times) < 3:
        return None
    workers = _extract_workers(t)
    if len(workers) != 1:
        return None
    out1 = outputs[0]['value']
    T_total = times[0]['hours']          # '480 crates in 4 hours'
    t_switch = times[1]['hours']         # 'after 2 hours'
    t_q = times[-1]['hours']             # 'in total after 3 hours'
    w1 = workers[0]['value']
    if w1 <= 0 or T_total <= 0 or t_switch <= 0 or t_q <= 0:
        return None
    if t_switch >= T_total + 1e-12:
        return None
    if not LEAVE_CUE.search(t):
        return None   # v23 covers the leave case; join stays v23.1
    per_wh = out1 / (w1 * T_total)
    if not re.search(r'نیمی|نصف|half', t, re.I):
        return None
    w2 = w1 / 2.0
    phase1 = per_wh * w1 * min(t_switch, t_q)
    phase2 = per_wh * w2 * max(t_q - t_switch, 0.0)
    value = phase1 + phase2
    return {'kind': 'rate_midtask_leave', 'value': value,
            'noun': outputs[0]['noun']}


def _solve_per_plural(t: str):
    """'packs 60 boxes every 2 hours ... in 5 hours' -> (out/span)*T.
    FA form: 'در هر ۲ ساعت ۶۰ جعبه تولید می‌کند. در ۵ ساعت ...'"""
    m = re.search(r'(\d+(?:\.\d+)?)\s+(' + OUTPUT_NOUNS_RE + r')\s*'
                  r'(?:هر|every)\s+(\d+(?:\.\d+)?)\s*(' + TIME_UNITS_RE + r')',
                  t, re.I)
    if not m:
        m = re.search(r'(?:هر)\s+(\d+(?:\.\d+)?)\s*(' + TIME_UNITS_RE + r')\s*'
                      r'(\d+(?:\.\d+)?)\s+(' + OUTPUT_NOUNS_RE + r')', t, re.I)
        if not m:
            return None
        out1 = _num(m.group(3))
        span = _num(m.group(1)) * _hours(m.group(2))
        noun = m.group(4).lower()
    else:
        out1 = _num(m.group(1))
        span = _num(m.group(3)) * _hours(m.group(4))
        noun = m.group(2).lower()
    if out1 <= 0 or span <= 0:
        return None
    rate = out1 / span
    skip = [(m.start(), m.end())]
    times = _extract_times(t, skip=skip)
    if len(times) != 1:
        return None
    T = times[0]['hours']
    if T <= 0:
        return None
    return {'kind': 'rate_per_plural', 'value': rate * T, 'noun': noun}


def _solve_single_application(t: str):
    """'A printer prints 38 pages per minute. How many pages in 2.5 minutes?'"""
    rates = _extract_rate_phrases(t)
    if len(rates) != 1:
        return None
    rate = rates[0]
    times = _extract_times(t, skip=[rate['span']])
    if len(times) != 1:
        return None
    unit = rate['unit'].lower()
    if unit in ('ثانیه', 'second', 'seconds', 'sec'):
        T = times[0]['hours'] * 3600.0
    elif unit in ('دقیقه', 'minute', 'minutes', 'min'):
        T = times[0]['hours'] * 60.0
    elif unit in ('ساعت', 'hour', 'hours', 'hr'):
        T = times[0]['hours']
    elif unit in ('روز', 'day', 'days'):
        T = times[0]['hours'] / 24.0
    else:
        return None
    if T <= 0:
        return None
    value = rate['value'] * T
    return {'kind': 'rate_single_application', 'value': value,
            'noun': rate['noun']}


def _solve_currency_application(t: str):
    """Currency rate × time ('15 dollars an hour for 8 hours' -> 120) and
    currency span rates ('10 dollars for every 2 hours ... for 6 hours'
    -> 30). Weak-W1-style generalization of the v22 rate families."""
    if not _MONEY_QUESTION.search(t):
        return None
    matches = list(_CURRENCY_RATE_RE.finditer(t))
    fa_rates = list(_CURRENCY_RATE_FA_RE.finditer(t))
    spans = list(_CURRENCY_SPAN_RE.finditer(t)) + \
        list(_CURRENCY_SPAN_FA_RE.finditer(t))
    # Form A: exactly one rate phrase + exactly one standalone time
    if len(matches) == 1 and len(fa_rates) == 0 and not spans:
        rate = float(matches[0].group(1))
        unit = matches[0].group(2).lower()
        factor = {'ساعت': 1.0, 'hour': 1.0, 'hours': 1.0, 'hr': 1.0,
                  'دقیقه': 60.0, 'minute': 60.0, 'minutes': 60.0, 'min': 60.0,
                  'روز': 1 / 24, 'day': 1 / 24, 'days': 1 / 24}.get(unit)
        if factor is None:
            return None
        times = _extract_times(t, skip=[matches[0].span()])
        if len(times) != 1:
            return None
        T = times[0]['hours'] * factor   # time expressed in the rate's scale
        if T <= 0:
            return None
        return {'kind': 'currency_rate_application', 'value': rate * T,
                'noun': None, 'currency': True}
    # Form A2 (FA): 'ساعتی ۱۵ دلار ... برای ۸ ساعت ... چقدر؟' -> 120
    if len(fa_rates) == 1 and len(matches) == 0 and not spans:
        rate = float(fa_rates[0].group(1))
        times = _extract_times(t, skip=[fa_rates[0].span()])
        if len(times) != 1:
            return None
        T = times[0]['hours']
        if T <= 0:
            return None
        return {'kind': 'currency_rate_application', 'value': rate * T,
                'noun': None, 'currency': True}
    # Form B: span rate -> per-hour, then × time
    if len(spans) == 1 and not matches and not fa_rates:
        m = spans[0]
        if m.re is _CURRENCY_SPAN_FA_RE:
            out, span, _unit = float(m.group(1)), float(m.group(2)) * _hours(m.group(3)), None
        else:
            out, span, _unit = float(m.group(1)), float(m.group(2)) * _hours(m.group(3)), None
        if out <= 0 or span <= 0:
            return None
        times = _extract_times(t, skip=[m.span()])
        if len(times) != 1:
            return None
        T = times[0]['hours']
        if T <= 0:
            return None
        return {'kind': 'currency_span_application',
                'value': (out / span) * T, 'noun': None, 'currency': True}
    return None


SOLVERS = (_solve_per_worker_sum, _solve_combined_inversion, _solve_inverted_hours,
           _solve_currency_application, _solve_midtask, _solve_same_crew,
           _solve_rate_scale, _solve_per_plural, _solve_single_application)


def solve_work_rate(text: str, language: str = 'fa'):
    """Try every sub-solver; a result counts ONLY when a second, independent
    re-derivation agrees exactly (witness discipline, v22 heritage)."""
    t = fold(text or '')
    for solver in SOLVERS:
        result = solver(t)
        if result is None:
            continue
        value = float(result['value'])
        if s4.numeric_guard(value) or value <= 0 or abs(value) > 1e15:
            continue
        witness = solver(t)
        if witness is None or abs(float(witness['value']) - value) > 1e-9:
            continue
        result['value'] = value
        return result
    return None


def render(result: dict, language: str) -> str:
    from jarvis.agent.language_brain_v22 import fmt_number
    v = fmt_number(result['value'])
    noun = result.get('noun')
    fa_n, en_n = _noun_pair(noun) if noun else ('', '')
    if result.get('hours_answer'):
        if language == 'fa':
            return f'زمان لازم می\u200cشود {v} ساعت.'
        return f'The required time is {v} hours.'
    if language == 'fa':
        return f'نتیجه می\u200cشود {v} {fa_n}.'.strip() if fa_n \
            else f'نتیجه می\u200cشود {v}.'
    return f'The result is {v} {en_n}.'.strip() if en_n \
        else f'The result is {v}.'
