"""JARVIS v23 — Unit Conversion Service (کارتابل واحد).

V23_LANGUAGE_BRAIN_PLAN.md §2/§4 (مرحله 1): the units family scored 0/10 on
the v22.4.2 diagnostic; standalone conversions ("How many grams are 2.5
kilograms?") never reached a verified solver.

Design (inherits the v22 discipline):
  * typed extraction via quantity_v22.extract_typed_quantities (exact spans,
    never silently substituted units);
  * fail-closed: the service answers ONLY when
      - the text is an explicit conversion question,
      - exactly one measured quantity exists,
      - source AND target units are known and share one base dimension;
    unknown units, temperature and mixed families yield None (abstention).
  * every result is verified by an independent witness (the inverse
    conversion must restore the source value) plus s4.numeric_guard.
  * word-numbers (نیم / ربع / سه / دوازده / ...) are folded BEFORE parsing;
    spans are not needed because the service re-derives everything from the
    folded text and never touches SourceSpan offsets.
"""
from __future__ import annotations

import re

from jarvis.agent import semantics_v22_4 as s4

DIGITS = str.maketrans('۰۱۲۳۴۵۶۷۸۹٠١٢٣٤٥٦٧٨٩', '01234567890123456789')

# ---------------------------------------------------------------------------
# unit tables: canonical -> base factor (base unit per family)
# ---------------------------------------------------------------------------
TABLES: dict[str, dict[str, float]] = {
    'mass':     {'MG': 0.001, 'G': 1.0, 'KG': 1000.0, 'TON': 1_000_000.0},
    'distance': {'MM': 0.001, 'CM': 0.01, 'M': 1.0, 'KM': 1000.0},
    'time':     {'SECOND': 1.0, 'MINUTE': 60.0, 'HOUR': 3600.0, 'DAY': 86400.0},
    'volume':   {'ML': 1.0, 'L': 1000.0},
}

# speed is a compound family with exact factors (base: m/s)
SPEED_TABLE: dict[str, float] = {
    'M_PER_SECOND': 1.0,
    'KM_PER_HOUR': 1.0 / 3.6,
    'MILE_PER_HOUR': 0.44704,     # exact definition: 1 mph = 0.44704 m/s
}

# word -> canonical unit, per family
UNIT_WORDS: dict[str, dict[str, str]] = {
    'mass': {
        'میلی‌گرم': 'MG', 'میلی گرم': 'MG', 'گرم': 'G', 'کیلوگرم': 'KG',
        'کیلو گرم': 'KG', 'کیلو': 'KG', 'تن': 'TON', 'هکتوگرم': 'HG',
        'milligram': 'MG', 'milligrams': 'MG', 'mg': 'MG',
        'gram': 'G', 'grams': 'G', 'g': 'G',
        'kilogram': 'KG', 'kilograms': 'KG', 'kgs': 'KG', 'kg': 'KG',
        'ton': 'TON', 'tons': 'TON', 'tonne': 'TON', 'tonnes': 'TON',
    },
    'distance': {
        'میلی‌متر': 'MM', 'میلی متر': 'MM', 'سانتی‌متر': 'CM', 'سانتی متر': 'CM',
        'متر': 'M', 'کیلومتر': 'KM', 'کیلو متر': 'KM',
        'millimeter': 'MM', 'millimeters': 'MM', 'mm': 'MM',
        'centimeter': 'CM', 'centimeters': 'CM', 'cm': 'CM',
        'meter': 'M', 'meters': 'M', 'metre': 'M', 'metres': 'M', 'm': 'M',
        'kilometer': 'KM', 'kilometers': 'KM', 'kilometre': 'KM',
        'kilometres': 'KM', 'km': 'KM',
    },
    'time': {
        'ثانیه': 'SECOND', 'دقیقه': 'MINUTE', 'ساعت': 'HOUR', 'روز': 'DAY',
        'هفته': 'WEEK',
        'second': 'SECOND', 'seconds': 'SECOND', 'sec': 'SECOND', 's': 'SECOND',
        'minute': 'MINUTE', 'minutes': 'MINUTE', 'min': 'MINUTE',
        'hour': 'HOUR', 'hours': 'HOUR', 'hr': 'HOUR', 'h': 'HOUR',
        'day': 'DAY', 'days': 'DAY',
    },
    'volume': {
        'میلی‌لیتر': 'ML', 'میلی لیتر': 'ML', 'لیتر': 'L', 'لیتـر': 'L',
        'milliliter': 'ML', 'milliliters': 'ML', 'millilitre': 'ML',
        'millilitres': 'ML', 'ml': 'ML',
        'liter': 'L', 'liters': 'L', 'litre': 'L', 'litres': 'L', 'l': 'L',
    },
    'speed': {
        'متر بر ثانیه': 'M_PER_SECOND', 'متر/ثانیه': 'M_PER_SECOND',
        'کیلومتر بر ساعت': 'KM_PER_HOUR', 'کیلومتر/ساعت': 'KM_PER_HOUR',
        'm/s': 'M_PER_SECOND', 'mps': 'M_PER_SECOND',
        'meters per second': 'M_PER_SECOND', 'metres per second': 'M_PER_SECOND',
        'km/h': 'KM_PER_HOUR', 'kph': 'KM_PER_HOUR', 'kmh': 'KM_PER_HOUR',
        'kilometers per hour': 'KM_PER_HOUR', 'kilometres per hour': 'KM_PER_HOUR',
        'mph': 'MILE_PER_HOUR', 'miles per hour': 'MILE_PER_HOUR',
    },
}

# multi-word units must be matched before single-word ones
_UNIT_WORD_SORTED: dict[str, list[tuple[str, str]]] = {
    fam: sorted(words.items(), key=lambda kv: -len(kv[0]))
    for fam, words in UNIT_WORDS.items()
}
_GLOBAL_UNIT_WORDS: list[tuple[str, tuple[str, str]]] = sorted(
    ((word, (fam, canon)) for fam, words in UNIT_WORDS.items()
     for word, canon in words.items()),
    key=lambda kv: -len(kv[0]))
_UNIT_LOOKUP: dict[str, tuple[str, str]] = {
    word: (fam, canon)
    for fam, words in UNIT_WORDS.items()
    for word, canon in words.items()
}
# units known to the typed extractor that this service deliberately refuses
# (non-linear / unsupported families) — they must ABSTAIN, never convert.
_REFUSED_HINTS = re.compile(
    r'سلسیوس|سانتیگراد|فارنهایت|celsius|fahrenheit|درجه|degrees?|'
    r'دسیبل|decibels?|مگاپیکسل|megapixels?|کیلوبایت|kilobytes?|مگابایت|megabytes?|'
    r'فرسخ|furlongs?|leagues?|parsecs?|nautical|acres?|hectares?|stones?|barrels?|'
    r'دلار|dollars?|یورو|euros?|تومان|tomans?|ریال|rials?|پوند|pounds?|'
    r'dollars?/hour|usd|items?|pieces?|کالا|قطعه',
    re.I)

_FA_WORD_NUMBERS = {
    'صفر': 0.0, 'یک': 1.0, 'دو': 2.0, 'سه': 3.0, 'چهار': 4.0, 'پنج': 5.0,
    'شش': 6.0, 'هفت': 7.0, 'هشت': 8.0, 'نه': 9.0, 'ده': 10.0,
    'یازده': 11.0, 'دوازده': 12.0, 'سیزده': 13.0, 'چهارده': 14.0,
    'پانزده': 15.0, 'پانزده': 15.0, 'شانزده': 16.0, 'هفده': 17.0,
    'هجده': 18.0, 'هجده': 18.0, 'نوزده': 19.0, 'بیست': 20.0,
    'سی': 30.0, 'چهل': 40.0, 'پنجاه': 50.0, 'شصت': 60.0,
    'هفتاد': 70.0, 'هشتاد': 80.0, 'نود': 90.0, 'صد': 100.0, 'یکصد': 100.0,
    'دویست': 200.0, 'سیصد': 300.0, 'چهارصد': 400.0, 'پانصد': 500.0,
    'ششصد': 600.0, 'هفتصد': 700.0, 'هشتصد': 800.0, 'نهصد': 900.0,
    'هزار': 1000.0,
}

_QUESTION_CUE = re.compile(
    r'چند|چقدر|برابر\s+است|how\s+many|how\s+much|what\s+is|equals\s+how|'
    r'is\s+that|in\s+\w+\s*\?|می\u200cشود\s*\?', re.I)


def fold_word_numbers(text: str) -> str:
    """Fold Persian word-numbers and fractions into ASCII digits so the typed
    extractor can see them ('نیم ساعت' -> '0.5 ساعت'). The article 'یک' is
    folded ONLY when followed by a known unit word ('یک ساعت' -> '1 ساعت'),
    never when it names a noun ('یک مسافت' stays untouched)."""
    t = (text or '').translate(DIGITS)

    def unit_follows(m: 're.Match') -> bool:
        rest = t[m.end():]
        lm = re.match(r'\s*', rest)
        pos = m.end() + lm.end()
        return _match_unit_word(t, pos) is not None

    def repl(m: 're.Match') -> str:
        w = m.group(1)
        if w == 'نیم':
            return ' 0.5 '
        if w == 'ربع':
            return ' 0.25 '
        if w == 'یک' and not unit_follows(m):
            return m.group(0)   # article 'یک مسافت' — not a number
        v = _FA_WORD_NUMBERS.get(w)
        return f' {int(v) if v == int(v) else v} ' if v is not None else m.group(0)

    return re.sub(r'\b(نیم|ربع|صفر|یک|دو|سه|چهار|پنج|شش|هفت|هشت|نه|ده|یازده|'
                  r'دوازده|سیزده|چهارده|پانزده|شانزده|هفده|هجده|نوزده|بیست|'
                  r'سی|چهل|پنجاه|شصت|هفتاد|هشتاد|نود|صد|یکصد|دویست|سیصد|'
                  r'چهارصد|پانصد|ششصد|هفتصد|هشتصد|نهصد|هزار)\b', repl, t)


def _match_unit_word(text: str, pos: int) -> tuple[str, str, str] | None:
    """Return (family, canonical, raw) for the unit word starting at pos.
    Global longest-first ordering: compound units ('m/s', 'متر بر ثانیه')
    always win over their prefixes ('m', 'متر')."""
    for word, (fam, canon) in _GLOBAL_UNIT_WORDS:
        if text[pos:pos + len(word)].lower() == word.lower():
            # word boundary for ASCII words
            nxt = text[pos + len(word):pos + len(word) + 1]
            if word[:1].isascii() and nxt and (nxt.isalnum()):
                continue
            return fam, canon, text[pos:pos + len(word)]
    return None


def _find_target_unit(text: str) -> tuple[str, str, str] | None:
    """Locate the TARGET unit: right after 'چند/how many/...' or after
    'in <unit>?' / 'برابر است با <unit>'. Returns (family, canonical, raw)."""
    anchors = [r'چند', r'چقدر', r'how\s+many', r'how\s+much',
               r'برابر\s+است\s+با', r'in\s+', r'equals']
    for anchor in anchors:
        for m in re.finditer(anchor, text, re.I):
            rest = text[m.end():]
            lm = re.match(r'\s*(?:many|much)?\s*', rest, re.I)
            pos = m.end() + lm.end()
            hit = _match_unit_word(text, pos)
            if hit is not None:
                return hit
    return None


def _local_source_scan(t: str) -> tuple[float, str, str] | None:
    """Fail-closed fallback for source values the typed extractor leaves
    dimensionless ('2 tons', '۵ تن'): exactly ONE number in the text, glued
    to a known unit word. Returns (value, family, canonical) or None."""
    numbers = list(re.finditer(r'(?<![\w.])(\d+(?:\.\d+)?)(?!\d)', t))
    if len(numbers) != 1:
        return None
    m = numbers[0]
    rest = t[m.end():]
    lm = re.match(r'\s*', rest)
    pos = m.end() + lm.end()
    hit = _match_unit_word(t, pos)
    if hit is None:
        return None
    return float(m.group(1)), hit[0], hit[1]


def _count_numbers(t: str) -> int:
    return len(re.findall(r'(?<![\w.])(\d+(?:\.\d+)?)(?!\d)', t))


def solve_conversion(text: str, language: str = 'fa') -> dict | None:
    """Solve an explicit standalone unit conversion. Returns
    {'value', 'family', 'target', 'fa_unit', 'en_unit'} or None (abstain)."""
    raw = text or ''
    if not _QUESTION_CUE.search(raw):
        return None
    if _REFUSED_HINTS.search(raw):
        # mixed/unsupported families never convert (fail-closed)
        return None
    t = fold_word_numbers(raw)
    # v23 guard: a conversion question contains exactly ONE numeric value.
    # Texts with several numbers are OTHER tasks (work-rate scenes, chains)
    # and an identity '6 hours -> hours' theft of a work-rate question is a
    # wrong answer, never a conversion.
    if _count_numbers(t) != 1:
        return None

    from jarvis.agent.quantity_v22 import extract_typed_quantities
    quantities = extract_typed_quantities(t)

    # exactly one measured quantity of a convertible family
    convertible = [q for q in quantities
                   if q.dimension in ('mass', 'distance', 'time', 'volume')
                   and (q.normalized_unit or q.unit) in
                   ('MG', 'G', 'KG', 'TON', 'HG', 'MM', 'CM', 'M', 'KM',
                    'SECOND', 'MINUTE', 'HOUR', 'DAY', 'WEEK', 'ML', 'L')]
    speeds = [q for q in quantities if q.dimension == 'speed'
              and q.unit != 'UNKNOWN_UNIT']

    src_value = src_family = src_canon = None
    if len(convertible) == 1 and not speeds:
        src = convertible[0]
        src_value = float(src.value)
        src_canon = src.normalized_unit or src.unit
        src_family = src.dimension
    elif len(convertible) == 0 and len(speeds) == 1:
        src_family = 'speed'   # handled by the dedicated speed path below
    elif len(convertible) == 0 and len(speeds) == 0:
        local = _local_source_scan(t)
        if local is not None:
            src_value, src_family, src_canon = local
    if src_family is None:
        return None

    if src_family != 'speed':
        hit = _find_target_unit(t)
        if hit is None:
            return None
        t_fam, tgt_canon, tgt_raw = hit
        if t_fam != src_family or src_canon not in TABLES.get(src_family, {}):
            return None
        if tgt_canon not in TABLES.get(t_fam, {}):
            return None
        if tgt_canon == src_canon:
            return None   # identity 'unit -> unit' is not a conversion
        table = TABLES[src_family]
        base = src_value * table[src_canon]
        value = base / table[tgt_canon]
        # witness: the inverse conversion must restore the source value
        restored = value * table[tgt_canon] / table[src_canon]
        if abs(restored - src_value) > 1e-9:
            return None
        if s4.numeric_guard(value) or s4.numeric_guard(base):
            return None
        if abs(value) < 1e-12:
            return None
        return {'kind': 'unit_conversion', 'value': float(value),
                'family': src_family, 'source': src_canon, 'target': tgt_canon,
                'fa_unit': _FA_CANON.get(tgt_canon, tgt_raw),
                'en_unit': _EN_CANON.get(tgt_canon, tgt_raw)}

    # speed family: typed extractor is the only source (compound units are
    # classified per family there); a locally scanned speed unit is NOT trusted
    if len(speeds) != 1:
        return None
    src = speeds[0]
    src_canon = 'M_PER_SECOND' if src.unit in ('M_PER_S', 'M_PER_SECOND') \
        else src.unit
    if src_canon not in SPEED_TABLE:
        return None
    hit = _find_target_unit(t)
    if hit is None or hit[0] != 'speed' or hit[1] not in SPEED_TABLE:
        return None
    if hit[1] == src_canon:
        return None   # identity 'unit -> unit' is not a conversion
    value = float(src.value) * SPEED_TABLE[src_canon] / SPEED_TABLE[hit[1]]
    restored = value * SPEED_TABLE[hit[1]] / SPEED_TABLE[src_canon]
    if abs(restored - float(src.value)) > 1e-9:
        return None
    if s4.numeric_guard(value) or abs(value) < 1e-12:
        return None
    return {'kind': 'unit_conversion_speed', 'value': float(value),
            'family': 'speed', 'source': src_canon, 'target': hit[1],
            'fa_unit': _FA_CANON.get(hit[1], hit[1]),
            'en_unit': _EN_CANON.get(hit[1], hit[1])}


_FA_CANON = {
    'MG': 'میلی‌گرم', 'G': 'گرم', 'HG': 'هکتوگرم', 'KG': 'کیلوگرم', 'TON': 'تن',
    'MM': 'میلی‌متر', 'CM': 'سانتی‌متر', 'M': 'متر', 'KM': 'کیلومتر',
    'SECOND': 'ثانیه', 'MINUTE': 'دقیقه', 'HOUR': 'ساعت', 'DAY': 'روز',
    'WEEK': 'هفته', 'ML': 'میلی‌لیتر', 'L': 'لیتر',
    'M_PER_SECOND': 'متر بر ثانیه', 'KM_PER_HOUR': 'کیلومتر بر ساعت',
    'MILE_PER_HOUR': 'مایل بر ساعت',
}
_EN_CANON = {
    'MG': 'milligrams', 'G': 'grams', 'HG': 'hectograms', 'KG': 'kilograms',
    'TON': 'tons',
    'MM': 'millimeters', 'CM': 'centimeters', 'M': 'meters', 'KM': 'kilometers',
    'SECOND': 'seconds', 'MINUTE': 'minutes', 'HOUR': 'hours', 'DAY': 'days',
    'WEEK': 'weeks', 'ML': 'milliliters', 'L': 'liters',
    'M_PER_SECOND': 'm/s', 'KM_PER_HOUR': 'km/h', 'MILE_PER_HOUR': 'mph',
}


def render(result: dict, language: str) -> str:
    from jarvis.agent.language_brain_v22 import fmt_number
    v = fmt_number(result['value'])
    unit = result['fa_unit'] if language == 'fa' else result['en_unit']
    if language == 'fa':
        return f'نتیجه می\u200cشود {v} {unit}.'.strip()
    return f'The result is {v} {unit}.'.strip()
