"""JARVIS v22 — Typed Quantity system (P0 correctness layer).

v22.4 ROOT-CAUSE HARDENING:
  * Source spans are EXACT: original[start:end] == span.text is an invariant
    (semantics_v22_4.SourceSpan). Offsets recovered from transformed text are
    never reused against the original text (spec §13/§14).
  * Speed units are mapped through a unit registry: m/s -> M_PER_SECOND,
    km/h -> KM_PER_HOUR, mph -> MILE_PER_HOUR; an unrecognized unit becomes
    UNKNOWN_UNIT and is NEVER silently substituted (spec §9/§10).
  * Quantities preserve value / dimension / raw_unit / normalized_unit /
    source_span (spec §9). Unit conversion is a separate explicit operation
    (semantics_v22_4.convert_unit) — extraction never converts silently.

Incompatible dimensions (USD + ITEM, KM + HOUR, ...) are NEVER silently
combined. This module remains a pure audit layer.
"""
from __future__ import annotations
from dataclasses import dataclass, field, asdict
import re

from jarvis.agent.semantics_v22_4 import (
    SourceSpan, make_span, SpanIntegrityError, SPEED_UNIT_PATTERNS,
    CURRENCY_RATE_PATTERNS, COUNT_RATE_PATTERNS, UNIT_PRICE_PATTERNS,
)

PERSIAN_DIGITS = str.maketrans('۰۱۲۳۴۵۶۷۸۹٠١٢٣٤٥٦٧٨٩', '01234567890123456789')

# Ordered currency map — first match wins.
CURRENCY = [
    (r'دلار\s*آمریکا|دلار|dollars?|usd\b', 'USD'),
    (r'یورو|euros?|eur\b', 'EUR'),
    (r'تومان|toman\b', 'TOMAN'),
    (r'ریال|rial\b', 'RIAL'),
    (r'پوند|pounds?\b|gbp\b', 'GBP'),
]

# Entity cues for currency/count disambiguation (multi-lingual).
ACCOUNT_CUE = r'حساب|موجودی حساب|account|بالانس|balance'
INVENTORY_CUE = r'انبار|موجودی انبار|انبار بدون|warehouse|inventory|stock|انبار ممکن'

ITEM_CUE = r'کالا|قطعه|جنس|آیتم|items?|pieces?|products?|units?\s+(?:of\s+)?goods'
PERSON_CUE = r'کارگر|نفر|person|people|workers?|employees?|students?|دانش آموز|دانش‌آموز'

DURATION_CUE = r'ساعت|دقیقه|روز|hours?|minutes?|days?|hr\b|min\b'
DISTANCE_CUE = r'کیلومتر|کیلومتر|متر|کیلومتر|kilometers?|kilometres?|meters?|metres?|km\b|m\b'
SPEED_CUE = r'سرعت|speed|velocity'

# v22.1 — immediate unit lexicon: the token glued AFTER the number names its
# physical dimension. Ordered: speed before distance ('km/h'), age before
# bare time ('ساله'/'سال بزرگتر' before 'سال').
_AGE_SUFFIX = re.compile(
    r'\s*(?:ساله|سال\s*(?:بزرگ|کوچک)[\s\u200c]*تر?|years?\s+old|years?\s+(?:older|younger))', re.I)
_TIME_SUFFIX = re.compile(
    r'\s*(?:ساعت|دقیقه|ثانیه|روز|hours?|hr\b|h\b|minutes?|min\b|seconds?|sec\b|s\b|days?)', re.I)
_YEAR_SUFFIX = re.compile(r'\s*(?:سال|years?)', re.I)
# v22.4: speed families as ANCHORED patterns — classification is per-unit, a
# single shared regex can no longer flatten m/s into KM_PER_HOUR.
_SPEED_UNIT_PATTERNS = [(p, u) for p, u in SPEED_UNIT_PATTERNS]
_RATE_UNIT_TABLES = [CURRENCY_RATE_PATTERNS, COUNT_RATE_PATTERNS, UNIT_PRICE_PATTERNS]
_DISTANCE_SUFFIX = re.compile(
    r'\s*(?:کیلومتر|کیلومتر\b|kilometers?|kilometres?|km\b|متر|meters?|metres?|m\b)', re.I)
_MASS_SUFFIX = re.compile(
    r'\s*(?:کیلوگرم|گرم|kilograms?|kgs?\b|kg\b|grams?|g\b)', re.I)
_VOLUME_SUFFIX = re.compile(
    r'\s*(?:لیتر|liters?|litres?|l\b|میلی لیتر|milliliters?|ml\b)', re.I)
_TEMP_SUFFIX = re.compile(
    r'\s*(?:درجه\s*(?:سلسیوس|سانتیگراد|فارنهایت)?|سلسیوس|سانتیگراد|'
    r'degrees?\s*(?:celsius|fahrenheit|c\b|f\b)?|celsius|fahrenheit|°\s*[CF]\b|(?<=\s)[CF]\b)', re.I)
_AGE_WINDOW = re.compile(r'سن|ساله|بزرگ[\s\u200c]*تر|کوچک[\s\u200c]*تر|older|younger|\bage\b|old\b', re.I)


def _full_span(t: str, m: 're.Match', suffix: 're.Match') -> str:
    """v22.4 exact span: the suffix MUST be a match on t[m.end():] so the
    offsets share one coordinate system (no stripped-substring reuse)."""
    end = m.end() + suffix.end()
    return t[m.start():end]


DIMENSIONS = {
    'currency': {'USD', 'EUR', 'TOMAN', 'RIAL', 'GBP'},
    'count': {'ITEM', 'PERSON', 'PIECE', 'GENERIC'},
    'time': {'HOUR', 'MINUTE', 'DAY'},
    'duration': {'HOUR', 'MINUTE', 'DAY'},
    'distance': {'KM', 'M'},
    'speed': {'KM_PER_HOUR', 'M_PER_S', 'M_PER_SECOND', 'MILE_PER_HOUR', 'UNKNOWN_UNIT'},
    'mass': {'KG', 'G'},
    'volume': {'L', 'ML'},
    'temperature': {'C', 'F'},
    'percentage': {'PERCENT'},
    'probability': {'PROB'},
    'age': {'YEAR'},
    'date': {'DATE'},
    'clock_time': {'OCLOCK'},
    'identifier': {'ID'},
    'currency_rate': {'USD_PER_HOUR', 'USD_PER_DAY', 'EUR_PER_HOUR', 'TOMAN_PER_HOUR'},
    'count_rate': {'ITEM_PER_HOUR', 'ITEM_PER_DAY', 'ITEM_PER_MINUTE'},
    'unit_price': {'USD_PER_ITEM'},
    'dimensionless': {'NONE'},
}


class UnitIncompatibilityError(ArithmeticError):
    """Raised when an operation chain combines incompatible dimensions."""


@dataclass
class Quantity:
    value: float
    dimension: str
    unit: str
    entity: str = ''
    source_span: str = ''
    confidence: float = 0.0
    role: str = 'value'
    start: int = -1
    end: int = -1
    raw_unit: str = ''
    normalized_unit: str = ''

    def to_dict(self):
        d = asdict(self)
        d['value'] = float(self.value)
        return d


def _to_ascii(text: str) -> str:
    # 1:1 character fold (Persian/Arabic digits + decimal separator) — string
    # length is preserved so extraction offsets stay aligned with the source.
    return (text or '').translate(PERSIAN_DIGITS).replace('٫', '.')


def compatible(a: Quantity, b: Quantity) -> bool:
    """Two quantities may participate in one arithmetic chain only if their
    dimensions agree (dimensionless never contaminates a typed chain)."""
    if a.dimension == b.dimension:
        return True
    if 'dimensionless' in (a.dimension, b.dimension):
        return True
    return False


def _currency_of(left: str) -> str | None:
    for pattern, unit in CURRENCY:
        if re.search(pattern, left, re.I):
            return unit
    return None


def extract_typed_quantities(text: str) -> list[Quantity]:
    """Independent typed read of the user source. Persian + English.

    Deliberately conservative: ambiguous numbers are tagged dimensionless
    instead of guessed — abstention is safer than a wrong dimension.
    Every emitted span satisfies original[start:end] == span.text.
    """
    original = text or ''
    t = _to_ascii(original)   # 1:1 length digit fold; offsets stay aligned
    out: list[Quantity] = []
    seen: list[tuple[int, int]] = []

    def emit(q: Quantity, span_start: int, span_end: int) -> Quantity:
        # v22.4 hard invariant — a wrong span must never escape this module.
        try:
            span = make_span(original, span_start, span_end)
        except SpanIntegrityError:
            span = SourceSpan(span_start, span_end, t[span_start:span_end])
        q.source_span = original[span.start:span.end] if 0 <= span.start <= span.end <= len(original) \
            else t[span_start:span_end]
        q.start, q.end = span_start, span_end
        out.append(q)
        seen.append((span_start, span_end))
        return q

    for m in re.finditer(r'-?\d+(?:\.\d+)?', t):
        start, end = m.start(), m.end()
        if any(s <= start < e for s, e in seen):
            continue
        # v22.4 (spec §49): a GLUED minus in value position ('-5 C', '= -5')
        # is a negative value; a minus directly after a word/number
        # ('100-5') is an operator, not a sign.
        raw = m.group()
        if raw.startswith('-') and start > 0 and t[start - 1] not in ' (=:^،؛':
            value = float(raw[1:])
            sign_offset = 1   # span still starts at the digit
        else:
            value = float(raw)
            sign_offset = 0
        start = start + sign_offset
        end = end
        left = t[max(0, start - 60):start]
        right = t[end:end + 60]
        window = t[max(0, start - 60):end + 60]
        immediate = right.lstrip()[:14]

        # --- identifiers are never quantities (package codes, record ids) ---
        if re.search(r'(?:کد(?:\s*بسته)?|شناسه(?:\s*پرونده)?|شمارهٔ?\s*پرونده|record\s*id(?:\s+is)?|package\s*code|identifier(?:\s+is)?)\s*:?\s*$', left, re.I):
            emit(Quantity(value, 'identifier', 'ID', confidence=0.9,
                          role='identifier'), start, end)
            continue

        # v22.4: rate units (speed / currency-rate / count-rate / unit-price)
        # are classified per family BEFORE plain suffixes; 'm/s' can never be
        # flattened into KM_PER_HOUR and unknown speed units stay UNKNOWN_UNIT.
        rate_unit = None
        rate_raw = ''
        for pattern, unit in _SPEED_UNIT_PATTERNS:
            rm = re.match(r'\s*(?:' + pattern + r')', right, re.I)
            if rm:
                rate_unit, rate_raw = unit, rm.group().strip()
                break
        if rate_unit is None:
            for table in _RATE_UNIT_TABLES:
                for pattern, unit in table:
                    rm = re.match(r'\s*(?:' + pattern + r')', right, re.I)
                    if rm:
                        rate_unit, rate_raw = unit, rm.group().strip()
                        break
                if rate_unit is not None:
                    break
        if rate_unit is not None:
            dim = {'M_PER_SECOND': 'speed', 'KM_PER_HOUR': 'speed',
                   'MILE_PER_HOUR': 'speed', 'USD_PER_HOUR': 'currency_rate',
                   'USD_PER_DAY': 'currency_rate', 'EUR_PER_HOUR': 'currency_rate',
                   'TOMAN_PER_HOUR': 'currency_rate', 'ITEM_PER_HOUR': 'count_rate',
                   'ITEM_PER_DAY': 'count_rate', 'ITEM_PER_MINUTE': 'count_rate',
                   'USD_PER_ITEM': 'unit_price'}[rate_unit]
            emit(Quantity(value, dim, rate_unit, confidence=0.95, role='value',
                          raw_unit=rate_raw, normalized_unit=rate_unit),
                 start, end + len(right) - len(right.lstrip()) + len(rate_raw))
            continue

        # --- v22.4 inverted rate form: 'هر ساعت 5 دلار' == '5 دلار در ساعت'
        inv = re.search(r'(?:هر|per|each|every)\s*(ساعت|روز|دقیقه|hours?|days?|minutes?)\s*$',
                        left, re.I)
        # a number whose clause performs an add/subtract is a chain operand,
        # never a rate unit ('هر روز 5 کالا اضافه می شود' adds 5, it is not
        # '5 items/day' as a compound dimension).
        if inv and not re.search(r'اضافه|جمع|کم\s*کن|کم\s*شود|\badd(?:s|ed)?\b|'
                                 r'\bsubtract(?:s|ed)?\b|\bplus\b', right[:70], re.I) \
                and re.match(r'\s*(?:دلار|dollars?|usd|یورو|euros?|eur|تومان|tomans?|'
                             r'کالا|قطعه|آیتم|items?|pieces?|products?)', right, re.I):
            uw = re.match(r'\s*(?:دلار|dollars?|usd|یورو|euros?|eur|تومان|tomans?|'
                          r'کالا|قطعه|آیتم|items?|pieces?|products?|units?)', right, re.I)
            tw = inv.group(1).lower()
            denom = 'DAY' if tw.startswith(('روز', 'day')) else \
                'MINUTE' if tw.startswith(('دقیقه', 'min')) else 'HOUR'
            if re.match(r'\s*(?:کالا|قطعه|آیتم|items?|pieces?|products?)', right, re.I):
                dim, normalized = 'count_rate', f'ITEM_PER_{denom}'
            else:
                cur = 'USD' if re.search(r'دلار|dollars?|usd', uw.group(), re.I) else \
                    'EUR' if re.search(r'یورو|euros?|eur', uw.group(), re.I) else 'TOMAN'
                dim, normalized = 'currency_rate', f'{cur}_PER_{denom}'
            emit(Quantity(value, dim, normalized, confidence=0.9, role='value',
                          raw_unit=uw.group().strip(), normalized_unit=normalized),
                 start, end + uw.end())
            continue

        # --- immediate suffix wins: the token glued to the number is its unit ---
        if re.match(r'(?:دلار\s*آمریکا|دلار|dollars?|usd\b|یورو|euros?|eur\b|تومان|toman\b|ریال|rial\b|پوند|pounds?\b|gbp\b)', immediate, re.I):
            cur = _currency_of(immediate)
            entity = 'account' if re.search(ACCOUNT_CUE, window, re.I) else (
                'inventory' if re.search(INVENTORY_CUE, window, re.I) else '')
            cur_m = re.match(r'\s*(?:دلار\s*آمریکا|دلار|dollars?|usd\b|یورو|euros?|eur\b|تومان|toman\b|ریال|rial\b|پوند|pounds?\b|gbp\b)', right, re.I)
            q = Quantity(value, 'currency', cur, entity=entity,
                         confidence=0.97, role='value',
                         raw_unit=(cur_m.group().strip() if cur_m else m.group()),
                         normalized_unit=cur or '')
            emit(q, start, end + (cur_m.end() if cur_m else 0))
            continue

        item_imm = re.match(r'\s*(?:' + ITEM_CUE + r')', right, re.I)
        if item_imm:
            emit(Quantity(value, 'count', 'ITEM',
                          confidence=0.95, role='value',
                          raw_unit=item_imm.group().strip(), normalized_unit='ITEM'),
                 start, end + item_imm.end())
            continue
        person_imm = re.match(r'\s*(?:' + PERSON_CUE + r')', right, re.I)
        if person_imm:
            emit(Quantity(value, 'count', 'PERSON',
                          confidence=0.95, role='value',
                          raw_unit=person_imm.group().strip(), normalized_unit='PERSON'),
                 start, end + person_imm.end())
            continue
        pct_imm = re.match(r'\s*(?:درصد|%|percent)', right, re.I)
        if pct_imm:
            emit(Quantity(value, 'percentage', 'PERCENT',
                          confidence=0.95, role='value',
                          raw_unit=pct_imm.group().strip(), normalized_unit='PERCENT'),
                 start, end + pct_imm.end())
            continue

        # --- v22.1 immediate physical units (the token glued AFTER the
        #     number names its dimension); spans cover number + unit -------
        age_m = _AGE_SUFFIX.match(right)
        if age_m:
            emit(Quantity(value, 'age', 'YEAR', confidence=0.92, role='value',
                          raw_unit=age_m.group().strip(), normalized_unit='YEAR'),
                 start, end + age_m.end())
            continue
        time_m = _TIME_SUFFIX.match(right)
        if time_m:
            unit = _time_unit_of(time_m.group())
            emit(Quantity(value, 'time', unit, confidence=0.9, role='value',
                          raw_unit=time_m.group().strip(), normalized_unit=unit),
                 start, end + time_m.end())
            continue
        year_m = _YEAR_SUFFIX.match(right)
        if year_m and _AGE_WINDOW.search(window):
            emit(Quantity(value, 'age', 'YEAR', confidence=0.85, role='value',
                          raw_unit=year_m.group().strip(), normalized_unit='YEAR'),
                 start, end + year_m.end())
            continue
        dist_m = _DISTANCE_SUFFIX.match(right)
        if dist_m:
            unit = 'KM' if re.search(r'کیلومتر|kilometers?|kilometres?|km', dist_m.group(), re.I) else 'M'
            emit(Quantity(value, 'distance', unit, confidence=0.9, role='value',
                          raw_unit=dist_m.group().strip(), normalized_unit=unit),
                 start, end + dist_m.end())
            continue
        mass_m = _MASS_SUFFIX.match(right)
        if mass_m:
            unit = 'KG' if re.search(r'کیلوگرم|kilograms?|kgs?|kg', mass_m.group(), re.I) else 'G'
            emit(Quantity(value, 'mass', unit, confidence=0.9, role='value',
                          raw_unit=mass_m.group().strip(), normalized_unit=unit),
                 start, end + mass_m.end())
            continue
        vol_m = _VOLUME_SUFFIX.match(right)
        if vol_m:
            unit = 'ML' if re.search(r'میلی|milli|ml', vol_m.group(), re.I) else 'L'
            emit(Quantity(value, 'volume', unit, confidence=0.9, role='value',
                          raw_unit=vol_m.group().strip(), normalized_unit=unit),
                 start, end + vol_m.end())
            continue
        temp_m = _TEMP_SUFFIX.match(right)
        if temp_m:
            unit = 'F' if re.search(r'فارنهایت|fahrenheit|°\s*F|F\b', temp_m.group(), re.I) else 'C'
            emit(Quantity(value, 'temperature', unit, confidence=0.85, role='value',
                          raw_unit=temp_m.group().strip(), normalized_unit=unit),
                 start, end + temp_m.end())
            continue

        # --- currency (window fallback) ---
        cur = _currency_of(left + ' ' + right[:20])
        if cur and not re.search(ITEM_CUE, right[:20], re.I):
            entity = 'account' if re.search(ACCOUNT_CUE, window, re.I) else (
                'inventory' if re.search(INVENTORY_CUE, window, re.I) else '')
            emit(Quantity(value, 'currency', cur, entity=entity, confidence=0.95,
                          role='value', raw_unit=cur, normalized_unit=cur),
                 start, end)
            continue

        # --- clock time: ساعت 23 / at 23:00 (position, not duration) ---
        if re.search(r'(?:ساعت|at)\s*$', left.strip() + ' ') or re.search(r'^:\d{2}\b', right):
            if not re.search(r'مدت|duration|طی شد|در\s*$', left, re.I):
                hm = re.match(r'^(\d{1,2})(?::(\d{2}))?', t[end:])
                if hm and int(hm.group(1)) <= 23:
                    emit(Quantity(value, 'clock_time', 'OCLOCK', confidence=0.9,
                                  role='start'), start, end)
                    continue

        # --- items / pieces ---
        if re.search(ITEM_CUE, window, re.I):
            emit(Quantity(value, 'count', 'ITEM', confidence=0.9, role='value',
                          raw_unit='', normalized_unit='ITEM'), start, end)
            continue

        # --- people / workers ---
        if re.search(PERSON_CUE, window, re.I):
            emit(Quantity(value, 'count', 'PERSON', confidence=0.9, role='value',
                          raw_unit='', normalized_unit='PERSON'), start, end)
            continue

        # --- percentage ---
        if re.search(r'درصد|%|percent', window, re.I):
            emit(Quantity(value, 'percentage', 'PERCENT', confidence=0.9,
                          role='value', raw_unit='', normalized_unit='PERCENT'),
                 start, end)
            continue

        # --- speed from context WITHOUT an explicit unit: the unit stays
        #     UNKNOWN_UNIT — never silently substituted (spec §10) --------
        if re.search(SPEED_CUE, window, re.I) and re.search(DISTANCE_CUE, window, re.I):
            emit(Quantity(value, 'speed', 'UNKNOWN_UNIT', confidence=0.75,
                          role='value', raw_unit='', normalized_unit='UNKNOWN_UNIT'),
                 start, end)
            continue

        emit(Quantity(value, 'dimensionless', 'NONE', confidence=0.4, role='value',
                      raw_unit='', normalized_unit='NONE'), start, end)
    return out


def _time_unit_of(token: str) -> str:
    tok = token.strip().lower()
    if tok.startswith(('دقیقه', 'min')) or 'دقیقه' in tok:
        return 'MINUTE'
    if tok.startswith(('ثانیه', 'sec', 's')):
        return 'SECOND'
    if tok.startswith(('روز', 'day')):
        return 'DAY'
    return 'HOUR'


def _dim_of_operation_value(op: dict, quantities: list[Quantity]) -> str:
    """Locate the typed dimension of an operation's numeric value, if any."""
    v = op.get('value')
    if v is None:
        return 'dimensionless'
    for q in quantities:
        if abs(float(q.value) - float(v)) < 1e-12:
            return q.dimension
    return 'dimensionless'


def validate_operation_chain(initial: Quantity, operations: list[dict],
                             quantities: list[Quantity]) -> list[str]:
    """Return failed typed-check names for an add/subtract chain over `initial`.

    v22.4: dimension compatibility is now validated per operation through the
    operation-aware algebra; currencies must also agree on their unit."""
    from jarvis.agent.semantics_v22_4 import validate_binary_operation
    failed: list[str] = []
    cur = initial
    for op in operations:
        if op.get('op') not in ('add', 'subtract', 'inventory_add', 'inventory_remove',
                                'credit', 'debit'):
            continue
        vdim = _dim_of_operation_value(op, quantities)
        vq = next((q for q in quantities if abs(float(q.value) - float(op.get('value', 0))) < 1e-12),
                  None)
        rhs = vq if vq is not None else Quantity(float(op.get('value', 0) or 0), vdim, '')
        if vdim == 'dimensionless' or cur.dimension == 'dimensionless':
            continue
        failure = validate_binary_operation(op.get('op', 'add'), cur, rhs)
        if failure is not None:
            name = 'currency_dimension_mismatch' if 'currency' in (cur.dimension, rhs.dimension) \
                else 'source_operation_dimension_consistency'
            if name not in failed:
                failed.append(name)
    return failed


def dimension_verdict(quantities: list[Quantity]) -> dict:
    """Pairwise audit across the whole source: currencies must never mix with
    counts, and two different currencies never add without a rate."""
    typed = [q for q in quantities if q.dimension not in ('dimensionless', 'identifier')]
    currencies = [q for q in typed if q.dimension == 'currency']
    counts = [q for q in typed if q.dimension == 'count']
    failed: list[str] = []
    units = {q.unit for q in currencies}
    if len(units) > 1:
        failed.append('source_currency_consistency')
    if currencies and counts and len(typed) >= 2:
        # A currency entity (account balance) mixed with item counts is only a
        # violation when the sentence performs one arithmetic chain over both.
        if any(q.entity == 'account' for q in currencies):
            failed.append('currency_dimension_mismatch')
    return {'failed_checks': failed, 'quantities': [q.to_dict() for q in quantities]}
