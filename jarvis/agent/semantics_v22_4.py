"""JARVIS v22.4 — TRUE ROOT-CAUSE SEMANTIC CORE.

This module implements the architectural invariants that eliminate the v22.3
defect classes at their root (never per-example patches):

  * SourceSpan            — exact provenance type with a hard invariant
                            original[start:end] == text (never approximate).
  * DimensionExpr         — compound dimensions (currency/time, distance/time)
                            as first-class typed values; rates are no longer
                            "sentences containing two dimensions".
  * UNIT registry         — raw unit -> normalized unit -> dimension mapping
                            (m/s is M_PER_SECOND, never silently KM_PER_HOUR).
  * validate_binary_operation — operation-aware dimension algebra. The SAME
                            pair of dimensions may be legal for one operator
                            and illegal for another (USD/hour x hour is legal,
                            hour + USD is not).
  * DimensionFailure      — structured failure metadata (operator, left/right
                            dimension + unit) so the renderer never guesses.
  * convert_unit          — explicit conversion separate from extraction,
                            with full provenance metadata.

The module is pure: no I/O, no global mutable state except explicit ContextVars
owned by the verifier layer.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import math
import re

# ======================================================================
# 1. SourceSpan — exact provenance (spec §13/§14/§15)
# ======================================================================


class SpanIntegrityError(ValueError):
    """Raised when a span does not satisfy original[start:end] == text."""


@dataclass(frozen=True)
class SourceSpan:
    start: int
    end: int
    text: str

    def validate(self, original: str) -> 'SourceSpan':
        if not isinstance(original, str):
            raise SpanIntegrityError('original_not_text')
        if not (0 <= self.start <= self.end <= len(original)):
            raise SpanIntegrityError(
                f'span_out_of_range: [{self.start},{self.end}) len={len(original)}')
        if original[self.start:self.end] != self.text:
            raise SpanIntegrityError(
                f'span_mismatch: original[{self.start}:{self.end}]='
                f'{original[self.start:self.end]!r} != {self.text!r}')
        return self


def make_span(original: str, start: int, end: int) -> SourceSpan:
    """Factory that guarantees the invariant at construction time."""
    if not isinstance(original, str):
        raise SpanIntegrityError('original_not_text')
    if not (0 <= start <= end <= len(original)):
        raise SpanIntegrityError(f'span_out_of_range: [{start},{end}) len={len(original)}')
    return SourceSpan(start, end, original[start:end])


class IndexMap:
    """Index mapping for transformed text (strip/normalize/digit-fold).

    Any transformation that changes indices MUST be paired with an IndexMap so
    offsets recovered from the transformed string can be projected back onto
    the ORIGINAL source (spec §14). Constructed from an alignment list where
    aligned[i] = index in original of transformed char i (or -1 when the char
    was introduced by the transform).
    """

    def __init__(self, aligned: list[int], original_length: int):
        self.aligned = aligned
        self.original_length = original_length

    @classmethod
    def identity(cls, length: int) -> 'IndexMap':
        return cls(list(range(length)), length)

    def to_original(self, index: int) -> int:
        """Project a transformed-text index onto the original text."""
        if 0 <= index < len(self.aligned):
            v = self.aligned[index]
            if v >= 0:
                return v
        # Fall back to the nearest aligned predecessor (boundaries of
        # inserted/removed regions resolve to the enclosing original span).
        for j in range(min(index, len(self.aligned) - 1), -1, -1):
            if self.aligned[j] >= 0:
                return self.aligned[j]
        return 0

    def map_span(self, start: int, end: int) -> tuple[int, int]:
        s = self.to_original(start)
        e = self.to_original(max(start, end - 1)) + (
            1 if 0 < end <= len(self.aligned) and self.aligned[end - 1] >= 0 else 0)
        return s, max(s, e)


# ======================================================================
# 2. Dimensions & compound dimension expressions (spec §5/§6)
# ======================================================================

class Dimension:
    CURRENCY = 'currency'
    COUNT = 'count'
    TIME = 'time'
    DISTANCE = 'distance'
    SPEED = 'speed'
    MASS = 'mass'
    VOLUME = 'volume'
    TEMPERATURE = 'temperature'
    PERCENTAGE = 'percentage'
    PROBABILITY = 'probability'
    AGE = 'age'
    DATE = 'date'
    CLOCK_TIME = 'clock_time'
    IDENTIFIER = 'identifier'
    DIMENSIONLESS = 'dimensionless'
    CURRENCY_RATE = 'currency_rate'
    COUNT_RATE = 'count_rate'
    UNIT_PRICE = 'unit_price'          # currency/count
    UNKNOWN_UNIT = 'unknown_unit'


@dataclass(frozen=True)
class DimensionExpr:
    """A (possibly compound) physical dimension: numerator/denominator."""
    numerator: str
    denominator: str = ''

    def key(self) -> str:
        return self.numerator if not self.denominator \
            else f'{self.numerator}/{self.denominator}'

    def is_rate(self) -> bool:
        return bool(self.denominator)

    def __str__(self):
        return self.key()


# Canonical base dimension expressions.
BASE_DIMS = {
    Dimension.CURRENCY: DimensionExpr(Dimension.CURRENCY),
    Dimension.COUNT: DimensionExpr(Dimension.COUNT),
    Dimension.TIME: DimensionExpr(Dimension.TIME),
    Dimension.DISTANCE: DimensionExpr(Dimension.DISTANCE),
    Dimension.MASS: DimensionExpr(Dimension.MASS),
    Dimension.VOLUME: DimensionExpr(Dimension.VOLUME),
    Dimension.TEMPERATURE: DimensionExpr(Dimension.TEMPERATURE),
    Dimension.PERCENTAGE: DimensionExpr(Dimension.PERCENTAGE),
    Dimension.PROBABILITY: DimensionExpr(Dimension.PROBABILITY),
    Dimension.AGE: DimensionExpr(Dimension.AGE),
    Dimension.DIMENSIONLESS: DimensionExpr(Dimension.DIMENSIONLESS),
}

RATE_DIMS = {
    Dimension.SPEED: DimensionExpr(Dimension.DISTANCE, Dimension.TIME),
    Dimension.CURRENCY_RATE: DimensionExpr(Dimension.CURRENCY, Dimension.TIME),
    Dimension.COUNT_RATE: DimensionExpr(Dimension.COUNT, Dimension.TIME),
    Dimension.UNIT_PRICE: DimensionExpr(Dimension.CURRENCY, Dimension.COUNT),
}


def dimension_expr(dimension: str | DimensionExpr) -> DimensionExpr:
    if isinstance(dimension, DimensionExpr):
        return dimension
    return RATE_DIMS.get(dimension) or BASE_DIMS.get(
        dimension, DimensionExpr(str(dimension)))


# Dimensions that never participate in the strict chain guard.
SAFE_DIMS = {Dimension.DIMENSIONLESS, Dimension.IDENTIFIER, Dimension.CLOCK_TIME,
             Dimension.PERCENTAGE, Dimension.PROBABILITY, Dimension.DATE,
             Dimension.AGE, Dimension.UNKNOWN_UNIT}


# ======================================================================
# 3. Unit registry — raw unit -> normalized unit -> dimension (spec §9/§10)
# ======================================================================

# normalized_unit -> (dimension_expr_key, si_factor_to_base)
UNIT_REGISTRY: dict[str, tuple[str, float]] = {
    # --- speed (distance/time) ---
    'M_PER_SECOND': (Dimension.SPEED, 1.0),
    'KM_PER_HOUR': (Dimension.SPEED, 1.0 / 3.6),
    'MILE_PER_HOUR': (Dimension.SPEED, 0.44704),
    # --- currency rate (currency/time) ---
    'USD_PER_HOUR': (Dimension.CURRENCY_RATE, 1.0),
    'USD_PER_DAY': (Dimension.CURRENCY_RATE, 1.0),
    'EUR_PER_HOUR': (Dimension.CURRENCY_RATE, 1.0),
    'TOMAN_PER_HOUR': (Dimension.CURRENCY_RATE, 1.0),
    # --- count rate (count/time) ---
    'ITEM_PER_HOUR': (Dimension.COUNT_RATE, 1.0),
    'ITEM_PER_DAY': (Dimension.COUNT_RATE, 1.0),
    'ITEM_PER_MINUTE': (Dimension.COUNT_RATE, 1.0),
    # --- unit price (currency/count) ---
    'USD_PER_ITEM': (Dimension.UNIT_PRICE, 1.0),
    # --- time ---
    'HOUR': (Dimension.TIME, 3600.0),
    'MINUTE': (Dimension.TIME, 60.0),
    'SECOND': (Dimension.TIME, 1.0),
    'DAY': (Dimension.TIME, 86400.0),
    # --- distance ---
    'KM': (Dimension.DISTANCE, 1000.0),
    'M': (Dimension.DISTANCE, 1.0),
    'MILE': (Dimension.DISTANCE, 1609.344),
    # --- mass ---
    'KG': (Dimension.MASS, 1.0),
    'G': (Dimension.MASS, 0.001),
    # --- volume ---
    'L': (Dimension.VOLUME, 1.0),
    'ML': (Dimension.VOLUME, 0.001),
    # --- temperature ---
    'C': (Dimension.TEMPERATURE, 1.0),
    'F': (Dimension.TEMPERATURE, 1.0),
    # --- currency ---
    'USD': (Dimension.CURRENCY, 1.0),
    'EUR': (Dimension.CURRENCY, 1.0),
    'TOMAN': (Dimension.CURRENCY, 1.0),
    'RIAL': (Dimension.CURRENCY, 1.0),
    'GBP': (Dimension.CURRENCY, 1.0),
    # --- count ---
    'ITEM': (Dimension.COUNT, 1.0),
    'PERSON': (Dimension.COUNT, 1.0),
    'PIECE': (Dimension.COUNT, 1.0),
    'GENERIC': (Dimension.COUNT, 1.0),
}


def unit_dimension(normalized_unit: str) -> DimensionExpr:
    entry = UNIT_REGISTRY.get(normalized_unit)
    if entry is None:
        return DimensionExpr(Dimension.UNKNOWN_UNIT)
    return dimension_expr(entry[0])


def unit_si_factor(normalized_unit: str) -> float:
    entry = UNIT_REGISTRY.get(normalized_unit)
    return entry[1] if entry else 1.0


# Ordered raw-unit patterns. FIRST MATCH WINS; speed families are matched
# before their distance components (m/s before m, km/h before km).
SPEED_UNIT_PATTERNS: list[tuple[str, str]] = [
    # metres per second
    (r'm\s*/\s*s\b|meters?\s+per\s+seconds?|metres?\s+per\s+seconds?|متر[\s\u200c]*بر[\s\u200c]*ثانیه', 'M_PER_SECOND'),
    # miles per hour
    (r'mph\b|miles?\s+per\s+hours?|مایل[\s\u200c]*بر[\s\u200c]*ساعت', 'MILE_PER_HOUR'),
    # kilometres per hour (after mph!)
    (r'km\s*/\s*(?:hour|hr|h)\b|kmph\b|kph\b|kilometers?\s+per\s+hours?|kilometres?\s+per\s+hours?|کیلومتر[\s\u200c]*بر[\s\u200c]*ساعت', 'KM_PER_HOUR'),
]

CURRENCY_RATE_PATTERNS: list[tuple[str, str]] = [
    (r'(دلار|dollars?|usd)[\s\u200c]*(?:/|(?:در|بر)[\s\u200c]*|per\s)(?:ساعت|hour|hr)', 'USD_PER_HOUR'),
    (r'(دلار|dollars?|usd)[\s\u200c]*(?:/|(?:در|بر)[\s\u200c]*|per\s)(?:روز|day)', 'USD_PER_DAY'),
    (r'(یورو|euros?|eur)[\s\u200c]*(?:/|(?:در|بر)[\s\u200c]*|per\s)(?:ساعت|hour|hr)', 'EUR_PER_HOUR'),
    (r'(تومان|toman)[\s\u200c]*(?:/|(?:در|بر)[\s\u200c]*|per\s)(?:ساعت|hour|hr)', 'TOMAN_PER_HOUR'),
]

COUNT_RATE_PATTERNS: list[tuple[str, str]] = [
    (r'(کالا|قطعه|جنس|آیتم|items?|pieces?|products?|units?)[\s\u200c]*(?:/|(?:در|بر)[\s\u200c]*|per\s)(?:ساعت|hour|hr)', 'ITEM_PER_HOUR'),
    (r'(کالا|قطعه|جنس|آیتم|items?|pieces?|products?|units?)[\s\u200c]*(?:/|(?:در|بر)[\s\u200c]*|per\s)(?:دقیقه|minute|min)', 'ITEM_PER_MINUTE'),
    (r'(کالا|قطعه|جنس|آیتم|items?|pieces?|products?|units?)[\s\u200c]*(?:/|(?:در|بر)[\s\u200c]*|per\s)(?:روز|day)', 'ITEM_PER_DAY'),
]

UNIT_PRICE_PATTERNS: list[tuple[str, str]] = [
    (r'(دلار|dollars?|usd)[\s\u200c]*(?:/|(?:در|بر)[\s\u200c]*|per\s)(?:کالا|قطعه|آیتم|item|piece)', 'USD_PER_ITEM'),
]


def classify_rate_unit(text: str) -> tuple[str, str] | None:
    """Return (raw_unit_text, normalized_unit) for a rate suffix in `text`,
    or None when no rate unit is present."""
    for pattern, unit in SPEED_UNIT_PATTERNS:
        m = re.search(pattern, text, re.I)
        if m:
            return m.group(0).strip(), unit
    for table in (CURRENCY_RATE_PATTERNS, COUNT_RATE_PATTERNS, UNIT_PRICE_PATTERNS):
        for pattern, unit in table:
            m = re.search(pattern, text, re.I)
            if m:
                return m.group(0).strip(), unit
    return None


# ======================================================================
# 4. Operation-aware dimension algebra (spec §5)
# ======================================================================

ADD_OPS = frozenset({'add', 'subtract', 'inventory_add', 'inventory_remove',
                     'credit', 'debit', 'sum'})
MUL_OPS = frozenset({'multiply', 'scale'})
DIV_OPS = frozenset({'divide'})


@dataclass
class DimensionFailure:
    """Structured failure metadata (spec §23). The renderer MUST use these
    fields instead of guessing a dimension pair."""
    failure_type: str            # dimension_mismatch | currency_mix | rate_mismatch
    operator: str                # add | subtract | multiply | divide
    left_dimension: str
    left_unit: str
    right_dimension: str
    right_unit: str
    message_fa: str = ''
    message_en: str = ''

    @property
    def key(self) -> str:
        """Renderer key derived from the ACTUAL dimensions (never generic)."""
        if self.failure_type == 'rate_mismatch':
            return 'rate_mismatch'
        pair = {self.left_dimension, self.right_dimension}
        if self.failure_type == 'currency_mix':
            return 'currency_mix'
        if pair == {'time', 'distance'}:
            return 'time_distance'
        if pair == {'mass', 'volume'}:
            return 'mass_volume'
        if pair == {'time', 'currency'} or (self.left_dimension == Dimension.CURRENCY_RATE
                                            or self.right_dimension == Dimension.CURRENCY_RATE):
            return 'time_currency'
        if pair == {'time', 'count'} or Dimension.COUNT_RATE in (
                self.left_dimension, self.right_dimension):
            return 'time_count'
        if pair == {'currency', 'count'}:
            return 'currency_item'
        return 'dimension'

    def to_dict(self) -> dict:
        return {
            'failure_type': self.failure_type,
            'operator': self.operator,
            'left_dimension': self.left_dimension,
            'left_unit': self.left_unit,
            'right_dimension': self.right_dimension,
            'right_unit': self.right_unit,
            'message_fa': self.message_fa,
            'message_en': self.message_en,
            'key': self.key,
        }


def _as_q(quantity):
    """Accept Quantity-like objects or (dimension, unit) tuples."""
    if hasattr(quantity, 'dimension'):
        return (str(quantity.dimension), str(getattr(quantity, 'unit', '') or ''))
    if isinstance(quantity, (tuple, list)) and len(quantity) == 2:
        return (str(quantity[0]), str(quantity[1]))
    return (str(quantity), '')


# Structured messages per renderer key (spec §24).
_FAILURE_MESSAGES = {
    'time_distance': ('زمان و مسافت را نمی‌توان مستقیماً جمع کرد.',
                      'Time and distance cannot be directly added.'),
    'mass_volume': ('جرم و حجم دو کمیت متفاوت هستند.',
                    'Mass and volume are different quantities.'),
    'currency_item': ('مبلغ پول و تعداد کالا را نمی‌توان مستقیماً جمع کرد.',
                      'Money and item counts cannot be directly added.'),
    'currency_mix': ('دو ارز متفاوت بدون نرخ تبدیل قابل جمع نیستند.',
                     'Two different currencies cannot be combined without an exchange rate.'),
    'time_currency': ('زمان و مبلغ پول را نمی‌توان مستقیماً جمع کرد.',
                      'Time and money cannot be directly added.'),
    'time_count': ('زمان و تعداد را نمی‌توان مستقیماً جمع کرد.',
                   'Time and counts cannot be directly added.'),
    'rate_mismatch': ('این دو نرخ کمیت متفاوتی را می‌سنجند و جمع‌پذیر نیستند.',
                      'These rates measure different quantities and cannot be combined.'),
    'dimension': ('یکاهای این عددها برای یک عمل ریاضی سازگار نیستند.',
                  'The units of these numbers are incompatible for this operation.'),
}


def _failure(operator, left, right, failure_type='dimension_mismatch') -> DimensionFailure:
    ld, lu = _as_q(left)
    rd, ru = _as_q(right)
    expr = dimension_expr(ld)
    expr_r = dimension_expr(rd)
    # Derive the renderer key from the BASE dimensions of the operands.
    probe_l = (expr.numerator, lu)
    probe_r = (expr_r.numerator, ru)
    f = DimensionFailure(failure_type, operator, ld, lu, rd, ru)
    if failure_type == 'rate_mismatch':
        f.message_fa, f.message_en = _FAILURE_MESSAGES['rate_mismatch']
    else:
        key = DimensionFailure('x', operator, probe_l[0], probe_l[1],
                               probe_r[0], probe_r[1]).key
        if failure_type == 'currency_mix':
            key = 'currency_mix'
        f.message_fa, f.message_en = _FAILURE_MESSAGES.get(key, _FAILURE_MESSAGES['dimension'])
    return f


def validate_binary_operation(operator: str, left_quantity, right_quantity,
                              relation: str | None = None):
    """Operation-aware dimension algebra (spec §5).

    Returns None when the operation is dimensionally valid, otherwise a
    structured DimensionFailure. Dimensionless operands never contaminate a
    typed chain; rates are compound dimensions (numerator/denominator).
    """
    if relation == 'rate':  # caller already knows it is a rate composition
        return None
    op = str(operator).lower()
    left, right = _as_q(left_quantity), _as_q(right_quantity)
    ld, rd = dimension_expr(left[0]), dimension_expr(right[0])

    # Untyped operands never trigger the guard (conservative by design).
    if ld.numerator in (Dimension.DIMENSIONLESS, Dimension.UNKNOWN_UNIT) \
            or rd.numerator in (Dimension.DIMENSIONLESS, Dimension.UNKNOWN_UNIT):
        return None
    if ld.numerator in SAFE_DIMS or rd.numerator in SAFE_DIMS:
        return None

    if op in ADD_OPS:
        if ld.key() == rd.key():
            # Same compound dimension: currencies must also agree on unit.
            if ld.numerator == Dimension.CURRENCY and left[1] and right[1] \
                    and left[1] != right[1]:
                return _failure(op, left, right, 'currency_mix')
            return None
        if ld.is_rate() and rd.is_rate():
            # Two rates measuring different quantities (USD/hour + ITEM/hour)
            # are never combinable — a structured rate mismatch, not a generic
            # dimension confusion.
            return _failure(op, left, right, 'rate_mismatch')
        return _failure(op, left, right)

    if op in MUL_OPS:
        # rate × denominator-base -> numerator (SPEED×TIME→DISTANCE,
        # CURRENCY_RATE×TIME→CURRENCY, COUNT_RATE×TIME→COUNT).
        if ld.is_rate() and not rd.is_rate() and ld.denominator == rd.numerator:
            return None
        if rd.is_rate() and not ld.is_rate() and rd.denominator == ld.numerator:
            return None
        # dimensionless scaling
        if ld.numerator == Dimension.DIMENSIONLESS or rd.numerator == Dimension.DIMENSIONLESS:
            return None
        return _failure(op, left, right)

    if op in DIV_OPS:
        # same/same -> dimensionless ratio
        if ld.key() == rd.key():
            return None
        # DISTANCE / TIME -> SPEED, CURRENCY / TIME -> CURRENCY_RATE,
        # COUNT / TIME -> COUNT_RATE, CURRENCY / COUNT -> UNIT_PRICE
        if rd.key() == Dimension.TIME and not ld.is_rate():
            return None
        if rd.key() == Dimension.COUNT and ld.numerator == Dimension.CURRENCY:
            return None
        if ld.is_rate() and rd.numerator == Dimension.DIMENSIONLESS:
            return None
        return _failure(op, left, right)

    # Unknown operator: be strict.
    return _failure(op, left, right)


def validate_rate_pair_addition(left_unit: str, right_unit: str):
    """Adding two rates requires the SAME normalized rate family
    (10 USD/hour + 3 USD/hour -> valid; USD/hour + ITEM/hour -> invalid)."""
    if not left_unit or not right_unit:
        return None
    ld, rd = unit_dimension(left_unit), unit_dimension(right_unit)
    if ld.key() == rd.key():
        return None
    return DimensionFailure(
        'rate_mismatch', 'add', ld.key(), left_unit, rd.key(), right_unit,
        *_FAILURE_MESSAGES['rate_mismatch'])


# ======================================================================
# 5. Explicit unit conversion, separate from extraction (spec §11)
# ======================================================================

CONVERSION_FACTORS = {
    # (from_unit, to_unit) -> factor  (to = from * factor)
    ('M_PER_SECOND', 'KM_PER_HOUR'): 3.6,
    ('KM_PER_HOUR', 'M_PER_SECOND'): 1 / 3.6,
    ('M_PER_SECOND', 'MILE_PER_HOUR'): 2.23694,
    ('KM_PER_HOUR', 'MILE_PER_HOUR'): 0.621371,
    ('MILE_PER_HOUR', 'KM_PER_HOUR'): 1.60934,
    ('HOUR', 'MINUTE'): 60.0,
    ('MINUTE', 'HOUR'): 1 / 60.0,
    ('HOUR', 'SECOND'): 3600.0,
    ('KM', 'M'): 1000.0,
    ('M', 'KM'): 0.001,
    ('KG', 'G'): 1000.0,
    ('L', 'ML'): 1000.0,
}


def convert_unit(value: float, from_unit: str, to_unit: str) -> dict | None:
    """Explicit conversion with full provenance. Never mutates the extracted
    quantity; returns metadata the renderer may present verbatim."""
    if from_unit == to_unit:
        return {'source_value': float(value), 'source_unit': from_unit,
                'result_value': float(value), 'result_unit': to_unit,
                'factor': 1.0}
    factor = CONVERSION_FACTORS.get((from_unit, to_unit))
    if factor is None:
        # Derive through SI factors when both units share one dimension.
        d_from, d_to = unit_dimension(from_unit), unit_dimension(to_unit)
        if d_from.key() == d_to.key() and d_from.key() != Dimension.UNKNOWN_UNIT:
            si = unit_si_factor(from_unit) / unit_si_factor(to_unit)
            return {'source_value': float(value), 'source_unit': from_unit,
                    'result_value': float(value) * si, 'result_unit': to_unit,
                    'factor': si}
        return None
    result = float(value) * factor
    if not math.isfinite(result):
        return None
    return {'source_value': float(value), 'source_unit': from_unit,
            'result_value': result, 'result_unit': to_unit, 'factor': factor}


# ======================================================================
# 6. Chain-operand extraction for the guard (spec §5: no blanket exemptions)
# ======================================================================

# Chain language includes explicit operators '+'/'-' between numbers.
_CHAIN_LANG = re.compile(
    r'جمع\s*کن|جمع(?=\s*\d)|جمع\s*می[\s\u200c]*شود|اضافه\s*کن|اضافه\s*می[\s\u200c]*شود|کم\s*کن|'
    r'کم\s*می[\s\u200c]*شود|با\s*هم\s*جمع|جمع\s*زدن|جمع[\s\u200c]*ش\b|'
    r'مجموع[\s\u200c]*ش\b|'
    r'\badd(?:s|ed)?\b|\bplus\b|'
    r'\bsubtract(?:s|ed)?\b|\bsum(?:s|med)?\b|\btotal\b|\bcombine[ds]?\b|'
    r'(?<=\s)\+(?=\s)|(?<=\d)\s*\+\s*(?=\d)', re.I)

# A time quantity preceded by these cues QUALIFIES the sentence (rate window,
# after X days, ...) instead of acting as an addition operand.
_TEMPORAL_QUALIFIER_LEFT = re.compile(
    r'(?:بعد\s*از|پس\s*از|طی|ظرف|در\s*طول|هر|after|within|in|for|per|each|'
    r'every|over|during)\s*$', re.I)

_RATE_WINDOW = re.compile(
    r'(?:در|هر|per|each|every)\s*(?:ساعت|روز|دقیقه|hours?|days?|minutes?)\s*'
    r'(?:فقط|تنها|یک\s*بار)?\s*$', re.I)

_CLAUSE_SPLIT = re.compile(r'[؛;،,\.؟?\!\n]|و\s*بعد|سپس|آنگاه|\bthen\b|\band\b|\bو\b')


def _clause_bounds(text: str) -> list[tuple[int, int]]:
    bounds, start = [], 0
    for m in _CLAUSE_SPLIT.finditer(text):
        if m.start() > start:
            bounds.append((start, m.start()))
        start = m.end()
    if start < len(text):
        bounds.append((start, len(text)))
    return bounds or [(0, len(text))]


def chain_operands(text: str, quantities) -> list:
    """Quantities that participate in an explicit add/subtract chain.

    Operation-aware instead of dimension-pair-based:
      * the chain must be explicitly requested somewhere in the sentence;
      * a quantity joins the chain when its own clause performs the
        operation, or when it is the FIRST typed quantity preceding a verb
        clause (the chain initial: '100 دلار است؛ 5 دلار اضافه کن');
      * time quantities acting as temporal qualifiers ('بعد از 3 روز',
        'after 2 hours', 'هر روز') are NOT operands — that is the correct
        root-cause replacement for the removed {time,count}/{time,currency}
        blanket exemptions.
    """
    if not (text and _CHAIN_LANG.search(text)):
        return []
    bounds = _clause_bounds(text)
    typed = [q for q in quantities
             if getattr(q, 'dimension', 'dimensionless') not in SAFE_DIMS]
    if len(typed) < 2:
        return []

    def clause_index(q) -> int:
        start = getattr(q, 'start', -1)
        if start < 0:
            return 0
        for i, (s, e) in enumerate(bounds):
            if s <= start < e:
                return i
        return len(bounds) - 1

    def temporal_qualified(q) -> bool:
        if getattr(q, 'dimension', '') != Dimension.TIME:
            return False
        start = getattr(q, 'start', -1)
        end = getattr(q, 'end', -1)
        left = text[max(0, start - 24):start]
        if _TEMPORAL_QUALIFIER_LEFT.search(left):
            return True
        right = text[end:end + 12]
        return bool(re.match(r'\s*(?:بعد|پس|later|after)', right, re.I))

    verb_clauses = {i for i, (s, e) in enumerate(bounds)
                    if _CHAIN_LANG.search(text[s:e])}
    if not verb_clauses:
        return []
    first_verb_clause = min(verb_clauses)
    operands: list = []
    # Persian verb-first ordering ('جمع 5 لیتر و 8 کیلوگرم'): when the FIRST
    # clause itself requests the operation, the whole typed sentence is the
    # operation scope.
    verb_first = 0 in verb_clauses
    for idx, q in enumerate(typed):
        if temporal_qualified(q):
            continue
        ci = clause_index(q)
        if ci in verb_clauses:
            operands.append(q)
        elif idx == 0 and ci < first_verb_clause:
            # chain initial asserted before the verb clause
            operands.append(q)
        elif verb_first and not temporal_qualified(q):
            operands.append(q)
    # A lone operand never makes a chain.
    return operands if len(operands) >= 2 else []


def cross_dimension_chain_failure_v4(text: str, quantities) -> list[DimensionFailure]:
    """Structured replacement for v22.3's `cross_dimension_chain_failure`.

    Validates EVERY consecutive operand pair with the operation-aware algebra
    instead of whitelisting dimension pairs globally.
    """
    operands = chain_operands(text, quantities)
    failures: list[DimensionFailure] = []
    for a, b in zip(operands, operands[1:]):
        failure = validate_binary_operation('add', a, b)
        if failure is not None and not any(
                f.left_dimension == failure.left_dimension
                and f.right_dimension == failure.right_dimension for f in failures):
            failures.append(failure)
    return failures


# ======================================================================
# 7. AGE SEMANTICS — entity -> attribute -> value (spec §16-§21)
# ======================================================================

# Name tokens: Persian letters or Latin words (no digits).
_NAME = r"[A-Za-z\u0600-\u06FF\u200c]+"

AGE_DIFFERENCE_QUERY = re.compile(
    r'اختلاف\s*سن|فاصله\s*سن'
    r'|چند\s*سال[^؟?;؛.\n]{0,24}?(?:بزرگ|کوچک)[\s\u200c]*تر'
    r'|چند\s*سال[\s\u200c]*(?:متفاوت|فرق|فاصله)'
    r'|چند\s*سال\s*اختلاف'
    r'|how\s+many\s+years\s+(?:older|younger|apart)'
    r'|how\s+much\s+(?:older|younger)'
    r'|what\s+is\s+the\s+age\s+difference|age\s+difference'
    r'|how\s+many\s+years\s+between', re.I)

_AGE_PRONOUNS = frozenset({
    'او', 'اوست', 'ایشان', 'آنها', 'her', 'his', 'him', 'their', 'them',
})
_PRONOUN_MARK = '__pronoun__'

_AGE_PATTERNS = [
    # EN: "Ali is 35 (years old)", "Ali is 35."
    re.compile(rf'({_NAME})\s+is\s+(\d+(?:\.\d+)?)(?:\s+years?\s+old)?(?=\s|,|\.|;|؛|؟|\?|$)', re.I),
    # EN: "Ali, aged 35," / "Ali aged 35"
    re.compile(rf'({_NAME}),?\s+aged\s+(\d+(?:\.\d+)?)', re.I),
    # EN: "Ali's age is 35" / "age of Ali is 35"
    re.compile(rf"({_NAME})(?:'s|\u2019s)?\s+age\s+is\s+(\d+(?:\.\d+)?)", re.I),
    re.compile(rf'age\s+of\s+({_NAME})\s+is\s+(\d+(?:\.\d+)?)', re.I),
    # FA: "سن علی 35 است" / "سن علی 35 سال است" / "سن علی 35"
    re.compile(rf'سن\s+({_NAME})\s+(\d+(?:\.\d+)?)(?:\s+سال)?(?:\s+است)?', re.I),
    # FA: "علی 35 سال دارد"
    re.compile(rf'({_NAME})\s+(\d+(?:\.\d+)?)\s+سال\s+دارد', re.I),
    # FA: "علی 35 ساله است" / "علی 35 ساله"
    re.compile(rf'({_NAME})\s+(\d+(?:\.\d+)?)\s*ساله(?:\s+است)?', re.I),
    # FA: "علی 35 سالش است"
    re.compile(rf'({_NAME})\s+(\d+(?:\.\d+)?)\s+سالش\s+است', re.I),
]

# Relation form: "Reza is 22 years older than Ali" / "رضا 22 سال از علی بزرگتر است"
# Each entry: (compiled pattern, group indices) — (subject, value, object, direction)
_AGE_RELATION_PATTERNS = [
    (re.compile(
        rf'({_NAME})\s+is\s+(\d+(?:\.\d+)?)\s+years?\s+(older|younger)\s+than\s+({_NAME})', re.I),
     (1, 2, 4, 3)),
    (re.compile(
        rf'({_NAME})\s+(\d+(?:\.\d+)?)\s+سال\s+(?:از|than)\s+({_NAME})\s+(بزرگ|کوچک)[\s\u200c]*(?:تر|tar)?', re.I),
     (1, 2, 3, 4)),
    (re.compile(
        rf'({_NAME})\s+(?:از|than)\s+({_NAME})\s+(\d+(?:\.\d+)?)\s+سال\s+(بزرگ|کوچک)[\s\u200c]*تر', re.I),
     (1, 3, 2, 4)),
    # FA: 'مادرش 18 سال بزرگتر از اوست' — (A) N سال بزرگتر از (B)
    (re.compile(
        rf'({_NAME})\s+(\d+(?:\.\d+)?)\s+سال\s+(بزرگ|کوچک)[\s\u200c]*تر\s+از\s+({_NAME})', re.I),
     (1, 2, 4, 3)),
    # EN implicit object: 'her mother is 6 years older.' — object anaphora
    (re.compile(
        rf'({_NAME})\s+is\s+(\d+(?:\.\d+)?)\s+years?\s+(older|younger)\b(?!\s*than)', re.I),
     (1, 2, None, 3)),
]


_NAME_STRIP = ' \u200c،,؛;:؟?!.«»"\''


def _clean_name(token: str) -> str:
    """Persian punctuation (؟ ! . ، ؛) sits inside the \u0600-\u06FF block
    and would otherwise glue onto captured names ('سارا؟')."""
    return (token or '').strip(_NAME_STRIP)


def _is_name(token: str) -> bool:
    t = token.strip(' \u200c،,')
    if not t or any(ch.isdigit() for ch in t):
        return False
    stopwords = {'the', 'a', 'an', 'and', 'his', 'her', 'their', 'age', 'years',
                 'year', 'old', 'is', 'was', 'he', 'she', 'آن', 'که', 'و', 'سن',
                 'سال', 'است', 'دارد', 'ساله', 'این',
                 # v22.4: possession/auxiliary verbs are never owners — the
                 # optional-verb balance grammar would otherwise read
                 # 'has 100 dollars' as the owner 'has' (spec §26).
                 'has', 'had', 'have', 'were', 'are', 'am', 'been', 'being',
                 'does', 'did', 'do', 'gives', 'gave', 'pays', 'paid',
                 'transfers', 'transferred', 'receives', 'received', 'sent'}
    return t.lower() not in stopwords and len(t) <= 24


@dataclass
class AgeFact:
    entity: str
    value: float | None = None
    unit: str = 'YEAR'
    relation: str | None = None       # older_than | younger_than
    relation_object: str | None = None
    difference: float | None = None
    start: int = -1
    end: int = -1
    span: str = ''

    def to_dict(self) -> dict:
        return {'entity': self.entity, 'attribute': 'age', 'value': self.value,
                'unit': self.unit, 'relation': self.relation,
                'relation_object': self.relation_object,
                'difference': self.difference, 'source_span': self.span}


def extract_age_facts(text: str) -> list[AgeFact]:
    """Entity → attribute → value extraction for ages (spec §17).

    Generalizes ALL phrasings (aged / possessive / دارد / ساله / سن X N است);
    a new surface form adds one pattern to the table, never a new code path.
    """
    t = (text or '').translate(str.maketrans('۰۱۲۳۴۵۶۷۸۹٠١٢٣٤٥٦٧٨٩', '01234567890123456789'))
    facts: list[AgeFact] = []
    claimed: list[tuple[int, int]] = []

    def claim(s: int, e: int) -> bool:
        if any(s < ce and cs < e for cs, ce in claimed):
            return False
        claimed.append((s, e))
        return True

    for pattern, (si, vi, oi, di) in _AGE_RELATION_PATTERNS:
        for m in pattern.finditer(t):
            subject = _clean_name(m.group(si))
            obj = _clean_name(m.group(oi)) if oi else _PRONOUN_MARK
            value, direction = m.group(vi), m.group(di)
            if not _is_name(subject):
                continue
            if obj != _PRONOUN_MARK and not _is_name(obj):
                continue
            if not claim(m.start(), m.end()):
                continue
            facts.append(AgeFact(
                entity=subject, value=None,
                relation='older_than' if direction.lower() in ('older', 'بزرگ') else 'younger_than',
                relation_object=obj, difference=float(value),
                start=m.start(), end=m.end(), span=m.group(0)))
    for pattern in _AGE_PATTERNS:
        for m in pattern.finditer(t):
            name, value = m.group(1), float(m.group(2))
            if not _is_name(name):
                continue
            if not (0 <= value <= 150):
                continue
            if not claim(m.start(), m.end()):
                continue
            facts.append(AgeFact(entity=_clean_name(name), value=value,
                                 start=m.start(), end=m.end(), span=m.group(0)))
    # anaphora: 'او/her/his' resolves to the nearest OTHER named person that
    # carries age evidence in the same sentence (spec §17 entity binding).
    for f in facts:
        if f.relation_object in _AGE_PRONOUNS or f.relation_object == _PRONOUN_MARK:
            others = [g.entity for g in facts
                      if g is not f and g.entity != f.entity and g.value is not None]
            if others:
                f.relation_object = others[0]
            elif len({g.entity for g in facts if g is not f}) == 1:
                f.relation_object = next(g.entity for g in facts if g is not f)
    return facts


def bind_age_query(text: str, facts: list[AgeFact]) -> dict | None:
    """Bind the age question to NAMED entities (spec §18) — never to the
    first two numbers found in the text."""
    t = (text or '').translate(str.maketrans('۰۱۲۳۴۵۶۷۸۹٠١٢٣٤٥٦٧٨٩', '01234567890123456789'))
    # --- sum-of-ages query (spec §20/§21 family) ---
    if re.search(r'جمع\s*سن|مجموع\s*سن|سن\s*آن\s*دو'
                 r'|sum\s+of\s+(?:their\s+)?ages|total\s+age|ages?\s+combined', t, re.I):
        people: list[str] = []
        for f in facts:
            for n in [f.entity] + ([f.relation_object] if f.relation_object else []):
                if n and n != _PRONOUN_MARK and n not in people:
                    people.append(n)
        if len(people) == 2:
            return {'kind': 'sum', 'a': people[0], 'b': people[1]}
        return None
    if not AGE_DIFFERENCE_QUERY.search(t):
        return None
    names: list[str] = []
    # explicit "between X and Y" / "اختلاف سن X و Y" bindings first
    for m in re.finditer(
            rf'(?:between|اختلاف\s*سن|فاصله\s*سن|سن)\s+({_NAME})\s+(?:و|and)\s+({_NAME})', t, re.I):
        a, b = _clean_name(m.group(1)), _clean_name(m.group(2))
        if _is_name(a) and _is_name(b):
            names = [a, b]
            break
    # "how much older is Reza than Ali" — query-internal binding
    if not names:
        m = re.search(rf'how\s+much\s+(?:older|younger)\s+is\s+({_NAME})\s+than\s+({_NAME})', t, re.I)
        if m and _is_name(m.group(1)) and _is_name(m.group(2)):
            names = [_clean_name(m.group(1)), _clean_name(m.group(2))]
    if not names:
        m = re.search(rf'({_NAME})\s+چند\s+سال\s+(?:از|than)\s+({_NAME})', t, re.I)
        if m and _is_name(m.group(1)) and _is_name(m.group(2)):
            names = [_clean_name(m.group(1)), _clean_name(m.group(2))]
    # fall back to the entities actually carrying age evidence
    if not names:
        present: list[str] = []
        for f in facts:
            for n in ([f.entity] + ([f.relation_object] if f.relation_object else [])):
                if n not in present:
                    present.append(n)
        if len(present) == 2:
            names = present
    if len(names) != 2 or names[0] == names[1]:
        return None
    return {'kind': 'difference', 'a': names[0], 'b': names[1]}


def resolve_age(facts: list[AgeFact], entity: str, depth: int = 0) -> float | None:
    """Resolve an entity's absolute age through absolute values or relations."""
    if depth > 4:
        return None
    for f in facts:
        if f.entity == entity and f.value is not None:
            return f.value
    for f in facts:
        if f.entity == entity and f.relation and f.relation_object and f.difference is not None:
            base = resolve_age(facts, f.relation_object, depth + 1)
            if base is not None:
                return base + f.difference if f.relation == 'older_than' else base - f.difference
    return None


def solve_age_query(text: str) -> dict | None:
    """Entity-bound age solving. Returns metadata with the structured facts
    and the answer, or None when binding fails (never guesses)."""
    facts = extract_age_facts(text)
    query = bind_age_query(text, facts)
    if not query or not facts:
        return None
    a, b = query['a'], query['b']
    age_a, age_b = resolve_age(facts, a), resolve_age(facts, b)
    if age_a is None or age_b is None:
        return None
    if query.get('kind') == 'sum':
        return {'kind': 'sum', 'query': query,
                'ages': {a: age_a, b: age_b}, 'facts': facts,
                'value': float(age_a) + float(age_b)}
    value = abs(float(age_a) - float(age_b))   # invariant: absolute difference >= 0
    return {'kind': 'difference', 'query': query,
            'ages': {a: age_a, b: age_b}, 'facts': facts,
            'difference': value,
            'older': a if age_a >= age_b else b}


# ======================================================================
# 8. OWNERSHIP EVENT MODEL (spec §25-§31)
# ======================================================================

_NAMES = r"[A-Za-z\u0600-\u06FF\u200c]+"
_MONEY = r'(?:دلار|dollars?|usd|یورو|euros?|eur|تومان|tomans?|ریال|rials?|gbp|پوند|pounds?)'
_GOODS = r'(?:کالا|قطعه|جنس|آیتم|items?|pieces?|products?)'

# (pattern, group_map) — every pattern resolves direction explicitly.
_OWNERSHIP_TRANSFERS: list[tuple[str, tuple[int, int, int]]] = [
    # A gives/pays/transfers N (money) to B        (sender, amount, receiver)
    (rf'({_NAMES})\s+(?:gives?|gave|pays?|paid|transfers?|transferred)\s+'
     rf'(\d+(?:\.\d+)?)\s*(?:{_MONEY}|{_GOODS})?\s+to\s+({_NAMES})', (1, 2, 3)),
    # A gives B N (money)
    (rf'({_NAMES})\s+(?:gives?|gave|pays?|paid)\s+({_NAMES})\s+(\d+(?:\.\d+)?)'
     rf'\s*(?:{_MONEY}|{_GOODS})?', (1, 3, 2)),
    # B receives N (money) from A
    (rf'({_NAMES})\s+(?:receives?|received|gets?|got)\s+(\d+(?:\.\d+)?)'
     rf'\s*(?:{_MONEY}|{_GOODS})?\s+from\s+({_NAMES})', (3, 2, 1)),
    # N is sent/transferred from A to B
    (rf'(\d+(?:\.\d+)?)\s*(?:{_MONEY}|{_GOODS})?\s+(?:is\s+|was\s+)?'
     rf'(?:sent|transferred|given|paid)\s+from\s+({_NAMES})\s+to\s+({_NAMES})', (2, 1, 3)),
    # FA: A به B N (money) می دهد / می پردازد
    (rf'({_NAMES})\s+به\s+({_NAMES})\s+(\d+(?:\.\d+)?)\s*(?:{_MONEY}|{_GOODS})?\s*'
     rf'(?:می[\s\u200c]*)?(?:دهد|دهم|پردازد|داد|دادند|دادیم|می دهد)', (1, 3, 2)),
    # FA: A N (money) به B می دهد
    (rf'({_NAMES})\s+(\d+(?:\.\d+)?)\s*(?:{_MONEY}|{_GOODS})\s+به\s+({_NAMES})\s*'
     rf'(?:می[\s\u200c]*)?(?:دهد|دهم|پردازد|داد|دادند|دادیم|می دهد)', (1, 2, 3)),
    # FA: A برای B N (money) می فرستد
    (rf'({_NAMES})\s+برای\s+({_NAMES})\s+(\d+(?:\.\d+)?)\s*(?:{_MONEY}|{_GOODS})?\s*'
     rf'می[\s\u200c]*فرستد', (1, 3, 2)),
    # FA: A N (money) برای B می فرستد
    (rf'({_NAMES})\s+(\d+(?:\.\d+)?)\s*(?:{_MONEY}|{_GOODS})\s+برای\s+({_NAMES})\s*'
     rf'می[\s\u200c]*فرستد', (1, 2, 3)),
    # FA: B N (money) از A دریافت می کند
    (rf'({_NAMES})\s+(\d+(?:\.\d+)?)\s*(?:{_MONEY}|{_GOODS})\s+از\s+({_NAMES})\s*'
     rf'دریافت[\s\u200c]*می[\s\u200c]*کند', (3, 2, 1)),
    # FA: A N (money) به B منتقل کرد / منتقل می کند / منتقل شد
    (rf'({_NAMES})\s+(\d+(?:\.\d+)?)\s*(?:{_MONEY}|{_GOODS})\s+به\s+({_NAMES})\s*'
     rf'(?:منتقل[\s\u200c]*(?:می[\s\u200c]*کند|کرد|شده?\s*است|شد))', (1, 2, 3)),
    # FA: N (money) از A به B منتقل/ارسال شد
    (rf'(\d+(?:\.\d+)?)\s*(?:{_MONEY}|{_GOODS})\s+از\s+({_NAMES})\s+به\s+({_NAMES})\s*'
     rf'(?:منتقل|ارسال|واریز)[\s\u200c]*(?:می[\s\u200c]*شود|شد)', (2, 1, 3)),
    # v22.4 (spec §27): implicit receiver — 'مینا 43 دلار می‌دهد' / 'she pays 43'
    # the receiver is anaphoric and resolves to the other balance holder.
    (rf'({_NAMES})\s+(\d+(?:\.\d+)?)\s*(?:{_MONEY}|{_GOODS})\s*'
     rf'(?:را\s*)?(?:می[\s\u200c]*)?(?:دهد|دهد|پردازد|فرستد|داد|کرد)\s*[.؛،?!؟]', (1, 2)),
    (rf'({_NAMES})\s+(?:\btransfers?|\bpays?|\bgives?)\s+(\d+(?:\.\d+)?)'
     rf'\s*(?:{_MONEY}|{_GOODS})?\s*[.؛،?!؟]', (1, 2)),
]

_OWNERSHIP_BALANCES: list[str] = [
    # FA: 'X N دلار دارد' — and the SHARED-VERB forms 'X N دلار و Y M دلار
    # دارند' / 'X N دلار دارد، Y M دلار' where the verb (and even the unit)
    # may be omitted on the second owner: the unit comes from the system.
    rf'({_NAMES})\s+(\d+(?:\.\d+)?)\s*(?:{_MONEY}|{_GOODS})\s*(?:دارد|داشت|دارند|داشته)?',
    # EN: 'X has N (dollars)?' — the unit may be dropped in compact sentences.
    rf'({_NAMES})\s+has\s+(\d+(?:\.\d+)?)(?:\s*(?:{_MONEY}|{_GOODS}))?',
    rf'(?:balance\s+of)\s+({_NAMES})\s+(?:is|است|=)?\s*'
    rf'(\d+(?:\.\d+)?)\s*(?:{_MONEY}|{_GOODS})?',
    rf'({_NAMES})(?:\'s|\u2019s)?\s+(?:balance)\s+(?:is|=)\s*(\d+(?:\.\d+)?)',
    rf'موجودی\s+({_NAMES})\s+(\d+(?:\.\d+)?)\s*(?:{_MONEY}|{_GOODS})?\s*(?:است|دارد)?',
]

# v22.4 (spec §29): multi-transfer shorthand 'Ali 100. Sara 50. ...' — a bare
# 'Name N' statement is ONLY a balance when it does not overlap a transfer
# span and the sentence carries explicit transfer language.
_OWNERSHIP_BARE_BALANCE = (
    rf'({_NAMES})\s+(\d+(?:\.\d+)?)(?=\s*[.؛،;؟?!]|$)')


@dataclass
class OwnershipEvent:
    type: str                 # transfer
    source: str               # sender (money leaves)
    target: str               # receiver (money arrives)
    amount: float
    unit: str
    start: int
    end: int
    source_span: str

    def to_dict(self) -> dict:
        return {'type': self.type, 'source': self.source, 'target': self.target,
                'amount': self.amount, 'unit': self.unit,
                'source_span': self.source_span, 'direction': f'{self.source}->{self.target}'}


_OWN_PRONOUNS = frozenset({'او', 'اوست', 'ایشان', 'آنها', 'her', 'his', 'him',
                           'them', 'their', 'she', 'he'})
_OWN_PRONOUN_MARK = '__own_pronoun__'


def extract_ownership_model(text: str) -> dict | None:
    """Full ownership state model: balances + ordered transfer events +
    query binding (sender / receiver / combined / difference)."""
    t = (text or '').translate(str.maketrans('۰۱۲۳۴۵۶۷۸۹٠١٢٣٤٥٦٧٨٩', '01234567890123456789'))
    balances: dict[str, float] = {}
    units: set[str] = set()

    def unit_of(seg: str) -> str:
        if re.search(r'دلار|dollars?|usd', seg, re.I):
            return 'USD'
        if re.search(r'یورو|euros?|eur', seg, re.I):
            return 'EUR'
        if re.search(r'تومان|tomans?', seg, re.I):
            return 'TOMAN'
        if re.search(r'ریال|rials?', seg, re.I):
            return 'RIAL'
        if re.search(r'پوند|pounds?|gbp', seg, re.I):
            return 'GBP'
        if re.search(r'کالا|قطعه|جنس|آیتم|items?|pieces?|products?', seg, re.I):
            return 'ITEM'
        return 'UNSPECIFIED'

    events: list[OwnershipEvent] = []
    claimed: list[tuple[int, int]] = []
    for pattern, groups in _OWNERSHIP_TRANSFERS:
        for m in re.finditer(pattern, t, re.I):
            if len(groups) == 3:
                gi, ai, ti = groups
            else:
                gi, ai = groups   # implicit-receiver form
                ti = None
            if any(m.group(i) is None for i in (gi, ai)) or (ti is not None and m.group(ti) is None):
                continue
            sender = _clean_name(m.group(gi))
            receiver = _clean_name(m.group(ti)) if ti is not None else _OWN_PRONOUN_MARK
            if receiver != _OWN_PRONOUN_MARK:
                if not (_is_name(sender) and _is_name(receiver)) or sender == receiver:
                    continue
            elif not _is_name(sender):
                continue
            if any(m.start() < ce and cs < m.end() for cs, ce in claimed):
                continue
            claimed.append((m.start(), m.end()))
            events.append(OwnershipEvent(
                'transfer', sender, receiver, float(m.group(ai)),
                unit_of(m.group(0)), m.start(), m.end(), m.group(0)))
            units.add(unit_of(m.group(0)))
    # balances are extracted AFTER transfer spans are known so a transfer
    # amount ('Ali gives Sara 10.') can never masquerade as a balance.
    for pattern in _OWNERSHIP_BALANCES + [_OWNERSHIP_BARE_BALANCE]:
        for m in re.finditer(pattern, t, re.I):
            name = _clean_name(m.group(1))
            if not _is_name(name) or name in balances:
                continue
            if any(m.start() < ce and cs < m.end() for cs, ce in claimed):
                continue   # number belongs to a transfer event, not a balance
            balances[name] = float(m.group(2))
            units.add(unit_of(m.group(0)))
    if not events or not balances:
        return None
    # pronoun anaphora — runs AFTER balances exist: 'به او منتقل می‌کند' /
    # implicit receiver resolves to the OTHER balance holder (spec §26).
    for e in events:
        if e.target in _OWN_PRONOUNS or e.target == _OWN_PRONOUN_MARK:
            others = [n for n in balances if n != e.source]
            if len(others) == 1:
                e.target = others[0]
        if e.source in _OWN_PRONOUNS:
            others = [n for n in balances if n != e.target]
            if len(others) == 1:
                e.source = others[0]

    # currency safety: mixed units in one closed system are refused (spec §30);
    # 'UNSPECIFIED' means the sentence never names a unit and stays safe.
    concrete_units = {u for u in units if u != 'UNSPECIFIED'}
    if len(concrete_units) > 1:
        return {'conflict': 'ownership_unit_conflict', 'units': sorted(concrete_units),
                'balances': balances,
                'events': [e.to_dict() for e in events]}
    # query binding (spec §28)
    q = _ownership_query(t, list(balances))
    if q is None:
        return None
    unit = sorted(concrete_units)[0] if concrete_units else 'UNSPECIFIED'
    return {'balances': balances, 'events': [e.to_dict() for e in events],
            'unit': unit, 'query': q}


def _ownership_query(t: str, names: list[str]) -> dict | None:
    name_alt = '|'.join(re.escape(n) for n in names) if names else _NAMES
    # combined total
    if re.search(r'(?:مجموع|جمع)[^؟?;؛]{0,20}(?:موجودی|پول|دلار|تومان)'
                 r'|together|in\s+total|combined|total\s+(?:money|balance|amount)', t, re.I):
        return {'kind': 'combined'}
    # explicit difference
    if re.search(r'اختلاف[^؟?;؛]{0,20}(?:موجودی|پول|دلار|تومان)|difference\s+(?:between|in)\s+'
                 r'(?:the\s+)?(?:balance|money)', t, re.I):
        m = re.search(rf'({_NAMES})\s+(?:و|and)\s+({_NAMES})', t)
        if m:
            return {'kind': 'difference', 'a': m.group(1).strip(' \u200c،,'),
                    'b': m.group(2).strip(' \u200c،,')}
        return {'kind': 'difference', 'a': names[0] if names else None,
                'b': names[1] if len(names) > 1 else None}
    # named balance — bind the ASKED entity, not the first owner (spec §28)
    for m in re.finditer(rf'(?:موجودی|سهم|پول|balance\s+of|how\s+much\s+(?:does|money\s+does))\s+'
                         rf'({_NAMES})', t, re.I):
        name = _clean_name(m.group(1))
        if _is_name(name):
            return {'kind': 'entity', 'entity': name}
    for m in re.finditer(rf'({_NAMES})(?:\'s|\u2019s)?\s+(?:balance|موجودی)', t, re.I):
        name = _clean_name(m.group(1))
        if _is_name(name):
            return {'kind': 'entity', 'entity': name}
    for m in re.finditer(rf'(?:how\s+much\s+does|چند\s*(?:دلار|تومان)\s+دارد)\s+({_NAMES})', t, re.I):
        name = _clean_name(m.group(1))
        if _is_name(name):
            return {'kind': 'entity', 'entity': name}
    for m in re.finditer(rf'({_NAMES})\s+چند\s+(?:دلار|تومان)\s+دارد', t, re.I):
        name = _clean_name(m.group(1))
        if _is_name(name):
            return {'kind': 'entity', 'entity': name}
    return None


def execute_ownership(model: dict) -> dict:
    """Execute ordered transfers with conservation + non-negativity invariants.
    Same-currency closed system conserves total funds (spec §54)."""
    balances = dict(model['balances'])
    # An entity that only APPEARS in a transfer (never stated a balance)
    # exists in the closed system with balance 0 — the event model defines
    # them, the conservation invariant still holds (spec §26/§29).
    for e in model['events']:
        balances.setdefault(e['source'], 0.0)
        balances.setdefault(e['target'], 0.0)
    total_before = sum(balances.values())
    for e in model['events']:
        src, tgt, amount = e['source'], e['target'], float(e['amount'])
        if amount < 0:
            return {'ok': False, 'reason': 'negative_transfer'}
        if src not in balances or tgt not in balances:
            return {'ok': False, 'reason': 'unknown_entity'}
        if balances[src] < amount:
            return {'ok': False, 'reason': 'impossible_transfer'}
        balances[src] -= amount
        balances[tgt] += amount
    total_after = sum(balances.values())
    if not math.isclose(total_before, total_after, rel_tol=1e-9, abs_tol=1e-9):
        return {'ok': False, 'reason': 'conservation_failure'}
    q = model.get('query') or {}
    if q.get('kind') == 'combined':
        return {'ok': True, 'value': total_after, 'balances': balances}
    if q.get('kind') == 'difference':
        a, b = q.get('a'), q.get('b')
        if a not in balances or b not in balances:
            return {'ok': False, 'reason': 'unknown_entity'}
        return {'ok': True, 'value': abs(balances[a] - balances[b]), 'balances': balances}
    entity = q.get('entity')
    if entity not in balances:
        return {'ok': False, 'reason': 'unknown_query_entity'}
    return {'ok': True, 'value': balances[entity], 'balances': balances}


# ======================================================================
# 9. INVENTORY EVENT MODEL (spec §32-§35)
# ======================================================================

INVENTORY_EVENT_VERBS = {
    'restock': (r'restock|replenish|رسید|تأمین|تامین|شارژ', +1),
    'purchase_in': (r'purchas\w*|خرید(?:اری)?\s*(?:شد)?', +1),
    'sale': (r'sell|sold|sale|فروش|فروخت', -1),
    'remove': (r'remove|بردار|حذف|خارج|کم', -1),
    'damage': (r'damag\w*|spoiled?|destroyed?|آسیب|خراب|ضایع', -1),
    'return': (r'return\w*|مرجوع|برگشت|بازگشت', +1),
    'correction': (r'correct\w*|اصلاح|تصحیح', 0),   # direction from verb context
    'add': (r'add\w*|اضافه|افزایش', +1),
    'subtract': (r'subtract\w*|decrease|کاهش', -1),
}


@dataclass
class InventoryEvent:
    event_id: str
    type: str
    direction: int
    quantity: float
    unit: str
    product: str
    start: int
    end: int
    source_span: str

    def to_dict(self) -> dict:
        return {'event_id': self.event_id, 'type': self.type,
                'direction': self.direction, 'quantity': self.quantity,
                'unit': self.unit, 'product': self.product,
                'source_span': self.source_span}


def _inventory_direction(text: str) -> int:
    POS = r'اضافه|زیاد|افزایش|واریز|دریافت|وارد|ورودی|رسید|مرجوع|برگشت|بازگشت|restock|arrive|return'
    NEG = r'کم|کاهش|برداشت|فروخت|فروش|آسیب|خراب|ضایع|خارج|خروجی|sold|sell|damag|spoiled|remove|subtract'
    pos = bool(re.search(POS, text, re.I))
    neg = bool(re.search(NEG, text, re.I))
    if pos == neg:
        return 0
    return +1 if pos else -1


def _inventory_event_type(clause: str) -> str:
    for etype, (pattern, _) in INVENTORY_EVENT_VERBS.items():
        if re.search(pattern, clause, re.I):
            return etype
    return 'add' if _inventory_direction(clause) > 0 else 'subtract'


# Persian function words that may never form a product/entity name.
_FA_FUNCTION_WORDS = frozenset({
    'و', 'است', 'که', 'را', 'از', 'به', 'برای', 'با', 'در', 'شد', 'شود',
    'دارد', 'کالا', 'قطعه', 'تا', 'هم', 'نیز', 'اما', 'هر', 'این', 'آن',
})


def extract_inventory_model(text: str) -> dict | None:
    """Typed inventory event model: event_id/type/direction/quantity/unit/
    source_span + product binding when multiple products exist (spec §33-35)."""
    t = (text or '').translate(str.maketrans('۰۱۲۳۴۵۶۷۸۹٠١٢٣٤٥٦٧٨٩', '01234567890123456789'))
    if not re.search(r'انبار|موجودی|warehouse|inventory|stock|کالا', t, re.I):
        return None
    # --- multi-product binding (spec §35) ---
    # product name = the word(s) immediately carrying the stock assertion;
    # 'Product A stock = 10' / 'موجودی محصول الف 10' / 'stock of A is 10'.
    products: dict[str, float] = {}

    def _register(candidate: str, value: float):
        candidate = re.sub(r'^(?:stock\s+of|موجودی\s+انبار|موجودی)\s+', '',
                           candidate.strip(' \u200c،,:='), flags=re.I).strip()
        # a product name may never be (or contain) a function word — the name
        # slot sits before 'موجودی' in Persian and routinely captures 'است و'.
        tokens = {tok.lower() for tok in re.split(r'[\s\u200c]+', candidate) if tok}
        if not candidate or (tokens & _FA_FUNCTION_WORDS):
            return
        if _is_name(candidate) and candidate.lower() not in (
                'inventory', 'stock', 'warehouse', 'the'):
            products.setdefault(candidate, value)

    # pass 1: 'X stock = N' / 'X موجودی N'  (name carries the assertion)
    for m in re.finditer(
            rf'({_NAMES}(?:\s+{_NAMES})?)\s*[:=]?\s*(?:stock|موجودی)[^0-9\n]{{0,14}}?'
            rf'(\d+(?:\.\d+)?)', t, re.I):
        _register(m.group(1), float(m.group(2)))
    # pass 2: 'stock of X N' / 'موجودی X N'
    for m in re.finditer(
            rf'(?:stock\s+of|موجودی)\s+({_NAMES}(?:\s+{_NAMES})?)[^0-9\n]{{0,14}}?'
            rf'(\d+(?:\.\d+)?)', t, re.I):
        _register(m.group(1), float(m.group(2)))
    if len(products) < 2:
        return None   # single-product chains stay on the v21 path (proven grammar)
    events: list[InventoryEvent] = []
    for m in re.finditer(r'(\d+(?:\.\d+)?)\s*(?:کالا|قطعه|items?|pieces?|units?)?\s*'
                         rf'(?:از|of|from)\s+({_NAMES}(?:\s+{_NAMES})?)', t, re.I):
        qty = float(m.group(1))
        name = m.group(2).strip(' \u200c،,')
        hit = None
        for pname in products:
            if pname == name or pname.endswith(name) or name.endswith(pname):
                hit = pname
                break
        if hit is None:
            continue
        clause = t[max(0, m.start() - 50):m.end() + 50]
        direction = _inventory_direction(clause)
        if direction == 0:
            continue
        events.append(InventoryEvent(
            f'event_{len(events)}', _inventory_event_type(clause), direction,
            qty, 'ITEM', hit, m.start(), m.end(), m.group(0)))
    if not events:
        return None
    return {'products': products, 'events': [e.to_dict() for e in events],
            'unit': 'ITEM'}


def execute_inventory_model(model: dict) -> dict:
    """Execute typed inventory events with order preservation + product binding."""
    products = dict(model['products'])
    for e in model['events']:
        name, qty, direction = e['product'], float(e['quantity']), int(e['direction'])
        if name not in products:
            return {'ok': False, 'reason': 'unknown_product'}
        products[name] = products[name] + direction * qty
        if products[name] < 0:
            return {'ok': False, 'reason': 'impossible_inventory'}
    q = _inventory_query(model.get('text', ''), list(products))
    if q is None:
        return {'ok': False, 'reason': 'query_unbound'}
    if q not in products:
        return {'ok': False, 'reason': 'unknown_query_product'}
    return {'ok': True, 'value': products[q], 'products': products,
            'query_product': q}


def _inventory_query(t: str, names: list[str]) -> str | None:
    """Bind the asked product; partial names resolve against known products."""
    def resolve(candidate: str) -> str | None:
        candidate = candidate.strip(' \u200c،,:?؟. ')
        for n in names:
            if n == candidate or n.endswith(candidate) or candidate.endswith(n):
                return n
        return None

    for m in re.finditer(rf'(?:stock\s+of|موجودی|inventario)\s+({_NAMES}(?:\s+{_NAMES})?)', t, re.I):
        n = resolve(m.group(1))
        if n:
            return n
    for m in re.finditer(rf'({_NAMES}(?:\s+{_NAMES})?)[^.؟?\n]{{0,20}}(?:چند|how\s+many|چند\s*تا)', t, re.I):
        n = resolve(m.group(1))
        if n:
            return n
    return names[0] if len(names) == 1 else None


# ======================================================================
# 10. TEMPORAL FRAME (spec §36-§40)
# ======================================================================

_START_CUES = re.compile(
    r'(?:شروع|شروع می شود|خروج|حرکت|زمان شروع|starts?|start(?:s)? at|begins?|leaves?|'
    r'departs?|opens?|departing|arrival is)', re.I)
_END_CUES = re.compile(
    r'(?:تمام|پایان|می رسد|ends?|end(?:s)? at|finish(?:es)?|arrives?|arrival|'
    r'be over)', re.I)
_DURATION_CUES = re.compile(
    r'(?:مدت(?:\s*کار)?|طول می کشد|طول می\u200cکشد|دیرش|lasts?|takes?|duration(?:\s+is)?|'
    r'for\s+\d+(?:\.\d+)?\s*(?:hours?|minutes?))', re.I)


@dataclass
class TemporalFrame:
    start_time: str | None = None       # 'HH:MM'
    end_time: str | None = None         # 'HH:MM'
    durations: list[tuple[float, str]] = field(default_factory=list)  # (value, unit)
    day_offset: int = 0
    query: str = 'end'                  # end | start

    def to_dict(self) -> dict:
        return {'start_time': self.start_time, 'end_time': self.end_time,
                'durations': [list(d) for d in self.durations],
                'day_offset': self.day_offset, 'query': self.query}


def extract_temporal_frame(text: str) -> TemporalFrame | None:
    """Generalized start/duration/end frame: a train, a shift, a meeting, a
    shop opening — any temporal evidence, not a fixed phrase list (spec §36)."""
    t = (text or '').translate(str.maketrans('۰۱۲۳۴۵۶۷۸۹', '0123456789'))
    if not (_START_CUES.search(t) or _END_CUES.search(t)):
        return None
    if not _DURATION_CUES.search(t) and not re.search(
            r'\d+(?:\.\d+)?\s*(?:ساعت|دقیقه|hours?|minutes?)', t, re.I):
        return None
    frame = TemporalFrame()
    # explicit end instant FIRST: 'ends at 22:00' binds 'at' to the END, so it
    # must never double as the start instant (spec §37: end+duration -> start).
    me = re.search(r'(?:ends?|finishes?|تمام می شود|تمام می\u200cشود|به پایان می رسد)\s*'
                   r'(?:ساعت|at)?\s*(\d{1,2})(?::(\d{2}))?', t, re.I)
    if me:
        frame.end_time = f'{int(me.group(1)):02d}:{int(me.group(2) or 0):02d}'
    # start instant: cued form (ساعت X / at X[:MM]) — but never the clock that
    # an END verb already claimed.
    def _start_candidate(mm: 're.Match') -> bool:
        left = t[max(0, mm.start() - 24):mm.start()]
        return not _END_CUES.search(left)
    m = None
    for cand in re.finditer(r'(?:ساعت|at|از)\s*(\d{1,2})(?::(\d{2}))?', t, re.I):
        if _start_candidate(cand):
            m = cand
            break
    if m:
        frame.start_time = f'{int(m.group(1)):02d}:{int(m.group(2) or 0):02d}'
    else:
        # bare clock token requires temporal frame evidence (spec §39) and a
        # frame whose end instant is not already claimed by an END verb.
        if frame.end_time is None:
            m = re.search(r'\b(\d{1,2}):(\d{2})\b', t)
            if m and (_DURATION_CUES.search(t) or _END_CUES.search(t)):
                frame.start_time = f'{int(m.group(1)):02d}:{int(m.group(2)):02d}'
    # secondary end form: 'at 23:00' AFTER the end verb ('تمام می شود ساعت 01')
    if frame.end_time is None:
        m = re.search(r'(?:ساعت|at)\s*(\d{1,2})(?::(\d{2}))?\s*(?:تمام|پایان|به پایان|ends?|finishes?)', t, re.I)
        if m:
            frame.end_time = f'{int(m.group(1)):02d}:{int(m.group(2) or 0):02d}'
    # durations
    for dm in re.finditer(
            r'(?:مدت(?:\s*کار)?|duration|دیرش)\D{0,12}?(\d+(?:\.\d+)?)\s*(ساعت|دقیقه|hours?|minutes?|hr|min)'
            r'|\b(\d+(?:\.\d+)?)\s*(hours?|minutes?|ساعت|دقیقه)\s*(?:طول|تمام|کشید|of\s*work|long)?'
            r'|\b(?:lasts?|takes?)\s+(\d+(?:\.\d+)?)\s*(hours?|minutes?)', t, re.I):
        value = dm.group(1) or dm.group(3) or dm.group(5)
        unit = dm.group(2) or dm.group(4) or dm.group(6)
        unit = str(unit).lower()
        frame.durations.append(
            (float(value), 'hour' if unit.startswith(('ساعت', 'hour', 'hr', 'h')) else 'minute'))
    if frame.end_time and not frame.start_time and frame.durations:
        frame.query = 'start'
    if (frame.start_time or frame.end_time) and frame.durations:
        return frame
    return None


# ======================================================================
# 11. Non-finite / domain invariants (spec §50-§52)
# ======================================================================

def numeric_guard(value) -> str | None:
    """Return a structured guard name when a value may never be presented:
    NaN / ±Infinity / out-of-domain probability."""
    try:
        v = float(value)
    except (TypeError, ValueError, OverflowError):
        return 'non_numeric_answer'
    if math.isnan(v):
        return 'non_finite_nan'
    if math.isinf(v):
        return 'non_finite_infinity'
    if not math.isfinite(v):
        return 'non_finite'
    return None


def probability_guard(p: float) -> str | None:
    try:
        v = float(p)
    except (TypeError, ValueError, OverflowError):
        return 'non_numeric_answer'
    if math.isnan(v) or math.isinf(v):
        return 'non_finite_probability'
    if not (0.0 <= v <= 1.0):
        return 'probability_range'
    return None


# ======================================================================
# 12. RATE ARITHMETIC — compound-dimension computation (spec §6/§7)
# ======================================================================

_TIME_HOURS = {'HOUR': 1.0, 'MINUTE': 1 / 60.0, 'SECOND': 1 / 3600.0, 'DAY': 24.0}
_TIME_SECONDS = {'HOUR': 3600.0, 'MINUTE': 60.0, 'SECOND': 1.0, 'DAY': 86400.0}
_TIME_DAYS = {'HOUR': 1 / 24.0, 'MINUTE': 1 / 1440.0, 'SECOND': 1 / 86400.0, 'DAY': 1.0}

# rate family -> base quantity the numerator measures, with fact-locked unit
# words per language (the renderer may never substitute another unit).
_RATE_BASE_UNITS = {
    'M_PER_SECOND': ('distance', 'متر', 'meters'),
    'KM_PER_HOUR': ('distance', 'کیلومتر', 'kilometers'),
    'MILE_PER_HOUR': ('distance', 'مایل', 'miles'),
    'USD_PER_HOUR': ('currency', 'دلار', 'dollars'),
    'USD_PER_DAY': ('currency', 'دلار', 'dollars'),
    'EUR_PER_HOUR': ('currency', 'یورو', 'euros'),
    'TOMAN_PER_HOUR': ('currency', 'تومان', 'toman'),
    'ITEM_PER_HOUR': ('count', 'کالا', 'items'),
    'ITEM_PER_DAY': ('count', 'کالا', 'items'),
    'ITEM_PER_MINUTE': ('count', 'کالا', 'items'),
    'USD_PER_ITEM': ('currency', 'دلار', 'dollars'),
}

_RATE_X_TIME_LANGUAGE = re.compile(
    r'×|\btimes?\b|ضرب|هزینه|دستمزد|حقوق|کرایه|cost|earn[s]?|charged|bill|invoice'
    r'|محاسبه|total|چند\s*(?:دلار|تومان|کالا|کیلومتر|متر)|how\s+much|how\s+many'
    r'|travel[s]?|moved?|covers?', re.I)
_RATE_ADD_LANGUAGE = re.compile(
    r'\+|با\s*هم\s*جمع|جمع\s*کن|جمع\s*می|add(?:ed)?\s+to|combined|plus|sum', re.I)


def _rate_numerator_dimension(normalized_unit: str) -> str:
    d = unit_dimension(normalized_unit)
    return d.numerator


def solve_rate(text: str) -> dict | None:
    """Typed rate arithmetic over compound dimensions:
      * rate × time -> base (5 USD/hour × 2 hours -> 10 USD)
      * rate + rate -> rate (10 USD/hour + 3 USD/hour -> 13 USD/hour)
      * mismatched rate families are REFUSED with structured metadata.
    Returns None when no unambiguous rate structure exists (never guesses)."""
    from jarvis.agent.quantity_v22 import extract_typed_quantities
    t = (text or '').translate(str.maketrans('۰۱۲۳۴۵۶۷۸۹٠١٢٣٤٥٦٧٨٩', '01234567890123456789'))
    quantities = extract_typed_quantities(t)
    rates = [q for q in quantities if q.dimension in (
        'currency_rate', 'count_rate', 'unit_price', 'speed')]
    times = [q for q in quantities if q.dimension == 'time']
    if not rates:
        return None

    def hours_of(q) -> float:
        return float(q.value) * _TIME_HOURS.get(q.unit, 1.0)

    # --- rate × time -> base ---------------------------------------------
    if len(rates) == 1 and len(times) == 1 and _RATE_X_TIME_LANGUAGE.search(t):
        rate, time_q = rates[0], times[0]
        rate_unit = rate.normalized_unit or rate.unit
        # duration expressed in the rate's own denominator scale
        if rate_unit == 'M_PER_SECOND':
            duration = float(time_q.value) * _TIME_SECONDS.get(time_q.unit, 1.0)
        elif rate_unit in ('USD_PER_DAY', 'ITEM_PER_DAY'):
            duration = float(time_q.value) * _TIME_DAYS.get(time_q.unit, 1.0)
        else:
            duration = float(time_q.value) * _TIME_HOURS.get(time_q.unit, 1.0)
        numerator, fa_unit, en_unit = _RATE_BASE_UNITS.get(
            rate_unit, (_rate_numerator_dimension(rate_unit), '', ''))
        value = float(rate.value) * duration
        return {'kind': 'rate_application', 'value': value,
                'rate': float(rate.value), 'rate_unit': rate_unit,
                'duration': duration, 'numerator': numerator,
                'fa_unit': fa_unit, 'en_unit': en_unit}

    # --- rate + rate -> rate (same family) or structured refusal ---------
    if len(rates) == 2 and _RATE_ADD_LANGUAGE.search(t):
        a, b = rates
        failure = validate_binary_operation('add', a, b)
        if failure is not None:
            return {'kind': 'rate_conflict', 'failure': failure.to_dict()}
        family = a.normalized_unit or a.unit
        return {'kind': 'rate_addition', 'value': float(a.value) + float(b.value),
                'rate_unit': family}
    return None
