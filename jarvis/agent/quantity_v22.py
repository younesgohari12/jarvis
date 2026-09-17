"""JARVIS v22 — Typed Quantity system (P0 correctness layer).

Every number the system touches becomes a structured object:

    Quantity(value=100, dimension='currency', unit='USD',
             entity='حساب', source_span='100 دلار', confidence=0.98)

Incompatible dimensions (USD + ITEM, KM + HOUR, ...) are NEVER silently
combined. This module is a pure audit layer: it changes no v21 behaviour,
it only gives the v22 verifier an independent witness.
"""
from __future__ import annotations
from dataclasses import dataclass, field, asdict
import re

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

DIMENSIONS = {
    'currency': {'USD', 'EUR', 'TOMAN', 'RIAL', 'GBP'},
    'count': {'ITEM', 'PERSON', 'PIECE', 'GENERIC'},
    'time': {'HOUR', 'MINUTE', 'DAY'},
    'duration': {'HOUR', 'MINUTE', 'DAY'},
    'distance': {'KM', 'M'},
    'speed': {'KM_PER_HOUR', 'M_PER_S'},
    'mass': {'KG', 'G'},
    'volume': {'L', 'ML'},
    'temperature': {'C', 'F'},
    'percentage': {'PERCENT'},
    'probability': {'PROB'},
    'age': {'YEAR'},
    'date': {'DATE'},
    'clock_time': {'OCLOCK'},
    'identifier': {'ID'},
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

    def to_dict(self):
        d = asdict(self)
        d['value'] = float(self.value)
        return d


def _to_ascii(text: str) -> str:
    return text.translate(PERSIAN_DIGITS)


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
    """
    t = _to_ascii(text or '')
    out: list[Quantity] = []
    seen: list[tuple[int, int]] = []
    for m in re.finditer(r'\d+(?:\.\d+)?', t):
        start, end = m.start(), m.end()
        if any(s <= start < e for s, e in seen):
            continue
        value = float(m.group())
        left = t[max(0, start - 60):start]
        right = t[end:end + 60]
        window = t[max(0, start - 60):end + 60]
        immediate = right.lstrip()[:14]

        # --- identifiers are never quantities (package codes, record ids) ---
        if re.search(r'(?:کد(?:\s*بسته)?|شناسه(?:\s*پرونده)?|شمارهٔ?\s*پرونده|record\s*id(?:\s+is)?|package\s*code|identifier(?:\s+is)?)\s*:?\s*$', left, re.I):
            out.append(Quantity(value, 'identifier', 'ID', source_span=m.group(), confidence=0.9,
                                role='identifier', start=start, end=end))
            seen.append((start, end))
            continue

        # --- immediate suffix wins: the token glued to the number is its unit ---
        if re.match(r'(?:دلار\s*آمریکا|دلار|dollars?|usd\b|یورو|euros?|eur\b|تومان|toman\b|ریال|rial\b|پوند|pounds?\b|gbp\b)', immediate, re.I):
            cur = _currency_of(immediate)
            entity = 'account' if re.search(ACCOUNT_CUE, window, re.I) else (
                'inventory' if re.search(INVENTORY_CUE, window, re.I) else '')
            out.append(Quantity(value, 'currency', cur, entity=entity, source_span=m.group(),
                                confidence=0.97, role='value', start=start, end=end))
            seen.append((start, end))
            continue
        if re.match(ITEM_CUE, immediate, re.I):
            out.append(Quantity(value, 'count', 'ITEM', source_span=m.group(), confidence=0.95,
                                role='value', start=start, end=end))
            seen.append((start, end))
            continue
        if re.match(PERSON_CUE, immediate, re.I):
            out.append(Quantity(value, 'count', 'PERSON', source_span=m.group(), confidence=0.95,
                                role='value', start=start, end=end))
            seen.append((start, end))
            continue
        if re.match(r'(?:درصد|%|percent)', immediate, re.I):
            out.append(Quantity(value, 'percentage', 'PERCENT', source_span=m.group(),
                                confidence=0.95, role='value', start=start, end=end))
            seen.append((start, end))
            continue

        # --- currency (window fallback) ---
        cur = _currency_of(left + ' ' + right[:20])
        if cur and not re.search(ITEM_CUE, right[:20], re.I):
            entity = 'account' if re.search(ACCOUNT_CUE, window, re.I) else (
                'inventory' if re.search(INVENTORY_CUE, window, re.I) else '')
            out.append(Quantity(value, 'currency', cur, entity=entity, source_span=m.group(),
                                confidence=0.95, role='value', start=start, end=end))
            seen.append((start, end))
            continue

        # --- clock time: ساعت 23 / at 23:00 (position, not duration) ---
        if re.search(r'(?:ساعت|at)\s*$', left.strip() + ' ') or re.search(r'^:\d{2}\b', right):
            if not re.search(r'مدت|duration|طی شد|در\s*$', left, re.I):
                hm = re.match(r'^(\d{1,2})(?::(\d{2}))?', t[end:])
                if hm and int(hm.group(1)) <= 23:
                    out.append(Quantity(value, 'clock_time', 'OCLOCK', source_span=m.group(),
                                        confidence=0.9, role='start', start=start, end=end))
                    seen.append((start, end))
                    continue

        # --- items / pieces ---
        if re.search(ITEM_CUE, window, re.I):
            out.append(Quantity(value, 'count', 'ITEM', source_span=m.group(), confidence=0.9,
                                role='value', start=start, end=end))
            seen.append((start, end))
            continue

        # --- people / workers ---
        if re.search(PERSON_CUE, window, re.I):
            out.append(Quantity(value, 'count', 'PERSON', source_span=m.group(), confidence=0.9,
                                role='value', start=start, end=end))
            seen.append((start, end))
            continue

        # --- percentage ---
        if re.search(r'درصد|%|percent', window, re.I):
            out.append(Quantity(value, 'percentage', 'PERCENT', source_span=m.group(),
                                confidence=0.9, role='value', start=start, end=end))
            seen.append((start, end))
            continue

        # --- distance / speed / duration windows are tagged only with cues ---
        if re.search(SPEED_CUE, window, re.I) and re.search(DISTANCE_CUE, window, re.I):
            out.append(Quantity(value, 'speed', 'KM_PER_HOUR', source_span=m.group(),
                                confidence=0.75, role='value', start=start, end=end))
            seen.append((start, end))
            continue

        out.append(Quantity(value, 'dimensionless', 'NONE', source_span=m.group(),
                            confidence=0.4, role='value', start=start, end=end))
        seen.append((start, end))
    return out


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
    """Return failed typed-check names for an add/subtract chain over `initial`."""
    failed: list[str] = []
    cur_dim = initial.dimension
    for op in operations:
        if op.get('op') not in ('add', 'subtract', 'inventory_add', 'inventory_remove',
                                'credit', 'debit'):
            continue
        vdim = _dim_of_operation_value(op, quantities)
        if vdim == 'dimensionless' or cur_dim == 'dimensionless':
            continue
        if not compatible(Quantity(cur_dim and 0 or 0, cur_dim, ''),
                          Quantity(0, vdim, '')):
            name = 'currency_dimension_mismatch' if 'currency' in (cur_dim, vdim) \
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
