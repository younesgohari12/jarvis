"""JARVIS v22 — Numeric Role V2.

v21 roles were coarse (initial/value/distractor). v22 gives every source
number an expressive role (workers_initial, workers_target, price, discount,
tax, inventory_add, ratio_a, probability, identifier, distractor, ...) and
checks that IR slots honour those roles — catching slot swaps like
workers_initial=8 when the text says '3 workers ... 8 workers ...'.
"""
from __future__ import annotations
import re
from jarvis.agent.quantity_v22 import _to_ascii, Quantity

# Expressive role taxonomy (spec §6).
ROLES = [
    'initial_value', 'target_value',
    'workers_initial', 'workers_target', 'hours_initial', 'hours_target',
    'output_initial', 'price', 'cost', 'revenue', 'discount', 'tax',
    'inventory_initial', 'inventory_add', 'inventory_remove',
    'ratio_a', 'ratio_b', 'total', 'n', 'k', 'p', 'sample_size',
    'sequence_term', 'start', 'duration', 'distance', 'speed_value',
    'age_value', 'age_difference', 'transfer_amount', 'identifier',
    'distractor', 'value',
]

ROLE_CHECK = 'numeric_role_consistency'
SLOT_CHECK = 'source_slot_consistency'


def close(a, b):
    import math
    try:
        return math.isclose(float(a), float(b), rel_tol=1e-8, abs_tol=1e-9)
    except (ValueError, TypeError, OverflowError):
        return False

_WORKERS_PAT = r'کارگر|workers?|نفر|people'
_HOURS_PAT = r'ساعت|hours?|hr\b'
_OUTPUT_PAT = r'تولید|خروجی|قطعه|واحد|جعبه|produces?|output|pieces?|units?|boxes?|items?|products?'


def classify_source_numbers(text: str) -> list[dict]:
    """Tag every source number with an expressive role + span.

    Covers ASCII/Persian digits AND spelled-out numbers (شش، هشت، Ten، five…).
    Purely span/context driven — never reads IR slots, neural labels or
    execution traces (independence from the parser interpretation).
    """
    t = _to_ascii(text or '')
    found = []
    claimed: list[tuple[int, int]] = []
    for m in re.finditer(r'\d+(?:\.\d+)?', t):
        claimed.append((m.start(), m.end()))
        left = t[max(0, m.start() - 70):m.start()]
        right = t[m.end():m.end() + 70]
        role, conf = _role_for(left, right, float(m.group()), t)
        found.append({'value': float(m.group()), 'start': m.start(), 'end': m.end(),
                      'span': m.group(), 'role': role, 'confidence': conf})
    # spelled-out numbers (digits already claimed are skipped)
    fa_units = {'صفر': 0, 'یک': 1, 'دو': 2, 'سه': 3, 'چهار': 4, 'پنج': 5, 'شش': 6,
                'هفت': 7, 'هشت': 8, 'نه': 9, 'ده': 10, 'یازده': 11, 'دوازده': 12,
                'سیزده': 13, 'چهارده': 14, 'پانزده': 15, 'شانزده': 16, 'هفده': 17,
                'هجده': 18, 'نوزده': 19, 'بیست': 20}
    fa_tens = {'سی': 30, 'چهل': 40, 'پنجاه': 50, 'شصت': 60, 'هفتاد': 70,
               'هشتاد': 80, 'نود': 90, 'صد': 100}
    en_words = {'zero': 0, 'one': 1, 'two': 2, 'three': 3, 'four': 4, 'five': 5,
                'six': 6, 'seven': 7, 'eight': 8, 'nine': 9, 'ten': 10, 'eleven': 11,
                'twelve': 12, 'thirteen': 13, 'fourteen': 14, 'fifteen': 15,
                'sixteen': 16, 'seventeen': 17, 'eighteen': 18, 'nineteen': 19,
                'twenty': 20, 'thirty': 30, 'forty': 40, 'fifty': 50, 'sixty': 60,
                'seventy': 70, 'eighty': 80, 'ninety': 90, 'hundred': 100}
    WORD_VALUES = {**{k: float(v) for k, v in fa_units.items()},
                   **{k: float(v) for k, v in fa_tens.items()},
                   **{k: float(v) for k, v in en_words.items()}}
    word_re = re.compile(
        r'\b(?:صفر|یک|دو|سه|چهار|پنج|شش|هفت|هشت|نه|ده|یازده|دوازده|سیزده|چهارده|پانزده|'
        r'شانزده|هفده|هجده|نوزده|بیست|سی|چهل|پنجاه|شصت|هفتاد|هشتاد|نود|صد|'
        r'zero|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|thirteen|'
        r'fourteen|fifteen|sixteen|seventeen|eighteen|nineteen|twenty|thirty|forty|fifty|'
        r'sixty|seventy|eighty|ninety|hundred)\b', re.I)
    for m in word_re.finditer(t):
        if any(s <= m.start() < e for s, e in claimed):
            continue
        word = m.group().lower()
        value = WORD_VALUES.get(word)
        if value is None:
            continue
        left = t[max(0, m.start() - 70):m.start()]
        right = t[m.end():m.end() + 70]
        role, conf = _role_for(left, right, value, t)
        found.append({'value': value, 'start': m.start(), 'end': m.end(),
                      'span': m.group(), 'role': role, 'confidence': conf})
    # sequence terms: a run of >=2 numbers separated only by separators
    found = _mark_sequence_terms(t, found)
    found.sort(key=lambda f: f['start'])
    return found


def _role_for(left: str, right: str, value: float, full: str) -> tuple[str, float]:
    if re.search(r'(?:کد(?:\s*بسته)?|شناسه(?:\s*پرونده)?|شمارهٔ?\s*پرونده|record\s*id(?:\s+is)?|package\s*code|identifier(?:\s+is)?)\s*:?\s*$', left, re.I):
        return 'identifier', 0.92

    # --- immediate suffix wins: the token directly after the number ---
    immediate = right.lstrip()[:10]
    if re.match(_HOURS_PAT, immediate, re.I):
        return 'hours', 0.88
    if re.match(r'(?:دقیقه|minutes?|min\b|روز|days?|دوم|seconds?|sec\b)', immediate, re.I):
        return 'hours', 0.85  # duration family (may be minute-denominated)
    if re.match(_WORKERS_PAT, immediate, re.I):
        return 'workers', 0.88
    if re.match(r'(?:واحد|قطعه|جعبه|pieces?|units?\b|products?|boxes?|items?|output)', immediate, re.I):
        return 'output', 0.88

    # --- window-based fallbacks ---
    if re.search(r'درصد|%|percent', left + right, re.I) and \
            re.search(r'احتمال|probability|شانس|chance', full, re.I):
        return 'p', 0.9
    if re.search(r'درصد|%|percent', left + right, re.I):
        return 'discount' if re.search(r'تخفیف|discount', left + right, re.I) else 'value', 0.7
    if re.search(_WORKERS_PAT, left, re.I) or re.search(_WORKERS_PAT, right[:30], re.I):
        return 'workers', 0.8
    if re.search(r'مدت|duration', left, re.I) and re.search(_HOURS_PAT, left + right, re.I):
        return 'duration', 0.85
    if re.search(_OUTPUT_PAT, left + right, re.I):
        return 'output', 0.7
    if re.search(r'سرعت|speed', left + right, re.I):
        return 'speed_value', 0.8
    if re.search(r'سال|ساله|aged|years?[- ]?old', left + right, re.I):
        return 'age_value', 0.8
    if re.search(r'بزرگتر|کوچکتر|older|younger', left + right, re.I):
        return 'age_difference', 0.8
    if re.search(r'انتقال|transfer|می دهد|می‌دهد', full, re.I):
        return 'transfer_amount', 0.7
    return 'value', 0.4


def _mark_sequence_terms(t: str, found: list[dict]) -> list[dict]:
    seq = re.search(r'(?:دنباله|sequence)\s+([^؛;؟?\.]+)', t, re.I)
    if not seq:
        return found
    body = seq.group(1)
    for m in re.finditer(r'\d+(?:\.\d+)?', body):
        abs_start = seq.start(1) + m.start()
        for f in found:
            if f['start'] == abs_start:
                f['role'] = 'sequence_term'
                f['confidence'] = 0.9
    return found


def upgrade_roles(source_numbers: list[dict], text: str = '') -> list[dict]:
    """Assign *_initial / *_target using clause semantics, not raw order:

    the clause that ASSERTS the known output ('... خروجی 84 قطعه است.') holds
    the initial side; the INTERROGATIVE clause ('چند/چه خروجی...؟') holds the
    target side. Falls back to source order when clause type is unclear.
    """
    clauses = [c for c in re.split(r'[؟?!؛;\.\n]', text or '') if c.strip()]
    INTERROG = re.compile(r'چه\s|چند\s|how\s+many|what\s', re.I)

    def clause_of(pos: int) -> str:
        acc = 0
        for c in clauses:
            acc_next = acc + len(c) + 1
            if acc <= pos < acc_next:
                return c
            acc = acc_next
        return text or ''

    out = []
    counts: dict[str, int] = {}
    workers_roles: dict[int, str] = {}
    for i, f in enumerate(source_numbers):
        if f['role'] == 'workers':
            clause = clause_of(f['start'])
            if INTERROG.search(clause):
                workers_roles[i] = 'workers_target'
    # second pass: fill remaining workers pairwise in order (initial, target)
    pending = [i for i, f in enumerate(source_numbers) if f['role'] == 'workers' and i not in workers_roles]
    for j, i in enumerate(pending):
        workers_roles[i] = 'workers_target' if j % 2 == 1 else 'workers_initial'
    for i, f in enumerate(source_numbers):
        role = f['role']
        if role == 'workers':
            role = workers_roles.get(i, 'workers_initial')
        elif role == 'hours':
            clause = clause_of(f['start'])
            if INTERROG.search(clause):
                role = 'hours_target'
            else:
                role = 'hours_initial' if counts.get('hours_i', 0) == 0 else 'hours_target'
                if role == 'hours_initial':
                    counts['hours_i'] = 1
        elif role == 'output':
            clause = clause_of(f['start'])
            if INTERROG.search(clause):
                role = 'output_target'
            else:
                role = 'output_initial' if counts.get('out_i', 0) == 0 else 'output_target'
                if role == 'output_initial':
                    counts['out_i'] = 1
        out.append({**f, 'role': role})
    return out


def classify_source_numbers_v2(text: str) -> list[dict]:
    return upgrade_roles(classify_source_numbers(text), text)


def role_slot_consistency(ir_dict: dict, source_numbers: list[dict]) -> dict:
    """Independent check: IR slots must honour the expressive source roles.

    Returns {'failed_checks': [...], 'details': {...}} — never mutates IR.
    """
    failed: list[str] = []
    details: dict = {}
    slots = ir_dict.get('slots', {}) or {}
    if ir_dict.get('task') == 'work_rate':
        workers_initial_src = [f for f in source_numbers if f.get('role') == 'workers_initial']
        workers_target_src = [f for f in source_numbers if f.get('role') == 'workers_target']
        if workers_initial_src and workers_target_src and 'workers_initial' in slots and 'workers_target' in slots:
            wi, wt = float(slots['workers_initial']), float(slots['workers_target'])
            first, second = float(workers_initial_src[0]['value']), float(workers_target_src[0]['value'])
            ok = close(wi, first) and close(wt, second)
            details['workers_order'] = {'slots': [wi, wt], 'source': [first, second]}
            if not ok:
                failed.append(ROLE_CHECK)
    # every non-probability slot value must be traceable to some source span
    # (hour<->minute converted slots are accepted when the source uses minutes)
    traceable_misses = []
    source_vals = [float(f['value']) for f in source_numbers if f.get('role') != 'distractor']
    minute_source = bool(re.search(r'دقیقه|minute', source_numbers and ir_dict.get('source_text', '') or '', re.I)) \
        if isinstance(ir_dict.get('source_text'), str) else False
    for key, val in slots.items():
        if isinstance(val, (int, float)) and not isinstance(val, bool):
            if key in ('p',):
                continue
            ok = any(close(float(val), sv) for sv in source_vals)
            if not ok and minute_source:
                ok = any(close(float(val) * 60, sv) or close(float(val) / 60, sv) for sv in source_vals)
            if source_vals and not ok:
                traceable_misses.append(key)
    if traceable_misses:
        failed.append(SLOT_CHECK)
        details['untraceable_slots'] = traceable_misses
    return {'failed_checks': failed, 'details': details}
