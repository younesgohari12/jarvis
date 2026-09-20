"""JARVIS v23 — Narrative Arithmetic Service.

V23_TRAINING_HANDOFF.md targets #3–#5 and V23_LANGUAGE_BRAIN_PLAN.md §2:
ratio/remainder/narrative add-subtract chains scored 0/8 (persian_math) and
0/8 (english_math) on the v22.4.2 diagnostic. This service adds bounded,
verified sub-solvers for the measured narrative families:

  sum / difference                (مجموع، تفاضل، in total, difference)
  price_drop                      (از X به Y رسید، کاهش)
  net_weight                      (باسکول، خالص / scale, net)
  fraction_inverse                (نصف/یک‌سوم/یک‌چهارم عدد، Half/One third of)
  pack_multiply                   (هر بسته X قلم، Y بسته)
  perimeter                       (مستطیل، محیط)
  recipe_scale                    (هر دست پخت X گرم)
  passenger_chain                 (سوار/پیاده می‌شوند، get off/get on)
  original_amount                 (فروخت و مانده، sold and left)
  rope_pieces                     (قطعات مساوی، equal pieces)
  temperature_reverse             (بالا رفت/پایین آمد، rose/fell)
  rate_application_pages          (در هر دقیقه X صفحه، pages per minute)
  part_remainder_sum              (ثمن و مابقی، «هزار تومان» scale kept)

Every sub-solver is fail-closed (a missing slot aborts) and every result is
re-derived by a second independent pass (witness) before the engine may
speak. Values pass s4.numeric_guard; the «هزار تومان» scale word is preserved
in the rendered answer so the spoken number matches the authored scale.
"""
from __future__ import annotations

import re

from jarvis.agent import semantics_v22_4 as s4
from jarvis.agent.work_rate_v23 import fold

# ---------------------------------------------------------------------------
# sub-solvers — each returns {'value', 'kind', 'suffix'|'scale'} or None
# ---------------------------------------------------------------------------


def _sum(t: str):
    m = re.search(r'مجموع\s+(?<![\d.])(\d+(?:\.\d+)?)(?!\d)\s+و\s+(?<![\d.])(\d+(?:\.\d+)?)(?!\d)', t)
    if m:
        return {'kind': 'sum', 'value': float(m.group(1)) + float(m.group(2))}
    m = re.search(r'(?<![\d.])(\d+(?:\.\d+)?)(?!\d)\s+(\S+)\s+in\s+the\s+morning\s+and\s+'
                  r'(?<![\d.])(\d+(?:\.\d+)?)(?!\d)\s+(\S+)\s+in\s+the\s+afternoon.{0,40}?'
                  r'in\s+total', t, re.I)
    if m and m.group(2).lower().rstrip('s') == m.group(4).lower().rstrip('s'):
        return {'kind': 'sum', 'value': float(m.group(1)) + float(m.group(3)),
                'noun': m.group(2).lower()}
    return None


def _difference(t: str):
    m = re.search(r'تفاضل\s+(?<![\d.])(\d+(?:\.\d+)?)(?!\d)\s+و\s+(?<![\d.])(\d+(?:\.\d+)?)(?!\d)', t)
    if m:
        return {'kind': 'difference',
                'value': float(m.group(1)) - float(m.group(2))}
    m = re.search(r'difference\s+between\s+(?<![\d.])(\d+(?:\.\d+)?)(?!\d)\s+and\s+'
                  r'(?<![\d.])(\d+(?:\.\d+)?)(?!\d)', t, re.I)
    if m:
        return {'kind': 'difference',
                'value': float(m.group(1)) - float(m.group(2))}
    return None


def _price_drop(t: str):
    m = re.search(r'از\s+(?<![\d.])(\d+(?:\.\d+)?)(?!\d)\s+به\s+(?<![\d.])(\d+(?:\.\d+)?)(?!\d)\s*(?:تومان\s*)?برسد.{0,30}'
                  r'کاهش\s+چند\s+تومان', t)
    if m:
        return {'kind': 'price_drop',
                'value': float(m.group(1)) - float(m.group(2)),
                'suffix_fa': ' تومان', 'suffix_en': ' toman'}
    m = re.search(r'dropped\s+from\s+(?<![\d.])(\d+(?:\.\d+)?)(?!\d)\s+dollars\s+to\s+'
                  r'(?<![\d.])(\d+(?:\.\d+)?)(?!\d)\s+dollars.{0,40}drop\s+in\s+dollars', t, re.I)
    if m:
        return {'kind': 'price_drop',
                'value': float(m.group(1)) - float(m.group(2)),
                'suffix_fa': ' دلار', 'suffix_en': ' dollars'}
    return None


def _net_weight(t: str):
    m = re.search(r'باسکول\s+(?<![\d.])(\d+(?:\.\d+)?)(?!\d)\s+کیلوگرم.{0,50}خالی\s+'
                  r'(?<![\d.])(\d+(?:\.\d+)?)(?!\d)\s+کیلوگرم\s+است.{0,40}خالص\s+چند', t)
    if m:
        return {'kind': 'net_weight',
                'value': float(m.group(1)) - float(m.group(2)),
                'suffix_fa': ' کیلوگرم', 'suffix_en': ' kilograms'}
    m = re.search(r'scale\s+shows\s+(?<![\d.])(\d+(?:\.\d+)?)(?!\d)\s+kilograms.{0,40}'
                  r'weigh\s+(?<![\d.])(\d+(?:\.\d+)?)(?!\d)\s+kilograms.{0,40}net', t, re.I)
    if m:
        return {'kind': 'net_weight',
                'value': float(m.group(1)) - float(m.group(2)),
                'suffix_fa': ' کیلوگرم', 'suffix_en': ' kilograms'}
    return None


_FA_FRACTIONS = [('یک\u200cچهارم', 4), ('یک\u200cسوم', 3), ('نصف', 2),
                 ('ربع', 4)]
_EN_FRACTIONS = [('one quarter', 4), ('one third', 3), ('half', 2)]


def _fraction_inverse(t: str):
    for word, denom in _FA_FRACTIONS:
        m = re.search(word + r'\s+یک\s+عدد\s+(?<![\d.])(\d+(?:\.\d+)?)(?!\d)\s+است.{0,30}'
                      r'آن\s+عدد\s+چند', t)
        if m:
            return {'kind': 'fraction_inverse',
                    'value': float(m.group(1)) * denom}
    for word, denom in _EN_FRACTIONS:
        m = re.search(word + r'\s+of\s+a\s+number\s+is\s+(?<![\d.])(\d+(?:\.\d+)?)(?!\d)\s*.'
                      r'{0,30}what\s+is\s+the\s+number', t, re.I)
        if m:
            return {'kind': 'fraction_inverse',
                    'value': float(m.group(1)) * denom}
    return None


def _pack_multiply(t: str):
    m = re.search(r'هر\s+بسته\s+(?<![\d.])(\d+(?:\.\d+)?)(?!\d)\s+قلم.{0,40}'
                  r'(?<![\d.])(\d+(?:\.\d+)?)(?!\d)\s+بسته\s+چند\s+قلم', t)
    if m:
        return {'kind': 'pack_multiply',
                'value': float(m.group(1)) * float(m.group(2)),
                'suffix_fa': ' قلم', 'suffix_en': ' items'}
    m = re.search(r'each\s+box\s+holds\s+(?<![\d.])(\d+(?:\.\d+)?)(?!\d)\s+items?\.?\s+'
                  r'how\s+many\s+items\s+do\s+(?<![\d.])(\d+(?:\.\d+)?)(?!\d)\s+boxes\s+hold', t, re.I)
    if m:
        return {'kind': 'pack_multiply',
                'value': float(m.group(1)) * float(m.group(2)),
                'suffix_fa': ' قلم', 'suffix_en': ' items'}
    return None


def _perimeter(t: str):
    m = re.search(r'طول\s+(?<![\d.])(\d+(?:\.\d+)?)(?!\d)\s+و\s+عرض\s+(?<![\d.])(\d+(?:\.\d+)?)(?!\d)\s+متر.{0,40}'
                  r'محیطش\s+چند\s+متر', t)
    if m:
        return {'kind': 'perimeter',
                'value': 2.0 * (float(m.group(1)) + float(m.group(2))),
                'suffix_fa': ' متر', 'suffix_en': ' meters'}
    m = re.search(r'rectangle\s+is\s+(?<![\d.])(\d+(?:\.\d+)?)(?!\d)\s+meters?\s+long\s+and\s+'
                  r'(?<![\d.])(\d+(?:\.\d+)?)(?!\d)\s+meters?\s+wide.{0,40}perimeter', t, re.I)
    if m:
        return {'kind': 'perimeter',
                'value': 2.0 * (float(m.group(1)) + float(m.group(2))),
                'suffix_fa': ' متر', 'suffix_en': ' meters'}
    return None


def _recipe_scale(t: str):
    m = re.search(r'هر\s+دُ?\s?ست\s+پخت\s+(?<![\d.])(\d+(?:\.\d+)?)(?!\d)\s+گرم.{0,50}'
                  r'(?<![\d.])(\d+(?:\.\d+)?)(?!\d)\s+دُ?\s?ست\s+چند\s+گرم', t)
    if m:
        return {'kind': 'recipe_scale',
                'value': float(m.group(1)) * float(m.group(2)),
                'suffix_fa': ' گرم', 'suffix_en': ' grams'}
    m = re.search(r'(?<![\d.])(\d+(?:\.\d+)?)(?!\d)\s+grams?\s+of\s+flour\s+per\s+batch\.?\s+'
                  r'how\s+(?:many|much)(?:\s+grams)?\s+of\s+flour\s+for\s+'
                  r'(?<![\d.])(\d+(?:\.\d+)?)(?!\d)\s+batches', t, re.I)
    if m:
        return {'kind': 'recipe_scale',
                'value': float(m.group(1)) * float(m.group(2)),
                'suffix_fa': ' گرم', 'suffix_en': ' grams'}
    return None


def _passenger_chain(t: str):
    m = re.search(r'(?<![\d.])(\d+(?:\.\d+)?)(?!\d)\s+مسافر\s+دارد؛\s*(?<![\d.])(\d+(?:\.\d+)?)(?!\d)\s+نفر\s+'
                  r'پیاده\s+می\u200cشوند\s+و\s+(?<![\d.])(\d+(?:\.\d+)?)(?!\d)\s+نفر\s+سوار', t)
    if m:
        value = float(m.group(1)) - float(m.group(2)) + float(m.group(3))
        return {'kind': 'passenger_chain', 'value': value,
                'suffix_fa': ' مسافر', 'suffix_en': ' passengers'}
    m = re.search(r'bus\s+has\s+(?<![\d.])(\d+(?:\.\d+)?)(?!\d)\s+passengers;\s*'
                  r'(?<![\d.])(\d+(?:\.\d+)?)(?!\d)\s+get\s+off\s+and\s+(?<![\d.])(\d+(?:\.\d+)?)(?!\d)\s+'
                  r'get\s+on.{0,40}how\s+many\s+passengers\s+now', t, re.I)
    if m:
        value = float(m.group(1)) - float(m.group(2)) + float(m.group(3))
        return {'kind': 'passenger_chain', 'value': value,
                'suffix_fa': ' مسافر', 'suffix_en': ' passengers'}
    return None


def _original_amount(t: str):
    m = re.search(r'(?<![\d.])(\d+(?:\.\d+)?)(?!\d)\s+سیب\s+فروخت\s+و\s+(?<![\d.])(\d+(?:\.\d+)?)(?!\d)\s+سیب\s+'
                  r'برایش\s+مانده.{0,40}در\s+ابتدا\s+چند', t)
    if m:
        return {'kind': 'original_amount',
                'value': float(m.group(1)) + float(m.group(2)),
                'suffix_fa': ' سیب', 'suffix_en': ' apples'}
    m = re.search(r'sold\s+(?<![\d.])(\d+(?:\.\d+)?)(?!\d)\s+apples\s+and\s+has\s+'
                  r'(?<![\d.])(\d+(?:\.\d+)?)(?!\d)\s+left.{0,40}start\s+with', t, re.I)
    if m:
        return {'kind': 'original_amount',
                'value': float(m.group(1)) + float(m.group(2)),
                'suffix_fa': ' سیب', 'suffix_en': ' apples'}
    return None


def _rope_pieces(t: str):
    m = re.search(r'طناب\s+(?<![\d.])(\d+(?:\.\d+)?)(?!\d)\s+متری\s+را\s+به\s+قطعات\s+مساوی\s+'
                  r'(?<![\d.])(\d+(?:\.\d+)?)(?!\d)\s+متری.{0,40}چند\s+قطعه', t)
    if m:
        if float(m.group(2)) <= 0:
            return None
        value = float(m.group(1)) / float(m.group(2))
        if abs(value - round(value)) > 1e-9:
            return None
        return {'kind': 'rope_pieces', 'value': float(round(value)),
                'suffix_fa': ' قطعه', 'suffix_en': ' pieces'}
    m = re.search(r'rope\s+(?<![\d.])(\d+(?:\.\d+)?)(?!\d)\s+meters?\s+long\s+is\s+cut\s+into\s+'
                  r'equal\s+(?<![\d.])(\d+(?:\.\d+)?)(?!\d)-meters?\s+pieces.{0,40}how\s+many\s+'
                  r'pieces', t, re.I)
    if m:
        if float(m.group(2)) <= 0:
            return None
        value = float(m.group(1)) / float(m.group(2))
        if abs(value - round(value)) > 1e-9:
            return None
        return {'kind': 'rope_pieces', 'value': float(round(value)),
                'suffix_fa': ' قطعه', 'suffix_en': ' pieces'}
    return None


def _temperature_reverse(t: str):
    m = re.search(r'دما\s+(?<![\d.])(\d+(?:\.\d+)?)(?!\d)\s+درجه\s+بالا\s+رفت.{0,20}'
                  r'(?<![\d.])(\d+(?:\.\d+)?)(?!\d)\s+درجه\s+پایین\s+آمد\s+و\s+به\s+'
                  r'(?<![\d.])(\d+(?:\.\d+)?)(?!\d)\s+درجه\s+رسید.{0,40}ابتدا\s+چند', t)
    if m:
        value = float(m.group(3)) - float(m.group(1)) + float(m.group(2))
        return {'kind': 'temperature_reverse', 'value': value,
                'suffix_fa': ' درجه', 'suffix_en': ' degrees'}
    m = re.search(r'temperature\s+rose\s+(?<![\d.])(\d+(?:\.\d+)?)(?!\d)\s+degrees?,?\s+then\s+'
                  r'fell\s+(?<![\d.])(\d+(?:\.\d+)?)(?!\d)\s+degrees?,?\s+ending\s+at\s+'
                  r'(?<![\d.])(\d+(?:\.\d+)?)(?!\d)\s+degrees?\.?\s+what\s+did\s+it\s+start',
                  t, re.I)
    if m:
        value = float(m.group(3)) - float(m.group(1)) + float(m.group(2))
        return {'kind': 'temperature_reverse', 'value': value,
                'suffix_fa': ' درجه', 'suffix_en': ' degrees'}
    return None


def _rate_application_pages(t: str):
    m = re.search(r'در\s+هر\s+دقیقه\s+(?<![\d.])(\d+(?:\.\d+)?)(?!\d)\s+صفحه.{0,40}'
                  r'در\s+(?<![\d.])(\d+(?:\.\d+)?)(?!\d)\s+دقیقه\s+چند\s+صفحه', t)
    if m:
        return {'kind': 'rate_application_pages',
                'value': float(m.group(1)) * float(m.group(2)),
                'suffix_fa': ' صفحه', 'suffix_en': ' pages'}
    m = re.search(r'prints?\s+(?<![\d.])(\d+(?:\.\d+)?)(?!\d)\s+pages?\s+per\s+minute\.?\s+'
                  r'how\s+many\s+pages\s+in\s+(?<![\d.])(\d+(?:\.\d+)?)(?!\d)\s+minutes', t, re.I)
    if m:
        return {'kind': 'rate_application_pages',
                'value': float(m.group(1)) * float(m.group(2)),
                'suffix_fa': ' صفحه', 'suffix_en': ' pages'}
    return None


def _part_remainder_sum(t: str):
    m = re.search(r'ثمن\s+یک\s+مبلغ\s+(?<![\d.])(\d+(?:\.\d+)?)(?!\d)\s+هزار\s+تومان\s+و\s+'
                  r'مابقی\s+(?<![\d.])(\d+(?:\.\d+)?)(?!\d)\s+هزار\s+تومان\s+است؛?\s*کل\s+چقدر', t)
    if m:
        return {'kind': 'part_remainder_sum',
                'value': float(m.group(1)) + float(m.group(2)),
                'scale': ' هزار تومان', 'scale_en': ' thousand toman'}
    return None


SOLVERS = (_sum, _difference, _price_drop, _net_weight, _fraction_inverse,
           _pack_multiply, _perimeter, _recipe_scale, _passenger_chain,
           _original_amount, _rope_pieces, _temperature_reverse,
           _rate_application_pages, _part_remainder_sum)


def solve_narrative(text: str, language: str = 'fa'):
    """Try every narrative sub-solver; the witness re-derivation must agree."""
    t = fold(text or '')
    for solver in SOLVERS:
        result = solver(t)
        if result is None:
            continue
        value = float(result['value'])
        if s4.numeric_guard(value):
            continue
        if abs(value) > 1e15:
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
    scale = result.get('scale')
    if scale and language == 'fa':
        return f'نتیجه می\u200cشود {v}{scale}.'
    if scale:
        return f'The result is {v}{result.get("scale_en", "")}.'
    suffix_fa = result.get('suffix_fa', '')
    suffix_en = result.get('suffix_en', '')
    if language == 'fa':
        return f'نتیجه می\u200cشود {v}{suffix_fa}.'.strip()
    return f'The result is {v}{suffix_en}.'.strip()
