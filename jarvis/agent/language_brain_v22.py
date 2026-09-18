"""JARVIS v22 — Language Brain V2.

Separation of concerns (spec §12): the Language Model understands and speaks;
JARVIS reasons. This module therefore owns ONLY:
  * NLU normalisation  — informal Persian/English -> parseable text
  * NLG rendering      — verified results spoken naturally (never robotic)
  * clarification      — abstention messages that ask the right question
It never replaces JARVIS reasoning and never invents numeric content.
"""
from __future__ import annotations
import re

PERSIAN_DIGITS = str.maketrans('۰۱۲۳۴۵۶۷۸۹٠١٢٣٤٥٦٧٨٩', '01234567890123456789')

_CHAR_MAP = {'ي': 'ی', 'ك': 'ک', 'ۀ': 'هٔ', 'أ': 'ا', 'إ': 'ا', 'ؤ': 'و', 'ة': 'ه'}

# Bounded colloquial -> formal map (safe retry vocabulary only).
COLLOQUIAL = [
    (r'\bمیخوام\b', 'می‌خواهم'), (r'\bمی خوام\b', 'می‌خواهم'), (r'\bمیخوای\b', 'می‌خواهی'),
    (r'\bمیکن\b', 'می‌کنم'), (r'\bمیکنم\b', 'می‌کنم'), (r'\bمیشه\b', 'می‌شود'),
    (r'\bمیشه\b', 'می‌شود'), (r'\bچندتا\b', 'چند'), (r'\bچند تا\b', 'چند'),
    (r'\bچنده\b', 'چند است'), (r'\bچقدره\b', 'چقدر است'), (r'\bچند میشه\b', 'چند می‌شود'),
    (r'\bتعدادش\b', 'تعداد آن'), (r'\bموجودیش\b', 'موجودی آن'),
    (r'\bبده\b', 'بده'), (r'\bبنویس\b', 'بنویس'), (r'\bحساب کن\b', 'حساب کن'),
]

_ROBOTIC = [
    r'^بر اساس محاسبات انجام[\u200c\s]*شده[،,:\s]*',
    r'^با توجه به محاسبات[،,:\s]*',
    r'^based on the calculations[,\s]*',
    r'^after performing the calculations[,\s]*',
]


def normalize_chars(text: str) -> str:
    out = text or ''
    for src, dst in _CHAR_MAP.items():
        out = out.replace(src, dst)
    return out.translate(PERSIAN_DIGITS)


def normalize_nlu(text: str) -> str:
    """One bounded normalisation retry for informal input. Never rewrites
    numbers, never reorders words, never touches code/identifiers."""
    t = normalize_chars(text or '').strip()
    t = re.sub(r'[\u200c]', '\u200c', t)
    for pattern, replacement in COLLOQUIAL:
        t = re.sub(pattern, replacement, t, flags=re.I)
    t = _paraphrase_rewrite(t)
    t = re.sub(r'\s+', ' ', t).strip()
    t = re.sub(r'\s+([،؛؟?!])', r'\1', t)
    return t


# ----------------------------------------------------------------------
# Bounded paraphrase rewriter: maps common natural phrasings onto the
# canonical verified grammar. Fires ONLY on the v22 NLU retry path (the
# first solve attempt always sees the user's original wording).
_PARAPHRASE = [
    # Persian counting: several natural phrasings -> canonical combination form
    (r'از\s*(\d+)\s*(?:نفر|دانش\s*آموز|دانش‌آموز|worker|کارگر)\s*(?:چگونه\s*می[\s\u200c]*توان|چطور\s*می[\s\u200c]*شود|چند\s*گروه)\s*'
     r'(\d+)\s*نفره(?:\s*گروه)?\s*(?:انتخاب\s*کرد|ساخت|تشکیل\s*داد|درست\s*کرد)?',
     'به چند روش می\u200cتوان \\2 نفر از \\1 نفر را انتخاب کرد؟'),
    (r'تعداد\s*ترکیب\s*های?\s*(\d+)\s*نفره(?:\s*از)?\s*(\d+)\s*نفر',
     'به چند روش می\u200cتوان \\1 نفر از \\2 نفر را انتخاب کرد؟'),
    # English split phrasings -> canonical ratio form
    (r'[Ss]plit\s+([\d.]+)\s+into\s+two\s+parts\s+in\s+the\s+ratio\s+(\d+)\s*(?::|to)\s*(\d+)',
     r'Divide \1 in the ratio \2:\3'),
    (r'[Dd]ivide\s+([\d.]+)\s+between\s+two\s+parts\s+in\s+the\s+ratio\s+(\d+)\s*(?::|to)\s*(\d+)',
     r'Divide \1 in the ratio \2:\3'),
    # English ownership tail -> canonical witness form
    (r'[Ww]hat\s+is\s+(\w+)(?:\u2019|\')s\s+balance[^?]*\?', r'Balance of \1?'),
    # Persian shift wording -> canonical scheduling wording
    (r'شیفت\s*کاری', 'کار'),
    (r'پایان\s*شیفت', 'پایان کار'),
    (r'و\s*(\d+(?:\.\d+)?)\s*ساعت\s*طول\s*می[\s\u200c]*کشد', r'؛ مدت کار \1 ساعت است'),
]
# v22.1 P0 fix: the two former age-difference rewrites ("موجودی X است؛ Y کم کن"
# / "start with X; subtract Y") were REMOVED — they re-titled an age task as
# an inventory/finance chain and produced 35−57 = −22 verified against the
# rewrite itself. Age differences are now computed by the dedicated
# entity-bound operation in LocalIntelligenceV22._age_difference_answer,
# verified with abs(age_a − age_b) against the immutable source.


def _paraphrase_rewrite(text: str) -> str:
    for pattern, replacement in _PARAPHRASE:
        new = re.sub(pattern, replacement, text, flags=re.I)
        if new != text:
            text = new
            break  # one bounded rewrite per retry
    return text


# ----------------------------------------------------------------------
# v22.1 — age-difference detection (independent, span-based).
# Reads ONLY the immutable user text: exactly two distinct named ages plus
# difference language -> the entity-bound abs(a−b) operation may answer.
AGE_DIFFERENCE_LANGUAGE = re.compile(
    r'اختلاف\s*سن|چند\s*سال\s*(?:بزرگ|کوچک)[\s\u200c]*تر'
    r'|چند\s*سال[\s\u200c]*(?:متفاوت|فرق)'
    r'|age\s+difference|how\s+many\s+years\s+(?:older|younger)'
    r'|what\s+is\s+the\s+age\s+difference', re.I)

_AGE_FA_SALEH = re.compile(r'([\u0600-\u06FF]+)\s+(\d+)\s*ساله')
_AGE_FA_SEN = re.compile(r'سن\s+([\u0600-\u06FF]+)\s+(\d+)')
_AGE_EN_IS = re.compile(r'([A-Za-z\u0600-\u06FF]+)\s+is\s+(\d+)(?:\s+years?\s+old)?')


def detect_age_difference(text: str):
    """Return ((name_a, age_a), (name_b, age_b)) when the immutable source
    asks an age-difference question about exactly two distinct named ages;
    otherwise None. Never invents or reorders values."""
    t = normalize_chars(text or '')
    if not AGE_DIFFERENCE_LANGUAGE.search(t):
        return None
    ages: list[tuple[str, float]] = []
    seen_names: set[str] = set()
    for pattern in (_AGE_FA_SALEH, _AGE_FA_SEN, _AGE_EN_IS):
        for m in pattern.finditer(t):
            name = m.group(1).strip('\u200c ').strip()
            value = float(m.group(2))
            if not name or name in seen_names or not (0 <= value <= 150):
                continue
            seen_names.add(name)
            ages.append((name, value))
        if len(ages) >= 2:
            break
    if len(ages) != 2:
        return None
    return (ages[0], ages[1])


# ----------------------------------------------------------------------
# v22.1 — clock-token normalization for the PARSE INPUT only.
# 'ساعت 23:30' -> 'ساعت 23.5' so the v21 parser can read minute-precision
# starts. The immutable user source itself is never rewritten; every
# witness (typed audit, temporal witness) keeps reading the original.
_CLOCK_TOKEN = re.compile(r'(\d{1,2}):(\d{2})')
_SCHED_LANGUAGE = re.compile(
    r'شروع|پایان|تمام|مدت|کار|شیفت|duration|lasts?|shift|start|end|finish', re.I)


def normalize_clock_tokens(text: str) -> str:
    t = normalize_chars(text or '')
    if not _SCHED_LANGUAGE.search(t):
        return text or ''

    def _repl(m: 're.Match') -> str:
        h, mm = int(m.group(1)), int(m.group(2))
        left = t[max(0, m.start() - 14):m.start()]
        if h <= 23 and mm < 60 and re.search(r'(?:ساعت|at|از)\s*$', left, re.I):
            dec = h + mm / 60.0
            return f'{dec:.10g}'.rstrip('0').rstrip('.') if mm else str(h)
        return m.group(0)

    out = _CLOCK_TOKEN.sub(_repl, t)
    return out


def render_age_difference(older_name: str, value: float, language: str = 'fa') -> str:
    v = fmt_number(value)
    if language == 'fa':
        return f'اختلاف سن {v} سال است.'
    return f'The age difference is {v} years.'


def fmt_number(v, language: str = 'fa') -> str:
    """ASCII-digit number formatting (10 significant digits, v21 compatible)."""
    f = float(v) if not isinstance(v, float) else v
    if f.is_integer() and abs(f) < 1e15:
        return str(int(f))
    return f'{f:.10g}'


def sanitize_response(text: str) -> str:
    """Post-generation hygiene: no degenerate repetition, no robotic openers,
    no double punctuation. Applied to every rendered response."""
    t = (text or '').strip()
    # collapse immediate word repetitions (است است است -> است)
    prev = None
    while prev != t:
        prev = t
        t = re.sub(r'(\S+)(\s+\1)+', r'\1', t, flags=re.I)
    for pattern in _ROBOTIC:
        t = re.sub(pattern, '', t, flags=re.I)
    t = re.sub(r'([،,؛;])\s*\1+', r'\1', t)
    t = re.sub(r'\s+', ' ', t).strip()
    if t and t[-1] not in '.,!?:؟،؛':
        t += '.'
    return t


# ----------------------------------------------------------------------
# Natural rendering — templates per task family.
# ----------------------------------------------------------------------
_UNIT_FA = {'USD': 'دلار', 'EUR': 'یورو', 'TOMAN': 'تومان', 'RIAL': 'ریال', 'GBP': 'پوند'}
_UNIT_EN = {'USD': 'dollars', 'EUR': 'euros', 'TOMAN': 'toman', 'RIAL': 'rial', 'GBP': 'pounds'}


def _currency_word(qs: list, language: str) -> str:
    for q in qs:
        if q.get('dimension') == 'currency':
            return (_UNIT_FA if language == 'fa' else _UNIT_EN).get(q.get('unit'), '')
    return ''


def render_word_problem(value, ir_dict: dict, language: str = 'fa', mode: str = 'concise') -> str:
    """Speak a verified numeric answer naturally. Falls back to '' when no
    clean template applies (caller keeps the v21 canonical rendering then)."""
    task = ir_dict.get('task')
    qs = ir_dict.get('quantities') if 'quantities' in ir_dict else []
    v = fmt_number(value, language)
    if language == 'fa':
        if task == 'inventory':
            return f'نتیجه می\u200cشود {v} کالا.'
        if task == 'finance':
            unit = _currency_word(qs, 'fa')
            if task and unit and ir_dict.get('slots', {}).get('initial') is not None:
                return f'موجودی جدید می\u200cشود {v} {unit}.'
            return f'نتیجه می\u200cشود {v}.'
        if task == 'work_rate':
            unit = 'قطعه' if ir_dict.get('units', {}).get('output') == 'piece' else 'واحد'
            return f'نتیجه می\u200cشود {v} {unit}.'
        if task == 'ratio':
            if isinstance(value, (list, tuple)) and len(value) == 2:
                return f'هر بخش می\u200cشود {fmt_number(value[0])} و {fmt_number(value[1])}.'
            return ''
        if task == 'speed':
            return f'سرعت می\u200cشود {v} کیلومتر بر ساعت.'
        if task == 'age':
            return f'نتیجه می\u200cشود {v} سال.'
        if task == 'sequence':
            return f'جملهٔ خواسته\u200cشده می\u200cشود {v}.'
        return ''
    # English
    if task == 'inventory':
        return f'The result is {v} items.'
    if task == 'finance':
        unit = _currency_word(qs, 'en')
        if unit and ir_dict.get('slots', {}).get('initial') is not None:
            return f'The new balance is {v} {unit}.'
        return f'The result is {v}.'
    if task == 'work_rate':
        unit = 'pieces' if ir_dict.get('units', {}).get('output') == 'piece' else 'units'
        return f'The output is {v} {unit}.'
    if task == 'ratio':
        if isinstance(value, (list, tuple)) and len(value) == 2:
            return f'The parts are {fmt_number(value[0])} and {fmt_number(value[1])}.'
        return ''
    if task == 'speed':
        return f'The speed is {v} km/h.'
    if task == 'age':
        return f'The result is {v} years.'
    if task == 'sequence':
        return f'The requested term is {v}.'
    return ''


# ----------------------------------------------------------------------
_CLARIFY = {
    'currency_item': {
        'fa': 'این دو عدد یکای متفاوت دارند (پول و تعداد کالا) و بدون نرخ تبدیل یا تناسب مشخص قابل جمع نیستند؛ لطفاً بگویید دقیقاً چه چیزی را با چه چیزی می‌خواهید جمع کنید.',
        'en': 'These two numbers have different units (money and item count) and cannot be added without a conversion rate. Could you clarify what exactly should be added?',
    },
    'currency_mix': {
        'fa': 'دو یکای پولی متفاوت بدون نرخ تبدیل قابل جمع نیستند؛ لطفاً ارزها را یکسان کنید یا نرخ تبدیل را بگویید.',
        'en': 'Two different currencies cannot be added without an exchange rate. Please use one currency or provide the rate.',
    },
    'dimension': {
        'fa': 'یکای این عددها برای یک محاسبهٔ واحد مناسب نیست؛ لطفاً صورت مسئله را روشن‌تر بنویسید.',
        'en': 'The units of these numbers do not fit one calculation. Please clarify the problem statement.',
    },
}


def clarification(language: str, reason: str) -> str:
    key = reason if reason in _CLARIFY else 'dimension'
    return _CLARIFY[key]['fa' if language == 'fa' else 'en']
