"""V23 shared authoring vocabulary + templates (train_core / dev_paraphrase / blind).

Every generator here is NEW v23 authoring. No row from any frozen evaluation
set is copied or paraphrased-verbatim from it (contamination guard verifies
this separately, see benchmarks/v23_contamination_guard.py).

Authoring labels (honest, from day one — spec §5):
  literal            hand-written row, not produced from a template
  authored-template  produced from a bounded template with a seeded PRNG
  synthetic          parameterized family expansion (values sampled)
"""
from __future__ import annotations

import json
import random
import re

# ---------------------------------------------------------------- digits ----
FA_DIGITS = str.maketrans('0123456789', '۰۱۲۳۴۵۶۷۸۹')


def fa_num(v) -> str:
    """Persian numeral rendering (2.5 -> ۲.۵)."""
    return str(v).translate(FA_DIGITS)


def en_num(v) -> str:
    f = float(v)
    if f.is_integer() and abs(f) < 1e15:
        return str(int(f))
    return f'{f:g}'


def num(v, language: str) -> str:
    return fa_num(v) if language == 'fa' else en_num(v)


# ---------------------------------------------------------------- people ----
FA_NAMES = ['علی', 'رضا', 'سارا', 'مریم', 'حسین', 'زهرا', 'امیر', 'نگار', 'مهدی', 'الهام']
EN_NAMES = ['Ali', 'Sara', 'Reza', 'Maryam', 'Hossein', 'Zahra', 'Amir', 'Negar', 'Mehdi', 'Elahe']

# ------------------------------------------------------------- work nouns ---
FA_OUT = ['قطعه', 'کالا', 'جعبه', 'صندلی', 'میز', 'کیسه', 'قطعات', 'بسته']
EN_OUT = ['pieces', 'items', 'boxes', 'chairs', 'tables', 'bags', 'parts', 'packages']


def pick(rng, seq):
    return rng.choice(seq)


def clamp_values(rng, lo, hi, count=4, integers=True):
    out = []
    for _ in range(count):
        v = rng.randint(lo, hi) if integers else round(rng.uniform(lo, hi), 2)
        out.append(v)
    return out


# ------------------------------------------------------ row normalization ---
_STRIP = re.compile(r'\s+')


def norm_row(text: str) -> str:
    """Normalization shared with the contamination guard: ASCII fold, lowercase,
    ZWNJ/space folding."""
    t = (text or '').translate(str.maketrans('۰۱۲۳۴۵۶۷۸۹٠١٢٣٤٥٦٧٨٩', '01234567890123456789'))
    t = t.replace('\u200c', ' ').replace('٫', '.').lower()
    return _STRIP.sub(' ', t).strip()


def ngrams(text: str, n: int = 3) -> set:
    toks = norm_row(text).split()
    if len(toks) < n:
        return {' '.join(toks)} if toks else set()
    return {' '.join(toks[i:i + n]) for i in range(len(toks) - n + 1)}


def jaccard(a: set, b: set) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def load_frozen_ngrams(paths) -> list[set]:
    frozen = []
    for p in paths:
        try:
            with open(p, encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    row = json.loads(line)
                    frozen.append(ngrams(row.get('question', '')))
        except FileNotFoundError:
            continue
    return frozen


def row(id_: str, domain: str, kind: str, question: str, expected, must_abstain: bool,
        authoring: str) -> dict:
    return {'id': id_, 'domain': domain, 'kind': kind, 'question': question,
            'expected': float(expected) if expected is not None else None,
            'must_abstain': bool(must_abstain), 'authoring': authoring}
