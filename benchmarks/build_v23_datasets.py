"""V23 train_core + dev_paraphrase dataset builders (spec §3).

train_core       ~15k parameterized synthetic cases over the v23 families
                 (units conversion, work-rate algebra, narrative arithmetic,
                 currency rates). Every row carries (text, expected, kind,
                 authoring) so the slot mapping can be verified against the
                 deterministic services.

dev_paraphrase   2k author-centric FA/EN/mixed sentences over the same
                 abstractions — used for byte-compat + paraphrase checks.

Both outputs are SHA-256 recorded and audited for contamination against the
frozen evaluation sets (zero literal rows; 3-gram guard, threshold 0.34).
"""
from __future__ import annotations

import hashlib
import json
import os
import random
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__))))

from build_v23_blind_set import (  # noqa: E402  (reuse the frozen vocabulary)
    UNITS, FA_UNIT, EN_UNIT_PLURAL, EN_UNIT_SING, _unit_value,
    gen_units, gen_work_rate, gen_narrative, gen_ownership, gen_finance,
    gen_age, gen_rates, gen_extra, gen_hard_negatives,
    FROZEN_PATHS, CONTAM_THRESHOLD, SPEED_KMH, SPEED_MS, FA_WORK_VERB,
    EN_WORK_VERB, FA_WORKER, EN_WORKER, FA_OUT, EN_OUT, FA_NAMES, EN_NAMES,
)
from v23_authoring_common import (  # noqa: E402
    fa_num, en_num, pick, norm_row, ngrams, jaccard, load_frozen_ngrams,
)

SEED = 20260921
TRAIN_TARGET = 15000
DEV_TARGET = 2000


# ---------------------------------------------------------------------------
# parameterized train_core families (fresh authoring, NOT blind-set rows)
# ---------------------------------------------------------------------------
def train_units(rows, rng, n):
    fams = list(UNITS.items())
    i = 0
    while i < n:
        fam, lst = pick(rng, fams)
        (u1, f1), (u2, f2) = pick(rng, lst), pick(rng, lst)
        if u1 == u2:
            continue
        # wide value sampling keeps the unique space large (spec: ~15k)
        v = rng.choice([0.25, 0.5, 0.75, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5,
                        6, 7, 7.5, 8, 9, 10, 12, 15, 20, 25, 30, 40, 45,
                        50, 60, 75, 80, 90, 120, 150, 180, 200, 240, 250,
                        270, 300, 350, 400, 450, 500, 600, 750, 800, 900,
                        1000, 1200, 1500, 2000, 2500, 3000, 4000, 5000])
        exp = v * f1 / f2
        if abs(exp - round(exp, 6)) > 1e-9:
            continue
        exp = float(round(exp, 6))
        if rng.random() < 0.5:
            q = (f'How many {EN_UNIT_PLURAL[u2]} are {en_num(v)} '
                 f'{EN_UNIT_PLURAL[u1] if v != 1 else EN_UNIT_SING[u1]}?')
        else:
            q = f'{fa_num(v)} {FA_UNIT[u1]} چند {FA_UNIT[u2]} است؟'
        rows.append({'text': q, 'expected': exp, 'kind': 'unit_convert',
                     'family': fam, 'authoring': 'synthetic'})
        i += 1


def train_work_scale(rows, rng, n):
    i = 0
    while i < n:
        w1, w2 = rng.randint(2, 30), rng.randint(2, 30)
        t1, t2 = rng.choice([2, 3, 4, 5, 6, 7, 8, 9, 10, 12]), rng.choice([2, 3, 4, 5, 6, 7, 8, 9, 10, 12])
        per = rng.choice([2, 3, 4, 5, 6, 7, 8, 9, 11, 12, 14, 15])
        out1, out2 = per * w1 * t1, per * w2 * t2
        vf, ve = pick(rng, FA_WORK_VERB), pick(rng, EN_WORK_VERB)
        wf, we = pick(rng, FA_WORKER), pick(rng, EN_WORKER)
        of, oe = pick(rng, FA_OUT), pick(rng, EN_OUT)
        if rng.random() < 0.5:
            q = (f'اگر {fa_num(w1)} {wf} در {fa_num(t1)} ساعت {fa_num(out1)} '
                 f'{of} {vf}، {fa_num(w2)} {wf} در {fa_num(t2)} ساعت '
                 f'چند {of} {vf}؟')
        else:
            q = (f'If {w1} {we} {ve} {out1} {oe} in {t1} hours, how many '
                 f'{oe} do {w2} {we} {ve} in {t2} hours?')
        rows.append({'text': q, 'expected': float(out2), 'kind': 'rate_scale',
                     'family': 'work_rate', 'authoring': 'synthetic'})
        i += 1


def train_work_scale_minutes(rows, rng, n):
    i = 0
    while i < n:
        w1, w2 = rng.randint(2, 15), rng.randint(2, 15)
        m1, m2 = rng.choice([20, 30, 40, 45, 60, 90, 120, 150, 180]), rng.choice([20, 30, 40, 45, 60, 90, 120, 150, 180])
        per = rng.choice([2, 3, 4, 5, 6])
        out1, out2 = per * w1 * (m1 // 10), per * w2 * (m2 // 10)
        of, oe = pick(rng, FA_OUT), pick(rng, EN_OUT)
        ve = pick(rng, EN_WORK_VERB)
        if rng.random() < 0.5:
            q = (f'اگر {fa_num(w1)} کارگر در {fa_num(m1)} دقیقه {fa_num(out1)} '
                 f'{of} تولید کنند، {fa_num(w2)} کارگر در {fa_num(m2)} دقیقه '
                 f'چند {of} تولید می‌کنند؟')
        else:
            q = (f'{w1} workers {ve} {out1} {oe} in {m1} minutes. How many '
                 f'{oe} do {w2} workers {ve} in {m2} minutes?')
        rows.append({'text': q, 'expected': float(out2),
                     'kind': 'rate_scale_minutes', 'family': 'work_rate',
                     'authoring': 'synthetic'})
        i += 1


def train_work_inverted(rows, rng, n):
    i = 0
    while i < n:
        w1, w2 = rng.randint(2, 24), rng.randint(2, 24)
        t1 = rng.choice([2, 3, 4, 5, 6, 8, 10, 12])
        per = rng.choice([2, 3, 4, 5, 6, 7, 8])
        out = per * w1 * t1
        h2 = (w1 * t1) / w2
        if abs(h2 - round(h2, 4)) > 1e-9:
            continue
        of = pick(rng, FA_OUT)
        oe = pick(rng, EN_OUT)
        if rng.random() < 0.5:
            q = (f'{fa_num(w1)} کارگر برای تولید {fa_num(out)} {of} به '
                 f'{fa_num(t1)} ساعت نیاز دارند؛ {fa_num(w2)} کارگر چند ساعت '
                 f'نیاز دارند؟')
        else:
            q = (f'{w1} workers need {t1} hours to produce {out} {oe}. '
                 f'How many hours do {w2} workers need for the same work?')
        rows.append({'text': q, 'expected': float(round(h2, 4)),
                     'kind': 'rate_inverted_hours', 'family': 'work_rate',
                     'authoring': 'synthetic'})
        i += 1


def train_narrative_ops(rows, rng, n):
    kinds = ['sum', 'difference', 'price_drop', 'net_weight', 'pack_multiply',
             'perimeter', 'recipe_scale']
    i = 0
    while i < n:
        kind = pick(rng, kinds)
        if kind == 'sum':
            a, b = rng.randint(50, 2000), rng.randint(20, 900)
            q = f'مجموع {fa_num(a)} و {fa_num(b)} چند می‌شود؟'
            exp = float(a + b)
        elif kind == 'difference':
            a, b = rng.randint(100, 900), rng.randint(10, 99)
            q = f'تفاضل {fa_num(a)} و {fa_num(b)} چند است؟'
            exp = float(a - b)
        elif kind == 'price_drop':
            hi = rng.randint(300, 3000)
            lo = hi - rng.randint(40, 280)
            q = (f'اگر قیمت کالا از {fa_num(hi)} به {fa_num(lo)} برسد، '
                 f'کاهش چند تومان بوده است؟')
            exp = float(hi - lo)
        elif kind == 'net_weight':
            gross, tare = rng.randint(200, 900), rng.randint(10, 90)
            q = (f'باسکول {fa_num(gross)} کیلوگرم نشان می‌دهد و کیسه‌های خالی '
                 f'{fa_num(tare)} کیلوگرم است؛ خالص چند کیلوگرم است؟')
            exp = float(gross - tare)
        elif kind == 'pack_multiply':
            per, packs = rng.randint(4, 20), rng.randint(5, 60)
            q = (f'هر بسته {fa_num(per)} قلم کالا دارد؛ {fa_num(packs)} بسته '
                 f'چند قلم دارد؟')
            exp = float(per * packs)
        elif kind == 'perimeter':
            ln, wd = rng.randint(3, 40), rng.randint(2, 30)
            q = (f'مستطیلی با طول {fa_num(ln)} و عرض {fa_num(wd)} متر، '
                 f'محیطش چند متر است؟')
            exp = float(2 * (ln + wd))
        else:
            per, batches = rng.randint(100, 900), rng.randint(2, 9)
            q = (f'هر دست پخت {fa_num(per)} گرم آرد می‌خواهد؛ '
                 f'{fa_num(batches)} دست چند گرم می‌خواهد؟')
            exp = float(per * batches)
        rows.append({'text': q, 'expected': exp, 'kind': kind,
                     'family': 'narrative', 'authoring': 'synthetic'})
        i += 1


def train_currency_rates(rows, rng, n):
    i = 0
    while i < n:
        rate = rng.choice([4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 15, 16, 18, 20,
                           22, 24, 25, 27, 30, 32, 35, 36, 40, 45, 48, 50,
                           54, 60, 70, 75, 80, 90])
        t = rng.choice([0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5, 5.5, 6, 6.5,
                        7, 7.5, 8, 9, 10, 11, 12])
        exp = float(rate * t)
        if rng.random() < 0.5:
            q = (f'A worker earns {rate} dollars an hour. How much for {t} '
                 f'hours?')
        else:
            q = (f'یک کارگر ساعتی {fa_num(rate)} دلار دریافت می‌کند. برای '
                 f'{fa_num(t)} ساعت چقدر دریافت می‌کند؟')
        rows.append({'text': q, 'expected': exp,
                     'kind': 'currency_rate_application', 'family': 'rates',
                     'authoring': 'synthetic'})
        i += 1


# ---------------------------------------------------------------------------
def build(split: str, target: int, seed: int) -> list[dict]:
    rng = random.Random(seed)
    rows = []
    if split == 'train_core':
        # 15k mix: 45% work-rate, 25% units, 20% narrative, 10% currency
        train_work_scale(rows, rng, int(target * .25))
        train_work_scale_minutes(rows, rng, int(target * .10))
        train_work_inverted(rows, rng, int(target * .10))
        train_units(rows, rng, int(target * .25))
        train_narrative_ops(rows, rng, int(target * .20))
        train_currency_rates(rows, rng, int(target * .10))
    else:
        # dev_paraphrase: reuse the BLIND generators but with a different
        # seed so the phrasing distribution matches while values differ.
        # The blind set itself stays frozen and untouched (different file).
        i = gen_units(rows, rng, 1)
        i = gen_work_rate(rows, rng, i)
        i = gen_narrative(rows, rng, i)
        i = gen_ownership(rows, rng, i)
        i = gen_finance(rows, rng, i)
        i = gen_age(rows, rng, i)
        i = gen_rates(rows, rng, i)
        i = gen_extra(rows, rng, i)
        rows = [{'text': r['question'], 'expected': r['expected'],
                 'kind': r['kind'], 'family': r['domain'],
                 'authoring': r['authoring'],
                 'must_abstain': r['must_abstain']}
                for r in rows]
        # trim/pad to target
        while len(rows) < target:
            pad = []
            train_work_scale(pad, rng, 50)
            train_units(pad, rng, 50)
            rows.extend(pad)
        rows = rows[:max(target, 0)] if len(rows) > target * 1.5 else rows
    # dedupe + contamination guard against frozen sets
    frozen = load_frozen_ngrams(FROZEN_PATHS)
    kept, dropped = [], 0
    seen = set()
    for r in rows:
        key = norm_row(r['text'])
        if key in seen:
            continue
        seen.add(key)
        g = ngrams(r['text'])
        worst = max((jaccard(g, fg) for fg in frozen), default=0.0)
        if worst >= CONTAM_THRESHOLD:
            dropped += 1
            continue
        kept.append(r)
    return kept, dropped


def main():
    frozen = load_frozen_ngrams(FROZEN_PATHS)
    os.makedirs('datasets/v23', exist_ok=True)
    summary = {}
    for split, target, seed in [('train_core', TRAIN_TARGET, SEED),
                                ('dev_paraphrase', DEV_TARGET, SEED + 1)]:
        rows, dropped = build(split, target, seed)
        out = f'datasets/v23/{split}.jsonl'
        with open(out, 'w', encoding='utf-8') as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + '\n')
        sha = hashlib.sha256(open(out, 'rb').read()).hexdigest()
        summary[split] = {'file': out, 'rows': len(rows), 'sha256': sha,
                          'contam_dropped': dropped,
                          'frozen_sources': len(frozen)}
    with open('reports/v23/datasets_manifest.json', 'w', encoding='utf-8') as f:
        json.dump({'seed': SEED, 'contam_threshold': CONTAM_THRESHOLD,
                   **summary,
                   'note': ('no row from any frozen evaluation set is used '
                            'for training (contamination = 0, spec §6)')},
                  f, ensure_ascii=False, indent=1)
    print(json.dumps(summary, ensure_ascii=False, indent=1))
    return 0


if __name__ == '__main__':
    sys.exit(main())
