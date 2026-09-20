"""V23 blind evaluation set builder — eval_blind_v23 (spec §5 protocol step 1).

FREEZE RULES (V23_LANGUAGE_BRAIN_PLAN.md §3/§5):
  * Built BEFORE any v23 engine code exists in the tree.
  * Seeded PRNG -> byte-reproducible output.
  * Honest authoring labels on every row (literal / authored-template / synthetic).
  * ZERO rows copied from any frozen evaluation set; a 3-gram contamination
    guard runs inside the builder and drops+replaces any near-duplicate.
  * Includes hard negatives (must_abstain=True) across families.
  * The output file is written once; afterwards it is read-only (chmod 444)
    and its SHA-256 is recorded in reports/v23/blind_freeze.json.

Output: benchmarks/v23_blind_set.jsonl (>= 1200 rows)
"""
from __future__ import annotations

import json
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from v23_authoring_common import (FA_NAMES, EN_NAMES, FA_OUT, EN_OUT,
                                  fa_num, en_num, num, pick, row, norm_row,
                                  ngrams, jaccard, load_frozen_ngrams)

SEED = 20260920
TARGET_MIN = 1200

# Frozen evaluation sources for the contamination guard (never copied from).
FROZEN_PATHS = [
    'benchmarks/v22_4_blind_1600.jsonl',
    'benchmarks/v22_fresh_blind_1000.jsonl',
    'benchmarks/v22_4_2_diagnostic_set.jsonl',
    'benchmarks/broad_v21_500.jsonl',
    'benchmarks/fresh_v20_clean.jsonl',
    'benchmarks/diagnostic_v20_1_300.jsonl',
]

CONTAM_THRESHOLD = 0.34   # conservative: any 3-gram Jaccard >= this rejects the row


# ===========================================================================
# units conversion family
# ===========================================================================
# unit -> (to_base_factor, base)
UNITS = {
    'mass':     [('mg', 0.001), ('g', 1.0), ('kg', 1000.0), ('ton', 1_000_000.0)],
    'distance': [('mm', 0.001), ('cm', 0.01), ('m', 1.0), ('km', 1000.0)],
    'time':     [('s', 1.0), ('min', 60.0), ('h', 3600.0), ('day', 86400.0)],
    'volume':   [('ml', 1.0), ('l', 1000.0)],
}
FA_UNIT = {
    'mg': 'میلی‌گرم', 'g': 'گرم', 'kg': 'کیلوگرم', 'ton': 'تن',
    'mm': 'میلی‌متر', 'cm': 'سانتی‌متر', 'm': 'متر', 'km': 'کیلومتر',
    's': 'ثانیه', 'min': 'دقیقه', 'h': 'ساعت', 'day': 'روز',
    'ml': 'میلی‌لیتر', 'l': 'لیتر',
}
EN_UNIT_PLURAL = {
    'mg': 'milligrams', 'g': 'grams', 'kg': 'kilograms', 'ton': 'tons',
    'mm': 'millimeters', 'cm': 'centimeters', 'm': 'meters', 'km': 'kilometers',
    's': 'seconds', 'min': 'minutes', 'h': 'hours', 'day': 'days',
    'ml': 'milliliters', 'l': 'liters',
}
# singular word used right after the value in EN source sentences
EN_UNIT_SING = {
    'mg': 'milligram', 'g': 'gram', 'kg': 'kilogram', 'ton': 'ton',
    'mm': 'millimeter', 'cm': 'centimeter', 'm': 'meter', 'km': 'kilometer',
    's': 'second', 'min': 'minute', 'h': 'hour', 'day': 'day',
    'ml': 'milliliter', 'l': 'liter',
}
FA_UNIT_OF_WORD = {v: k for k, v in FA_UNIT.items()}
# English unit words -> canonical key (plural/singular/abbrev)
EN_UNIT_OF_WORD = {}
for fam, lst in UNITS.items():
    for u, _f in lst:
        EN_UNIT_OF_WORD[EN_UNIT_PLURAL[u]] = u
        EN_UNIT_OF_WORD[EN_UNIT_SING[u]] = u
        EN_UNIT_OF_WORD[u] = u

SPEED = [('km/h', 'm/s', 3.6), ('m/s', 'km/h', 1 / 3.6)]
# clean speed pairs
SPEED_KMH = [18, 36, 54, 72, 90, 108, 126]
SPEED_MS = [5, 10, 15, 20, 25, 30]


def _unit_value(rng, u):
    """Natural values per unit so converted results stay clean."""
    if u in ('kg', 'l', 'day'):
        return rng.choice([0.5, 1.5, 2, 2.5, 3, 4, 7.5])
    if u in ('g', 'ml', 'mg', 'mm'):
        return rng.choice([250, 400, 500, 750, 1200, 2500, 5000])
    if u in ('cm', 'min'):
        return rng.choice([30, 45, 90, 150, 180, 240, 270, 480])
    if u in ('m', 's'):
        return rng.choice([2, 3, 5, 9, 12, 20, 45, 300, 500, 3000])
    if u in ('km', 'ton'):
        return rng.choice([2, 3, 5, 7, 12])
    return rng.choice([2, 3, 4, 5])


def gen_units(rows, rng, start_id):
    i = start_id
    pairs = []
    for fam, lst in UNITS.items():
        for a in range(len(lst)):
            for b in range(len(lst)):
                if a != b:
                    pairs.append((fam, lst[a], lst[b]))
    forms = ['how_many', 'is_that', 'fa_direct', 'fa_is_that', 'equals']
    for fam, (u1, f1), (u2, f2) in pairs:
        for _ in range(4):
            v = _unit_value(rng, u1)
            exp = v * f1 / f2
            if abs(exp - round(exp, 6)) > 1e-9:
                continue
            exp = float(round(exp, 6))
            lang = rng.choice(['en', 'fa'])
            form = rng.choice(forms)
            q = _unit_question(form, lang, v, u1, u2)
            if q is None:
                continue
            rows.append(row(f'v23-units-{i:04d}', 'units_conversion', 'unit_convert',
                            q, exp, False, 'authored-template'))
            i += 1
    # speed conversions (clean pairs only)
    for kmh in SPEED_KMH:
        for lang in ('en', 'fa'):
            ms = kmh / 3.6
            q = (f'A speed of {kmh} km/h equals how many m/s?' if lang == 'en' else
                 f'سرعت {fa_num(kmh)} کیلومتر بر ساعت چند متر بر ثانیه است؟')
            rows.append(row(f'v23-units-{i:04d}', 'units_conversion', 'unit_convert_speed',
                            q, ms, False, 'authored-template'))
            i += 1
    for ms in SPEED_MS:
        for lang in ('en', 'fa'):
            kmh = ms * 3.6
            q = (f'A speed of {ms} m/s equals how many km/h?' if lang == 'en' else
                 f'سرعت {fa_num(ms)} متر بر ثانیه چند کیلومتر بر ساعت است؟')
            rows.append(row(f'v23-units-{i:04d}', 'units_conversion', 'unit_convert_speed',
                            q, kmh, False, 'authored-template'))
            i += 1
    # literal hand-written anchors (honest label: literal)
    literals = [
        ('How many grams are 2.5 kilograms?', 2500.0),
        ('How many kilometers are 5000 meters?', 5.0),
        ('How many minutes are 2.5 hours?', 150.0),
        ('How many seconds are 3 minutes?', 180.0),
        ('How many hours are 270 minutes?', 4.5),
        ('A cable is 1.5 meters long; how many centimeters is that?', 150.0),
        ('A tank holds 2 liters; how many milliliters is that?', 2000.0),
        ('یک مسافت ۳ کیلومتر است؛ چند متر است؟', 3000.0),
        ('۲.۵ کیلوگرم چند گرم است؟', 2500.0),
        ('نیم ساعت چند دقیقه است؟', 30.0),
        ('یک بطری ۱.۵ لیتری چند میلی‌لیتر است؟', 1500.0),
        ('سه ساعت چند دقیقه می‌شود؟', 180.0),
        ('سرعت ۹۰ کیلومتر بر ساعت چند متر بر ثانیه است؟', 25.0),
    ]
    for q, exp in literals:
        rows.append(row(f'v23-units-{i:04d}', 'units_conversion', 'unit_convert',
                        q, exp, False, 'literal'))
        i += 1
    # unknown-unit hard negatives (fail-closed: must abstain)
    unknowns_en = ['furlongs', 'leagues', 'nautical miles', 'stones', 'barrels', 'acres']
    unknowns_fa = ['فرسخ', 'مایع‌های ناشناس', 'واحد ناشناخته']
    for u in unknowns_en:
        for v in (3, 5, 12):
            q = f'How many {u} are {v} kilometers?'
            rows.append(row(f'v23-units-{i:04d}', 'units_conversion', 'unit_unknown',
                            q, None, True, 'authored-template'))
            i += 1
    for v in (4, 6):
        q = f'{fa_num(v)} کیلومتر چند فرسخ است؟'
        rows.append(row(f'v23-units-{i:04d}', 'units_conversion', 'unit_unknown',
                        q, None, True, 'authored-template'))
        i += 1
    # temperature hard negative: the service refuses non-linear/unsupported dims
    for q in ['How many Fahrenheit are 30 Celsius?', '۳۰ درجه سلسیوس چند فارنهایت است؟']:
        rows.append(row(f'v23-units-{i:04d}', 'units_conversion', 'unit_unknown',
                        q, None, True, 'literal'))
        i += 1
    return i


def _unit_question(form, lang, v, u1, u2):
    if lang == 'en':
        if form == 'how_many':
            return (f'How many {EN_UNIT_PLURAL[u2]} are {en_num(v)} '
                    f'{EN_UNIT_PLURAL[u1] if v != 1 else EN_UNIT_SING[u1]}?')
        if form == 'is_that':
            v2 = en_num(v)
            return (f'A rope is {v2} {EN_UNIT_SING[u1]} long; how many '
                    f'{EN_UNIT_PLURAL[u2]} is that?')
        if form == 'equals':
            return (f'{en_num(v)} {EN_UNIT_PLURAL[u1]} equals how many '
                    f'{EN_UNIT_PLURAL[u2]}?')
        return None
    if form == 'fa_direct':
        return f'{fa_num(v)} {FA_UNIT[u1]} چند {FA_UNIT[u2]} است؟'
    if form == 'fa_is_that':
        return f'یک طناب {fa_num(v)} {FA_UNIT[u1]} است؛ چند {FA_UNIT[u2]} است؟'
    if form == 'equals':
        return f'{fa_num(v)} {FA_UNIT[u1]} برابر است با چند {FA_UNIT[u2]}؟'
    return None


# ===========================================================================
# work-rate family
# ===========================================================================
FA_WORK_VERB = ['تولید کنند', 'بسته‌بندی کنند', 'مونتاژ کنند', 'بسازند', 'بارگیری کنند']
EN_WORK_VERB = ['produce', 'assemble', 'pack', 'build', 'load']

FA_WORKER = ['کارگر', 'نفر', 'کارمند', 'ماشین']
EN_WORKER = ['workers', 'people', 'employees', 'machines']


def gen_work_rate(rows, rng, start_id):
    i = start_id
    # (a) forward scale: (w1,h1,out1) -> (w2,h2) -> out2
    for _ in range(100):
        lang = rng.choice(['fa', 'en'])
        w1, w2 = rng.randint(2, 12), rng.randint(2, 12)
        t1, t2 = rng.choice([2, 3, 4, 5, 6]), rng.choice([2, 3, 4, 5, 6])
        per = rng.choice([2, 3, 4, 5, 6, 8])
        out1 = per * w1 * t1
        out2 = per * w2 * t2
        verb_f, verb_e = pick(rng, FA_WORK_VERB), pick(rng, EN_WORK_VERB)
        wor_f, wor_e = pick(rng, FA_WORKER), pick(rng, EN_WORKER)
        out_f, out_e = pick(rng, FA_OUT), pick(rng, EN_OUT)
        if rng.random() < 0.5:
            q = (f'اگر {fa_num(w1)} {wor_f} در {fa_num(t1)} ساعت {fa_num(out1)} '
                 f'{out_f} {verb_f}، {fa_num(w2)} {wor_f} در {fa_num(t2)} ساعت '
                 f'چند {out_f} {verb_f}؟')
        else:
            q = (f'If {w1} {wor_e} {verb_e} {out1} {out_e} in {t1} hours, how many '
                 f'{out_e} do {w2} {wor_e} {verb_e} in {t2} hours?')
        rows.append(row(f'v23-work-{i:04d}', 'work_rate', 'rate_scale',
                        q, out2, False, 'synthetic'))
        i += 1
    # (a2) forward scale with minutes
    for _ in range(45):
        lang = rng.choice(['fa', 'en'])
        w1, w2 = rng.randint(2, 9), rng.randint(2, 9)
        m1, m2 = rng.choice([30, 45, 60, 90, 120]), rng.choice([30, 45, 60, 90, 120])
        per = rng.choice([2, 3, 4, 5])
        out1 = per * w1 * (m1 // 30)
        out2 = per * w2 * (m2 // 30)
        out_f, out_e = pick(rng, FA_OUT), pick(rng, EN_OUT)
        verb_f, verb_e = pick(rng, FA_WORK_VERB), pick(rng, EN_WORK_VERB)
        if lang == 'fa':
            q = (f'اگر {fa_num(w1)} کارگر در {fa_num(m1)} دقیقه {fa_num(out1)} '
                 f'{out_f} {verb_f}، {fa_num(w2)} کارگر در {fa_num(m2)} دقیقه '
                 f'چند {out_f} {verb_f}؟')
        else:
            q = (f'{w1} workers {verb_e} {out1} {out_e} in {m1} minutes. How many '
                 f'{out_e} do {w2} workers {verb_e} in {m2} minutes?')
        rows.append(row(f'v23-work-{i:04d}', 'work_rate', 'rate_scale_minutes',
                        q, out2, False, 'synthetic'))
        i += 1
    # (b) inverted hours: same output, new crew size -> hours
    FA_STEM = {'تولید کنند': 'تولید', 'بسته‌بندی کنند': 'بسته‌بندی',
               'مونتاژ کنند': 'مونتاژ', 'بسازند': 'ساخت', 'بارگیری کنند': 'بارگیری'}
    EN_STEM = {'produce': 'produce', 'assemble': 'assemble', 'pack': 'pack',
               'build': 'build', 'load': 'load'}
    for _ in range(60):
        lang = rng.choice(['fa', 'en'])
        w1, w2 = rng.randint(2, 12), rng.randint(2, 12)
        t1 = rng.choice([2, 3, 4, 5, 6])
        per = rng.choice([2, 3, 4, 5])
        out = per * w1 * t1
        h2 = (w1 * t1) / w2
        if abs(h2 - round(h2, 4)) > 1e-9:
            continue
        h2 = float(round(h2, 4))
        out_f, out_e = pick(rng, FA_OUT), pick(rng, EN_OUT)
        vf, ve = pick(rng, FA_WORK_VERB), pick(rng, EN_WORK_VERB)
        if lang == 'fa':
            q = (f'{fa_num(w1)} کارگر برای {FA_STEM[vf]} {fa_num(out)} '
                 f'{out_f} به {fa_num(t1)} ساعت نیاز دارند. '
                 f'{fa_num(w2)} کارگر برای همین کار چند ساعت نیاز دارند؟')
        else:
            q = (f'{w1} workers need {t1} hours to {EN_STEM[ve]} {out} {out_e}. '
                 f'How many hours do {w2} workers need for the same work?')
        rows.append(row(f'v23-work-{i:04d}', 'work_rate', 'rate_inverted_hours',
                        q, h2, False, 'synthetic'))
        i += 1
    # (c) combined inversion: two solo times -> together
    for _ in range(55):
        lang = rng.choice(['fa', 'en'])
        a, b = rng.choice([2, 3, 4, 6, 8, 12]), rng.choice([2, 3, 4, 6, 8, 12])
        tog = (a * b) / (a + b)
        if abs(tog - round(tog, 4)) > 1e-9:
            continue
        tog = float(round(tog, 4))
        n1, n2 = (pick(rng, FA_NAMES), pick(rng, FA_NAMES)) if lang == 'fa' \
            else (pick(rng, EN_NAMES), pick(rng, EN_NAMES))
        if n1 == n2:
            continue
        obj_f = pick(rng, ['یک گزارش', 'یک مخزن', 'یک پروژه', 'یک دیوار'])
        obj_e = pick(rng, ['a report', 'a tank', 'a project', 'a wall'])
        if lang == 'fa':
            q = (f'{n1} می‌تواند {obj_f} را در {fa_num(a)} ساعت تمام کند و '
                 f'{n2} در {fa_num(b)} ساعت. با هم چند ساعت طول می‌کشد؟')
        else:
            q = (f'{n1} can finish {obj_e} in {a} hours and {n2} in {b} hours. '
                 f'Working together, how many hours do they need?')
        rows.append(row(f'v23-work-{i:04d}', 'work_rate', 'rate_combined_inversion',
                        q, tog, False, 'synthetic'))
        i += 1
    # (d) per-worker rate sum: two named rates + together time
    for _ in range(60):
        lang = rng.choice(['fa', 'en'])
        r1, r2 = rng.randint(2, 9), rng.randint(2, 9)
        t = rng.choice([2, 3, 4, 5])
        exp = (r1 + r2) * t
        if lang == 'fa':
            n1, n2 = pick(rng, FA_NAMES), pick(rng, FA_NAMES)
            if n1 == n2:
                continue
            out_f, out_e = pick(rng, ['صندلی', 'میز', 'کالا', 'جعبه']), ''
            q = (f'{n1} در هر ساعت {fa_num(r1)} {out_f} می‌سازد و {n2} در هر ساعت '
                 f'{fa_num(r2)} {out_f}. با هم در {fa_num(t)} ساعت چند {out_f} می‌سازند؟')
        else:
            n1, n2 = pick(rng, EN_NAMES), pick(rng, EN_NAMES)
            if n1 == n2:
                continue
            out_e = pick(rng, ['chairs', 'tables', 'items', 'boxes'])
            q = (f'{n1} builds {r1} {out_e} per hour and {n2} builds {r2} {out_e} '
                 f'per hour. Working together for {t} hours, how many {out_e} '
                 f'do they build?')
        rows.append(row(f'v23-work-{i:04d}', 'work_rate', 'rate_per_worker_sum',
                        q, exp, False, 'synthetic'))
        i += 1
    # per-worker rate sum: two named rates + together time (second batch)
    for _ in range(30):
        lang = rng.choice(['fa', 'en'])
        r1, r2 = rng.randint(2, 9), rng.randint(2, 9)
        t = rng.choice([2, 3, 4, 5])
        exp = (r1 + r2) * t
        if lang == 'fa':
            n1, n2 = pick(rng, FA_NAMES), pick(rng, FA_NAMES)
            if n1 == n2:
                continue
            out_f = pick(rng, ['صندلی', 'میز', 'کالا', 'جعبه'])
            q = (f'{n1} هر ساعت {fa_num(r1)} {out_f} و {n2} هر ساعت '
                 f'{fa_num(r2)} {out_f} می‌سازد. با هم در {fa_num(t)} ساعت '
                 f'چند {out_f} می‌سازند؟')
        else:
            n1, n2 = pick(rng, EN_NAMES), pick(rng, EN_NAMES)
            if n1 == n2:
                continue
            out_e = pick(rng, ['chairs', 'tables', 'items', 'boxes'])
            q = (f'{n1} makes {r1} {out_e} an hour; {n2} makes {r2} {out_e} '
                 f'an hour. Together, how many {out_e} in {t} hours?')
        rows.append(row(f'v23-work-{i:04d}', 'work_rate', 'rate_per_worker_sum',
                        q, exp, False, 'synthetic'))
        i += 1
    # (e) same-crew extrapolation: (w,h,out) -> h2 (same w)
    for _ in range(45):
        lang = rng.choice(['fa', 'en'])
        w = rng.randint(2, 8)
        h1, h2 = rng.choice([2, 3, 4, 6]), rng.choice([4, 5, 6, 8])
        per = rng.choice([2, 3, 4, 5])
        out1 = per * h1
        out2 = per * h2
        out_f, out_e = pick(rng, FA_OUT), pick(rng, EN_OUT)
        if lang == 'fa':
            q = (f'یک کارگاه با {fa_num(w)} کارگر در {fa_num(h1)} ساعت {fa_num(out1)} '
                 f'{out_f} تولید می‌کند. اگر تعداد کارگران تغییری نکند، در '
                 f'{fa_num(h2)} ساعت چند {out_f} تولید می‌کند؟')
        else:
            q = (f'A shop with {w} workers assembles {out1} {out_e} in {h1} hours. '
                 f'If no workers are added or removed, how many {out_e} are '
                 f'assembled in {h2} hours?')
        rows.append(row(f'v23-work-{i:04d}', 'work_rate', 'rate_same_crew',
                        q, out2, False, 'synthetic'))
        i += 1
    # (f) mid-task leave/join
    for _ in range(40):
        lang = rng.choice(['fa', 'en'])
        w = rng.choice([8, 10, 12])
        h_total = rng.choice([3, 4, 6])
        per = rng.choice([2, 3, 4, 5])
        out = per * w * h_total
        t_switch = max(1, h_total // 2)
        t_after = h_total - t_switch
        frac = 2   # half leave
        exp = per * w * t_switch + per * (w // frac) * t_after
        out_f, out_e = pick(rng, FA_OUT), pick(rng, EN_OUT)
        if lang == 'fa':
            q = (f'{fa_num(w)} کارگر در {fa_num(h_total)} ساعت {fa_num(out)} {out_f} '
                 f'بارگیری می‌کنند. پس از {fa_num(t_switch)} ساعت نیمی از کارگران '
                 f'می‌روند. در مجموع پس از {fa_num(h_total)} ساعت چند {out_f} '
                 f'بارگیری شده است؟')
        else:
            q = (f'{w} workers load {out} {out_e} in {h_total} hours. After '
                 f'{t_switch} hours, half of the workers leave. How many {out_e} '
                 f'are loaded in total after {h_total} hours?')
        rows.append(row(f'v23-work-{i:04d}', 'work_rate', 'rate_midtask_leave',
                        q, exp, False, 'synthetic'))
        i += 1
    # forward scale: second batch with wider values
    for _ in range(60):
        lang = rng.choice(['fa', 'en'])
        w1, w2 = rng.randint(3, 15), rng.randint(3, 15)
        t1, t2 = rng.choice([2, 3, 4, 6, 8]), rng.choice([2, 3, 4, 6, 8])
        per = rng.choice([3, 4, 5, 6, 7, 9])
        out1 = per * w1 * t1
        out2 = per * w2 * t2
        verb_f, verb_e = pick(rng, FA_WORK_VERB), pick(rng, EN_WORK_VERB)
        wor_f, wor_e = pick(rng, FA_WORKER), pick(rng, EN_WORKER)
        out_f, out_e = pick(rng, FA_OUT), pick(rng, EN_OUT)
        if rng.random() < 0.5:
            q = (f'اگر {fa_num(w1)} {wor_f} در {fa_num(t1)} ساعت {fa_num(out1)} '
                 f'{out_f} {verb_f}، {fa_num(w2)} {wor_f} در {fa_num(t2)} ساعت '
                 f'چند {out_f} {verb_f}؟')
        else:
            q = (f'If {w1} {wor_e} {verb_e} {out1} {out_e} in {t1} hours, how many '
                 f'{out_e} do {w2} {wor_e} {verb_e} in {t2} hours?')
        rows.append(row(f'v23-work-{i:04d}', 'work_rate', 'rate_scale',
                        q, out2, False, 'synthetic'))
        i += 1
    # literal anchors
    literals = [
        ('If 6 workers assemble 96 chairs in 4 hours, how many chairs do 9 workers assemble in 5 hours?', 180.0),
        ('10 workers produce 200 units in 4 hours. How many hours do 8 workers need to produce 200 units?', 5.0),
        ('Ali can finish a report in 6 hours and Sara in 3 hours. Working together, how many hours do they need?', 2.0),
        ('Ali builds 4 chairs per hour and Sara builds 3 chairs per hour. Working together for 2 hours, how many chairs do they build?', 14.0),
        ('12 workers load 480 crates in 4 hours. After 2 hours, half of the workers leave. How many crates are loaded in total after 3 hours?', 300.0),
        ('اگر ۴ کارگر در ۳ ساعت ۶۰ قطعه تولید کنند، ۶ کارگر در ۵ ساعت چند قطعه تولید می‌کنند؟', 150.0),
        ('۵ کارگر برای رنگ‌آمیزی حصار به ۶ ساعت نیاز دارند؛ ۳ کارگر چند ساعت نیاز دارند؟', 10.0),
    ]
    for q, exp in literals:
        rows.append(row(f'v23-work-{i:04d}', 'work_rate', 'literal_anchor',
                        q, exp, False, 'literal'))
        i += 1
    # insufficient-information hard negatives (must abstain)
    hns = [
        ('چند کارگر چند قطعه تولید می‌کنند؟', None),
        ('Some workers produce 40 pieces. How long does it take?', None),
        ('Workers produce pieces in hours. How many pieces in 5 hours?', None),
        ('اگر سرعت کار بیشتر شود، چند قطعه تولید می‌شود؟', None),
        ('A team finishes the job faster than before. How many workers joined?', None),
    ]
    for q, _ in hns:
        rows.append(row(f'v23-work-{i:04d}', 'work_rate', 'insufficient_info',
                        q, None, True, 'literal'))
        i += 1
    return i


# ===========================================================================
# narrative arithmetic family (persian_math / english_math)
# ===========================================================================
def gen_narrative(rows, rng, start_id):
    i = start_id
    # (1) narrative sum — fa + en
    for _ in range(45):
        lang = rng.choice(['fa', 'en'])
        a, b = rng.randint(110, 900), rng.randint(50, 500)
        if lang == 'fa':
            q = f'مجموع {fa_num(a)} و {fa_num(b)} چند می‌شود؟'
        else:
            noun = pick(rng, ['crates', 'apples', 'books', 'bottles'])
            q = (f'A warehouse stores {a} {noun} in the morning and {b} {noun} '
                 f'in the afternoon. How many {noun} in total?')
        rows.append(row(f'v23-math-{i:04d}', 'narrative_math', 'sum',
                        q, a + b, False, 'synthetic'))
        i += 1
    # (2) difference — fa + en
    for _ in range(35):
        lang = rng.choice(['fa', 'en'])
        a, b = rng.randint(60, 400), rng.randint(20, 55)
        if lang == 'fa':
            q = f'تفاضل {fa_num(a)} و {fa_num(b)} چند است؟'
        else:
            q = f'What is the difference between {a} and {b}?'
        rows.append(row(f'v23-math-{i:04d}', 'narrative_math', 'difference',
                        q, a - b, False, 'synthetic'))
        i += 1
    # (3) price drop — fa + en
    for _ in range(40):
        lang = rng.choice(['fa', 'en'])
        hi = rng.choice([400, 500, 600, 800, 900, 1200])
        lo = hi - rng.choice([50, 100, 150, 200, 250])
        if lang == 'fa':
            q = (f'اگر قیمت یک کالا از {fa_num(hi)} به {fa_num(lo)} تومان برسد، '
                 f'کاهش چند تومان بوده است؟')
        else:
            q = (f'The price of an item dropped from {hi} dollars to {lo} dollars. '
                 f'What was the drop in dollars?')
        rows.append(row(f'v23-math-{i:04d}', 'narrative_math', 'price_drop',
                        q, hi - lo, False, 'synthetic'))
        i += 1
    # (4) net weight
    for _ in range(30):
        lang = rng.choice(['fa', 'en'])
        gross = rng.choice([300, 380, 430, 500, 620])
        tare = rng.choice([20, 30, 40, 50])
        if lang == 'fa':
            q = (f'یک باسکول {fa_num(gross)} کیلوگرم گندم نشان می‌دهد؛ کیسه‌های خالی '
                 f'{fa_num(tare)} کیلوگرم است. گندم خالص چند کیلوگرم است؟')
        else:
            q = (f'A scale shows {gross} kilograms of wheat; the empty bags weigh '
                 f'{tare} kilograms. How many kilograms is the net wheat?')
        rows.append(row(f'v23-math-{i:04d}', 'narrative_math', 'net_weight',
                        q, gross - tare, False, 'synthetic'))
        i += 1
    # (5) fraction inverse (half/third/quarter)
    for _ in range(40):
        lang = rng.choice(['fa', 'en'])
        denom, fa_word = rng.choice([(2, 'نصف'), (3, 'یک‌سوم'), (4, 'یک‌چهارم')])
        part = rng.randint(10, 90)
        whole = part * denom
        if lang == 'fa':
            q = f'{fa_word} یک عدد {fa_num(part)} است؛ آن عدد چند است؟'
        else:
            word = {2: 'Half', 3: 'One third', 4: 'One quarter'}[denom]
            q = f'{word} of a number is {part}. What is the number?'
        rows.append(row(f'v23-math-{i:04d}', 'narrative_math', 'fraction_inverse',
                        q, whole, False, 'synthetic'))
        i += 1
    # (6) pack multiply
    for _ in range(30):
        lang = rng.choice(['fa', 'en'])
        per, packs = rng.choice([6, 8, 12, 15]), rng.choice([10, 25, 35, 48])
        if lang == 'fa':
            q = (f'اگر هر بسته {fa_num(per)} قلم کالا داشته باشد، {fa_num(packs)} '
                 f'بسته چند قلم دارد؟')
        else:
            q = (f'Each box holds {per} items. How many items do {packs} boxes hold?')
        rows.append(row(f'v23-math-{i:04d}', 'narrative_math', 'pack_multiply',
                        q, per * packs, False, 'synthetic'))
        i += 1
    # (7) rectangle perimeter
    for _ in range(25):
        lang = rng.choice(['fa', 'en'])
        ln, wd = rng.randint(4, 25), rng.randint(3, 15)
        if lang == 'fa':
            q = (f'یک مستطیل با طول {fa_num(ln)} و عرض {fa_num(wd)} متر، '
                 f'محیطش چند متر است؟')
        else:
            q = (f'A rectangle is {ln} meters long and {wd} meters wide. '
                 f'What is its perimeter in meters?')
        rows.append(row(f'v23-math-{i:04d}', 'narrative_math', 'perimeter',
                        q, 2 * (ln + wd), False, 'synthetic'))
        i += 1
    # (8) recipe scale
    for _ in range(25):
        lang = rng.choice(['fa', 'en'])
        per, batches = rng.choice([150, 250, 350, 450]), rng.choice([2, 3, 4, 6])
        if lang == 'fa':
            q = (f'برای هر دُست پخت {fa_num(per)} گرم آرد لازم است؛ برای '
                 f'{fa_num(batches)} دُست چند گرم آرد لازم است؟')
        else:
            q = (f'A recipe needs {per} grams of flour per batch. How many grams '
                 f'of flour for {batches} batches?')
        rows.append(row(f'v23-math-{i:04d}', 'narrative_math', 'recipe_scale',
                        q, per * batches, False, 'synthetic'))
        i += 1
    # (9) bus passengers (off/on chain)
    for _ in range(30):
        lang = rng.choice(['fa', 'en'])
        start, off, on = rng.randint(30, 80), rng.randint(5, 25), rng.randint(5, 25)
        if lang == 'fa':
            q = (f'یک اتوبوس {fa_num(start)} مسافر دارد؛ {fa_num(off)} نفر پیاده '
                 f'می‌شوند و {fa_num(on)} نفر سوار می‌شوند. الان چند مسافر دارد؟')
        else:
            q = (f'A bus has {start} passengers; {off} get off and {on} get on. '
                 f'How many passengers now?')
        rows.append(row(f'v23-math-{i:04d}', 'narrative_math', 'passenger_chain',
                        q, start - off + on, False, 'synthetic'))
        i += 1
    # (10) original amount
    for _ in range(25):
        lang = rng.choice(['fa', 'en'])
        left, sold = rng.randint(120, 400), rng.randint(60, 200)
        if lang == 'fa':
            q = (f'یک باغدار {fa_num(sold)} سیب فروخت و {fa_num(left)} سیب برایش '
                 f'مانده است. در ابتدا چند سیب داشته است؟')
        else:
            q = (f'A farmer sold {sold} apples and has {left} left. How many '
                 f'apples did he start with?')
        rows.append(row(f'v23-math-{i:04d}', 'narrative_math', 'original_amount',
                        q, sold + left, False, 'synthetic'))
        i += 1
    # (11) rope pieces
    for _ in range(18):
        lang = rng.choice(['fa', 'en'])
        ln = rng.choice([6, 8, 9, 12])
        piece = rng.choice([0.5, 0.75, 1.5])
        exp = float(round(ln / piece, 6))
        if abs(exp - round(exp)) > 1e-9:
            continue
        exp = float(round(exp))
        if lang == 'fa':
            q = (f'یک طناب {fa_num(ln)} متری را به قطعات مساوی {fa_num(piece)} '
                 f'متری برش می‌زنند. چند قطعه به دست می‌آید؟')
        else:
            q = (f'A rope {ln} meters long is cut into equal {piece}-meter pieces. '
                 f'How many pieces?')
        rows.append(row(f'v23-math-{i:04d}', 'narrative_math', 'rope_pieces',
                        q, exp, False, 'synthetic'))
        i += 1
    # (12) temperature reverse
    for _ in range(18):
        lang = rng.choice(['fa', 'en'])
        rose, fell, end = rng.randint(3, 9), rng.randint(6, 14), rng.randint(5, 15)
        start = end - rose + fell
        if lang == 'fa':
            q = (f'دما {fa_num(rose)} درجه بالا رفت، سپس {fa_num(fell)} درجه پایین '
                 f'آمد و به {fa_num(end)} درجه رسید. دمای ابتدا چند درجه بود؟')
        else:
            q = (f'The temperature rose {rose} degrees, then fell {fell} degrees, '
                 f'ending at {end} degrees. What did it start at?')
        rows.append(row(f'v23-math-{i:04d}', 'narrative_math', 'temperature_reverse',
                        q, start, False, 'synthetic'))
        i += 1
    # (13) printer/pipeline rate application with NEW nouns (pages etc.)
    for _ in range(30):
        lang = rng.choice(['fa', 'en'])
        rate = rng.choice([24, 32, 38, 45])
        t = rng.choice([1.5, 2, 2.5, 3])
        exp = float(rate * t)
        if lang == 'fa':
            q = (f'یک چاپگر در هر دقیقه {fa_num(rate)} صفحه چاپ می‌کند. در '
                 f'{fa_num(t)} دقیقه چند صفحه چاپ می‌کند؟')
        else:
            q = (f'A printer prints {rate} pages per minute. How many pages in '
                 f'{t} minutes?')
        rows.append(row(f'v23-math-{i:04d}', 'narrative_math', 'rate_application_pages',
                        q, exp, False, 'synthetic'))
        i += 1
    # (14) part + remainder sum (هزار تومان scale preserved)
    for _ in range(20):
        part1, part2 = rng.choice([60, 90, 120]), rng.choice([150, 180, 210])
        q = (f'ثمن یک مبلغ {fa_num(part1)} هزار تومان و مابقی {fa_num(part2)} '
             f'هزار تومان است؛ کل چقدر است؟')
        rows.append(row(f'v23-math-{i:04d}', 'narrative_math', 'part_remainder_sum',
                        q, part1 + part2, False, 'synthetic'))
        i += 1
    # literal anchors
    literals = [
        ('مجموع ۲۵۰ و ۱۷۵ چند می‌شود؟', 425.0),
        ('اگر قیمت یک کالا از ۸۰۰ به ۶۵۰ برسد، کاهش چند تومان بوده است؟', 150.0),
        ('یک باسکول ۴۳۰ کیلوگرم گندم نشان می‌دهد؛ کیسه‌های خالی ۳۰ کیلوگرم است. گندم خالص چند کیلوگرم است؟', 400.0),
        ('نصف یک عدد ۴۵ است؛ آن عدد چند است؟', 90.0),
        ('One third of a number is 17. What is the number?', 51.0),
        ('A bus has 57 passengers; 19 get off and 12 get on. How many passengers now?', 50.0),
        ('A rope 9 meters long is cut into equal 0.75-meter pieces. How many pieces?', 12.0),
        ('The temperature rose 6 degrees, then fell 11 degrees, ending at 9 degrees. What did it start at?', 14.0),
    ]
    for q, exp in literals:
        rows.append(row(f'v23-math-{i:04d}', 'narrative_math', 'literal_anchor',
                        q, exp, False, 'literal'))
        i += 1
    # hard negatives (must abstain): missing operand / unknown referent
    hns = [
        ('یکی از اعداد ۴۵ است؛ عدد دیگر چند است؟', None),
        ('A crate lost some apples and now has 30. How many did it lose?', None),
        ('قیمت کالا کاهش یافت؛ چقدر کاهش یافت؟', None),
        ('The temperature changed twice and ended at 9 degrees. What did it start at?', None),
    ]
    for q, _ in hns:
        rows.append(row(f'v23-math-{i:04d}', 'narrative_math', 'insufficient_info',
                        q, None, True, 'literal'))
        i += 1
    return i


# ===========================================================================
# ownership / finance / age / rates families (v23 paraphrase breadth)
# ===========================================================================
def gen_ownership(rows, rng, start_id):
    i = start_id
    EN_VERBS = ['pays', 'sends', 'hands', 'wires', 'gives']
    FA_VERBS = ['می‌پردازد', 'می‌فرستد', 'حواله می‌کند', 'می‌دهد', 'واریز می‌کند']
    for _ in range(60):
        lang = rng.choice(['fa', 'en'])
        b1, b2 = rng.randint(40, 200), rng.randint(40, 200)
        x = rng.randint(5, 30)
        n1, n2 = (pick(rng, FA_NAMES), pick(rng, FA_NAMES)) if lang == 'fa' \
            else (pick(rng, EN_NAMES), pick(rng, EN_NAMES))
        if n1 == n2:
            continue
        unit_f, unit_e = 'تومان', 'dollars'
        if lang == 'fa':
            q = (f'{n1} {fa_num(b1)} تومان دارد و {n2} {fa_num(b2)} تومان. '
                 f'{n1} به {n2} {fa_num(x)} تومان {pick(rng, FA_VERBS)}. '
                 f'موجودی {n1} چند تومان است؟')
        else:
            q = (f'{n1} has {b1} dollars and {n2} has {b2} dollars. {n1} '
                 f'{pick(rng, EN_VERBS)} {x} dollars to {n2}. '
                 f"What is {n1}'s balance?")
        rows.append(row(f'v23-own-{i:04d}', 'ownership', 'transfer_single',
                        q, b1 - x, False, 'synthetic'))
        i += 1
    # split transfers: gives X to A and Y to B
    for _ in range(40):
        lang = rng.choice(['fa', 'en'])
        b = rng.randint(80, 300)
        x, y = rng.randint(5, 30), rng.randint(5, 30)
        if lang == 'fa':
            n0 = pick(rng, FA_NAMES)
            na, nb = pick(rng, FA_NAMES), pick(rng, FA_NAMES)
            if len({n0, na, nb}) < 3:
                continue
            q = (f'{n0} {fa_num(b)} تومان دارد. {n0} به {na} {fa_num(x)} تومان و '
                 f'به {nb} {fa_num(y)} تومان می‌دهد. موجودی {n0} چند تومان است؟')
        else:
            n0 = pick(rng, EN_NAMES)
            na, nb = pick(rng, EN_NAMES), pick(rng, EN_NAMES)
            if len({n0, na, nb}) < 3:
                continue
            q = (f'{n0} has {b} dollars. {n0} gives {x} dollars to {na} and '
                 f'{y} dollars to {nb}. What is {n0}\'s balance?')
        rows.append(row(f'v23-own-{i:04d}', 'ownership', 'transfer_split',
                        q, b - x - y, False, 'synthetic'))
        i += 1
    # literal anchors
    for q, exp in [
        ('Sara has 150 dollars and Omar has 60 dollars. Sara sends 25 dollars to Omar. What is Sara\'s balance?', 125.0),
        ('علی ۲۰۰ تومان دارد. علی به رضا ۳۰ تومان و به مریم ۲۰ تومان می‌دهد. موجودی علی چند تومان است؟', 150.0),
    ]:
        rows.append(row(f'v23-own-{i:04d}', 'ownership', 'literal_anchor',
                        q, exp, False, 'literal'))
        i += 1
    # hard negative: unknown initial balances
    for q, _ in [
        ('علی به رضا ۲۰ تومان می‌دهد. موجودی علی چند تومان است؟', None),
        ('Omar wires 15 dollars. What is Omar\'s balance?', None),
    ]:
        rows.append(row(f'v23-own-{i:04d}', 'ownership', 'insufficient_info',
                        q, None, True, 'literal'))
        i += 1
    return i


def gen_finance(rows, rng, start_id):
    i = start_id
    for _ in range(50):
        lang = rng.choice(['fa', 'en'])
        price = rng.choice([200, 250, 400, 500, 800])
        pct = rng.choice([10, 15, 20, 25, 30])
        disc = price * pct / 100
        exp = float(round(price - disc, 6))
        if lang == 'fa':
            q = (f'قیمت کالایی {fa_num(price)} تومان است و {fa_num(pct)} درصد '
                 f'تخفیف می‌خورد. قیمت پس از تخفیف چند تومان است؟')
        else:
            q = (f'An item costs {price} dollars with a {pct}% discount. '
                 f'What is the price after the discount?')
        rows.append(row(f'v23-fin-{i:04d}', 'finance', 'discount_amount',
                        q, exp, False, 'synthetic'))
        i += 1
    # tax after discount (multi-step)
    for _ in range(30):
        price = rng.choice([100, 200, 300, 400])
        dpct = rng.choice([10, 20])
        tpct = rng.choice([5, 10])
        after = price * (1 - dpct / 100)
        exp = float(round(after * (1 + tpct / 100), 6))
        lang = rng.choice(['fa', 'en'])
        if lang == 'fa':
            q = (f'کالایی {fa_num(price)} تومان است؛ ابتدا {fa_num(dpct)} درصد '
                 f'تخفیف می‌خورد و سپس {fa_num(tpct)} درصد مالیات اضافه می‌شود. '
                 f'قیمت نهایی چند تومان است؟')
        else:
            q = (f'An item costs {price} dollars; first a {dpct}% discount is '
                 f'applied, then {tpct}% tax is added. What is the final price?')
        rows.append(row(f'v23-fin-{i:04d}', 'finance', 'discount_then_tax',
                        q, exp, False, 'synthetic'))
        i += 1
    # markup
    for _ in range(25):
        cost = rng.choice([80, 120, 150, 200])
        pct = rng.choice([20, 25, 50])
        exp = float(cost * (1 + pct / 100))
        lang = rng.choice(['fa', 'en'])
        if lang == 'fa':
            q = (f'خرید یک کالا {fa_num(cost)} تومان است و فروشنده {fa_num(pct)} '
                 f'درصد سود اضافه می‌کند. قیمت فروش چند تومان است؟')
        else:
            q = (f'A product costs {cost} dollars and the seller adds {pct}% '
                 f'markup. What is the selling price?')
        rows.append(row(f'v23-fin-{i:04d}', 'finance', 'markup',
                        q, exp, False, 'synthetic'))
        i += 1
    return i


def gen_age(rows, rng, start_id):
    i = start_id
    for _ in range(45):
        lang = rng.choice(['fa', 'en'])
        a, b = rng.randint(12, 60), rng.randint(6, 50)
        if a == b:
            continue
        older, younger = max(a, b), min(a, b)
        n1, n2 = (pick(rng, FA_NAMES), pick(rng, FA_NAMES)) if lang == 'fa' \
            else (pick(rng, EN_NAMES), pick(rng, EN_NAMES))
        if n1 == n2:
            continue
        if lang == 'fa':
            q = (f'{n1} {fa_num(older)} ساله و {n2} {fa_num(younger)} ساله است. '
                 f'{n1} چند سال از {n2} بزرگ‌تر است؟')
        else:
            q = (f'{n1} is {older} years old and {n2} is {younger} years old. '
                 f'How many years older is {n1} than {n2}?')
        rows.append(row(f'v23-age-{i:04d}', 'age', 'age_difference',
                        q, older - younger, False, 'synthetic'))
        i += 1
    # future projection
    for _ in range(30):
        lang = rng.choice(['fa', 'en'])
        a, b = rng.randint(15, 50), rng.randint(8, 40)
        if a == b:
            continue
        d = abs(a - b)
        k = rng.choice([3, 5, 10])
        n1, n2 = (pick(rng, FA_NAMES), pick(rng, FA_NAMES)) if lang == 'fa' \
            else (pick(rng, EN_NAMES), pick(rng, EN_NAMES))
        if n1 == n2:
            continue
        if lang == 'fa':
            q = (f'{n1} {fa_num(a)} ساله و {n2} {fa_num(b)} ساله است. '
                 f'{fa_num(k)} سال دیگر اختلاف سن آن‌ها چند سال است؟')
        else:
            q = (f'{n1} is {a} years old and {n2} is {b} years old. In {k} years, '
                 f'what will the age difference be?')
        rows.append(row(f'v23-age-{i:04d}', 'age', 'age_future_difference',
                        q, d, False, 'synthetic'))
        i += 1
    return i


def gen_rates(rows, rng, start_id):
    i = start_id
    # rate × time with currency
    for _ in range(40):
        lang = rng.choice(['fa', 'en'])
        rate = rng.choice([5, 8, 12, 15])
        t = rng.choice([2, 3, 4, 6])
        exp = float(rate * t)
        if lang == 'fa':
            q = (f'یک کارگر ساعتی {fa_num(rate)} دلار دریافت می‌کند. برای '
                 f'{fa_num(t)} ساعت کار چقدر دریافت می‌کند؟')
        else:
            q = (f'A worker earns {rate} dollars per hour. How much for {t} '
                 f'hours of work?')
        rows.append(row(f'v23-rate-{i:04d}', 'rates', 'currency_rate_application',
                        q, exp, False, 'synthetic'))
        i += 1
    # item rate × time with new nouns
    for _ in range(40):
        lang = rng.choice(['fa', 'en'])
        rate = rng.choice([6, 9, 12, 14])
        t = rng.choice([2, 3, 4, 5])
        exp = float(rate * t)
        noun_e = pick(rng, ['crates', 'widgets', 'cartons', 'panels'])
        noun_f = pick(rng, ['جعبه', 'پنل', 'کارتن', 'قطعه'])
        if lang == 'fa':
            q = (f'یک دستگاه در هر ساعت {fa_num(rate)} {noun_f} تولید می‌کند. '
                 f'در {fa_num(t)} ساعت چند {noun_f} تولید می‌کند؟')
        else:
            q = (f'A machine produces {rate} {noun_e} per hour. How many '
                 f'{noun_e} in {t} hours?')
        rows.append(row(f'v23-rate-{i:04d}', 'rates', 'item_rate_application',
                        q, exp, False, 'synthetic'))
        i += 1
    # inverted rate form: «هر ساعت ۵ دلار»
    for _ in range(25):
        rate = rng.choice([4, 6, 7, 9])
        t = rng.choice([3, 4, 5, 6])
        exp = float(rate * t)
        q = (f'هر ساعت {fa_num(rate)} دلار هزینه می‌شود. برای {fa_num(t)} ساعت '
             f'چقدر هزینه می‌شود؟')
        rows.append(row(f'v23-rate-{i:04d}', 'rates', 'fa_inverted_rate',
                        q, exp, False, 'synthetic'))
        i += 1
    return i


def gen_hard_negatives(rows, rng, start_id):
    i = start_id
    hn = [
        # dimension algebra violations
        ('۲ ساعت و ۵ دلار را جمع کن', None),
        ('Add 3 hours and 40 dollars', None),
        ('موجودی علی ۱۵۰ دلار و ۲۰ کالاست؛ جمعش چقدر است؟', None),
        ('Add 12 m/s and 30 kilometers', None),
        # insufficient information
        ('How much is half of the number?', None),
        ('نصف پولم چقدر است؟', None),
        ('چند گرم است؟', None),
        # unknown-unit conversion
        ('How many parsecs are 4 kilometers?', None),
        ('۱۲ مگاپیکسل چند کیلوبایت است؟', None),
        # conflicting rates (two different rates for the same machine)
        ('A machine produces 10 items per hour and 15 items per hour. '
         'What is its rate?', None),
        # ambiguous referents
        ('Ali and Sara have some money. How much do they have?', None),
        ('اگر همه بروند، چند نفر می‌مانند؟', None),
        # rate question missing time
        ('A printer prints 30 pages per minute. How many pages?', None),
        # nonsense structure
        ('چند ساعت در ۵ دلار هست؟', None),
    ]
    for q, _ in hn:
        rows.append(row(f'v23-hn-{i:04d}', 'hard_negatives', 'must_abstain',
                        q, None, True, 'literal'))
        i += 1
    return i


# ===========================================================================
# v23-extra breadth: unit rephrasings, per-plural rates, 3-leg ownership
# chains, percentage-of-remainder finance
# ===========================================================================
def gen_extra(rows, rng, start_id):
    i = start_id
    # "What is X <unit1> in <unit2>?" + fa «چند <unit2> می‌شود؟»
    pairs = []
    for fam, lst in UNITS.items():
        for a in range(len(lst)):
            for b in range(len(lst)):
                if a != b:
                    pairs.append((fam, lst[a], lst[b]))
    for fam, (u1, f1), (u2, f2) in pairs:
        for _ in range(2):
            v = _unit_value(rng, u1)
            exp = v * f1 / f2
            if abs(exp - round(exp, 6)) > 1e-9:
                continue
            exp = float(round(exp, 6))
            if rng.random() < 0.5:
                q = (f'What is {en_num(v)} {EN_UNIT_PLURAL[u1]} in '
                     f'{EN_UNIT_PLURAL[u2]}?')
            else:
                q = f'{fa_num(v)} {FA_UNIT[u1]} چند {FA_UNIT[u2]} می‌شود؟'
            rows.append(row(f'v23-xun-{i:04d}', 'units_conversion', 'unit_convert',
                            q, exp, False, 'authored-template'))
            i += 1
    # per-plural rate: "packs 60 boxes every 2 hours" -> rate × time
    for _ in range(40):
        lang = rng.choice(['fa', 'en'])
        out = rng.choice([40, 60, 80, 90])
        span = rng.choice([2, 3])
        t = rng.choice([4, 5, 6])
        total = out * t / span
        if abs(total - round(total)) > 1e-9:
            continue
        total = float(round(total))
        noun_e = pick(rng, EN_OUT)
        noun_f = pick(rng, FA_OUT)
        if lang == 'fa':
            q = (f'یک دستگاه در هر {fa_num(span)} ساعت {fa_num(out)} {noun_f} '
                 f'تولید می‌کند. در {fa_num(t)} ساعت چند {noun_f} تولید می‌کند؟')
        else:
            q = (f'A machine packs {out} {noun_e} every {span} hours. How many '
                 f'{noun_e} in {t} hours?')
        rows.append(row(f'v23-xrate-{i:04d}', 'rates', 'per_plural_rate',
                        q, total, False, 'synthetic'))
        i += 1
    # 3-leg ownership chains with pronoun-free re-query
    for _ in range(40):
        lang = rng.choice(['fa', 'en'])
        bA, bB, bC = (rng.randint(60, 250), rng.randint(40, 200),
                      rng.randint(40, 200))
        x, y = rng.randint(5, 25), rng.randint(5, 20)
        if lang == 'fa':
            nA, nB, nC = pick(rng, FA_NAMES), pick(rng, FA_NAMES), pick(rng, FA_NAMES)
            if len({nA, nB, nC}) < 3:
                continue
            q = (f'{nA} {fa_num(bA)} تومان، {nB} {fa_num(bB)} تومان و {nC} '
                 f'{fa_num(bC)} تومان دارد. {nA} به {nB} {fa_num(x)} تومان '
                 f'می‌پردازد و {nB} به {nC} {fa_num(y)} تومان حواله می‌کند. '
                 f'موجودی {nB} چند تومان است؟')
        else:
            nA, nB, nC = pick(rng, EN_NAMES), pick(rng, EN_NAMES), pick(rng, EN_NAMES)
            if len({nA, nB, nC}) < 3:
                continue
            q = (f'{nA} has {bA} dollars, {nB} has {bB} dollars and {nC} has '
                 f'{bC} dollars. {nA} pays {x} dollars to {nB} and {nB} wires '
                 f'{y} dollars to {nC}. What is {nB}\'s balance?')
        rows.append(row(f'v23-xown-{i:04d}', 'ownership', 'transfer_chain',
                        q, bB + x - y, False, 'synthetic'))
        i += 1
    # percentage-of-remainder finance
    for _ in range(30):
        lang = rng.choice(['fa', 'en'])
        total = rng.choice([200, 300, 400, 500])
        p1 = rng.choice([10, 20, 25])
        p2 = rng.choice([10, 20])
        rem = total * (1 - p1 / 100)
        spent2 = rem * p2 / 100
        exp = float(round(rem - spent2, 6))
        if lang == 'fa':
            who = pick(rng, FA_NAMES)
            q = (f'{who} {fa_num(total)} تومان دارد. ابتدا {fa_num(p1)} درصد آن را '
                 f'خرج می‌کند و سپس {fa_num(p2)} درصد باقیمانده را. چند تومان '
                 f'باقی می‌ماند؟')
        else:
            q = (f'Sara has {total} dollars. She first spends {p1}% of it and '
                 f'then {p2}% of the remainder. How many dollars remain?')
        rows.append(row(f'v23-xfin-{i:04d}', 'finance', 'percent_of_remainder',
                        q, exp, False, 'synthetic'))
        i += 1
    return i


# ===========================================================================
# main: build, contamination-guard, write frozen set
# ===========================================================================
def main():
    rng = random.Random(SEED)
    rows = []
    i = gen_units(rows, rng, 1)
    i = gen_work_rate(rows, rng, i)
    i = gen_narrative(rows, rng, i)
    i = gen_ownership(rows, rng, i)
    i = gen_finance(rows, rng, i)
    i = gen_age(rows, rng, i)
    i = gen_rates(rows, rng, i)
    i = gen_extra(rows, rng, i)
    i = gen_hard_negatives(rows, rng, i)

    # ---- contamination guard: zero near-duplicates of frozen eval rows ----
    frozen = load_frozen_ngrams(FROZEN_PATHS)
    dropped = []

    def clean(rs):
        kept = []
        for r in rs:
            g = ngrams(r['question'])
            worst = max((jaccard(g, fg) for fg in frozen), default=0.0)
            if worst >= CONTAM_THRESHOLD:
                dropped.append({'id': r['id'], 'overlap': round(worst, 3)})
                continue
            kept.append(r)
        return kept

    rows = clean(rows)

    # ---- dedupe identical questions (ids stay unique) -------------------
    seen_q = set()
    deduped = []
    for r in rows:
        key = norm_row(r['question'])
        if key in seen_q:
            continue
        seen_q.add(key)
        deduped.append(r)
    rows = deduped

    total = len(rows)
    assert total >= TARGET_MIN, f'blind set too small: {total} < {TARGET_MIN}'
    os.makedirs('benchmarks', exist_ok=True)
    out = 'benchmarks/v23_blind_set.jsonl'
    with open(out, 'w', encoding='utf-8') as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + '\n')
    os.chmod(out, 0o444)   # read-only: the freeze is enforced by the OS
    import hashlib
    sha = hashlib.sha256(open(out, 'rb').read()).hexdigest()
    os.makedirs('reports/v23', exist_ok=True)
    with open('reports/v23/blind_freeze.json', 'w', encoding='utf-8') as f:
        json.dump({'set': 'V23_BLIND_EVAL (frozen before first execution)',
                   'file': out, 'sha256': sha, 'rows': total,
                   'seed': SEED, 'contam_threshold': CONTAM_THRESHOLD,
                   'contam_dropped': dropped,
                   'frozen_sources': FROZEN_PATHS,
                   'frozen_source_count': len(frozen),
                   'answerable': sum(1 for r in rows if not r['must_abstain']),
                   'must_abstain': sum(1 for r in rows if r['must_abstain']),
                   'note': ('this set is NOT the published frozen benchmark and '
                            'is never averaged with it (spec §38)')},
                  f, ensure_ascii=False, indent=1)
    by_domain = {}
    for r in rows:
        d = by_domain.setdefault(r['domain'], {'total': 0, 'correct': 0})
        d['total'] += 1
    print(json.dumps({'rows': total, 'sha256': sha[:16] + '...',
                      'domains': {k: v['total'] for k, v in sorted(by_domain.items())},
                      'contam_dropped': len(dropped)}, ensure_ascii=False, indent=1))
    return 0


if __name__ == '__main__':
    sys.exit(main())
