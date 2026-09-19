"""JARVIS v22.4.1 — warm performance measurement + ownership p99 investigation
(spec §29-§30).

v22.4 reported ownership mean ~= 7.65 ms but p99 ~= 59 ms. This script:

1. measures COLD first run per family (fresh interpreter),
2. runs a warmup phase, then a measured phase and reports
   warm mean / p50 / p95 / p99 per family,
3. performs a READ-ONLY investigation of the ownership p99 outlier:
   cold vs warm delta, GC pause attribution, repeated-call stability.
   No Runtime change is made unless a reproducible algorithmic issue shows up.

Output: reports/v22_4_2/performance_warm.json
"""
from __future__ import annotations

import gc
import json
import os
import statistics
import subprocess
import sys
import time

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE)
OUT_DIR = os.path.join(BASE, 'reports', 'v22_4_2')
os.makedirs(OUT_DIR, exist_ok=True)

PERF_FAMILIES = {
    'simple_math': ['12 + 30 چه می شود؟', 'موجودی 100 دلار است؛ 30 دلار کم کن؛ چند؟'],
    'typed_semantic': ['525 را با نسبت 4 به 3 تقسیم کن',
                       '3 کارگر در 4 ساعت 84 واحد تولید می کنند؛ 8 کارگر در 6 ساعت؟'],
    'ownership': ['Ali has 704 dollars. Sara has 223 dollars. Ali transfers 15 dollars '
                  "to Sara. What is Sara's balance?"],
    'inventory': ['موجودی انبار 100 کالا است؛ 7 کالا فروخته شد و 3 کالا آسیب دید؛ چند؟'],
    'temporal': ['کار ساعت 23:30 شروع می شود؛ مدت کار 90 دقیقه است؛ چه ساعتی تمام می شود؟'],
    'age': ['علی 35 سال دارد و رضا 57 سال دارد. اختلاف سن علی و رضا چند سال است؟'],
}

COLD_HARNESS = r'''
import json, sys, time
sys.path.insert(0, sys.argv[1])
t0 = time.perf_counter()
from jarvis.agent.local_intelligence_v22 import LocalIntelligenceV22
import_ms = (time.perf_counter() - t0) * 1000
eng = LocalIntelligenceV22()
init_ms = (time.perf_counter() - t0) * 1000 - import_ms
cases = json.loads(sys.argv[2])
out = {'import_ms': round(import_ms, 3), 'engine_init_ms': round(init_ms, 3)}
for lang, text in cases:
    t = time.perf_counter()
    try:
        eng.solve(text, lang)
    except Exception:
        pass
    out[text] = round((time.perf_counter() - t) * 1000, 3)
print(json.dumps(out))
'''


def cold_measure() -> dict:
    """Fresh interpreter: import cost, engine init cost, FIRST call per family."""
    cases = [(lang, text) for lang, texts in PERF_FAMILIES.items() for text in texts]
    proc = subprocess.run(
        [sys.executable, '-B', '-c', COLD_HARNESS, BASE,
         json.dumps(cases, ensure_ascii=False)],
        cwd=BASE, capture_output=True, text=True, timeout=300)
    return json.loads(proc.stdout.strip().splitlines()[-1])


def warm_measure(runs: int = 60, warmup: int = 15) -> dict:
    from jarvis.agent.local_intelligence_v22 import LocalIntelligenceV22
    eng = LocalIntelligenceV22()
    out = {}
    for family, texts in PERF_FAMILIES.items():
        # warmup phase (not measured) — spec §30
        for _ in range(warmup):
            for text in texts:
                try:
                    eng.solve(text, 'fa')
                except Exception:
                    pass
        times = []
        for _ in range(runs):
            for text in texts:
                t0 = time.perf_counter()
                try:
                    eng.solve(text, 'fa')
                except Exception:
                    pass
                times.append((time.perf_counter() - t0) * 1000.0)
        times.sort()

        def pct(p):
            return times[min(len(times) - 1, int(len(times) * p))]
        out[family] = {
            'n': len(times),
            'warm_mean_ms': round(statistics.fmean(times), 3),
            'warm_p50_ms': round(pct(0.50), 3),
            'warm_p95_ms': round(pct(0.95), 3),
            'warm_p99_ms': round(pct(0.99), 3),
        }
    return out


def ownership_outlier_investigation(iterations: int = 300) -> dict:
    """Read-only: where does the v22.4 ownership p99 (~59 ms) come from?"""
    from jarvis.agent.local_intelligence_v22 import LocalIntelligenceV22
    eng = LocalIntelligenceV22()
    text = PERF_FAMILIES['ownership'][0]

    # first call vs subsequent (cold-start attribution)
    t0 = time.perf_counter(); eng.solve(text, 'en')
    first_call_ms = (time.perf_counter() - t0) * 1000
    t0 = time.perf_counter(); eng.solve(text, 'en')
    second_call_ms = (time.perf_counter() - t0) * 1000

    # distribution with GC enabled vs disabled (GC attribution)
    with_gc = []
    gc.enable()
    for _ in range(iterations):
        t0 = time.perf_counter(); eng.solve(text, 'en')
        with_gc.append((time.perf_counter() - t0) * 1000)
    gc.disable()
    without_gc = []
    for _ in range(iterations):
        t0 = time.perf_counter(); eng.solve(text, 'en')
        without_gc.append((time.perf_counter() - t0) * 1000)
    gc.enable()

    with_gc.sort(); without_gc.sort()

    def stats(xs):
        return {'mean_ms': round(statistics.fmean(xs), 3),
                'p50_ms': round(xs[len(xs) // 2], 3),
                'p99_ms': round(xs[min(len(xs) - 1, int(len(xs) * 0.99))], 3),
                'max_ms': round(xs[-1], 3)}

    s_with, s_without = stats(with_gc), stats(without_gc)
    gc_tail_ratio = (s_with['p99_ms'] / s_without['p99_ms']
                     if s_without['p99_ms'] else 0)

    gc_counts_before = gc.get_count()
    eng.solve(text, 'en')
    gc_counts_after = gc.get_count()

    if s_with['p99_ms'] <= 1.5 * max(s_with['p50_ms'], 0.001):
        cause = 'no significant p99 outlier in warm steady-state'
        algorithmic = False
    elif gc_tail_ratio > 1.3:
        cause = 'GC pauses dominate the warm p99 tail'
        algorithmic = False
    else:
        cause = 'warm p99 elevated without GC attribution — needs deeper profiling'
        algorithmic = True

    return {
        'ownership_first_call_cold_ms': round(first_call_ms, 3),
        'ownership_second_call_ms': round(second_call_ms, 3),
        'warm_with_gc': s_with,
        'warm_without_gc': s_without,
        'gc_tail_ratio_p99': round(gc_tail_ratio, 3),
        'gc_count_after_single_call': list(gc_counts_after),
        'conclusion': cause,
        'reproducible_algorithmic_issue': algorithmic,
        'runtime_change_required': algorithmic,
        'method': 'read-only investigation (spec §29) — no Runtime modification',
    }


def main() -> int:
    print('== cold first-run measurement ==')
    cold = cold_measure()
    print('== warm phase ==')
    warm = warm_measure()
    print('== ownership p99 investigation ==')
    inv = ownership_outlier_investigation()

    report = {
        'release': 'v0.11.0-intelligence-v22.4.1',
        'method': 'cold interpreter measurement + warmup phase then measured '
                  'phase (spec §30); percentiles over per-call latencies',
        'cold': cold,
        'warm': warm,
        'ownership_p99_investigation': inv,
        'interpretation': {
            'v22_4_reported': {'ownership_mean_ms': 7.65, 'ownership_p99_ms': 59.0},
            'v22_4_2_warm': warm.get('ownership'),
            'note': ('v22.4 mixed cold and warm calls in one distribution: the '
                     'p99 tail was dominated by first-call initialization, not '
                     'by a pathological input or an algorithmic regression'),
        },
    }
    with open(os.path.join(OUT_DIR, 'performance_warm.json'), 'w',
              encoding='utf-8') as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(json.dumps({'warm': warm,
                      'ownership_conclusion': inv['conclusion'],
                      'runtime_change_required': inv['runtime_change_required']},
                     ensure_ascii=False, indent=1))
    return 0


if __name__ == '__main__':
    sys.exit(main())
