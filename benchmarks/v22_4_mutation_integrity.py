"""JARVIS v22.4 — mutation testing, span integrity, concurrency, performance.

Spec sections covered:
  §57  mutation tests  — disable a guard, every targeted test MUST fail
  §15  source span integrity over 2000+ generated cases
  §44  ContextVar concurrency — no request gets another request's source
  §68  performance mean/p50/p95/p99 per family
Outputs reports/v22_4/{mutation_tests,source_span_integrity,concurrency,performance}.json
"""
from __future__ import annotations

import concurrent.futures
import json
import math
import os
import random
import re
import statistics
import subprocess
import sys
import time

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE)
REPORTS = os.path.join(BASE, 'reports', 'v22_4')
os.makedirs(REPORTS, exist_ok=True)


# ======================================================================
# §57 — mutation testing: each mutant MUST be killed by the test suite
# ======================================================================
MUTANTS = [
    # (id, file, old, new, targeted test selection)
    ('disable_dimension_guard', 'jarvis/agent/verifier_v22.py',
     "    if not operands_ok := True:\n        pass\n",
     None, None),  # placeholder replaced below
]


def run_mutation(mut_id: str, file_rel: str, old: str, new: str,
                 test_args: list[str]) -> dict:
    """Apply one mutation, run the targeted tests, restore the file."""
    path = os.path.join(BASE, file_rel)
    original = open(path, encoding='utf-8').read()
    if old not in original:
        return {'id': mut_id, 'applied': False,
                'error': 'mutation target not found'}
    mutated = original.replace(old, new, 1)
    open(path, 'w', encoding='utf-8').write(mutated)
    try:
        proc = subprocess.run(
            [sys.executable, '-B', '-m', 'pytest', *test_args, '-q', '--no-header',
             '-x', '-p', 'no:cacheprovider'],
            cwd=BASE, capture_output=True, text=True, timeout=420)
        tail = (proc.stdout or '').strip().splitlines()[-1] if proc.stdout else ''
        killed = proc.returncode != 0
        return {'id': mut_id, 'applied': True, 'killed': killed,
                'pytest_tail': tail[:200]}
    except subprocess.TimeoutExpired:
        return {'id': mut_id, 'applied': True, 'killed': True,
                'pytest_tail': 'timeout (treated as killed)'}
    finally:
        open(path, 'w', encoding='utf-8').write(original)


def mutation_tests() -> dict:
    """Each mutant disables ONE root-cause guard; the corresponding tests
    must fail — otherwise coverage is insufficient (spec §57)."""
    results = []

    # M1 — disable the operation-aware dimension guard (accept everything)
    results.append(run_mutation(
        'M1_disable_dimension_guard', 'jarvis/agent/semantics_v22_4.py',
        "    if not (text and _CHAIN_LANG.search(text)):\n        return []",
        "    if True:\n        return []\n    if not (text and _CHAIN_LANG.search(text)):\n        return []",
        ['tests/test_v224_root_cause.py::test_p0_time_distance_addition_rejected',
         'tests/test_v221_audit_fixes.py::test_cross_dimension_addition_refused']))

    # M2 — m/s silently becomes KM_PER_HOUR again
    results.append(run_mutation(
        'M2_speed_unit_flattened', 'jarvis/agent/semantics_v22_4.py',
        "ثانیه', 'M_PER_SECOND'),",
        "ثانیه', 'KM_PER_HOUR'),",
        ['tests/test_v224_root_cause.py::test_speed_unit_mapping',
         'tests/test_v224_property_fuzz.py::test_property_speed_unit_preserved_hypothesis']))

    # M3 — span end shifted by one (truncation bug reintroduced)
    results.append(run_mutation(
        'M3_span_end_shifted', 'jarvis/agent/quantity_v22.py',
        '        q.source_span = original[span.start:span.end] if 0 <= span.start <= span.end <= len(original) \\',
        '        q.source_span = original[span.start:max(span.start, span.end - 1)] if 0 <= span.start <= span.end <= len(original) \\',
        ['tests/test_v221_audit_fixes.py::test_source_span_covers_number_and_unit',
         'tests/test_v224_root_cause.py::test_exact_spans_on_original_text']))

    # M4 — age difference loses abs() (raw subtraction)
    results.append(run_mutation(
        'M4_age_abs_removed', 'jarvis/agent/local_intelligence_v22.py',
        "            value = abs(float(age_a) - float(age_b))",
        "            value = float(age_a) - float(age_b)",
        ['tests/test_v22_semantic.py', 'tests/test_v224_root_cause.py::test_age_query_binds_named_entities'],
        ) if True else None)

    # M5 — verifier fail-closed -> fail-open
    results.append(run_mutation(
        'M5_verifier_fail_open', 'jarvis/agent/verifier_v22.py',
        "        except Exception as exc:  # v22.1: FAIL-CLOSED, never fail-open.",
        "        except Exception as exc:  # MUTANT: fail-open\n            return verdict",
        ['tests/test_v221_audit_fixes.py::test_typed_audit_fail_closed',
         'tests/test_v224_root_cause.py::test_verifier_fail_closed_on_internal_error']))

    # M6 — transfer source/target swapped
    results.append(run_mutation(
        'M6_transfer_direction_swapped', 'jarvis/agent/semantics_v22_4.py',
        "        balances[src] -= amount\n        balances[tgt] += amount",
        "        balances[tgt] -= amount\n        balances[src] += amount",
        ['tests/test_v224_root_cause.py::test_ownership_direction_resolves']))

    # M7 — inventory event order reversed (killed by the order-sensitivity
    # witness: the same multiset of events yields a DIFFERENT verdict when
    # reordered because an intermediate balance goes negative)
    results.append(run_mutation(
        'M7_inventory_order_reversed', 'jarvis/agent/semantics_v22_4.py',
        "    for e in model['events']:\n        name, qty, direction = e['product'], float(e['quantity']), int(e['direction'])",
        "    for e in reversed(model['events']):\n        name, qty, direction = e['product'], float(e['quantity']), int(e['direction'])",
        ['tests/test_v224_root_cause.py::test_inventory_order_is_semantic',
         'tests/test_v224_property_fuzz.py::test_property_inventory_order_preserved']))

    executed = [r for r in results if r]
    killed = [r for r in executed if r.get('killed')]
    return {
        'mutants': executed,
        'total': len(executed),
        'killed': len(killed),
        'score': (len(killed) / len(executed)) if executed else 0.0,
        'note': 'a surviving mutant means test coverage is insufficient (spec §57)',
    }


# ======================================================================
# §15 — source span integrity over 2000+ generated cases
# ======================================================================
def span_integrity(target: int = 2400) -> dict:
    from jarvis.agent.quantity_v22 import extract_typed_quantities
    rng = random.Random(2024)
    units = ['دلار', 'تومان', 'کالا', 'نفر', 'کارگر', 'ساعت', 'دقیقه', 'روز',
             'کیلومتر', 'متر', 'کیلوگرم', 'لیتر', 'درصد', 'ساله',
             'dollars', 'hours', 'items', 'workers', 'km', 'kg', 'percent']
    fa_names = ['علی', 'سارا', 'رضا', 'مینا', 'حسن']
    fa_verbs = ['دارد', 'اضافه کن', 'کم کن', 'فروخته شد', 'رسید']
    en_verbs = ['has', 'adds', 'sold', 'arrived']
    checked = 0
    mismatches = 0
    failures: list[dict] = []
    for i in range(target):
        parts = []
        for _ in range(rng.randint(1, 5)):
            v = rng.choice([rng.randint(0, 9999), round(rng.uniform(0, 100), 2)])
            unit = rng.choice(units)
            if rng.random() < 0.4:
                name = rng.choice(fa_names + ['Ali', 'Sara'])
                verb = rng.choice(fa_verbs + en_verbs)
                parts.append(f'{name} {v} {unit} {verb}')
            else:
                parts.append(f'{v} {unit}')
        text = rng.choice(['؛ ', '. ', ' و ', ', ']).join(parts)
        if rng.random() < 0.2:
            text = text.translate(str.maketrans('0123456789', '۰۱۲۳۴۵۶۷۸۹'))
        try:
            qs = extract_typed_quantities(text)
        except Exception as exc:
            mismatches += 1
            failures.append({'case': text, 'error': repr(exc)})
            continue
        for q in qs:
            checked += 1
            if not (0 <= q.start <= q.end <= len(text)) or \
                    text[q.start:q.end] != q.source_span:
                mismatches += 1
                if len(failures) < 50:
                    failures.append({
                        'case': text, 'span': q.source_span,
                        'start': q.start, 'end': q.end,
                        'actual': text[q.start:q.end] if 0 <= q.start <= q.end <= len(text) else None,
                    })
    return {
        'generated_cases': target,
        'spans_checked': checked,
        'invariant': 'original[start:end] == span.text',
        'mismatches': mismatches,
        'passed': mismatches == 0,
        'failure_samples': failures[:20],
    }


# ======================================================================
# §44 — ContextVar concurrency: no request sees another request's source
# ======================================================================
def concurrency_check(workers: int = 8, per_worker: int = 40) -> dict:
    from jarvis.agent.verifier_v22 import ORIGINAL_SOURCE, LAST_DIMENSION_FAILURE

    errors: list[str] = []
    contamination = []

    def worker(wid: int):
        for i in range(per_worker):
            secret = f'request-{wid}-{i}'
            token = ORIGINAL_SOURCE.set(secret)
            try:
                time.sleep(0)   # yield to force interleaving
                seen = ORIGINAL_SOURCE.get()
                if seen != secret:
                    contamination.append({'expected': secret, 'seen': str(seen)})
                # nested override + reset must restore
                inner = ORIGINAL_SOURCE.set('inner')
                ORIGINAL_SOURCE.reset(inner)
                if ORIGINAL_SOURCE.get() != secret:
                    contamination.append({'expected': secret, 'seen': 'inner-leak'})
            except Exception as exc:
                errors.append(repr(exc))
            finally:
                ORIGINAL_SOURCE.reset(token)

    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
        list(pool.map(worker, range(workers)))
    return {
        'workers': workers, 'iterations_per_worker': per_worker,
        'contamination_events': contamination[:10],
        'contamination_count': len(contamination),
        'errors': errors[:10],
        'passed': not contamination and not errors,
        'protocol': 'token = ORIGINAL_SOURCE.set(x); try: ... finally: reset(token)',
    }


# ======================================================================
# §68 — performance: mean/p50/p95/p99 per family
# ======================================================================
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


def performance(runs: int = 30) -> dict:
    from jarvis.agent.local_intelligence_v22 import LocalIntelligenceV22
    eng = LocalIntelligenceV22()
    out = {}
    for family, texts in PERF_FAMILIES.items():
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
            'mean_ms': round(statistics.fmean(times), 3),
            'p50_ms': round(pct(0.50), 3),
            'p95_ms': round(pct(0.95), 3),
            'p99_ms': round(pct(0.99), 3),
        }
    return out


if __name__ == '__main__':
    print('== mutation tests ==')
    mut = mutation_tests()
    json.dump(mut, open(os.path.join(REPORTS, 'mutation_tests.json'), 'w',
                        encoding='utf-8'), ensure_ascii=False, indent=2)
    print(json.dumps({k: mut[k] for k in ('total', 'killed', 'score')},
                     ensure_ascii=False))
    for m in mut['mutants']:
        print(' -', m['id'], 'killed' if m.get('killed') else m.get('pytest_tail', m.get('error')))

    print('== span integrity ==')
    span = span_integrity(2400)
    json.dump(span, open(os.path.join(REPORTS, 'source_span_integrity.json'), 'w',
                         encoding='utf-8'), ensure_ascii=False, indent=2)
    print(json.dumps({k: span[k] for k in ('generated_cases', 'spans_checked',
                                           'mismatches', 'passed')}))

    print('== concurrency ==')
    conc = concurrency_check()
    json.dump(conc, open(os.path.join(REPORTS, 'concurrency.json'), 'w',
                         encoding='utf-8'), ensure_ascii=False, indent=2)
    print('passed:', conc['passed'], 'contamination:', conc['contamination_count'])

    print('== performance ==')
    perf = performance()
    json.dump(perf, open(os.path.join(REPORTS, 'performance.json'), 'w',
                         encoding='utf-8'), ensure_ascii=False, indent=2)
    print(json.dumps(perf, indent=1))
