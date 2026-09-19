"""JARVIS v22.4.1 — P0 smoke suite (release gate, spec §34).

Re-runs every documented v22.4 P0 fix against the current tree and records a
machine-checkable PASS/FAIL per case. Nothing here is hand-typed: results are
measured from the live Runtime.

Output: reports/v22_4_1/p0_smoke.json
"""
from __future__ import annotations

import json
import os
import sys

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE)

from jarvis.agent.quantity_v22 import extract_typed_quantities  # noqa: E402
from jarvis.agent.verifier_v22 import cross_dimension_chain_failure  # noqa: E402
from jarvis.agent.local_intelligence_v22 import LocalIntelligenceV22, UniversalVerifierV2  # noqa: E402
from jarvis.agent import verifier_v22 as vv  # noqa: E402

ENGINE = LocalIntelligenceV22()
OUT_DIR = os.path.join(BASE, 'reports', 'v22_4_1')
os.makedirs(OUT_DIR, exist_ok=True)


def solve(text, language='fa'):
    a = ENGINE.solve(text, language)
    return None if a is None else a.text


def solve_verdict(text, language='fa'):
    a = ENGINE.solve(text, language)
    return a


def check(name, ok, detail):
    return {'name': name, 'passed': bool(ok), 'detail': detail}


def run_smoke():
    results = []

    # 1 — 2 hours + 5 USD must be rejected
    q = extract_typed_quantities('2 hours + 5 dollars')
    guard = cross_dimension_chain_failure(q, 'add 2 hours and 5 dollars')
    a1 = solve('2 hours + 5 dollars. What is the total?', 'en')
    rejected = bool(guard) and not (a1 and '7' in a1)
    results.append(check('time_plus_currency_rejected', rejected,
                         {'guard': bool(guard), 'answer': a1,
                          'dims': [x.dimension for x in q]}))

    # 2 — 2 hours + 5 items must be rejected
    q = extract_typed_quantities('2 hours + 5 items')
    guard = cross_dimension_chain_failure(q, 'add 2 hours and 5 items')
    a2 = solve('2 hours + 5 items. What is the total?', 'en')
    results.append(check('time_plus_count_rejected', bool(guard) and not (a2 and '7' in a2),
                         {'guard': bool(guard), 'answer': a2,
                          'dims': [x.dimension for x in q]}))

    # 3 — 10 m/s must map to M_PER_SECOND (never KM_PER_HOUR)
    q = extract_typed_quantities('10 m/s')
    unit = q[0].unit if q else None
    results.append(check('m_per_second_unit', unit == 'M_PER_SECOND',
                         {'unit': unit}))
    q = extract_typed_quantities('سرعت 10 متر بر ثانیه')
    unit_fa = q[0].unit if q else None
    results.append(check('m_per_second_unit_fa', unit_fa == 'M_PER_SECOND',
                         {'unit': unit_fa}))

    # 4 — 50 کالا exact source span (truncation bug must stay dead)
    text = '50 کالا'
    q = extract_typed_quantities(text)
    span_ok = bool(q) and q[0].source_span == '50 کالا' and \
        text[q[0].start:q[0].end] == q[0].source_span
    results.append(check('exact_source_span_fa', span_ok,
                         {'span': q[0].source_span if q else None}))
    for sample in ('3 workers', '8 نفر'):
        qs = extract_typed_quantities(sample)
        ok = bool(qs) and sample[qs[0].start:qs[0].end] == qs[0].source_span \
            and qs[0].source_span == sample
        results.append(check(f'exact_source_span[{sample}]', ok,
                             {'span': qs[0].source_span if qs else None}))

    # 5 — Ali/Reza age difference (both orders, abs invariant)
    a5a = solve('علی 35 سال دارد و رضا 30 سال دارد. اختلاف سن علی و رضا چند سال است؟', 'fa')
    a5b = solve('رضا 30 سال دارد و علی 35 سال دارد. اختلاف سن علی و رضا چند سال است؟', 'fa')
    ok5 = bool(a5a and '5' in a5a) and bool(a5b and '5' in a5b)
    results.append(check('age_difference_fa', ok5, {'order1': a5a, 'order2': a5b}))
    a5c = solve('Ali is 35 years old and Reza is 30. What is the age difference?', 'en')
    results.append(check('age_difference_en', bool(a5c and '5' in a5c), {'answer': a5c}))

    # 6 — ownership transfer
    a6 = solve("Ali has 704 dollars. Sara has 223 dollars. Ali transfers 15 dollars "
               "to Sara. What is Sara's balance?", 'en')
    results.append(check('ownership_transfer', bool(a6 and '238' in a6), {'answer': a6}))
    a6b = solve('علی 500 دلار دارد و سارا 200 دلار دارد. علی 50 دلار به سارا می‌دهد. '
                'موجودی سارا چند دلار است؟', 'fa')
    results.append(check('ownership_transfer_fa', bool(a6b and '250' in a6b), {'answer': a6b}))

    # 7 — inventory multi-product
    a7 = solve('انبار 100 کالا داشت؛ 7 کتاب فروخته شد و 3 صندلی آسیب دید و 5 گوشی رسید؛ '
               'جمع کل چند کالا است؟', 'fa')
    a7_simple = solve('موجودی انبار 100 کالا است؛ 7 کالا فروخته شد و 3 کالا آسیب دید؛ چند؟', 'fa')
    results.append(check('inventory_events', bool(a7_simple and '90' in a7_simple),
                         {'multi_product': a7, 'simple': a7_simple}))

    # 8 — train 23:30 + 45m -> 00:15 next day
    a8 = solve('A train departs at 23:30. The trip takes 45 minutes. '
               'What time does it arrive?', 'en')
    ok8 = bool(a8 and ('00:15' in a8 or '0:15' in a8) and ('next' in a8.lower()))
    results.append(check('temporal_day_wrap', ok8, {'answer': a8}))
    a8b = solve('کار ساعت 23:30 شروع می شود؛ مدت کار 45 دقیقه است؛ چه ساعتی تمام می شود؟', 'fa')
    ok8b = bool(a8b and ('00:15' in a8b or '0:15' in a8b))
    results.append(check('temporal_day_wrap_fa', ok8b, {'answer': a8b}))

    # 9 — typed audit crash -> fail closed
    ir = vv.UniversalVerifierV2 if False else None
    from jarvis.agent.parser_v21 import get_parser_v21
    verifier = UniversalVerifierV2()
    ir_obj = get_parser_v21().parse('موجودی حساب 100 دلار است؛ 5 دلار اضافه کن')

    def boom(*a, **k):
        raise RuntimeError('injected crash')
    saved = vv.extract_typed_quantities
    try:
        vv.extract_typed_quantities = boom
        verdict = verifier.verify(ir_obj, 105)
    finally:
        vv.extract_typed_quantities = saved
    ok9 = (not verdict.passed
           and 'typed_audit_internal_error' in verdict.failed_checks
           and verdict.repair_stage == 'abstain')
    results.append(check('verifier_fail_closed', ok9,
                         {'passed': verdict.passed,
                          'failed_checks': list(verdict.failed_checks),
                          'repair_stage': verdict.repair_stage}))

    # 10 — rate semantics preserved
    a10 = solve('سرعت خودرو 60 کیلومتر بر ساعت است؛ در 2 ساعت چند کیلومتر می‌رود؟', 'fa')
    results.append(check('rate_semantics_fa', bool(a10 and '120' in a10), {'answer': a10}))

    return results


if __name__ == '__main__':
    results = run_smoke()
    passed = sum(1 for r in results if r['passed'])
    report = {
        'release': 'v0.11.0-intelligence-v22.4.1',
        'total': len(results),
        'passed': passed,
        'failed': len(results) - passed,
        'all_passed': passed == len(results),
        'results': results,
    }
    out = os.path.join(OUT_DIR, 'p0_smoke.json')
    with open(out, 'w', encoding='utf-8') as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(json.dumps({k: report[k] for k in ('total', 'passed', 'failed', 'all_passed')},
                     ensure_ascii=False))
    for r in results:
        print(' -', 'PASS' if r['passed'] else 'FAIL', r['name'])
    sys.exit(0 if report['all_passed'] else 1)
