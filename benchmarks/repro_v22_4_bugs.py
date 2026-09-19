"""v22.4 P0 bug reproduction against the v22.3 baseline.

Every case here documents ONE architectural defect of v22.3 that v22.4 must
eliminate at its root. Output: reports/v22_4/p0_reproduction.json
"""
from __future__ import annotations
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from jarvis.agent.quantity_v22 import extract_typed_quantities  # noqa: E402
from jarvis.agent.verifier_v22 import cross_dimension_chain_failure  # noqa: E402
from jarvis.agent.language_brain_v22 import detect_age_difference  # noqa: E402
from jarvis.agent.local_intelligence_v22 import LocalIntelligenceV22  # noqa: E402
from jarvis.agent import temporal_v22  # noqa: E402

ENGINE = LocalIntelligenceV22()
REPORTS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'reports', 'v22_4')
os.makedirs(REPORTS, exist_ok=True)


def solve(text, language='fa'):
    answer = ENGINE.solve(text, language)
    return {
        'text': text,
        'answer': None if answer is None else answer.text,
        'intent': None if answer is None else answer.intent,
        'checks': None if answer is None else list(answer.checks)[:6],
    }


def repro():
    out = {}

    # ------------------------------------------------------------------
    # BUG 1 — time + currency escapes the dimension guard
    # ------------------------------------------------------------------
    q = extract_typed_quantities('2 hours + 5 dollars')
    guard = cross_dimension_chain_failure(q, '2 hours + 5 dollars را جمع کن')
    out['bug1_time_currency_escape'] = {
        'input': '2 hours + 5 dollars',
        'dimensions': [x.dimension for x in q],
        'cross_dimension_guard': guard,
        'rejected': bool(guard),
        'expected': 'rejected (time + currency must not add)',
    }

    # ------------------------------------------------------------------
    # BUG 2 — speed units: 10 m/s silently becomes KM_PER_HOUR
    # ------------------------------------------------------------------
    q = extract_typed_quantities('10 m/s')
    out['bug2_speed_unit_wrong'] = {
        'input': '10 m/s',
        'unit': q[0].unit if q else None,
        'expected_unit': 'M_PER_SECOND',
        'wrong': bool(q) and q[0].unit == 'KM_PER_HOUR',
    }
    q = extract_typed_quantities('سرعت 10 متر بر ثانیه')
    out['bug2b_speed_unit_fa_wrong'] = {
        'input': 'سرعت 10 متر بر ثانیه',
        'unit': q[0].unit if q else None,
        'expected_unit': 'M_PER_SECOND',
    }

    # ------------------------------------------------------------------
    # BUG 3 — source span truncated (stripped-substring offsets reused)
    # ------------------------------------------------------------------
    spans = {}
    for text in ('50 کالا', '3 workers', '8 نفر'):
        qs = extract_typed_quantities(text)
        spans[text] = [
            {'span': x.source_span, 'start': x.start, 'end': x.end,
             'invariant_ok': 0 <= x.start <= x.end <= len(text) and text[x.start:x.end] == x.source_span}
            for x in qs
        ]
    out['bug3_source_span_truncated'] = spans

    # ------------------------------------------------------------------
    # BUG 4 — age semantics regex-bound (aged / possessive / سال دارد)
    # ------------------------------------------------------------------
    age_cases = {
        'en_aged': 'Ali, aged 35, and Reza, aged 57. What is the age difference between Ali and Reza?',
        'en_possessive': "Ali's age is 35 and Reza's age is 57. What is the age difference between Ali and Reza?",
        'fa_dare': 'علی 35 سال دارد و رضا 57 سال دارد. اختلاف سن علی و رضا چند سال است؟',
        'fa_sen_sal_est': 'سن علی 35 سال است و سن رضا 57 سال است. اختلاف سن علی و رضا چند سال است؟',
    }
    out['bug4_age_regex_bound'] = {
        k: {'detection': str(detect_age_difference(v)), 'detected': detect_age_difference(v) is not None}
        for k, v in age_cases.items()
    }

    # ------------------------------------------------------------------
    # BUG 5 — dimension failure metadata not structured
    # ------------------------------------------------------------------
    r = solve('2 ساعت و 120 کیلومتر را جمع کن')
    out['bug5_unstructured_failure'] = {
        'input': '2 ساعت و 120 کیلومتر را جمع کن',
        'answer': r['answer'],
        'note': 'message claims money/items although the true mismatch is time vs distance',
    }

    # ------------------------------------------------------------------
    # BUG 6 — ownership transfer patterns too narrow
    # ------------------------------------------------------------------
    ownership_cases = {
        'en_pays': 'Ali has 704 dollars. Sara has 223 dollars. Ali pays 15 dollars to Sara. What is Sara\'s balance?',
        'en_receives': 'Ali has 704 dollars. Sara has 223 dollars. Sara receives 15 dollars from Ali. What is Sara\'s balance?',
        'fa_miferestad': 'علی 704 دلار دارد. سارا 223 دلار دارد. علی 15 دلار برای سارا می‌فرستد. موجودی سارا چند است؟',
    }
    out['bug6_ownership_narrow'] = {k: solve(v, 'en' if k.startswith('en') else 'fa')
                                    for k, v in ownership_cases.items()}

    # ------------------------------------------------------------------
    # BUG 7 — scheduling frame gap (train departs)
    # ------------------------------------------------------------------
    out['bug7_scheduling_train'] = solve(
        'A train departs at 23:30. The trip takes 45 minutes. What time does it arrive?', 'en')

    # ------------------------------------------------------------------
    # BUG 8 — price delta misclassification
    # ------------------------------------------------------------------
    from jarvis.agent.numeric_roles_v22 import classify_source_numbers_v2
    roles = classify_source_numbers_v2('The price was reduced by 20 dollars.')
    out['bug8_price_delta_role'] = {
        'input': 'The price was reduced by 20 dollars.',
        'roles': [(r_['value'], r_['role']) for r_ in roles],
        'expected': '20 -> price_delta',
    }

    # ------------------------------------------------------------------
    # BUG 9 — inventory damage/return events
    # ------------------------------------------------------------------
    inv = solve('موجودی انبار 100 کالا است؛ 7 کالا فروخته شد و 3 کالا آسیب دید و 2 کالا مرجوع شد. موجودی چند است؟')
    out['bug9_inventory_events'] = inv

    # ------------------------------------------------------------------
    # BUG 10 — release metadata inconsistency (static facts)
    # ------------------------------------------------------------------
    out['bug10_metadata_inconsistency'] = {
        'RELEASE_V22.json metrics.regression': '1031 passed + 585 subtests',
        'RELEASE_V22.json test_suites': 'full tests/ regression (1042)',
        'v22_2_addendum verification': '1037 tests + 585 subtests',
        'actual_baseline_run': '1042 tests + 585 subtests, 0 failures',
    }

    return out


if __name__ == '__main__':
    data = repro()
    path = os.path.join(REPORTS, 'p0_reproduction.json')
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    print(json.dumps(data, ensure_ascii=False, indent=1)[:6000])
    print('saved ->', path)
