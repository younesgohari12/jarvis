"""JARVIS v22.4.1 — blind benchmark accounting from RAW chunks (spec §24-§26).

Do not trust the summary: independently aggregate chunk_00..chunk_04 and
assert the totals are mathematically consistent with reports/v22_4/new_blind.json.

Also records per-chunk SHA256 provenance for raw chunks that are NOT shipped
inside the release ZIP (spec §39).

Output: reports/v22_4_1/benchmark_accounting.json
"""
from __future__ import annotations

import hashlib
import json
import os
import sys

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR = os.path.join(BASE, 'reports', 'v22_4_1')
os.makedirs(OUT_DIR, exist_ok=True)
CHUNK_DIR = os.path.join(BASE, 'reports', 'v22_4', 'blind_chunks')

EXPECTED = {
    'total': 2075,
    'correct_total': 1891,
    'answerable': 1749,
    'correct_answerable': 1568,
}


def main() -> int:
    chunks = []
    agg = {'total': 0, 'correct_total': 0, 'answerable': 0,
           'correct_answerable': 0, 'expected_abstain': 0,
           'correct_expected_abstain': 0}
    for i in range(5):
        path = os.path.join(CHUNK_DIR, f'chunk_0{i}.json')
        raw = open(path, 'rb').read()
        d = json.loads(raw)
        t = d.get('tallies', {})
        row = {
            'chunk': f'chunk_0{i}',
            'sha256': hashlib.sha256(raw).hexdigest(),
            'total': d['total'],
            'answerable': d['answerable'],
            'expected_abstain': d.get('expected_abstain'),
            'correct_total': t.get('correct_total'),
            'correct_answerable': t.get('correct_answerable'),
            'correct_expected_abstain': t.get('correct_expected_abstain'),
            'score_numeric': d.get('score_numeric'),
        }
        chunks.append(row)
        agg['total'] += row['total'] or 0
        agg['answerable'] += row['answerable'] or 0
        agg['correct_total'] += row['correct_total'] or 0
        agg['correct_answerable'] += row['correct_answerable'] or 0
        agg['expected_abstain'] += row['expected_abstain'] or 0
        agg['correct_expected_abstain'] += row['correct_expected_abstain'] or 0

    final = json.load(open(os.path.join(BASE, 'reports', 'v22_4', 'new_blind.json')))
    ft = final['tallies']

    # per-chunk internal consistency: correct_total == correct_answerable +
    # correct_expected_abstain, total == answerable + expected_abstain
    per_chunk_consistent = all(
        c['total'] == (c['answerable'] or 0) + (c['expected_abstain'] or 0)
        and c['correct_total'] == (c['correct_answerable'] or 0) + (c['correct_expected_abstain'] or 0)
        for c in chunks)

    checks = {
        'sum_chunk_total_equals_final': agg['total'] == final['total'],
        'sum_chunk_correct_total_equals_final': agg['correct_total'] == ft['correct_total'],
        'sum_chunk_answerable_equals_final': agg['answerable'] == final['answerable'],
        'sum_chunk_correct_answerable_equals_final': agg['correct_answerable'] == ft['correct_answerable'],
        'expected_total_2075': agg['total'] == EXPECTED['total'],
        'expected_correct_1891': agg['correct_total'] == EXPECTED['correct_total'],
        'expected_answerable_1749': agg['answerable'] == EXPECTED['answerable'],
        'expected_correct_answerable_1568': agg['correct_answerable'] == EXPECTED['correct_answerable'],
        'per_chunk_partition_consistent': per_chunk_consistent,
        'score_reproduced': round(100.0 * agg['correct_total'] / agg['total'], 2) == 91.13,
        'answerable_score_reproduced': round(100.0 * agg['correct_answerable'] / agg['answerable'], 2) == 89.65,
    }
    all_ok = all(checks.values())

    # family accounting from new_blind (spec §26 — honest family results)
    by_family = final.get('by_family', {})
    family_total = sum(v['total'] for v in by_family.values())
    family_correct = sum(v['correct'] for v in by_family.values())
    checks['family_totals_sum_to_total'] = family_total == agg['total']
    checks['family_corrects_sum_to_correct'] = family_correct == agg['correct_total']

    report = {
        'release': 'v0.11.0-intelligence-v22.4.1',
        'purpose': 'independent raw-chunk aggregation of the frozen v22.4 blind run',
        'benchmark_sha256': final['benchmark_sha256'],
        'frozen_sha_required': '063aef8f732c845e77a34c6916749d732e96489bbb74edfe12a3a8afc13f7be1',
        'benchmark_sha_matches_frozen': final['benchmark_sha256'] == FROZEN_SHA_VAL,
        'chunks': chunks,
        'aggregated': agg,
        'final_summary': {'total': final['total'], 'answerable': final['answerable'],
                          'tallies': ft, 'score_numeric': final['score_numeric'],
                          'score_answerable_only': final['score_answerable_only']},
        'checks': checks,
        'all_consistent': all_ok,
        'family_results': {k: f"{v['correct']}/{v['total']}"
                           for k, v in sorted(by_family.items())},
        'chunk_provenance_note': ('raw chunk files are excluded from the release '
                                  'ZIP; their SHA256 hashes above + the frozen '
                                  'benchmark SHA allow full reproduction'),
    }
    with open(os.path.join(OUT_DIR, 'benchmark_accounting.json'), 'w',
              encoding='utf-8') as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(json.dumps({'aggregated': agg, 'all_consistent': all_ok,
                      'failed_checks': [k for k, v in checks.items() if not v]},
                     ensure_ascii=False))
    return 0 if all_ok else 1


FROZEN_SHA_VAL = '063aef8f732c845e77a34c6916749d732e96489bbb74edfe12a3a8afc13f7be1'

if __name__ == '__main__':
    sys.exit(main())
