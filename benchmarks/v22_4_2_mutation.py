"""JARVIS v22.4.2 — mutation testing for the NEW security guards (spec §24).

Reuses the v22.4.1 valid-kill framework (benchmarks/v22_4_1_mutation.py:
a mutant is KILLED_BY_ASSERTION only when applied AND collected>=1 AND
failed>=1 AND errors==0). The original M1-M7 mutants remain available there;
this runner adds:

  M8  route readonly-set over-extended  -> restores the BUG-001 safe fall-through
  M9  scanner prefix group dropped      -> restores the BUG-002 detection gap
  M10 placeholder filter disabled       -> scanner reports placeholders
  M11 markdown gate forced green        -> BUG-003 gate cannot fail

Every mutant restores the original bytes afterwards (SHA before/after
recorded; spec §57).

Output: reports/v22_4_2/mutation_tests.json
"""
from __future__ import annotations

import json
import os
import sys

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE)

from benchmarks.v22_4_1_mutation import (  # noqa: E402
    run_mutation, KILLED, SURVIVED, INVALID_MUTANT, NOT_FOUND, NO_TESTS,
    COLLECTION_ERROR, ENV_ERROR, TIMEOUT,
)

REPORTS = os.path.join(BASE, 'reports', 'v22_4_2')
os.makedirs(REPORTS, exist_ok=True)

NEW_TESTS = ['tests/test_v2242_security_and_rootcause.py']

MUTANTS = [
    ('M8_route_readonly_set_overextended', 'jarvis/tools/terminal.py',
     'ROUTE_READONLY_OPERATIONS = frozenset({"print"})',
     'ROUTE_READONLY_OPERATIONS = frozenset({"print", "add", "delete", "change"})',
     NEW_TESTS),

    ('M9_scanner_prefix_group_dropped', 'benchmarks/v22_4_1_security.py',
     "r\"(?ix)\\b((?:[A-Za-z0-9]+_)*(?:\" + names + r\"))['\\\"]?\\s*[:=]\\s*\"",
     "r\"(?ix)\\b((?:\" + names + r\"))['\\\"]?\\s*[:=]\\s*\"",
     NEW_TESTS),

    ('M10_placeholder_filter_disabled', 'benchmarks/v22_4_1_security.py',
     "def _is_placeholder_value(value: str | None) -> bool:\n    if not value:",
     "def _is_placeholder_value(value: str | None) -> bool:\n    return False\n    if not value:",
     NEW_TESTS),

    ('M11_markdown_gate_forced_green', 'benchmarks/v22_4_1_metadata_consistency.py',
     "        'problems': problems,\n        'passed': not problems,",
     "        'problems': problems,\n        'passed': True,",
     NEW_TESTS),
]


def main() -> int:
    results = []
    for mid, frel, old, new, tests in MUTANTS:
        result = run_mutation(mid, frel, old, new, tests)
        results.append(result)
        print(f"  {mid}: status={result['status']} killed={result.get('killed')} "
              f"restored={result.get('source_restored')}")
    valid = [r for r in results if r.get('applied') and r['status'] != INVALID_MUTANT]
    killed = [r for r in valid if r.get('valid_kill')]
    survived = [r for r in valid if r['status'] == SURVIVED]
    invalid = [r for r in results if r['status'] in
               (INVALID_MUTANT, NOT_FOUND, NO_TESTS, COLLECTION_ERROR, ENV_ERROR)]
    timeouts = [r for r in valid if r['status'] == TIMEOUT]
    report = {
        'framework': 'v22.4.1 valid-kill semantics (spec §13-§18), reused by v22.4.2',
        'kill_definition': 'applied AND collected>=1 AND failed>=1 AND errors==0',
        'mutants': results,
        'valid_mutants': len(valid),
        'killed': len(killed),
        'survived': len(survived),
        'invalid': len(invalid),
        'timeouts': len(timeouts),
        'all_source_restored': all(r.get('source_restored') for r in results),
        'target': {'valid_mutants': len(MUTANTS), 'killed': len(MUTANTS),
                   'survived': 0, 'invalid': 0},
        'meets_target': (len(valid) == len(MUTANTS) and len(killed) == len(MUTANTS)
                         and len(survived) == 0 and len(invalid) == 0
                         and all(r.get('source_restored') for r in results)),
    }
    with open(os.path.join(REPORTS, 'mutation_tests.json'), 'w',
              encoding='utf-8') as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(json.dumps({k: report[k] for k in
                      ('valid_mutants', 'killed', 'survived', 'invalid',
                       'all_source_restored', 'meets_target')}))
    return 0 if report['meets_target'] else 1


if __name__ == '__main__':
    sys.exit(main())
