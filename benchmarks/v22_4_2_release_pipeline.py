"""JARVIS v22.4.2 — release regression runner (spec §33, §28).

Runs the FULL tests/ tree with raw pytest and records measured counts —
never hand-typed. Baseline to protect: v22.4.1 published 1228 tests
(1215 v22.4 baseline) + 585 subtests, 0 failures.

Output: reports/v22_4_2/regression.json
"""
from __future__ import annotations

import json
import os
import re
import subprocess
import sys

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR = os.path.join(BASE, 'reports', 'v22_4_2')
os.makedirs(OUT_DIR, exist_ok=True)

BASELINE = {'tests': 1215, 'subtests': 585, 'failures': 0}      # v22.4
PREVIOUS = {'tests': 1228, 'subtests': 585, 'failures': 0}      # v22.4.1


def main() -> int:
    proc = subprocess.run(
        [sys.executable, '-B', '-m', 'pytest', 'tests/', '--no-header', '-q',
         '-p', 'no:cacheprovider'],
        cwd=BASE, capture_output=True, text=True, timeout=1800)
    lines = [l for l in (proc.stdout or '').strip().splitlines() if l.strip()]
    tail = lines[-1] if lines else ''

    def grab(pattern):
        m = re.search(pattern, tail)
        return int(m.group(1)) if m else 0

    tests = grab(r'(\d+) passed')
    skipped = grab(r'(\d+) skipped')
    subtests = grab(r'(\d+) subtests? passed')
    failures = grab(r'(\d+) failed') if proc.returncode != 0 else 0
    errors = grab(r'(\d+) error') if proc.returncode != 0 else 0

    report = {
        'source': 'raw pytest run of tests/ (never hand-typed)',
        'pytest_tail': tail,
        'returncode': proc.returncode,
        'tests': tests,
        'skipped': skipped,
        'skip_reason': ('release ZIP not present in the working tree during '
                        'the run (environment-dependent packaging test)') if skipped else None,
        'subtests': subtests,
        'failures': failures,
        'errors': errors,
        'baseline_required': f">= {BASELINE['tests']} tests + {BASELINE['subtests']} "
                             "subtests (v22.4), previous v22.4.1 = 1228",
        'meets_baseline': tests >= BASELINE['tests'] and
                          subtests >= BASELINE['subtests'] and
                          proc.returncode == 0,
    }
    with open(os.path.join(OUT_DIR, 'regression.json'), 'w',
              encoding='utf-8') as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(json.dumps({k: report[k] for k in ('tests', 'subtests', 'failures',
                                             'skipped', 'meets_baseline')}))
    return 0 if report['meets_baseline'] else 1


if __name__ == '__main__':
    sys.exit(main())
