"""JARVIS v22.4.1 — regression pipeline (spec §33).

Runs the FULL tests/ tree with raw pytest and records measured counts —
never hand-typed. The baseline (v22.4: 1215 tests + 585 subtests, 0 failures)
must not drop; new release-pipeline tests may raise it.

Output: reports/v22_4_1/regression.json
"""
from __future__ import annotations

import json
import os
import re
import subprocess
import sys

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR = os.path.join(BASE, 'reports', 'v22_4_1')
os.makedirs(OUT_DIR, exist_ok=True)

BASELINE = {'tests': 1215, 'subtests': 585, 'failures': 0}


def main() -> int:
    proc = subprocess.run(
        [sys.executable, '-B', '-m', 'pytest', 'tests/', '--no-header', '-q',
         '-p', 'no:cacheprovider'],
        cwd=BASE, capture_output=True, text=True, timeout=1800)
    tail = (proc.stdout or '').strip().splitlines()[-1] if proc.stdout else ''

    def grab(pattern):
        m = re.search(pattern, tail)
        return int(m.group(1)) if m else 0

    tests = grab(r'(\d+) passed')
    subtests = grab(r'(\d+) subtests? passed')
    failed = (grab(r'(\d+) failed') + grab(r'(\d+) error')
              if proc.returncode != 0 else 0)
    failures = grab(r'(\d+) failed') if proc.returncode != 0 else 0

    report = {
        'source': 'raw pytest run of tests/ (never hand-typed)',
        'pytest_tail': tail,
        'returncode': proc.returncode,
        'tests': tests,
        'subtests': subtests,
        'failures': failures,
        'baseline_required': f">= {BASELINE['tests']} tests + {BASELINE['subtests']} "
                             "subtests (v22.4)",
        'meets_baseline': tests >= BASELINE['tests'] and
                          subtests >= BASELINE['subtests'] and
                          proc.returncode == 0,
    }
    with open(os.path.join(OUT_DIR, 'regression.json'), 'w',
              encoding='utf-8') as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(json.dumps({k: report[k] for k in ('tests', 'subtests', 'failures',
                                             'meets_baseline', 'pytest_tail')}))
    return 0 if report['meets_baseline'] else 1


if __name__ == '__main__':
    sys.exit(main())
