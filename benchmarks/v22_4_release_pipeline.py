"""JARVIS v22.4 — release result pipeline (spec §59/§60/§71/§72).

raw pytest output -> result parser -> regression.json
security scan (secrets must be ZERO)
Outputs: reports/v22_4/{regression,security}.json
NEVER hand-type test counts.
"""
from __future__ import annotations
import json
import os
import re
import subprocess
import sys

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REPORTS = os.path.join(BASE, 'reports', 'v22_4')
os.makedirs(REPORTS, exist_ok=True)


def run_regression() -> dict:
    """Run the FULL tests/ suite and parse the raw pytest summary."""
    proc = subprocess.run(
        [sys.executable, '-B', '-m', 'pytest', 'tests/', '-q', '--no-header',
         '-p', 'no:cacheprovider'],
        cwd=BASE, capture_output=True, text=True, timeout=560)
    tail = (proc.stdout or '').strip().splitlines()[-1] if proc.stdout else ''
    m = re.search(r'(\d+) passed(?:, (\d+) subtests? passed)?', tail)
    if not m:
        raise RuntimeError(f'unparseable pytest tail: {tail!r}')
    tests = int(m.group(1))
    subtests = int(m.group(2) or 0)
    failures = 0 if proc.returncode == 0 else None
    if proc.returncode != 0:
        fm = re.search(r'(\d+) failed', tail)
        failures = int(fm.group(1)) if fm else -1
    return {
        'source': 'raw pytest run of tests/ (never hand-typed)',
        'pytest_tail': tail,
        'tests': tests,
        'subtests': subtests,
        'failures': failures if failures else 0,
        'returncode': proc.returncode,
        'baseline_required': '>= 1042 tests + 585 subtests (v22.3)',
        'meets_baseline': tests >= 1042 and (failures or 0) == 0,
    }


SECRET_PATTERNS = [
    (r'github_pat_[A-Za-z0-9_]{20,}', 'github_pat'),
    (r'ghp_[A-Za-z0-9]{20,}', 'github_token'),
    (r'gho_[A-Za-z0-9]{20,}', 'github_oauth'),
    (r'ghs_[A-Za-z0-9]{20,}', 'github_app'),
    (r'xox[baprs]-[A-Za-z0-9\-]{10,}', 'slack_token'),
    (r'sk-[A-Za-z0-9]{20,}', 'api_key'),
    (r'AKIA[0-9A-Z]{16}', 'aws_key'),
    (r'-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----', 'private_key'),
    (r'bot\d+:[A-Za-z0-9_\-]{30,}', 'bot_token'),
    (r'(?i)(?:password|passwd|secret)\s*[:=]\s*[\'"][^\'"]{8,}[\'"]', 'hardcoded_secret'),
]
SKIP_DIRS = {'.git', '__pycache__', 'node_modules', '.pytest_cache'}


def security_scan() -> dict:
    findings = []
    files_scanned = 0
    for root, dirs, names in os.walk(BASE):
        dirs[:] = [d for d in dirs if d not in SKIP_DIRS]
        for name in names:
            path = os.path.join(root, name)
            if os.path.getsize(path) > 3_000_000:
                continue
            try:
                text = open(path, encoding='utf-8', errors='ignore').read()
            except OSError:
                continue
            files_scanned += 1
            for pattern, label in SECRET_PATTERNS:
                for m in re.finditer(pattern, text):
                    findings.append({
                        'file': os.path.relpath(path, BASE),
                        'type': label,
                    })
    return {
        'files_scanned': files_scanned,
        'findings': findings,
        'secrets_found': len(findings),
        'required': 0,
        'passed': len(findings) == 0,
    }


if __name__ == '__main__':
    print('== full regression (raw pytest) ==')
    reg = run_regression()
    json.dump(reg, open(os.path.join(REPORTS, 'regression.json'), 'w',
                        encoding='utf-8'), ensure_ascii=False, indent=2)
    print(json.dumps({k: reg[k] for k in ('tests', 'subtests', 'failures',
                                          'meets_baseline')}))
    print('== security scan ==')
    sec = security_scan()
    json.dump(sec, open(os.path.join(REPORTS, 'security.json'), 'w',
                        encoding='utf-8'), ensure_ascii=False, indent=2)
    print(json.dumps({k: sec[k] for k in ('files_scanned', 'secrets_found', 'passed')}))
