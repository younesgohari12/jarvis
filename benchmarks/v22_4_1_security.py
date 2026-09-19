"""JARVIS v22.4.1 — security scanning (spec §40-§41).

TWO scans for this release:
  1. SOURCE TREE scan  — every file in the working tree (minus exclusions)
  2. FINAL PACKAGE scan — every file inside the release ZIP (authoritative)

The scanner reports only (file, secret_type) — never the secret value itself.

Reusable module: benchmarks/v22_4_1_security.py
Standalone:      python benchmarks/v22_4_1_security.py   (source scan)
"""
from __future__ import annotations

import json
import os
import re
import sys
import zipfile

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR = os.path.join(BASE, 'reports', 'v22_4_1')
os.makedirs(OUT_DIR, exist_ok=True)

# (secret_type, compiled pattern)
PATTERNS = [
    ('github_pat', re.compile(r'github_pat_[A-Za-z0-9_]{20,}')),
    ('github_token_classic', re.compile(r'\bghp_[A-Za-z0-9]{30,}\b')),
    ('github_oauth', re.compile(r'\bgho_[A-Za-z0-9]{30,}\b')),
    ('api_key_generic', re.compile(
        r'''(?ix)\b(api[_-]?key|apikey)\b\s*[:=]\s*['"][A-Za-z0-9_\-]{16,}['"]''')),
    ('secret_generic', re.compile(
        r'''(?ix)\b(secret[_-]?(?:key|token))\b\s*[:=]\s*['"][A-Za-z0-9_\-]{16,}['"]''')),
    ('aws_access_key', re.compile(r'\bAKIA[0-9A-Z]{16}\b')),
    ('aws_secret_key', re.compile(
        r'''(?i)aws.{0,20}['"][A-Za-z0-9/+=]{40}['"]''')),
    ('private_key_block', re.compile(r'-----BEGIN (?:RSA |EC |DSA |OPENSSH )?PRIVATE KEY-----')),
    ('bot_token', re.compile(r'\b\d{8,10}:[A-Za-z0-9_-]{30,}\b')),
    ('slack_token', re.compile(r'\bxox[baprs]-[A-Za-z0-9-]{10,}\b')),
    ('google_api_key', re.compile(r'\bAIza[0-9A-Za-z_-]{35}\b')),
    ('hardcoded_password', re.compile(
        r'''(?ix)\b(password|passwd|pwd)\b\s*[:=]\s*['"][^'"\s]{8,}['"]''')),
]

# files/dirs never scanned (binary or irrelevant)
SCAN_SKIP_DIRS = {'.git', '.venv', '__pycache__', '.pytest_cache', '.hypothesis',
                  '.mypy_cache', '.ruff_cache', '.tox', 'node_modules'}
SCAN_SKIP_SUFFIX = ('.png', '.jpg', '.jpeg', '.gif', '.zip', '.pdf', '.woff',
                    '.woff2', '.ttf', '.ico', '.db', '.pyc', '.so')


def scan_text(rel: str, text: str) -> list[dict]:
    findings = []
    for stype, pattern in PATTERNS:
        m = pattern.search(text)
        if m:
            findings.append({'file': rel, 'secret_type': stype})  # value NEVER recorded
    return findings


def scan_source_tree(root: str = BASE) -> dict:
    findings = []
    files_scanned = 0
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d not in SCAN_SKIP_DIRS]
        for fn in filenames:
            p = os.path.join(dirpath, fn)
            rel = os.path.relpath(p, root)
            if fn.endswith(SCAN_SKIP_SUFFIX):
                continue
            try:
                text = open(p, encoding='utf-8', errors='ignore').read()
            except OSError:
                continue
            files_scanned += 1
            findings.extend(scan_text(rel, text))
    return {'scan': 'source_tree', 'root': root, 'files_scanned': files_scanned,
            'findings': findings, 'secret_count': len(findings),
            'passed': len(findings) == 0,
            'patterns_checked': [t for t, _ in PATTERNS],
            'note': 'only (file, secret_type) is reported; values are never recorded'}


def scan_package(zip_path: str) -> dict:
    findings = []
    files_scanned = 0
    with zipfile.ZipFile(zip_path) as z:
        for name in z.namelist():
            if name.endswith(SCAN_SKIP_SUFFIX) or name.endswith('/'):
                continue
            try:
                text = z.read(name).decode('utf-8', errors='ignore')
            except Exception:
                continue
            files_scanned += 1
            findings.extend(scan_text(name, text))
    return {'scan': 'final_package', 'zip': os.path.basename(zip_path),
            'files_scanned': files_scanned, 'findings': findings,
            'secret_count': len(findings), 'passed': len(findings) == 0,
            'patterns_checked': [t for t, _ in PATTERNS],
            'authoritative_for_release': True}


if __name__ == '__main__':
    report = scan_source_tree()
    with open(os.path.join(OUT_DIR, 'security_source.json'), 'w',
              encoding='utf-8') as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(json.dumps({'files_scanned': report['files_scanned'],
                      'secret_count': report['secret_count'],
                      'passed': report['passed']}))
    for f_ in report['findings']:
        print(' FINDING:', f_)
    sys.exit(0 if report['passed'] else 1)
