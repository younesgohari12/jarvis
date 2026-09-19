"""JARVIS v22.4.1/v22.4.2 — security scanning (spec §40-§41, BUG-002 fix).

TWO scans for this release:
  1. SOURCE TREE scan  — every file in the working tree (minus exclusions)
  2. FINAL PACKAGE scan — every file inside the release ZIP (authoritative)

The scanner reports only (file, secret_type) — never the secret value itself.

BUG-002 (v22.4.2): the previous generic API-key pattern anchored on
``\\b(api[_-]?key)\\b``, which cannot match prefixed variable names such as
``OPENAI_API_KEY`` / ``AVALAI_API_KEY`` because ``_`` is a regex word
character and there is no boundary before ``API``. Credential-name detection
now consumes the FULL variable name (optional ``<prefix>_`` chunks plus the
credential suffix) and covers the API_KEY / APIKEY / ACCESS_TOKEN /
SECRET_KEY / SECRET_TOKEN / BOT_TOKEN families in Python, JSON and
.env-style assignment forms. Obvious placeholder values (YOUR_API_KEY,
CHANGE_ME, <API_KEY>, ${API_KEY}, os.getenv lookups, empty/None, ...) are
filtered so that only realistic credential assignments are reported
(spec §6.4-§6.5).

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
#
# BUG-002: name-based credential patterns share one grammar:
#   \b (<prefix>_)* <credential-suffix> ["']? \s* [: =] \s* <quoted | .env value>
# The leading \b plus full-name consumption is what makes prefixed names
# (OPENAI_API_KEY, MY_SERVICE_ACCESS_TOKEN, ...) detectable. Values shorter
# than 16 chars never match, so `API_KEY = os.getenv("API_KEY")`, empty and
# None assignments stay clean.
_CREDENTIAL_VALUE = (
    r"(?:['\"](?P<q>[A-Za-z0-9_\-/+=:.]{16,})['\"]"
    r"|(?P<u>[A-Za-z0-9_\-/+=:.]{16,}))"
)


def _credential_pattern(names: str) -> re.Pattern:
    return re.compile(
        r"(?ix)\b((?:[A-Za-z0-9]+_)*(?:" + names + r"))['\"]?\s*[:=]\s*"
        + _CREDENTIAL_VALUE
    )


# Values that are obviously not real credentials (spec §6.5) — checked only
# against the captured value, which is never recorded or printed.
_PLACEHOLDER_TOKENS = (
    "your", "change_me", "changeme", "example", "placeholder", "dummy",
    "sample", "insert_", "todo", "fixme", "test_key", "test_token",
    "test_secret", "not_a_real", "xxxx",
)


def _is_placeholder_value(value: str | None) -> bool:
    if not value:
        return True
    v = value.strip().casefold()
    if v in {"none", "null", "nil", "true", "false", "undefined"}:
        return True
    if (v.startswith("<") and v.endswith(">")) or (v.startswith("${") and v.endswith("}")):
        return True
    if any(token in v for token in _PLACEHOLDER_TOKENS):
        return True
    if len(v) >= 8 and len(set(v)) <= 3:  # repetitive filler, e.g. aaaa...
        return True
    return False


PATTERNS = [
    ('github_pat', re.compile(r'github_pat_[A-Za-z0-9_]{20,}')),
    ('github_token_classic', re.compile(r'\bghp_[A-Za-z0-9]{30,}\b')),
    ('github_oauth', re.compile(r'\bgho_[A-Za-z0-9]{30,}\b')),
    ('github_token_finegrained', re.compile(r'\bgh[usr]_[A-Za-z0-9]{30,}\b')),
    ('api_key_generic', _credential_pattern(r'api[_-]?key|apikey')),
    ('access_token_generic', _credential_pattern(r'access[_-]?token')),
    ('secret_key_generic', _credential_pattern(r'secret[_-]?key')),
    ('secret_token_generic', _credential_pattern(r'secret[_-]?token')),
    ('bot_token_generic', _credential_pattern(r'bot[_-]?token')),
    ('aws_access_key', re.compile(r'\bAKIA[0-9A-Z]{16}\b')),
    ('aws_secret_key', re.compile(
        r'''(?i)aws.{0,20}['"][A-Za-z0-9/+=]{40}['"]''')),
    ('private_key_block', re.compile(r'-----BEGIN (?:RSA |EC |DSA |OPENSSH )?PRIVATE KEY-----')),
    ('bot_token', re.compile(r'\b\d{8,10}:[A-Za-z0-9_-]{30,}\b')),
    ('slack_token', re.compile(r'\bxox[baprs]-[A-Za-z0-9-]{10,}\b')),
    ('google_api_key', re.compile(r'\bAIza[0-9A-Za-z_-]{35}\b')),
    ('hardcoded_password', re.compile(
        r'''(?ix)\b((?:[A-Za-z0-9]+_)*(?:password|passwd|pwd))['\"]?\s*[:=]\s*['"](?P<q>[^'"\s]{8,})['"]''')),
]

# patterns whose captured VALUE must pass the placeholder filter; the value
# itself is never recorded — findings keep only (file, secret_type).
VALUE_FILTERED_TYPES = frozenset({
    'api_key_generic', 'access_token_generic', 'secret_key_generic',
    'secret_token_generic', 'bot_token_generic', 'hardcoded_password',
})

# files/dirs never scanned (binary or irrelevant)
SCAN_SKIP_DIRS = {'.git', '.venv', '__pycache__', '.pytest_cache', '.hypothesis',
                  '.mypy_cache', '.ruff_cache', '.tox', 'node_modules'}
SCAN_SKIP_SUFFIX = ('.png', '.jpg', '.jpeg', '.gif', '.zip', '.pdf', '.woff',
                    '.woff2', '.ttf', '.ico', '.db', '.pyc', '.so')


def scan_text(rel: str, text: str) -> list[dict]:
    findings = []
    for stype, pattern in PATTERNS:
        m = pattern.search(text)
        if not m:
            continue
        if stype in VALUE_FILTERED_TYPES:
            groups = m.groupdict()
            value = groups.get('q') or groups.get('u') or ''
            if _is_placeholder_value(value):
                continue  # placeholder — not a real credential (spec §6.5)
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
