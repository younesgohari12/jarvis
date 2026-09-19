"""JARVIS v22.4.2 — release finalizer.

Reads ONLY measured artifacts (never hand-typed numbers) and emits
RELEASE_V22_4_2.json. Historical metrics (v22.4 / v22.4.1) are carried under
metrics.history with explicit namespacing (spec §42/§47).

Output: RELEASE_V22_4_2.json (in-tree; the ZIP hash of the FINAL package is
recorded afterwards in the EXTERNAL reports/v22_4_2/package_integrity.json —
a file inside a ZIP cannot contain that ZIP's final hash).
"""
from __future__ import annotations

import json
import os
import platform
import sys

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
R422 = os.path.join(BASE, 'reports', 'v22_4_2')


def load(path):
    with open(path, encoding='utf-8') as f:
        return json.load(f)


def main() -> int:
    reg = load(os.path.join(R422, 'regression.json'))
    mut = load(os.path.join(R422, 'mutation_tests.json'))
    diag = load(os.path.join(R422, 'diagnostic.json'))['summary']
    perf = load(os.path.join(R422, 'performance_warm.json'))
    sec = load(os.path.join(R422, 'security_source.json'))
    integrity = load(os.path.join(R422, 'package_integrity.json'))

    release = {
        'product': 'JARVIS',
        'version': 'v0.11.0-intelligence-v22.4.2',
        'codename': 'SECURITY AUDITED / ROOT-CAUSE FIXED (BUG-001 BUG-002 BUG-003 + WEAK-W1)',
        'release_label': ('JARVIS v0.11.0 — Intelligence v22.4.2 '
                          'SECURITY_AUDITED_FIXED'),
        'base_tag': 'v0.11.0-intelligence-v22.4.1',
        'release_tag': 'v0.11.0-intelligence-v22.4.2',
        'date': '2026-09-20',
        'generated_by': 'benchmarks/v22_4_2_finalize.py — from measured artifacts',
        'bugs_fixed': {
            'BUG-001': 'Windows route command permission classification bypass '
                       '(route -4/-6 add/delete/change were classified read_only)',
            'BUG-002': 'secret scanner prefixed-variable detection gap '
                       '(OPENAI_API_KEY family, JSON/.env forms, placeholder filter)',
            'BUG-003': 'release metadata test-count inconsistency (1227 vs 1228); '
                       'Markdown release docs are now gated against regression.json',
        },
        'intelligence_fixes': {
            'WEAK-W1': ('work-rate noun vocabulary generalization across the four '
                        'regex layers (output nouns, worker nouns, role audit, '
                        'source-facts gate); verified by unit tests and the '
                        'v22.4.2 diagnostic; structural families (combined rates, '
                        'inverted hours, per-worker rates) remain open and are '
                        'planned in V23_LANGUAGE_BRAIN_PLAN.md'),
        },
        'metrics': {
            'current': {
                'regression': {'tests': reg['tests'], 'subtests': reg['subtests'],
                               'failures': reg['failures'],
                               'skipped_environment_only': reg.get('skipped', 0)},
                'security': {'source_files_scanned': sec['files_scanned'],
                             'source_secret_count': sec['secret_count'],
                             'patterns_checked': len(sec['patterns_checked'])},
                'mutation': {'valid_mutants': mut['valid_mutants'],
                             'killed': mut['killed'], 'survived': mut['survived'],
                             'invalid': mut['invalid'],
                             'framework': 'v22.4.1 valid-kill semantics reused',
                             'note': 'original M1-M7 re-measured 7/7 killed in the '
                                     'v22.4.2 environment (see CHANGESET.md)'},
                'new_diagnostic_v2242': {
                    'cases': diag['total'],
                    'sha256': diag['sha256'],
                    'score_numeric': diag['score_numeric'],
                    'score_answerable_only': diag['score_answerable_only'],
                    'by_domain': diag['by_domain'],
                    'scope_note': ('freshly authored generalization diagnostic; '
                                   'NOT comparable to the frozen blind benchmark '
                                   'and never averaged with it (§38)'),
                },
                'performance_warm': {
                    'families': {k: v for k, v in perf.items()
                                 if isinstance(v, dict)},
                    'environment_python': platform.python_version(),
                    'environment_platform': platform.platform(),
                },
                'package': {
                    'zip_name': integrity['zip_name'],
                    'archive_file_count': integrity['archive_file_count'],
                    'manifest_entries': integrity['manifest_entries'],
                    'crc_ok': integrity['crc_ok'],
                    'reproducible_rebuild': integrity['reproducibility'].get(
                        'zip_binary_hash_identical'),
                    'security_scan_passed': integrity['security_scan']['passed'],
                },
            },
            'history': {
                'v22.4': {'tests': 1215, 'subtests': 585, 'failures': 0,
                          'blind': '1891/2075'},
                'v22.4.1': {'tests': 1228, 'subtests': 585, 'failures': 0,
                            'blind': '1891/2075',
                            'blind_answerable_only': '1568/1749'},
            },
        },
        'frozen_benchmark_untouched': {
            'dataset': 'V22_4_FROZEN_EVALUATION (2075 cases)',
            'sha256': '063aef8f732c845e77a34c6916749d732e96489bbb74edfe12a3a8afc13f7be1',
            'published_result': '1891/2075 = 91.13% (HISTORICAL RELEASE CLAIM, '
                                'not re-run in v22.4.2)',
            'files_modified_by_v2242': 0,
        },
        'gates': {
            'regression_baseline_held': reg['meets_baseline'],
            'mutation_all_killed': mut['meets_target'],
            'security_source_clean': sec['passed'],
            'package_integrity_passed': integrity['passed'],
            'all_release_gates_green': None,  # filled below
        },
    }
    release['gates']['all_release_gates_green'] = all(
        v for k, v in release['gates'].items() if k != 'all_release_gates_green')

    with open(os.path.join(BASE, 'RELEASE_V22_4_2.json'), 'w',
              encoding='utf-8') as f:
        json.dump(release, f, ensure_ascii=False, indent=2)
    print(json.dumps({'version': release['version'],
                      'gates': release['gates']}, ensure_ascii=False))
    return 0 if release['gates']['all_release_gates_green'] else 1


if __name__ == '__main__':
    sys.exit(main())
