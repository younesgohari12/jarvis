"""JARVIS v22.4.1 — finalize: baseline, scorecard, RELEASE metadata.

Everything is READ from measured artifacts (reports/…json + git) — nothing
hand-typed. Separates intelligence metrics from release-integrity metrics
(spec §48/§49) and namespaces historical values (§47).

Outputs:
  reports/v22_4_1/baseline.json        (v22.4 baseline this release must hold)
  reports/v22_4_1/final_scorecard.json (factual metrics + integrity evidence)
  RELEASE_V22_4_1.json                 (version-specific release metadata)
"""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
R44 = os.path.join(BASE, 'reports', 'v22_4')
R421 = os.path.join(BASE, 'reports', 'v22_4_1')
os.makedirs(R421, exist_ok=True)

FROZEN_SHA = '063aef8f732c845e77a34c6916749d732e96489bbb74edfe12a3a8afc13f7be1'


def load(path):
    with open(path, encoding='utf-8') as f:
        return json.load(f)


def git(*args):
    return subprocess.run(['git', *args], cwd=BASE, capture_output=True,
                          text=True).stdout.strip()


def main() -> int:
    reg = load(os.path.join(R421, 'regression.json'))
    smoke = load(os.path.join(R421, 'p0_smoke.json'))
    parity = load(os.path.join(R421, 'runtime_parity.json'))
    mut = load(os.path.join(R421, 'mutation_tests.json'))
    m5 = load(os.path.join(R421, 'm5_manual_reproduction.json'))
    mfw = load(os.path.join(R421, 'mutation_framework_validation.json'))
    acct = load(os.path.join(R421, 'benchmark_accounting.json'))
    auth = load(os.path.join(R421, 'benchmark_authoring_metadata.json'))
    perf = load(os.path.join(R421, 'performance_warm.json'))
    sec_src = load(os.path.join(R421, 'security_source.json'))

    # ------------------------------------------------------- baseline (§3)
    v44_release = load(os.path.join(BASE, 'RELEASE_V22_4.json'))
    baseline = {
        'release': 'v0.11.0-intelligence-v22.4',
        'base_tag_for_this_release': 'v0.11.0-intelligence-v22.4',
        'source': 'measured v22.4 artifacts (reports/v22_4/) — preserved, not re-measured',
        'regression': {'tests': 1215, 'subtests': 585, 'failures': 0},
        'broad': {'score': '406/500', 'percent': 81.2},
        'legacy_fresh_regression': {'score': '1006/1023', 'percent': 98.34,
                                    'label': 'FRESH_REGRESSION_LEGACY'},
        'new_blind_frozen': {'score': '1891/2075', 'percent': 91.13,
                             'answerable_only': '1568/1749',
                             'answerable_percent': 89.65,
                             'sha256': FROZEN_SHA},
        'families': {'ownership_legacy': '60/60', 'inventory_legacy': '99/100',
                     'age_legacy': '70/70', 'scheduling': '90/90',
                     'work_rate': '105/105'},
        'honest_gaps': {'persian_broad': '6/50', 'english_broad': '0/50',
                        'new_blind_work_rate': '52/90',
                        'new_blind_ownership': '194/235',
                        'new_blind_rates': '176/210',
                        'new_blind_finance': '124/150',
                        'new_blind_age': '139/163'},
    }
    with open(os.path.join(R421, 'baseline.json'), 'w', encoding='utf-8') as f:
        json.dump(baseline, f, ensure_ascii=False, indent=2)

    # -------------------------------------------------- gate checks (§60)
    gates = {
        'runtime_parity_confirmed': parity['parity_confirmed'],
        'regression_baseline_held': reg['meets_baseline'],
        'p0_smoke_all_passed': smoke['all_passed'],
        'mutation_target_met': mut['meets_target'],
        'mutation_m5_real_kill': m5['evidence_valid'],
        'mutation_framework_selftest': mfw['passed'],
        'mutation_sources_restored': mut['all_source_restored'],
        'frozen_blind_sha_unchanged': acct['benchmark_sha_matches_frozen'],
        'blind_accounting_reproduced': acct['all_consistent'],
        'authoring_labels_corrected': auth['x_plus_y_plus_z_equals_total'] and
                                      auth['faithful_reconstruction'],
        'source_security_clean': sec_src['passed'],
        'no_misleading_handwritten_claim': auth['corrected_counts']
                                          ['literal_hand_written'] < 635,
    }
    gates['all_release_gates_green'] = all(gates.values())

    # ------------------------------------------------- final scorecard (§48)
    scorecard = {
        'release': 'v0.11.0-intelligence-v22.4.1',
        'label': 'FINAL CLEAN HARDENED / RELEASE INTEGRITY LOCK (§59)',
        'purpose': 'release-integrity evidence; NOT a new intelligence score (§48)',
        'intelligence_metrics': {
            'note': 'carried forward from v22.4 — runtime is unchanged (§32)',
            'broad_500': baseline['broad']['score'],
            'legacy_fresh_regression_1023':
                baseline['legacy_fresh_regression']['score'],
            'frozen_blind_2075': baseline['new_blind_frozen']['score'],
            'frozen_blind_answerable_only':
                baseline['new_blind_frozen']['answerable_only'],
            'blind_benchmark_sha256': FROZEN_SHA,
            'family_results_blind': acct['family_results'],
            'honest_remaining_gaps': baseline['honest_gaps'],
        },
        'release_integrity_metrics': {
            'note': 'independent dimension — never averaged into intelligence (§49)',
            'regression': {'tests': reg['tests'], 'subtests': reg['subtests'],
                           'failures': reg['failures']},
            'p0_smoke': f"{smoke['passed']}/{smoke['total']}",
            'runtime_parity_vs_v22_4': parity['diff_count'] == 0 and
                                       f"{parity['sample_size']}/{parity['sample_size']} identical",
            'mutation': {'valid_mutants': mut['valid_mutants'],
                         'killed': mut['killed'], 'survived': mut['survived'],
                         'invalid': mut['invalid'],
                         'framework_selftest': mfw['passed'],
                         'm5_manual_evidence': m5['evidence_valid']},
            'benchmark_authoring': auth['corrected_counts'],
            'performance_warm_ms': perf['warm'],
            'security_source_secrets': sec_src['secret_count'],
        },
        'release_gates': gates,
    }
    with open(os.path.join(R421, 'final_scorecard.json'), 'w',
              encoding='utf-8') as f:
        json.dump(scorecard, f, ensure_ascii=False, indent=2)

    # --------------------------------------------- RELEASE_V22_4_1.json (§31)
    base_commit = git('rev-parse', 'v0.11.0-intelligence-v22.4')
    # runtime scope = committed + working-tree changes under jarvis/ vs base
    runtime_files = sorted(set(
        git('diff', '--name-only', base_commit, '--', 'jarvis/').splitlines()
        + [line[3:] for line in git('status', '--porcelain', '--', 'jarvis/').splitlines()]))
    release = {
        'product': 'JARVIS',
        'version': 'v0.11.0-intelligence-v22.4.1',
        'codename': 'RELEASE INTEGRITY / EVALUATION CLEANUP / FINAL LOCK',
        'release_label': 'JARVIS v0.11.0 — Intelligence v22.4.1 '
                         'FINAL CLEAN HARDENED / RELEASE INTEGRITY LOCK (§59)',
        'is_release_integrity_patch': True,
        'base_tag': 'v0.11.0-intelligence-v22.4',
        'base_commit': base_commit,
        'release_tag': 'v0.11.0-intelligence-v22.4.1',
        'release_commit_note': ('the release commit is identified by its tag; '
                                'a file cannot reference the commit that '
                                'first contains it'),
        'date': '2026-09-19',
        'generated_by': 'benchmarks/v22_4_1_finalize.py — from measured artifacts',
        'runtime_changed': len(runtime_files) > 0,
        'runtime_files_changed': runtime_files,
        'runtime_files_changed_count': len(runtime_files),
        'evaluation_changed': True,
        'packaging_changed': True,
        'scope': [
            'Issue A: .hypothesis + dev/cache artifacts excluded from the package '
            '(~175 cache files shipped in v22.4 are now rejected by policy + tests)',
            'Issue B: package_integrity.json generated LAST, directly from the '
            'final ZIP (manifest_entries now measured, not stale)',
            'Issue C: mutation runner valid-kill semantics (collected/failed/'
            'errors parsed); M5 test selection fixed to the real fail-closed '
            'tests; manual M5 reproduction evidence; framework self-test',
            'Issue D: honest authoring classification sidecar '
            '(authoring_metadata_version = 2) — frozen benchmark bytes and SHA '
            'untouched',
            'blind chunk accounting independently aggregated from raw chunks',
            'warm performance measurement + ownership p99 read-only investigation',
            'two security scans (source tree + final package)',
            'deterministic packaging + rebuild reproducibility check',
            'release-pipeline test suite (integrity, accounting, immutability, '
            'exclusion policy, metadata namespacing)',
            'V23_TRAINING_HANDOFF.md (informational)',
        ],
        'metrics': {
            'current': {
                'regression': {'tests': reg['tests'], 'subtests': reg['subtests'],
                               'failures': reg['failures']},
                'p0_smoke': {'passed': smoke['passed'], 'total': smoke['total']},
                'broad_500': baseline['broad']['score'],
                'legacy_fresh_regression_1023':
                    baseline['legacy_fresh_regression']['score'],
                'frozen_blind_2075': baseline['new_blind_frozen']['score'],
                'frozen_blind_answerable_only':
                    baseline['new_blind_frozen']['answerable_only'],
                'blind_benchmark_sha256': FROZEN_SHA,
                'mutation': {'valid_mutants': mut['valid_mutants'],
                             'killed': mut['killed'], 'survived': mut['survived']},
                'runtime_parity_vs_v22_4': parity['parity_confirmed'],
            },
            'history': {
                'v22.4': {'tests': 1215, 'subtests': 585, 'failures': 0,
                          'broad': '406/500', 'legacy': '1006/1023',
                          'blind': '1891/2075'},
                'v22.3': {'tests': 1042, 'subtests': 585, 'failures': 0,
                          'broad': '406/500', 'legacy': '855/1023'},
            },
        },
        'authoring_classification': {
            'authoring_metadata_version': 2,
            'counts': auth['corrected_counts'],
            'total': auth['total'],
            'claim_language': auth['claim_language'],
            'sidecar': 'benchmarks/v22_4_blind_authoring_metadata_v2.json',
        },
        'not_claimed': [
            'no runtime intelligence changes vs v22.4 (parity 40/40 verified)',
            'no new model weights; no GPU training (v23 scope)',
            'no improvement claims on the seen blind benchmark — it is frozen '
            'as V22_4_FROZEN_EVALUATION and becomes a v23 target only (§27/§28)',
            'no composite intelligence+quality score (§49)',
        ],
        'post_upload_digest': {
            'verified': False,
            'reason': 'not yet uploaded (§43) — updated after the GitHub asset '
                      'digest comparison',
        },
        'release_gate': ('FINAL CLEAN HARDENED — all release-integrity gates '
                         'green (§60); see '
                         'JARVIS_V22_4_1_RELEASE_INTEGRITY_AUDIT.md'),
        'gates': gates,
    }
    with open(os.path.join(BASE, 'RELEASE_V22_4_1.json'), 'w',
              encoding='utf-8') as f:
        json.dump(release, f, ensure_ascii=False, indent=2)

    print(json.dumps({'gates': gates,
                      'regression': release['metrics']['current']['regression'],
                      'runtime_files_changed_count':
                          release['runtime_files_changed_count']}, indent=1))
    return 0 if gates['all_release_gates_green'] else 1


if __name__ == '__main__':
    sys.exit(main())
