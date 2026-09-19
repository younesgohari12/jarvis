"""JARVIS v22.4.1 — release pipeline tests (spec §7, §25, §45, §47, §54, §55).

These tests lock the release-integrity machinery itself:
  * forbidden-path matcher (§7)
  * benchmark accounting from raw chunk tallies (§25)
  * benchmark immutability — frozen blind SHA (§54)
  * authoring metadata sidecar consistency (§19-§24)
  * mutation runner classification semantics (§55, pure-function level)
  * metadata namespacing current/history (§47)
  * full final-ZIP verification (§45) — runs when the release ZIP exists
    (built by benchmarks/package_v22_4_1.py; path via JARVIS_RELEASE_ZIP).
"""
from __future__ import annotations

import hashlib
import json
import os
import sys

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE)
sys.path.insert(0, os.path.join(BASE, 'benchmarks'))

FROZEN_SHA = '063aef8f732c845e77a34c6916749d732e96489bbb74edfe12a3a8afc13f7be1'
FROZEN_JSONL = os.path.join(BASE, 'benchmarks', 'v22_4_blind_1600.jsonl')
R421 = os.path.join(BASE, 'reports', 'v22_4_1')


def _load(path):
    with open(path, encoding='utf-8') as f:
        return json.load(f)


# ----------------------------------------------------------------------
# §54 — benchmark immutability
# ----------------------------------------------------------------------
def test_frozen_blind_benchmark_sha_unchanged():
    data = open(FROZEN_JSONL, 'rb').read()
    assert hashlib.sha256(data).hexdigest() == FROZEN_SHA


def test_sidecar_keeps_benchmark_bytes_frozen():
    sc = _load(os.path.join(BASE, 'benchmarks',
                            'v22_4_blind_authoring_metadata_v2.json'))
    assert sc['original_benchmark_sha256'] == FROZEN_SHA
    assert sc['benchmark_bytes_modified'] is False
    assert sc['authoring_metadata_version'] == 2


# ----------------------------------------------------------------------
# §19-§24 — authoring metadata consistency
# ----------------------------------------------------------------------
def test_authoring_counts_sum_to_total():
    sc = _load(os.path.join(BASE, 'benchmarks',
                            'v22_4_blind_authoring_metadata_v2.json'))
    c = sc['counts']
    assert c['literal_hand_written'] + c['authored_template_generated'] + \
        c['synthetic_template_generated'] == sc['total'] == 2075
    # the misleading "635 hand-written" claim must not survive
    assert c['literal_hand_written'] < 635
    rep = _load(os.path.join(R421, 'benchmark_authoring_metadata.json'))
    assert rep['corrected_counts'] == c
    assert rep['x_plus_y_plus_z_equals_total'] is True
    assert rep['faithful_reconstruction'] is True
    assert rep['benchmark_bytes_unchanged'] is True


def test_authoring_sidecar_covers_every_row():
    sc = _load(os.path.join(BASE, 'benchmarks',
                            'v22_4_blind_authoring_metadata_v2.json'))
    rows = [json.loads(line) for line in
            open(FROZEN_JSONL, encoding='utf-8').read().splitlines()]
    assert len(sc['row_classes']) == len(rows) == 2075
    assert set(sc['row_classes']) == {r['id'] for r in rows}


# ----------------------------------------------------------------------
# §24-§25 — blind accounting from raw chunks
# ----------------------------------------------------------------------
def test_benchmark_accounting_is_consistent():
    rep = _load(os.path.join(R421, 'benchmark_accounting.json'))
    assert rep['all_consistent'] is True
    agg = rep['aggregated']
    assert agg['total'] == 2075
    assert agg['correct_total'] == 1891
    assert agg['answerable'] == 1749
    assert agg['correct_answerable'] == 1568
    chunks = _load(os.path.join(BASE, 'reports', 'v22_4', 'new_blind.json'))
    assert sum(c['total'] for c in rep['chunks']) == chunks['total']
    assert sum(c['correct_total'] for c in rep['chunks']) == \
        chunks['tallies']['correct_total']
    assert sum(c['answerable'] for c in rep['chunks']) == chunks['answerable']
    assert sum(c['correct_answerable'] for c in rep['chunks']) == \
        chunks['tallies']['correct_answerable']


# ----------------------------------------------------------------------
# §55 — mutation classification semantics (pure-function level)
# ----------------------------------------------------------------------
def test_mutation_classifier_rejects_non_assertion_kills():
    from v22_4_1_mutation import (KILLED, NOT_FOUND, NO_TESTS, SURVIVED,
                                  classify_run, parse_pytest_output)

    # real assertion failure
    counts = parse_pytest_output(
        'collected 2 items\n1 failed, 1 passed in 1.2s', '', 1)
    assert classify_run(1, counts, '', '') == KILLED
    # all pass -> survived
    counts = parse_pytest_output('collected 2 items\n2 passed in 1.0s', '', 0)
    assert classify_run(0, counts, '', '') == SURVIVED
    # test not found -> never a kill
    counts = parse_pytest_output('', 'ERROR: not found: x.py::test_missing', 4)
    assert classify_run(4, counts, '', '') == NOT_FOUND
    # no tests ran -> never a kill
    counts = parse_pytest_output('no tests ran in 1.17s', '', 5)
    assert classify_run(5, counts, '', '') == NO_TESTS
    # "no tests ran" must NOT be classified as a kill (the v22.4 M5 bug)
    counts = parse_pytest_output('no tests ran in 1.17s', '', 4)
    assert classify_run(4, counts, '', '') != KILLED


def test_mutation_suite_report_meets_target():
    rep = _load(os.path.join(R421, 'mutation_tests.json'))
    assert rep['valid_mutants'] == 7
    assert rep['killed'] == 7
    assert rep['survived'] == 0
    assert rep['invalid'] == 0
    assert rep['all_source_restored'] is True
    assert rep['meets_target'] is True
    for m in rep['mutants']:
        assert m['status'] == 'KILLED_BY_ASSERTION'
        assert m['pytest']['collected'] >= 1
        assert m['pytest']['failed'] >= 1
        assert m['pytest']['errors'] == 0


def test_m5_manual_reproduction_is_valid():
    rep = _load(os.path.join(R421, 'm5_manual_reproduction.json'))
    assert rep['evidence_valid'] is True
    assert rep['step1_pre_mutation']['status'] == 'SURVIVED'
    assert rep['step3_mutated_run']['collected'] >= 1
    assert rep['step3_mutated_run']['failed'] >= 1
    assert rep['step4_source_restored'] is True


def test_mutation_framework_selftest():
    rep = _load(os.path.join(R421, 'mutation_framework_validation.json'))
    assert rep['passed'] is True
    expected = {
        'assertion_failure': 'KILLED_BY_ASSERTION',
        'all_pass': 'SURVIVED',
        'invalid_node_id': 'TEST_NOT_FOUND',
        'collection_error': 'TEST_COLLECTION_ERROR',
        'timeout': 'TIMEOUT',
        'no_tests_collected': 'INVALID_TEST_SELECTION',
    }
    got = {c['scenario']: c['got'] for c in rep['cases']}
    assert got == expected


# ----------------------------------------------------------------------
# §7 — exclusion policy unit checks
# ----------------------------------------------------------------------
def test_package_exclusion_policy_rejects_caches():
    from package_v22_4_1 import included
    assert included('benchmarks/.hypothesis/x/entry.json', '/r/.hypothesis/x/e.json') is False
    assert included('reports/x/__pycache__/a.pyc', '/r/a.pyc') is False
    assert included('.pytest_cache/CACHEDIR.TAG', '/r/CACHEDIR.TAG') is False
    assert included('a.log', '/r/a.log') is False
    assert included('blind_run.log', '/r/blind_run.log') is False
    assert included('reports/v22_4_1/package_integrity.json',
                    '/r/package_integrity.json') is False
    assert included('jarvis/agent/verifier_v22.py', '/r/verifier_v22.py') is True
    assert included('README.md', '/r/README.md') is True


# ----------------------------------------------------------------------
# §46/§47 — metadata consistency & namespacing
# ----------------------------------------------------------------------
def test_release_metadata_namespaces_history():
    rel = _load(os.path.join(BASE, 'RELEASE_V22_4_1.json'))
    assert rel['version'] == 'v0.11.0-intelligence-v22.4.1'
    assert rel['base_tag'] == 'v0.11.0-intelligence-v22.4'
    assert 'current' in rel['metrics'] and 'history' in rel['metrics']
    # exactly ONE current test count — it must AGREE with the measured
    # regression report (no conflicting 1031/1037/1042/1215-style values, §46)
    cur = rel['metrics']['current']
    reg = _load(os.path.join(R421, 'regression.json'))
    assert cur['regression'] == {k: reg[k] for k in
                                 ('tests', 'subtests', 'failures')}
    # the 0-failure gate is enforced by finalize gates + packaging, and
    # verified here non-circularly whenever the suite is green overall
    assert rel['metrics']['history']['v22.4']['tests'] == 1215
    assert rel['metrics']['history']['v22.3']['tests'] == 1042


def test_p0_smoke_report_all_passed():
    rep = _load(os.path.join(R421, 'p0_smoke.json'))
    assert rep['all_passed'] is True and rep['failed'] == 0


# ----------------------------------------------------------------------
# §45 — final ZIP verification (runs when the release ZIP exists)
# ----------------------------------------------------------------------
def _release_zip_path():
    env = os.environ.get('JARVIS_RELEASE_ZIP')
    if env:
        return env
    default = os.path.join(BASE, 'dist_v22_4_1',
                           'Jarvis_v0.11.0_Intelligence_v22.4.1_'
                           'FINAL_CLEAN_HARDENED.zip')
    alt = '/home/z/my-project/download/' \
          'Jarvis_v0.11.0_Intelligence_v22.4.1_FINAL_CLEAN_HARDENED.zip'
    return default if os.path.exists(default) else alt


def test_release_zip_integrity():
    import zipfile
    zip_path = _release_zip_path()
    if not os.path.exists(zip_path):
        import pytest
        pytest.skip('release ZIP not built yet (run benchmarks/package_v22_4_1.py)')

    manifest = _load(os.path.join(BASE, 'SHA256SUMS.json'))
    integrity = _load(os.path.join(R421, 'package_integrity.json'))

    with zipfile.ZipFile(zip_path) as z:
        names = z.namelist()
        file_names = [n for n in names if not n.endswith('/')]
        # one root (§37)
        assert {n.split('/')[0] for n in names} == {'Jarvis_v0.11.0'}
        # no forbidden paths (§7/§38)
        for name in file_names:
            assert '/.hypothesis/' not in f'/{name}'
            assert '/.pytest_cache/' not in f'/{name}'
            assert '/__pycache__/' not in f'/{name}'
            assert '/.git/' not in f'/{name}'
            assert '/.venv/' not in f'/{name}'
            assert '/node_modules/' not in f'/{name}'
            assert not name.endswith('.pyc')
            assert 'blind_run.log' not in name
        # manifest == archive inventory, exactly (§12)
        stripped = {n.split('/', 1)[1] for n in file_names if '/' in n}
        listed = set(manifest['files'])
        assert listed == stripped - {'SHA256SUMS.json'}
        # recompute every hash (§45)
        for rel, digest in manifest['files'].items():
            assert hashlib.sha256(z.read('Jarvis_v0.11.0/' + rel)).hexdigest() == digest
        assert z.testzip() is None
    # package_integrity report matches the actual archive (§45)
    assert integrity['zip_sha256'] == hashlib.sha256(
        open(zip_path, 'rb').read()).hexdigest()
    assert integrity['manifest_entries'] == len(manifest['files'])
    assert integrity['archive_file_count'] == len(file_names)
    assert integrity['passed'] is True
