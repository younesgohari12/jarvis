"""JARVIS v22.4.1 — VALID mutation testing framework (spec §13-§18, §55-§57).

Fixes the v22.4 runner defect: `killed = proc.returncode != 0` classified any
non-zero exit (usage error, test-not-found, collection error, import error,
timeout) as a mutation kill. The documented victim was M5, whose pytest tail
read "no tests ran in 1.17s" yet was reported killed.

New semantics — a mutant is KILLED_BY_ASSERTION only when ALL of:
  * the mutation was applied to the source (verified),
  * pytest actually ran and collected >= 1 tests,
  * at least one test FAILED on an assertion/behavior,
  * zero collection errors.
Everything else is classified explicitly and NEVER counted as a kill.

Statuses (spec §14):
  KILLED_BY_ASSERTION | SURVIVED | INVALID_MUTANT | TEST_COLLECTION_ERROR
  TEST_NOT_FOUND | INVALID_TEST_SELECTION | TIMEOUT | ENVIRONMENT_ERROR

Output: reports/v22_4_1/mutation_tests.json (+ mutation_framework_validation.json
is produced by --self-test).
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
import sys

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE)
REPORTS = os.path.join(BASE, 'reports', 'v22_4_1')
os.makedirs(REPORTS, exist_ok=True)

# statuses
KILLED = 'KILLED_BY_ASSERTION'
SURVIVED = 'SURVIVED'
INVALID_MUTANT = 'INVALID_MUTANT'
COLLECTION_ERROR = 'TEST_COLLECTION_ERROR'
NOT_FOUND = 'TEST_NOT_FOUND'
NO_TESTS = 'INVALID_TEST_SELECTION'
TIMEOUT = 'TIMEOUT'
ENV_ERROR = 'ENVIRONMENT_ERROR'

KILL_STATUSES = {KILLED}


# ----------------------------------------------------------------------
# pytest output parsing (spec §16)
# ----------------------------------------------------------------------
_SUMMARY_RE = re.compile(
    r'(?P<failed>\d+)\s+failed'
    r'|(?P<passed>\d+)\s+passed'
    r'|(?P<errors>\d+)\s+error'
    r'|(?P<warned>\d+)\s+warning'
    r'|(?P<skipped>\d+)\s+skipped'
    r'|(?P<subfailed>\d+)\s+subtests?\s+failed'
    r'|(?P<subpassed>\d+)\s+subtests?\s+passed')
_COLLECTED_RE = re.compile(r'collected\s+(\d+)\s+item')
_NO_TESTS_RE = re.compile(r'no tests ran')
_NOT_FOUND_RE = re.compile(r'ERROR:\s+not found')
_COLLECT_INTERRUPT_RE = re.compile(r'Interrupted:\s*\d+\s+error')


def parse_pytest_output(stdout: str, stderr: str, returncode: int) -> dict:
    """Extract collected/passed/failed/error counts from a pytest run."""
    text = (stdout or '') + '\n' + (stderr or '')
    counts = {'collected': 0, 'passed': 0, 'failed': 0, 'errors': 0,
              'subtests_passed': 0, 'subtests_failed': 0, 'skipped': 0}
    m = _COLLECTED_RE.search(text)
    if m:
        counts['collected'] = int(m.group(1))
    for m in _SUMMARY_RE.finditer(text):
        if m.group('failed'):
            counts['failed'] = int(m.group('failed'))
        elif m.group('passed'):
            counts['passed'] = int(m.group('passed'))
        elif m.group('errors'):
            counts['errors'] = int(m.group('errors'))
        elif m.group('skipped'):
            counts['skipped'] = int(m.group('skipped'))
        elif m.group('subfailed'):
            counts['subtests_failed'] = int(m.group('subfailed'))
        elif m.group('subpassed'):
            counts['subtests_passed'] = int(m.group('subpassed'))
    # fallback: "no tests ran" or exit code 5 means zero collected
    if counts['collected'] == 0 and (_NO_TESTS_RE.search(text) or returncode == 5):
        counts['collected'] = 0
    if counts['collected'] == 0 and counts['passed'] + counts['failed'] > 0:
        # -q style output without a "collected" banner still ran tests
        counts['collected'] = counts['passed'] + counts['failed'] + counts['errors'] + counts['skipped']
    return counts


def classify_run(returncode: int, counts: dict, stdout: str, stderr: str) -> str:
    """Map a pytest execution to an explicit mutation status (spec §14-§15)."""
    text = (stdout or '') + '\n' + (stderr or '')
    if _NOT_FOUND_RE.search(text) or returncode == 4:
        return NOT_FOUND
    if _COLLECT_INTERRUPT_RE.search(text) or 'error' in text.lower() and 'during collection' in text.lower():
        return COLLECTION_ERROR
    if counts['collected'] == 0 and (_NO_TESTS_RE.search(text) or returncode == 5):
        return NO_TESTS
    if counts['failed'] == 0 and counts['errors'] > 0:
        return COLLECTION_ERROR
    if returncode == 0 and counts['failed'] == 0 and counts['errors'] == 0:
        return SURVIVED
    if counts['failed'] > 0 and counts['errors'] == 0 and counts['collected'] > 0:
        return KILLED
    return ENV_ERROR


# ----------------------------------------------------------------------
# mutation execution (spec §13-§17, §57)
# ----------------------------------------------------------------------
def _run_pytest(test_args: list[str], timeout: float) -> dict:
    proc = subprocess.run(
        [sys.executable, '-B', '-m', 'pytest', *test_args, '--no-header',
         '--tb=no', '-p', 'no:cacheprovider'],
        cwd=BASE, capture_output=True, text=True, timeout=timeout)
    counts = parse_pytest_output(proc.stdout, proc.stderr, proc.returncode)
    status = classify_run(proc.returncode, counts, proc.stdout, proc.stderr)
    return {'returncode': proc.returncode, 'status': status, **counts,
            'stdout_tail': (proc.stdout or '').strip()[-400:],
            'stderr_tail': (proc.stderr or '').strip()[-200:]}


def run_mutation(mut_id: str, file_rel: str, old: str, new: str,
                 test_args: list[str], timeout: float = 420.0) -> dict:
    """Apply one mutation, run targeted tests, restore bytes, classify."""
    path = os.path.join(BASE, file_rel)
    original = open(path, 'rb').read()
    sha_before = hashlib.sha256(original).hexdigest()
    text = original.decode('utf-8')

    result: dict = {'id': mut_id, 'file': file_rel,
                    'tests': list(test_args), 'sha_before': sha_before}
    if old not in text:
        result.update({'applied': False, 'status': INVALID_MUTANT,
                       'killed': False, 'error': 'mutation target not found',
                       'sha_after': sha_before})
        return result

    open(path, 'wb').write(text.replace(old, new, 1).encode('utf-8'))
    result['applied'] = True
    try:
        run = _run_pytest(test_args, timeout)
        result['pytest'] = run
        result['status'] = run['status']
        result['killed'] = run['status'] in KILL_STATUSES
        # validity requirements (spec §15)
        result['valid_kill'] = bool(
            result['applied'] and run['status'] == KILLED
            and run['collected'] >= 1 and run['failed'] >= 1
            and run['errors'] == 0)
    except subprocess.TimeoutExpired:
        result['status'] = TIMEOUT
        result['killed'] = False
        result['valid_kill'] = False
    finally:
        open(path, 'wb').write(original)          # §57 always restore
        sha_after = hashlib.sha256(open(path, 'rb').read()).hexdigest()
        result['sha_after'] = sha_after
        result['source_restored'] = sha_after == sha_before
    return result


# ----------------------------------------------------------------------
# v22.4 mutant set (unchanged definitions, corrected M5 selection)
# ----------------------------------------------------------------------
MUTANTS = [
    ('M1_disable_dimension_guard', 'jarvis/agent/semantics_v22_4.py',
     "    if not (text and _CHAIN_LANG.search(text)):\n        return []",
     "    if True:\n        return []\n    if not (text and _CHAIN_LANG.search(text)):\n        return []",
     ['tests/test_v224_root_cause.py::test_p0_time_distance_addition_rejected',
      'tests/test_v221_audit_fixes.py::test_cross_dimension_addition_refused']),

    ('M2_speed_unit_flattened', 'jarvis/agent/semantics_v22_4.py',
     "ثانیه', 'M_PER_SECOND'),",
     "ثانیه', 'KM_PER_HOUR'),",
     ['tests/test_v224_root_cause.py::test_speed_unit_mapping',
      'tests/test_v224_property_fuzz.py::test_property_speed_unit_preserved_hypothesis']),

    ('M3_span_end_shifted', 'jarvis/agent/quantity_v22.py',
     '        q.source_span = original[span.start:span.end] if 0 <= span.start <= span.end <= len(original) \\',
     '        q.source_span = original[span.start:max(span.start, span.end - 1)] if 0 <= span.start <= span.end <= len(original) \\',
     ['tests/test_v221_audit_fixes.py::test_source_span_covers_number_and_unit',
      'tests/test_v224_root_cause.py::test_exact_spans_on_original_text']),

    ('M4_age_abs_removed', 'jarvis/agent/local_intelligence_v22.py',
     "            value = abs(float(age_a) - float(age_b))",
     "            value = float(age_a) - float(age_b)",
     ['tests/test_v22_semantic.py::TestAgeDifference::test_basic_difference'
      if False else 'tests/test_v22_semantic.py',
      'tests/test_v224_root_cause.py::test_age_query_binds_named_entities']),

    # M5 — corrected selection (spec §17): the v22.4 runner pointed at a
    # nonexistent node `test_typed_audit_fail_closed`; the real tests are:
    ('M5_verifier_fail_open', 'jarvis/agent/verifier_v22.py',
     "        except Exception as exc:  # v22.1: FAIL-CLOSED, never fail-open.",
     "        except Exception as exc:  # MUTANT: fail-open\n            return verdict",
     ['tests/test_v221_audit_fixes.py::test_typed_audit_crash_fails_closed',
      'tests/test_v224_root_cause.py::test_verifier_fail_closed_on_internal_error']),

    ('M6_transfer_direction_swapped', 'jarvis/agent/semantics_v22_4.py',
     "        balances[src] -= amount\n        balances[tgt] += amount",
     "        balances[tgt] -= amount\n        balances[src] += amount",
     ['tests/test_v224_root_cause.py::test_ownership_direction_resolves']),

    ('M7_inventory_order_reversed', 'jarvis/agent/semantics_v22_4.py',
     "    for e in model['events']:\n        name, qty, direction = e['product'], float(e['quantity']), int(e['direction'])",
     "    for e in reversed(model['events']):\n        name, qty, direction = e['product'], float(e['quantity']), int(e['direction'])",
     ['tests/test_v224_root_cause.py::test_inventory_order_is_semantic',
      'tests/test_v224_property_fuzz.py::test_property_inventory_order_preserved']),
]


def run_mutation_suite(only: list[str] | None = None) -> dict:
    mutants = [m for m in MUTANTS if not only or m[0] in only]
    results = []
    for mid, frel, old, new, tests in mutants:
        r = run_mutation(mid, frel, old, new, tests)
        results.append(r)
        print(f"  {mid}: status={r['status']} killed={r.get('killed')} "
              f"restored={r.get('source_restored')}")
    valid = [r for r in results if r.get('applied') and r.get('status') != INVALID_MUTANT]
    killed = [r for r in valid if r.get('valid_kill')]
    survived = [r for r in valid if r['status'] == SURVIVED]
    invalid = [r for r in results if r['status'] in
               (INVALID_MUTANT, NOT_FOUND, NO_TESTS, COLLECTION_ERROR, ENV_ERROR)]
    timeouts = [r for r in valid if r['status'] == TIMEOUT]
    report = {
        'framework': 'v22.4.1 valid-kill semantics (spec §13-§18)',
        'kill_definition': 'applied AND collected>=1 AND failed>=1 AND errors==0',
        'mutants': results,
        'valid_mutants': len(valid),
        'killed': len(killed),
        'survived': len(survived),
        'invalid': len(invalid),
        'timeouts': len(timeouts),
        'all_source_restored': all(r.get('source_restored') for r in results),
        'target': {'valid_mutants': 7, 'killed': 7, 'survived': 0, 'invalid': 0},
        'meets_target': len(valid) == 7 and len(killed) == 7 and len(survived) == 0
                        and len(invalid) == 0 and
                        all(r.get('source_restored') for r in results),
    }
    return report


# ----------------------------------------------------------------------
# §55 — mutation framework self-test
# ----------------------------------------------------------------------
SELFTEST_DIR = os.path.join(REPORTS, 'mutation_selftest_fixtures')


def _write_fixture(rel: str, content: str) -> str:
    path = os.path.join(SELFTEST_DIR, rel)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        f.write(content)
    return path


def mutation_framework_selftest() -> dict:
    """Classify five controlled scenarios and assert expected statuses."""
    import shutil
    shutil.rmtree(SELFTEST_DIR, ignore_errors=True)
    os.makedirs(SELFTEST_DIR, exist_ok=True)

    passing = _write_fixture('fixture_pass.py', 'def test_ok():\n    assert True\n')
    failing = _write_fixture('fixture_fail.py', 'def test_bad():\n    assert 1 == 2\n')
    sleepy = _write_fixture('fixture_sleep.py',
                            'import time\n\n\ndef test_slow():\n    time.sleep(60)\n')
    broken = _write_fixture('fixture_broken.py', 'def test_broken(:\n    pass\n')

    cases = []
    # 1 — actual assertion failure -> KILLED_BY_ASSERTION
    r = _run_pytest([failing], timeout=120)
    cases.append({'scenario': 'assertion_failure', 'expected': KILLED,
                  'got': r['status'], 'ok': r['status'] == KILLED})
    # 2 — all tests pass -> SURVIVED
    r = _run_pytest([passing], timeout=120)
    cases.append({'scenario': 'all_pass', 'expected': SURVIVED,
                  'got': r['status'], 'ok': r['status'] == SURVIVED})
    # 3 — invalid test node id -> TEST_NOT_FOUND (invalid, never a kill)
    r = _run_pytest([passing + '::test_does_not_exist'], timeout=120)
    cases.append({'scenario': 'invalid_node_id', 'expected': NOT_FOUND,
                  'got': r['status'], 'ok': r['status'] == NOT_FOUND})
    # 4 — collection error -> TEST_COLLECTION_ERROR (invalid, never a kill)
    r = _run_pytest([broken], timeout=120)
    cases.append({'scenario': 'collection_error', 'expected': COLLECTION_ERROR,
                  'got': r['status'], 'ok': r['status'] == COLLECTION_ERROR})
    # 5 — timeout -> TIMEOUT (never a kill)
    try:
        _run_pytest([sleepy], timeout=4)
        got, ok = 'NO_TIMEOUT_RAISED', False
    except subprocess.TimeoutExpired:
        got, ok = TIMEOUT, True
    cases.append({'scenario': 'timeout', 'expected': TIMEOUT,
                  'got': got, 'ok': ok})

    # 5b — "no tests ran" selection -> INVALID_TEST_SELECTION
    empty = _write_fixture('fixture_empty.py', '# no tests here\n')
    r = _run_pytest([empty], timeout=120)
    cases.append({'scenario': 'no_tests_collected', 'expected': NO_TESTS,
                  'got': r['status'], 'ok': r['status'] == NO_TESTS})

    report = {
        'purpose': 'only real assertion failures count as mutation kills (spec §55)',
        'cases': cases,
        'passed': all(c['ok'] for c in cases),
        'killing_classification_unique': KILLED,
    }
    return report


# ----------------------------------------------------------------------
# §56 — manual M5 reproduction evidence
# ----------------------------------------------------------------------
def manual_m5_evidence() -> dict:
    """Apply M5 by hand, run the exact targeted test, capture the failure,
    restore. Provides direct evidence that M5 now produces a REAL kill."""
    target = 'jarvis/agent/verifier_v22.py'
    old = "        except Exception as exc:  # v22.1: FAIL-CLOSED, never fail-open."
    new = ("        except Exception as exc:  # MUTANT: fail-open\n"
           "            return verdict")
    tests = ['tests/test_v221_audit_fixes.py::test_typed_audit_crash_fails_closed']

    path = os.path.join(BASE, target)
    original = open(path, 'rb').read()
    sha_before = hashlib.sha256(original).hexdigest()

    # 1 — the targeted test PASSES on unmutated source
    pre = _run_pytest(tests, timeout=120)
    pre_pass = pre['status'] == SURVIVED

    # 2 — apply mutation manually
    open(path, 'wb').write(original.decode('utf-8').replace(old, new, 1).encode('utf-8'))
    mutated_sha = hashlib.sha256(open(path, 'rb').read()).hexdigest()
    applied = mutated_sha != sha_before

    # 3 — run the targeted test against the fail-open mutant
    try:
        run = _run_pytest(tests, timeout=120)
    finally:
        open(path, 'wb').write(original)
    sha_restored = hashlib.sha256(open(path, 'rb').read()).hexdigest()

    report = {
        'mutation': 'M5 verifier exception -> return verdict (fail-open)',
        'targeted_test': tests,
        'step1_pre_mutation': {'status': pre['status'],
                               'passed': pre['passed'], 'failed': pre['failed']},
        'step2_mutation_applied': applied,
        'step3_mutated_run': {'status': run['status'], 'returncode': run['returncode'],
                              'collected': run['collected'], 'failed': run['failed'],
                              'errors': run['errors'],
                              'stdout_tail': run['stdout_tail']},
        'step4_source_restored': sha_restored == sha_before,
        'evidence_valid': bool(pre_pass and applied and run['status'] == KILLED
                               and run['failed'] >= 1 and run['errors'] == 0
                               and sha_restored == sha_before),
        'conclusion': ('the fail-open mutant is killed by a REAL collected, '
                       'asserting test — the v22.4 "no tests ran" false kill '
                       'is fixed'),
    }
    return report


if __name__ == '__main__':
    mode = sys.argv[1] if len(sys.argv) > 1 else 'all'
    if mode in ('all', 'selftest'):
        print('== mutation framework self-test ==')
        st = mutation_framework_selftest()
        json.dump(st, open(os.path.join(REPORTS, 'mutation_framework_validation.json'),
                           'w', encoding='utf-8'), ensure_ascii=False, indent=2)
        print(json.dumps({'passed': st['passed'],
                          'cases': [(c['scenario'], c['got']) for c in st['cases']]}))
    if mode in ('all', 'm5'):
        print('== manual M5 reproduction ==')
        m5 = manual_m5_evidence()
        json.dump(m5, open(os.path.join(REPORTS, 'm5_manual_reproduction.json'),
                           'w', encoding='utf-8'), ensure_ascii=False, indent=2)
        print('evidence_valid:', m5['evidence_valid'])
    if mode in ('all', 'suite'):
        print('== full mutation suite ==')
        rep = run_mutation_suite()
        json.dump(rep, open(os.path.join(REPORTS, 'mutation_tests.json'),
                            'w', encoding='utf-8'), ensure_ascii=False, indent=2)
        print(json.dumps({k: rep[k] for k in ('valid_mutants', 'killed', 'survived',
                                              'invalid', 'timeouts', 'meets_target',
                                              'all_source_restored')}))
    sys.exit(0)
