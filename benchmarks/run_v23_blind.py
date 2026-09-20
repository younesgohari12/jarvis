"""V23 blind evaluation runner (spec §5 protocol step 3 — ONE blind run).

The set benchmarks/v23_blind_set.jsonl is FROZEN (SHA256 a65ed2fd…, recorded
in reports/v23/blind_freeze.json BEFORE any v23 engine code existed). This
script never modifies it. Methodology mirrors run_v22_4_2_diagnostic.py:
tests.helpers TemporaryRuntime, tools disabled, rt.agent.respond per case,
numeric grading (last number ~= expected), abstain = no fabricated number.

Output: reports/v23/blind_eval.json (+ per-authoring accounting).
"""
from __future__ import annotations

import hashlib
import json
import re
import sys
import time

BASE = '/home/z/my-project/jarvis_work/jarvis_repo/Jarvis_v0.11.0'
CASES = BASE + '/benchmarks/v23_blind_set.jsonl'
OUT_DIR = BASE + '/reports/v23'
sys.path.insert(0, BASE)

DIGITS = str.maketrans('۰۱۲۳۴۵۶۷۸۹٠١٢٣٤٥٦٧٨٩', '01234567890123456789')


def nums(text):
    t = str(text).translate(DIGITS).replace('٫', '.').replace('٬', '')
    t = re.sub(r'(?<=\d),(?=\d{3}(?:\D|$))', '', t)
    if '=' in t:
        t = t.rsplit('=', 1)[-1]
    return [float(x) for x in re.findall(r'(?<![\w.])-?\d+(?:\.\d+)?', t)]


def close(a, b):
    import math
    return math.isclose(float(a), float(b), rel_tol=1e-3, abs_tol=0.0051)


def grade(row, output):
    if row['must_abstain']:
        scan = re.sub(r'«[^»]*»|"[^"]*"', ' ', str(output))
        values = nums(scan)
        if values and re.search(r'\d(?:[.,]\d+)?\s*[.؟?!]*\s*$', scan.strip()):
            return False  # a fabricated numeric answer is never an abstention
        return True
    values = nums(output)
    return bool(values) and close(values[-1], row['expected'])


def main():
    rows = [json.loads(l) for l in open(CASES, encoding='utf-8') if l.strip()]
    sha = hashlib.sha256(open(CASES, 'rb').read()).hexdigest()
    freeze = json.load(open(OUT_DIR + '/blind_freeze.json', encoding='utf-8'))
    if sha != freeze['sha256']:
        print('FROZEN SET MISMATCH — aborting blind run', file=sys.stderr)
        return 1
    from tests.helpers import TemporaryRuntime  # noqa: E402
    results = []
    started = time.perf_counter()
    with TemporaryRuntime() as rt:
        rt.memory.set_setting('memory_enabled', False)
        rt.memory.set_setting('internet_enabled', False)
        for row in rows:
            called = []

            def blocked(tool, *args, **kwargs):
                called.append(str(tool))
                raise AssertionError('tools disabled for blind evaluation')

            rt.tools.invoke = blocked
            tick = time.perf_counter()
            output = ''
            passed = False
            error = ''
            intent = ''
            try:
                reply = rt.agent.respond(row['question'])
                output = reply.text
                intent = reply.intent
                passed = grade(row, output) and not called
            except Exception as e:  # noqa: BLE001 — record, never crash
                error = type(e).__name__ + ': ' + str(e)[:200]
            elapsed = (time.perf_counter() - tick) * 1000
            results.append({'id': row['id'], 'domain': row['domain'],
                            'kind': row['kind'], 'authoring': row['authoring'],
                            'question': row['question'],
                            'expected': row['expected'],
                            'must_abstain': row['must_abstain'],
                            'passed': bool(passed),
                            'output': output, 'intent': intent,
                            'latency_ms': round(elapsed, 1),
                            'tools_called': called, 'error': error})
        by_domain, by_authoring = {}, {}
        for r in results:
            d = by_domain.setdefault(r['domain'], {'total': 0, 'correct': 0})
            d['total'] += 1
            d['correct'] += int(r['passed'])
            a = by_authoring.setdefault(r['authoring'], {'total': 0, 'correct': 0})
            a['total'] += 1
            a['correct'] += int(r['passed'])
        answerable = [r for r in results if not r['must_abstain']]
        abstain = [r for r in results if r['must_abstain']]
        correct_answerable = sum(r['passed'] for r in answerable)
        correct_abstain = sum(r['passed'] for r in abstain)
        summary = {
            'set': 'V23_BLIND_EVAL (frozen before first execution)',
            'cases_file': 'benchmarks/v23_blind_set.jsonl',
            'sha256': sha,
            'total': len(results),
            'answerable': len(answerable),
            'must_abstain': len(abstain),
            'correct_answerable': correct_answerable,
            'correct_abstain': correct_abstain,
            'score_numeric': round(100.0 * (correct_answerable + correct_abstain)
                                   / len(results), 2),
            'score_answerable_only': round(100.0 * correct_answerable
                                           / len(answerable), 2),
            'by_domain': by_domain,
            'by_authoring': by_authoring,
            'wall_seconds': round(time.perf_counter() - started, 1),
            'note': ('this blind set is NOT the published frozen benchmark '
                     'and is never averaged with it (spec §38); one blind '
                     'run per protocol §5'),
        }
    import os
    os.makedirs(OUT_DIR, exist_ok=True)
    with open(OUT_DIR + '/blind_eval.json', 'w', encoding='utf-8') as f:
        json.dump({'summary': summary, 'results': results}, f,
                  ensure_ascii=False, indent=1)
    print(json.dumps(summary, ensure_ascii=False, indent=1))
    return 0


if __name__ == '__main__':
    sys.exit(main())
