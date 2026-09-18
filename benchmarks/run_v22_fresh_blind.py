"""JARVIS v22.1 — official Fresh Blind runner (honest scoring).

Runs benchmarks/v22_fresh_blind_1000.jsonl against a temporary runtime with
tools disabled and scores every family by its own kind:

  num         last number in the reply ~= expected
  first_part  last number ~= expected (first ratio share)
  parts       the two trailing numbers ~= expected pair (order-insensitive)
  pct         probability rendered in percent ~= expected
  clock       'HH:MM' (+ day words) == {clock, day_offset}
  abstain     PASS only when NO forbidden number appears and the reply is
              not a numeric answer  ->  an abstention on an ANSWERABLE
              question is a FAILURE (num/parts/... simply don't match).

Usage: python benchmarks/run_v22_fresh_blind.py <ROOT> <CASES_JSONL> <OUT_JSON>
"""
from pathlib import Path
import sys, json, re, time, hashlib, resource

def _bootstrap():
    global ROOT, CASES, OUT
    ROOT = Path(sys.argv[1]).resolve()
    CASES = Path(sys.argv[2]).resolve()
    OUT = Path(sys.argv[3]).resolve()
    sys.path.insert(0, str(ROOT))


ROOT = CASES = OUT = None

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


def day_offset_of(text):
    t = str(text).translate(DIGITS)
    if re.search(r'روز\s*بعد|روز\s*دیگر|next\s+day|روز\s*پس', t, re.I):
        return 1
    m = re.search(r'\+\s*(\d+)\s*days?|(\d+)\s*روز\s*(?:بعد|پس)', t, re.I)
    if m:
        return int(m.group(1) or m.group(2))
    return 0


def score(row, text):
    kind, expected = row['kind'], row['expected']
    if kind == 'num':
        values = nums(text)
        return bool(values) and close(values[-1], expected)
    if kind == 'first_part':
        values = nums(text)
        return bool(values) and close(values[-1], expected)
    if kind == 'parts':
        values = nums(text)
        if len(values) < 2:
            return False
        tail = values[-2:]
        return (close(tail[0], expected[0]) and close(tail[1], expected[1])) or \
               (close(tail[1], expected[0]) and close(tail[0], expected[1]))
    if kind == 'pct':
        values = nums(text)
        if not values:
            return False
        v = values[-1]
        if '%' not in text and '٪' not in text and abs(v) <= 1:
            v *= 100.0
        return close(v, expected)
    if kind == 'clock':
        t = str(text).translate(DIGITS)
        m = re.search(r'\b(\d{1,2}):(\d{2})\b', t)
        if not m:
            return False
        hh, mm = int(m.group(1)), int(m.group(2))
        if not (0 <= hh <= 23 and 0 <= mm <= 59):
            return False
        return f'{hh:02d}:{mm:02d}' == expected['clock'] and day_offset_of(t) == int(expected.get('day_offset', 0))
    if kind == 'abstain':
        t = str(text).translate(DIGITS)
        # A quoted echo of the user's own question («...») is not a computed
        # result; scan only the reply's own words for forbidden values.
        scan = re.sub(r'«[^»]*»|"[^"]*"', ' ', t)
        values = nums(scan)
        for forbidden in (row.get('forbidden') or []):
            try:
                f = float(str(forbidden).translate(DIGITS))
            except ValueError:
                continue
            if any(close(v, f) for v in values):
                return False
        if values and re.search(r'\d(?:[.,]\d+)?\s*[.؟?!]*\s*$', scan.strip()):
            return False  # a numeric answer is never an abstention
        return True
    return False


def main():
    _bootstrap()
    from tests.helpers import TemporaryRuntime  # noqa: E402
    rows = [json.loads(l) for l in CASES.read_text(encoding='utf-8').splitlines() if l.strip()]
    results = []
    started = time.perf_counter()
    with TemporaryRuntime() as rt:
        startup_ms = (time.perf_counter() - started) * 1000
        rt.memory.set_setting('memory_enabled', False)
        rt.memory.set_setting('internet_enabled', False)
        for i, row in enumerate(rows):
            called = []

            def blocked(tool, *args, **kwargs):
                called.append(str(tool)); raise AssertionError('tools disabled for fresh benchmark')

            rt.tools.invoke = blocked
            tick = time.perf_counter(); output = ''; passed = False; error = ''; intent = ''
            try:
                reply = rt.agent.respond(row['question'])
                output = reply.text; intent = reply.intent
                passed = score(row, output) and not called
            except Exception as e:
                error = type(e).__name__ + ': ' + str(e)
            elapsed = (time.perf_counter() - tick) * 1000
            results.append({**row, 'passed': bool(passed), 'output': output, 'intent': intent,
                            'latency_ms': elapsed, 'tools_called': called, 'error': error})
            if (i + 1) % 100 == 0:
                print(i + 1, '/', len(rows), 'correct', sum(r['passed'] for r in results), flush=True)

    # ---------------- honest tallies (audit-compliant) ----------------
    answerable = [r for r in results if r['kind'] != 'abstain']
    must_abstain = [r for r in results if r['kind'] == 'abstain']
    correct_answerable = sum(r['passed'] for r in answerable)
    correct_abstain = sum(r['passed'] for r in must_abstain)
    no_answer = sum(1 for r in answerable if not r['passed'] and not nums(r['output']) and len((r['output'] or '').strip()) < 220)
    wrong = len(answerable) - correct_answerable - 0  # wrong = answered-and-mismatched or abstained-on-answerable
    total = len(results)
    summary = {
        'benchmark': 'v22_fresh_blind',
        'scorer': 'v22.1 honest — abstention on answerable questions counts as failure',
        'benchmark_sha256': hashlib.sha256(CASES.read_bytes()).hexdigest(),
        'total': total,
        'answerable': len(answerable),
        'expected_abstain': len(must_abstain),
        'tallies': {
            'correct_answerable': correct_answerable,
            'correct_expected_abstain': correct_abstain,
            'correct_total': correct_answerable + correct_abstain,
            'wrong_or_abstained_on_answerable': len(answerable) - correct_answerable,
            'failed_expected_abstain': len(must_abstain) - correct_abstain,
        },
        'score_numeric': round(100.0 * (correct_answerable + correct_abstain) / total, 2),
        'score_answerable_only': round(100.0 * correct_answerable / len(answerable), 2),
        'wall_seconds': round(time.perf_counter() - started, 2),
        'startup_ms': round(startup_ms, 1),
        'peak_rss_mb': round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024, 1),
        'tools_enabled': False, 'memory_enabled': False,
        'by_family': {}, 'by_language': {},
        'results': results,
    }
    for fam in sorted({r['family'] for r in results}):
        part = [r for r in results if r['family'] == fam]
        summary['by_family'][fam] = {
            'total': len(part), 'correct': sum(r['passed'] for r in part),
        }
    for lang in sorted({r['language'] for r in results}):
        part = [r for r in results if r['language'] == lang]
        a = [r for r in part if r['kind'] != 'abstain']
        ab = [r for r in part if r['kind'] == 'abstain']
        summary['by_language'][lang] = {
            'total': len(part),
            'correct': sum(r['passed'] for r in part),
            'score': round(100.0 * sum(r['passed'] for r in part) / len(part), 2),
            'answerable_correct': sum(r['passed'] for r in a),
            'answerable_total': len(a),
            'abstain_correct': sum(r['passed'] for r in ab),
        }
    OUT.write_text(json.dumps(summary, ensure_ascii=False, indent=1), encoding='utf-8')
    print('COMPLETE', summary['tallies']['correct_total'], '/', total,
          '| honest score', summary['score_numeric'], '| answerable-only', summary['score_answerable_only'], flush=True)


if __name__ == "__main__":
    main()
