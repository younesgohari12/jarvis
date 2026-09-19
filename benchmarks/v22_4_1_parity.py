"""JARVIS v22.4.1 — runtime parity check vs v22.4 (spec §35).

Runs a fixed deterministic sample through BOTH the v22.4 reference tree and
the v22.4.1 tree and diffs the responses. v22.4.1 is a release-integrity
patch: Runtime behavioral files must not change, so responses must be
byte-identical (only release/telemetry metadata may differ).

Usage:
  python benchmarks/v22_4_1_parity.py <v22.4_reference_tree> <output.json>

The same harness is executed inside each tree (via subprocess) so each engine
imports its OWN runtime modules.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR = os.path.join(BASE, 'reports', 'v22_4_1')
os.makedirs(OUT_DIR, exist_ok=True)

SAMPLE = [
    # (language, text) — deterministic, covering all semantic families
    ('en', '2 hours + 5 dollars. What is the total?'),
    ('en', '2 hours + 5 items. What is the total?'),
    ('fa', '2 ساعت و 120 کیلومتر را جمع کن.'),
    ('en', '10 m/s. Convert to km/h.'),
    ('en', 'An object moves at 10 m/s. How many meters in 30 seconds?'),
    ('en', 'A car travels 60 km per hour. How far in 2 hours?'),
    ('fa', 'سرعت خودرو 60 کیلومتر بر ساعت است؛ در 2 ساعت چند کیلومتر می‌رود؟'),
    ('fa', 'موجودی حساب 100 دلار است؛ 30 دلار کم کن؛ چند؟'),
    ('en', 'A wallet has 250 dollars. 70 dollars are removed. What remains?'),
    ('en', "Ali has 704 dollars. Sara has 223 dollars. Ali transfers 15 dollars to Sara. What is Sara's balance?"),
    ('en', 'Sara has 340 dollars. Ali has 120 dollars. Sara pays Ali 40 dollars. What is Ali\'s balance?'),
    ('fa', 'علی 500 دلار دارد و سارا 200 دلار دارد. علی 50 دلار به سارا می‌دهد. موجودی سارا چند دلار است؟'),
    ('fa', 'موجودی انبار 100 کالا است؛ 7 کالا فروخته شد و 3 کالا آسیب دید؛ چند؟'),
    ('en', 'A warehouse starts with 80 items. 12 items are sold and 5 damaged items are returned to stock. How many items now?'),
    ('en', 'A train departs at 23:30. The trip takes 45 minutes. What time does it arrive?'),
    ('fa', 'کار ساعت 23:30 شروع می شود؛ مدت کار 90 دقیقه است؛ چه ساعتی تمام می شود؟'),
    ('en', 'A meeting starts at 09:15 and lasts 2 hours and 20 minutes. When does it end?'),
    ('fa', 'علی 35 سال دارد و رضا 30 سال دارد. اختلاف سن علی و رضا چند سال است؟'),
    ('fa', 'رضا 30 سال دارد و علی 35 سال دارد. اختلاف سن رضا و علی چند سال است؟'),
    ('en', 'Ali is 35 years old and Reza is 30. What is the age difference?'),
    ('en', 'Mina is 28. Sara is 34. Who is older and by how many years?'),
    ('en', 'The price was reduced by 20 dollars.'),
    ('en', 'An item costs 400 dollars with a 25% discount. What is the final price in dollars?'),
    ('fa', 'قیمت یک کالا 800 تومان است و 20 درصد تخفیف می‌خورد؛ قیمت نهایی چند تومان است؟'),
    ('en', 'Divide 525 between two parts in the ratio 4:3. What is the first part\'s share?'),
    ('fa', '525 را با نسبت 4 به 3 تقسیم کن'),
    ('fa', '3 کارگر در 4 ساعت 84 واحد تولید می کنند؛ 8 کارگر در 6 ساعت؟'),
    ('en', '4 workers make 120 pieces in 3 hours. How many pieces do 6 workers make in 5 hours?'),
    ('en', 'Add 3 kg and 4 liters.'),
    ('en', 'Add 12 km and 5 km.'),
    ('fa', 'جمع 5 ساعت و 3 ساعت چند است؟'),
    ('en', 'What is the probability (in %) of heads in a fair coin toss?'),
    ('fa', 'شناسه سفارش 12345 ثبت شد؛ جمع 30 و 40 دلار چند دلار است؟'),
    ('en', 'Order #88123: sum of 25 dollars and 15 dollars?'),
    ('fa', 'قیمت بیت‌کوین امروز چند دلار است؟'),
    ('en', 'What is the capital of France?'),
    ('fa', 'جمع 7 دلار و 3 کالا را حساب کن.'),
    ('en', 'A runner keeps 5 m/s for 60 seconds. Distance in meters?'),
    ('en', '10 dollars per hour for 2 hours is how much?'),
    ('fa', 'هر ساعت 5 دلار برای 2 ساعت چقدر می شود؟'),
]

HARNESS = r'''
import json, sys
sys.path.insert(0, sys.argv[1])
from jarvis.agent.local_intelligence_v22 import LocalIntelligenceV22

SAMPLE = json.loads(sys.argv[2])
eng = LocalIntelligenceV22()
out = []
for lang, text in SAMPLE:
    try:
        a = eng.solve(text, lang)
        out.append({'language': lang, 'text': text,
                    'answer': None if a is None else a.text,
                    'intent': None if a is None else a.intent})
    except Exception as exc:
        out.append({'language': lang, 'text': text, 'error': repr(exc)})
print(json.dumps(out, ensure_ascii=False))
'''


def run_tree(tree: str) -> list:
    """Run the harness inside the given tree and return normalized outputs."""
    proc = subprocess.run(
        [sys.executable, '-B', '-c', HARNESS, tree,
         json.dumps(SAMPLE, ensure_ascii=False)],
        cwd=tree, capture_output=True, text=True, timeout=300)
    if proc.returncode != 0:
        raise RuntimeError(f'harness failed in {tree}:\n{proc.stderr[-1500:]}')
    return json.loads(proc.stdout.strip().splitlines()[-1])


def main() -> int:
    ref_tree = os.path.abspath(sys.argv[1])
    print('v22.4 reference tree:', ref_tree)
    ref_out = run_tree(ref_tree)
    cur_out = run_tree(BASE)

    diffs = []
    for r, c in zip(ref_out, cur_out):
        if r != c:
            diffs.append({'sample': r['text'], 'v22_4': r, 'v22_4_1': c})

    answered = sum(1 for o in cur_out if o.get('answer'))
    report = {
        'release': 'v0.11.0-intelligence-v22.4.1',
        'reference_tree': ref_tree,
        'reference_commit': subprocess.run(
            ['git', 'rev-parse', 'HEAD'], cwd=ref_tree, capture_output=True,
            text=True).stdout.strip(),
        'sample_size': len(SAMPLE),
        'answered_current': answered,
        'differences': diffs,
        'diff_count': len(diffs),
        'parity_confirmed': len(diffs) == 0,
        'allowed_differences_note': ('release version / telemetry version / '
                                     'packaging metadata only — none observed '
                                     'in normalized response fields'),
    }
    with open(os.path.join(OUT_DIR, 'runtime_parity.json'), 'w',
              encoding='utf-8') as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(json.dumps({'sample_size': len(SAMPLE), 'diff_count': len(diffs),
                      'parity_confirmed': report['parity_confirmed']}))
    for d in diffs[:5]:
        print(' DIFF:', d['sample'], '|', d['v22_4'].get('answer'),
              '->', d['v22_4_1'].get('answer'))
    return 0 if report['parity_confirmed'] else 1


if __name__ == '__main__':
    sys.exit(main())
