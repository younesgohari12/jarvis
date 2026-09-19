"""Chunked execution wrapper for the v22.4 blind benchmark.

The sandbox reaps long background processes, so the single logical run is
executed in deterministic sequential chunks with a checkpoint file. The
benchmark file is frozen (SHA256 in the manifest); chunk boundaries do NOT
touch the cases. Final artifact = merged results of the one complete pass.
"""
import json
import os
import subprocess
import sys

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CASES = os.path.join(BASE, 'benchmarks', 'v22_4_blind_1600.jsonl')
OUT = os.path.join(BASE, 'reports', 'v22_4', 'new_blind.json')
CHUNK_DIR = os.path.join(BASE, 'reports', 'v22_4', 'blind_chunks')
TMP_RUNNER = os.path.join(BASE, 'benchmarks', 'run_v22_fresh_blind.py')

rows = [json.loads(l) for l in open(CASES, encoding='utf-8') if l.strip()]
os.makedirs(CHUNK_DIR, exist_ok=True)
CHUNK = 420
chunks = list(range(0, len(rows), CHUNK))

for ci, start in enumerate(chunks):
    end = min(start + CHUNK, len(rows))
    part_path = os.path.join(CHUNK_DIR, f'chunk_{ci:02d}.json')
    rows_path = os.path.join(CHUNK_DIR, f'rows_{ci:02d}.jsonl')
    if os.path.exists(part_path):
        print(f'chunk {ci} already done', flush=True)
        continue
    with open(rows_path, 'w', encoding='utf-8') as f:
        for r in rows[start:end]:
            f.write(json.dumps(r, ensure_ascii=False) + '\n')
    # the runner writes summary+results; for chunks we only need results
    proc = subprocess.run(
        [sys.executable, '-u', '-B', TMP_RUNNER, BASE, rows_path, part_path],
        cwd=BASE, capture_output=True, text=True, timeout=560)
    print(f'chunk {ci} [{start}:{end}] rc={proc.returncode}',
          (proc.stdout or '').strip().splitlines()[-1] if proc.stdout else '',
          flush=True)
    if proc.returncode != 0:
        print(proc.stderr[-1500:], flush=True)
        sys.exit(1)

# merge the one logical pass
results = []
for ci in range(len(chunks)):
    part = json.load(open(os.path.join(CHUNK_DIR, f'chunk_{ci:02d}.json'),
                          encoding='utf-8'))
    results.extend(part['results'])

answerable = [r for r in results if r['kind'] != 'abstain']
must_abstain = [r for r in results if r['kind'] == 'abstain']
correct_answerable = sum(r['passed'] for r in answerable)
correct_abstain = sum(r['passed'] for r in must_abstain)
total = len(results)
by_family = {}
for fam in sorted({r['family'] for r in results}):
    part = [r for r in results if r['family'] == fam]
    by_family[fam] = {'total': len(part), 'correct': sum(r['passed'] for r in part)}
manifest = json.load(open(os.path.join(BASE, 'benchmarks', 'v22_4_blind_manifest.json'),
                          encoding='utf-8'))
summary = {
    'benchmark': 'v22_4_blind (NEW BLIND, executed once, frozen)',
    'benchmark_sha256': manifest['sha256'],
    'scorer': 'v22.1 honest scorer — abstention on answerable = failure',
    'total': total,
    'answerable': len(answerable),
    'expected_abstain': len(must_abstain),
    'tallies': {
        'correct_answerable': correct_answerable,
        'correct_expected_abstain': correct_abstain,
        'correct_total': correct_answerable + correct_abstain,
    },
    'score_numeric': round(100.0 * (correct_answerable + correct_abstain) / total, 2),
    'score_answerable_only': round(100.0 * correct_answerable / len(answerable), 2),
    'by_family': by_family,
    'hand_written': manifest['hand_written'],
    'template_generated': manifest['template_generated'],
    'execution': 'one logical pass, deterministic sequential chunks (sandbox kills long background processes)',
    'results': results,
}
json.dump(summary, open(OUT, 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
print('COMPLETE', correct_answerable + correct_abstain, '/', total,
      '| honest', summary['score_numeric'], '| answerable-only',
      summary['score_answerable_only'], flush=True)
