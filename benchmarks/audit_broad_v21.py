from pathlib import Path
import json,sys,hashlib,re
from collections import Counter
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from jarvis.agent.normalization_v20 import normalize,skeleton
rows=[json.loads(x) for x in (ROOT/'benchmarks/broad_v21_500.jsonl').read_text().splitlines()]
exact=set();normal=set();templates=set();count=0
# Scan input text, not outputs; report outputs separately if exact question appears there.
def register(t):
 global count
 if not isinstance(t,str):return
 count+=1;exact.add(t);normal.add(normalize(t));templates.add(skeleton(t))
for p in (ROOT/'datasets').rglob('*.jsonl'):
 for line in p.read_text().splitlines():
  try:r=json.loads(line)
  except ValueError:continue
  if isinstance(r,dict):register(r.get('input') or r.get('text') or r.get('question') or r.get('prompt'))
for p in (ROOT/'data/training').glob('*.json'):
 try:stack=[json.loads(p.read_text())]
 except ValueError:continue
 while stack:
  item=stack.pop()
  if isinstance(item,list):stack.extend(item)
  elif isinstance(item,dict):
   for k,v in item.items():
    if k in ('input','text','question','prompt','user') and isinstance(v,str):register(v)
    elif isinstance(v,(list,dict)):stack.append(v)
report={'training_inputs_scanned':count,'exact_overlap':[r['id'] for r in rows if r['question'] in exact],'normalized_overlap':[r['id'] for r in rows if normalize(r['question']) in normal],'numeric_template_overlap':[r['id'] for r in rows if skeleton(r['question']) in templates],'unique_inputs':len({r['question'] for r in rows}),'unique_numeric_masked_templates':len({skeleton(r['question']) for r in rows}),'template_definition':'whole lexical text, number-masked; function names not canonicalized','independence_limit':'Repeated families and paraphrases remain; no claim of zero conceptual overlap.','benchmark_sha256':hashlib.sha256((ROOT/'benchmarks/broad_v21_500.jsonl').read_bytes()).hexdigest()}
(ROOT/'reports/v21/broad_overlap.json').write_text(json.dumps(report,indent=2,ensure_ascii=False));print(report,flush=True)
assert not report['exact_overlap'] and not report['normalized_overlap']
