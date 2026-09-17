from pathlib import Path
import json,sys,hashlib
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from jarvis.agent.normalization_v20 import normalize,skeleton
rows=[json.loads(x) for x in (ROOT/'benchmarks/diagnostic_v20_1_300.jsonl').read_text().splitlines()]
exact=set();norms=set();templates=set();count=0
def register(text):
 global count
 if not isinstance(text,str):return
 count+=1;exact.add(text);norms.add(normalize(text));templates.add(skeleton(text))
for p in (ROOT/'datasets').rglob('*.jsonl'):
 for line in p.read_text().splitlines():
  try:r=json.loads(line)
  except ValueError:continue
  if isinstance(r,dict):register(r.get('input') or r.get('text') or r.get('question') or r.get('prompt'))
for p in (ROOT/'data/training').glob('*.json'):
 try:stack=[json.loads(p.read_text())]
 except ValueError:continue
 while stack:
  obj=stack.pop()
  if isinstance(obj,list):stack.extend(obj)
  elif isinstance(obj,dict):
   for k,v in obj.items():
    if k in ('input','text','question','prompt','user') and isinstance(v,str):register(v)
    elif isinstance(v,(list,dict)):stack.append(v)
report={'training_texts_scanned':count,'cases':len(rows),'unique_questions':len({r['question'] for r in rows}),'unique_numeric_masked_templates':len({skeleton(r['question']) for r in rows}),'declared_families':len({r['family'] for r in rows}),'training_overlap':{}}
for label,collection,transform in [('exact',exact,lambda x:x),('normalized',norms,normalize),('template',templates,skeleton)]:
 report['training_overlap'][label]=[r['id'] for r in rows if transform(r['question']) in collection]
prior=[json.loads(x)['question'] for x in (ROOT/'benchmarks/fresh_v20_clean.jsonl').read_text().splitlines()]
report['prior_benchmark_exact_overlap']=[r['id'] for r in rows if r['question'] in prior]
report['limitations']='Diagnostic set contains parameterized shared templates. No claim of unseen template generalization or conceptual independence. No cases removed after evaluation.'
(ROOT/'reports/v20_1/diagnostic_overlap.json').write_text(json.dumps(report,ensure_ascii=False,indent=2))
print({**report,'training_overlap':{k:len(v) for k,v in report['training_overlap'].items()}})
