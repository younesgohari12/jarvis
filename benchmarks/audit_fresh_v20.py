from pathlib import Path
import json,sys,hashlib
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from jarvis.agent.normalization_v20 import normalize,skeleton
fresh=[json.loads(l) for l in (ROOT/'benchmarks/fresh_v20.jsonl').read_text().splitlines()]
exact={};norms={};templates={};total=0
def register(text,path):
 global total
 if not isinstance(text,str):return
 total+=1;rel=str(path.relative_to(ROOT));exact.setdefault(text,rel);norms.setdefault(normalize(text),rel);templates.setdefault(skeleton(text),rel)
for p in sorted((ROOT/'datasets').rglob('*.jsonl')):
 for line in p.read_text().splitlines():
  try:r=json.loads(line)
  except ValueError:continue
  if isinstance(r,dict):register(r.get('input') or r.get('text') or r.get('question') or r.get('prompt'),p)
for p in (ROOT/'data/training').glob('*.json'):
 try:obj=json.loads(p.read_text())
 except ValueError:continue
 stack=[obj]
 while stack:
  obj=stack.pop()
  if isinstance(obj,list):stack.extend(obj)
  elif isinstance(obj,dict):
   for k,v in obj.items():
    if k in ('input','text','question','prompt','user') and isinstance(v,str):register(v,p)
    elif isinstance(v,(list,dict)):stack.append(v)
collisions=[];retained=[]
for r in fresh:
 q=r['question']; match={'exact':exact.get(q),'normalized':norms.get(normalize(q)),'template':templates.get(skeleton(q))}
 if any(match.values()):collisions.append({'id':r['id'],'question':q,'matches':match})
 else:retained.append(r)
path=ROOT/'benchmarks/fresh_v20_clean.jsonl';path.write_text('\n'.join(json.dumps(r,ensure_ascii=False) for r in retained)+'\n')
report={'training_texts_scanned':total,'original_candidates':len(fresh),'excluded_for_overlap':len(collisions),'retained':len(retained),'collisions':collisions,'exact_overlap':0,'normalized_overlap':0,'template_overlap':0,'template_definition':'NFKC+FA/EN word-number normalization then mask numeric spans; whole lexical skeleton','semantic_overlap_claim':'No claim of zero conceptual overlap. Arithmetic/semantic families intentionally recur.','clean_sha256':hashlib.sha256(path.read_bytes()).hexdigest()}
(ROOT/'reports/v20/fresh_overlap.json').write_text(json.dumps(report,indent=2,ensure_ascii=False))
print({k:v for k,v in report.items() if k!='collisions'},flush=True)
assert len(retained)>=200
