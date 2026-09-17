from pathlib import Path
import sys,json,hashlib
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from training.train_heads_v21 import train
from jarvis.agent.normalization_v20 import normalize,NUMBER,skeleton
from jarvis.agent.numeric_v21 import numeric_context
rows=[];source_count=0
for name in ['semantic_ir','world_model','adversarial','persian']:
 for line in (ROOT/'datasets/v21'/f'{name}.jsonl').read_text().splitlines():
  r=json.loads(line);g=r['program']
  if not g.get('operations') or 'initial' not in g.get('slots',{}):continue
  t=normalize(r['text']);ns=list(NUMBER.finditer(t));source_count+=1;offset=1 if name=='adversarial' else 0
  for i,m in enumerate(ns):
   label='distractor' if i<offset else 'initial' if i==offset else 'value'
   context=numeric_context(t,(m.start(),m.end()),g['domain']);key=skeleton(context);bucket=int(hashlib.sha256(key.encode()).hexdigest()[:8],16)%10
   rows.append((context,label,'test' if bucket==0 else 'validation' if bucket==1 else 'train'))
report=train('numeric_binding',rows);report['parent_programs_used']=source_count;report['numeric_samples']=len(rows);report['split_group']='normalized local numeric context; duplicate contexts stay together'
(ROOT/'reports/v21/training_numeric.json').write_text(json.dumps(report,ensure_ascii=False,indent=2))
