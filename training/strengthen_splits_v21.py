"""Group by operation structure, not surface paraphrase; repair ineffective negatives."""
from pathlib import Path
import json,hashlib,copy
from collections import Counter
ROOT=Path(__file__).resolve().parents[1]
import sys
sys.path.insert(0,str(ROOT))
from jarvis.agent.normalization_v20 import skeleton
names=['semantic_ir','world_model','verification','adversarial','persian'];data={n:[json.loads(x) for x in (ROOT/'datasets/v21'/f'{n}.jsonl').read_text().splitlines()] for n in names};parents={r['id']:r for name in names if name!='verification' for r in data[name]}
def structural(g):
 slots=g.get('slots',{});ops=g.get('operations',[])
 return json.dumps({'domain':g.get('domain'),'operations':[{k:v for k,v in op.items() if k in ('op','comparison')} for op in ops],'slot_keys':sorted(slots),'query':slots.get('query'),'mode':slots.get('mode')},sort_keys=True)
def partition(group):
 n=int(hashlib.sha256(group.encode()).hexdigest()[:8],16)%10
 return 'test' if n==0 else 'validation' if n==1 else 'train'
for r in parents.values():
 group=structural(r['program']);r['concept_group']='semantic_program:'+hashlib.sha256(group.encode()).hexdigest();r['split']=partition(r['concept_group'])
fixed=[]
for r in data['verification']:
 parent=parents[r['target']['source_id']];r['concept_group']=parent['concept_group'];r['split']=parent['split']
 correct=r['target']['correct'];bad=r['program']
 if bad==correct:
  bad=copy.deepcopy(correct);events=bad['operations'];pair=next(((i,j) for i in range(len(events)) for j in range(i+1,len(events)) if events[i]!=events[j]),None)
  if pair:
   i,j=pair;events[i],events[j]=events[j],events[i]
  else:bad['answer']+=11;r['label']='answer_value';r['target']['failed_check']='answer_value'
  r['program']=bad;r['text']=parent['text']+'\nCandidate: '+json.dumps(bad,ensure_ascii=False,sort_keys=True);r['template_id']=hashlib.sha256(skeleton(r['text']).encode()).hexdigest()[:20];fixed.append(r['id'])
for name,rows in data.items():(ROOT/'datasets/v21'/f'{name}.jsonl').write_text('\n'.join(json.dumps(r,ensure_ascii=False) for r in rows)+'\n')
manifest=json.loads((ROOT/'datasets/v21/manifest.json').read_text());rows=sum(data.values(),[])
manifest['split_counts']=dict(Counter(r['split'] for r in rows));manifest['split_strategy']='operation-domain-slot-query structure grouped; verification follows source parent'
manifest['semantic_program_groups']=len({r['concept_group'] for r in rows});manifest['fixed_identity_mutations']=fixed
manifest['files']={n+'.jsonl':hashlib.sha256((ROOT/'datasets/v21'/f'{n}.jsonl').read_bytes()).hexdigest() for n in names}
(ROOT/'datasets/v21/manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2));print('strengthened',manifest['semantic_program_groups'],'fixed',len(fixed),manifest['split_counts'])
