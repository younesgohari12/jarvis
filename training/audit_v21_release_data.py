from pathlib import Path
from collections import Counter,defaultdict
import json,hashlib,sys,ast
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from jarvis.agent.normalization_v20 import skeleton
from jarvis.agent.code_v21 import SafePython
rows=[]
for name in ['semantic_ir','world_model','verification','adversarial','persian','code']:
 rows += [json.loads(l) for l in (ROOT/'datasets/v21'/f'{name}.jsonl').read_text().splitlines()]
byid={r['id']:r for r in rows};group_splits=defaultdict(set);template_splits=defaultdict(set)
for r in rows:
 group_splits[r['concept_group']].add(r['split']);template_splits[r['template_id']].add(r['split'])
violations=[];verification_unchanged=[];code_errors=[]
for r in rows:
 if r['section']=='verification':
  parent=byid.get(r['target']['source_id'])
  if parent is None or parent['split']!=r['split']:violations.append(r['id'])
  if r['program']==r['target']['correct']:verification_unchanged.append(r['id'])
 if r['section']=='code':
  for check in r['program']['checks']:
   try:
    observed=SafePython().run(r['program']['source'],json.loads(json.dumps(check['input'])))
    if observed!=check['expected']:code_errors.append(r['id']);break
   except Exception:code_errors.append(r['id']);break
report={'total':len(rows),'sections':dict(Counter(r['section'] for r in rows)),'unique_inputs':len({r['text'] for r in rows}),'unique_template_ids':len(template_splits),'cross_split_concept_groups':sum(len(s)>1 for s in group_splits.values()),'cross_split_template_ids':sum(len(s)>1 for s in template_splits.values()),'verification_parent_or_split_errors':violations,'verification_identity_mutations':verification_unchanged,'code_oracle_errors':code_errors,'human_authored_samples':0,'numerical_parameter_variants':'Limited to two per lexical skeleton for the 20k core; code uses 2000 structural groups across five tasks.','limitations':['Synthetic programs dominate; conceptual overlap across splits remains.','Core dataset does not deliver balanced linguistic coverage of all requested domains.','The 3000 Persian rows are mostly quantitative instructions, not a broad Persian conversation corpus.']}
# Reviewable near-duplicate audit on held-out source skeletons; duplicates are not hidden.
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
texts=[skeleton(r['text'])[:1400] for r in rows]
vector=TfidfVectorizer(analyzer='char',ngram_range=(3,4),max_features=20000,dtype=np.float32).fit_transform(texts)
train=[i for i,r in enumerate(rows) if r['split']=='train'];test=[i for i,r in enumerate(rows) if r['split']=='test'];maximum=[]
for offset in range(0,len(test),64):
 sim=vector[test[offset:offset+64]]@vector[train].T;maximum.extend(sim.max(axis=1).toarray().ravel().tolist())
report['near_duplicates']={'metric':'char3-4 TFIDF cosine on numeric-masked inputs truncated to 1400 characters','test_samples':len(test),'test_near_train_at_0_9':sum(x>=.9 for x in maximum),'max_cosine':max(maximum),'meaning':'Zero exact overlap is not evidence of independent linguistic concepts.'}
(ROOT/'reports/v21/dataset_audit.json').write_text(json.dumps(report,ensure_ascii=False,indent=2))
print({k:v for k,v in report.items() if k not in ['verification_parent_or_split_errors','verification_identity_mutations','code_oracle_errors']},flush=True)
print('errors',len(violations),len(verification_unchanged),len(code_errors),flush=True)
