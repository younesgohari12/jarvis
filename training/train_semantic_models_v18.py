from __future__ import annotations
import json,re,sys
from pathlib import Path
from collections import Counter
import numpy as np
from scipy.sparse import csr_matrix
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from jarvis.agent.semantic_models_v18 import SemanticFrameClassifierV18, NumericSlotTaggerV18

VERSION='dataset_v009';FRAME_SIZE=131072;SLOT_SIZE=65536;SEED=18092026

def rows(split):
 p=ROOT/'datasets'/'splits'/f'{VERSION}_{split}.jsonl';return [json.loads(x) for x in p.read_text(encoding='utf-8').splitlines() if x.strip()]

def sparse_matrix(samples, feature_fn, size):
 indptr=[0];indices=[];values=[]
 for sample in samples:
  idx,val=feature_fn(sample,size);order=np.argsort(idx);idx=idx[order];val=val[order]
  indices.extend(int(x) for x in idx);values.extend(float(x) for x in val);indptr.append(len(indices))
 return csr_matrix((np.asarray(values,np.float32),np.asarray(indices,np.int32),np.asarray(indptr,np.int32)),shape=(len(samples),size),dtype=np.float32)

def metrics(clf,x,y,labels):
 pred=clf.predict(x);cm=confusion_matrix(y,pred,labels=np.arange(len(labels)));pc={}
 for i,l in enumerate(labels):
  den=int(cm[i].sum());pc[l]=float(cm[i,i]/den) if den else None
 return {'examples':len(y),'accuracy':float(accuracy_score(y,pred)),'per_class_accuracy':pc,'confusion_matrix':cm.tolist()}

def frame_train():
 splitrows={s:[r for r in rows(s) if r.get('origin')=='jarvis-semantic-v18' and r.get('metadata',{}).get('semantic_frame') not in {'routing'}] for s in ('train','validation','test')}
 labels=tuple(sorted({r['metadata']['semantic_frame'] for arr in splitrows.values() for r in arr}))
 lid={l:i for i,l in enumerate(labels)}
 def feat(r,size):return SemanticFrameClassifierV18.sparse_features(r['input'],size)
 xs={s:sparse_matrix(arr,feat,FRAME_SIZE) for s,arr in splitrows.items()};ys={s:np.asarray([lid[r['metadata']['semantic_frame']] for r in arr],np.int64) for s,arr in splitrows.items()}
 clf=SGDClassifier(loss='log_loss',penalty='l2',alpha=3e-6,max_iter=220,tol=1e-6,random_state=SEED,class_weight='balanced',average=True).fit(xs['train'],ys['train'])
 rep={s:metrics(clf,xs[s],ys[s],labels) for s in xs}
 weights=np.asarray(clf.coef_,np.float32);bias=np.asarray(clf.intercept_,np.float32)
 np.savez_compressed(ROOT/'models'/'semantic_frame_v18.npz',format=np.asarray(SemanticFrameClassifierV18.FORMAT),labels=np.asarray(labels),weights=weights,bias=bias,parameter_count=np.asarray(weights.size+bias.size,np.int64),training_examples=np.asarray(len(splitrows['train']),np.int64),dataset_version=np.asarray(VERSION))
 return {'labels':labels,'parameter_count':int(weights.size+bias.size),'splits':{s:len(v) for s,v in splitrows.items()},'metrics':rep}

def number_samples(split):
 out=[]
 for r in rows(split):
  if r.get('origin')!='jarvis-semantic-v18':continue
  sv=r.get('metadata',{}).get('slot_values') or {}
  if not sv:continue
  # normalized numeric values expected in prompt; ambiguous equal-value roles are skipped.
  byval={}
  for role,val in sv.items():
   try:key=round(float(val),8)
   except (TypeError,ValueError):continue
   byval.setdefault(key,[]).append(role)
  text=r['input']
  for m in re.finditer(r'(?<![\w.])\d+(?:[.,]\d+)?',text.translate(str.maketrans('۰۱۲۳۴۵۶۷۸۹٠١٢٣٤٥٦٧٨٩','01234567890123456789'))):
   val=round(float(m.group().replace(',','.')),8);roles=byval.get(val,[])
   label=roles[0] if len(roles)==1 else 'other'
   out.append((text,m.start(),m.end(),label))
 return out

def slot_train():
 data={s:number_samples(s) for s in ('train','validation','test')};labels=tuple(sorted({x[3] for arr in data.values() for x in arr}));lid={l:i for i,l in enumerate(labels)}
 def feat(sample,size):text,a,b,_=sample;return NumericSlotTaggerV18.sparse_features(text,a,b,size)
 xs={s:sparse_matrix(arr,feat,SLOT_SIZE) for s,arr in data.items()};ys={s:np.asarray([lid[x[3]] for x in arr],np.int64) for s,arr in data.items()}
 clf=SGDClassifier(loss='log_loss',penalty='l2',alpha=4e-6,max_iter=240,tol=1e-6,random_state=SEED+1,class_weight='balanced',average=True).fit(xs['train'],ys['train'])
 rep={s:metrics(clf,xs[s],ys[s],labels) for s in xs};weights=np.asarray(clf.coef_,np.float32);bias=np.asarray(clf.intercept_,np.float32)
 np.savez_compressed(ROOT/'models'/'numeric_slot_tagger_v18.npz',format=np.asarray(NumericSlotTaggerV18.FORMAT),labels=np.asarray(labels),weights=weights,bias=bias,parameter_count=np.asarray(weights.size+bias.size,np.int64),training_examples=np.asarray(len(data['train']),np.int64),dataset_version=np.asarray(VERSION))
 return {'labels':labels,'parameter_count':int(weights.size+bias.size),'splits':{s:len(v) for s,v in data.items()},'metrics':rep,'label_counts':{s:dict(Counter(x[3] for x in v)) for s,v in data.items()}}

def main():
 frame=frame_train();slot=slot_train();report={'format':'jarvis-semantic-models-v18-report','dataset_version':VERSION,'frame_classifier':frame,'numeric_slot_tagger':slot}
 # Quality gates: these tasks are template-diverse but deterministic; demand strong held-out scores.
 report['passed']=frame['metrics']['test']['accuracy']>=.93 and slot['metrics']['test']['accuracy']>=.88
 (ROOT/'reports'/'semantic_models_v18_training_report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
 print(json.dumps(report,ensure_ascii=False,indent=2))
 if not report['passed']:raise SystemExit(2)
if __name__=='__main__':main()
