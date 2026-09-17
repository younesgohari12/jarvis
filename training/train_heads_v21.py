from pathlib import Path
from collections import Counter
import numpy as np,json,sys,hashlib,time
from scipy.sparse import csr_matrix
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score,f1_score,confusion_matrix,log_loss
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from jarvis.agent.models_v21 import model_features,SIZE

def train(name,rows):
 cached=ROOT/'models'/f'{name}_v21.npz'
 if cached.exists():
  with np.load(cached,allow_pickle=False) as z:meta=json.loads(str(z['metadata']))
  if meta.get('training_rows_sha256')==hashlib.sha256(json.dumps(rows,ensure_ascii=False).encode()).hexdigest():
   meta['sha256']=hashlib.sha256(cached.read_bytes()).hexdigest();return meta
 texts=[r[0] for r in rows];labels=np.array([r[1] for r in rows]);splits=np.array([r[2] for r in rows]);indices=[];values=[];ptr=[0]
 for t in texts:
  i,v=model_features(t);indices.extend(i);values.extend(v);ptr.append(len(indices))
 X=csr_matrix((values,indices,ptr),shape=(len(rows),SIZE));tr=np.where(splits=='train')[0];va=np.where(splits=='validation')[0];te=np.where(splits=='test')[0]
 classes=sorted(set(labels));best=None;trials=[]
 for alpha in [.00001,.0001]:
  model=SGDClassifier(loss='log_loss',alpha=alpha,max_iter=60,tol=1e-4,random_state=21)
  model.fit(X[tr],labels[tr]);score=accuracy_score(labels[va],model.predict(X[va]));trials.append({'alpha':alpha,'validation_accuracy':score})
  if best is None or score>best[0]:best=(score,model)
 model=best[1];metrics={}
 for split,ix in [('train',tr),('validation',va),('test',te)]:
  pred=model.predict(X[ix]);p=model.predict_proba(X[ix]);metrics[split]={'n':len(ix),'accuracy':float(accuracy_score(labels[ix],pred)),'macro_f1':float(f1_score(labels[ix],pred,average='macro')),'loss':float(log_loss(labels[ix],p,labels=model.classes_)),'confusion_matrix':confusion_matrix(labels[ix],pred,labels=model.classes_).tolist()}
 w=model.coef_.astype('float32');b=model.intercept_.astype('float32')
 if len(model.classes_)==2:w=np.concatenate([-w/2,w/2]);b=np.concatenate([-b/2,b/2])
 metadata={'architecture':'hashed_linear_softmax','parameters':int(w.size+b.size),'classes':model.classes_.tolist(),'training_selection':trials,'metrics':metrics,'training_rows_sha256':hashlib.sha256(json.dumps(rows,ensure_ascii=False).encode()).hexdigest()}
 p=ROOT/'models'/f'{name}_v21.npz';np.savez_compressed(p,weights=w,bias=b,labels=model.classes_,metadata=json.dumps(metadata))
 metadata['sha256']=hashlib.sha256(p.read_bytes()).hexdigest();print(name,metrics,flush=True);return metadata

def main():
 get=lambda name:[json.loads(l) for l in (ROOT/'datasets/v21'/f'{name}.jsonl').read_text().splitlines()]
 code=get('code');repair=get('verification');reasoning=sum([get(x) for x in ['semantic_ir','world_model','adversarial','persian']],[])
 jobs={'code_task':[(r['text'],r['label'],r['split']) for r in code],'verification_failure':[(r['text'],r['label'],r['split']) for r in repair]}
 # Clause-level training is derived from source and gold program labels.
 import re
 ops=[]
 for r in reasoning:
  clauses=re.split(r'[;؛،,]| و سپس | and then ',r['text']);events=r['program'].get('operations',[])
  if len(clauses)==len(events)+1:
   for c,e in zip(clauses[1:],events):
    if e['op'] in ('add','subtract','multiply','divide','percentage_add','percentage_remove'):ops.append((c,e['op'],r['split']))
 # Identical clauses cannot cross split boundaries even when parent programs differ.
 from jarvis.agent.normalization_v20 import skeleton
 canonical={}
 for t,label,split in ops:canonical[(skeleton(t),label)]=(t,label)
 clean=[]
 for (sk,label),(t,_) in canonical.items():
  n=int(hashlib.sha256(sk.encode()).hexdigest()[:8],16)%10;clean.append((t,label,'test' if n==0 else 'validation' if n==1 else 'train'))
 jobs['operation_language']=clean
 report={'models':{name:train(name,rows) for name,rows in jobs.items()},'core_model_unchanged_by_this_script':True}
 (ROOT/'reports/v21/training_heads.json').write_text(json.dumps(report,ensure_ascii=False,indent=2))
if __name__=='__main__':main()
