"""True next-token cross entropy LoRA training, frozen project-owned Transformer.
No torch or external pretrained model. Adapter A and B are optimized with Adam.
"""
from pathlib import Path
import json,hashlib,sys,time
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from jarvis.neural.transformer import JarvisTransformer
from jarvis.neural.tokenizer import JarvisTokenizer
base=ROOT/'models/jarvis_nano_v18.npz';model=JarvisTransformer.load(base,dequantize=True);tok=JarvisTokenizer.load(ROOT/'models/jarvis_tokenizer_v003.json')
print('base',model.config.d_model,model.config.vocab_size,flush=True)
rows=[]
for name in ['persian','code']:
 for line in (ROOT/'datasets/v21'/f'{name}.jsonl').read_text().splitlines():
  r=json.loads(line)
  if name=='code' and r['label'] not in ('generate','explain'):continue
  if not isinstance(r['target'],str):continue
  rows.append(r)
rng=np.random.default_rng(2109);rng.shuffle(rows)
# Deterministic stratified cap controls CPU cost; full source counts are reported.
selected=[];counts={}
for r in rows:
 key=(r['section'],r['split']);limit=1200 if r['split']=='train' else 160
 if counts.get(key,0)<limit:selected.append(r);counts[key]=counts.get(key,0)+1
capture=[];original=model._project_head
model._project_head=lambda x:(capture.append(x.copy()) or original(x))
X=[];Y=[];splits=[];domains=[];used=[]
start=time.time()
for i,r in enumerate(selected):
 prompt=[tok.bos_id,tok.special_to_id['<user>'],*tok.encode(r['text'])[-56:],tok.special_to_id['<assistant>']]
 answer=[*tok.encode(r['target'])[:48],tok.eos_id];tokens=prompt+answer
 capture.clear();model.forward(tokens[:-1]);hidden=capture[-1][0]
 # Uniformly sample up to 12 answer positions, including start and end.
 positions=np.unique(np.linspace(len(prompt)-1,len(tokens)-2,min(12,len(answer)),dtype=int))
 for pos in positions:X.append(hidden[pos]);Y.append(tokens[pos+1]);splits.append(r['split']);domains.append(r['section'])
 used.append(r['id'])
 if (i+1)%200==0:print('features',i+1,len(selected),'seconds',round(time.time()-start,1),flush=True)
X=np.asarray(X,dtype=np.float32);Y=np.asarray(Y);splits=np.asarray(splits);domains=np.asarray(domains)
W=model.parameters['tok_embeddings'].T if model.config.tie_embeddings else model.parameters['lm_head'];rank=16
A=rng.normal(0,.02,(X.shape[1],rank)).astype('float32');B=np.zeros((rank,W.shape[1]),dtype='float32')
ma=np.zeros_like(A);va=np.zeros_like(A);mb=np.zeros_like(B);vb=np.zeros_like(B);step=0
tr=np.where(splits=='train')[0];valid=np.where(splits=='validation')[0];test=np.where(splits=='test')[0]
def metrics(ix,aa=None,bb=None):
 total=0;correct=0
 for chunk in np.array_split(ix,max(1,(len(ix)+127)//128)):
  logits=X[chunk]@W
  if aa is not None:logits+=(X[chunk]@aa)@bb
  logits-=logits.max(axis=1,keepdims=True);p=np.exp(logits);p/=p.sum(axis=1,keepdims=True)
  total-=np.log(np.maximum(p[np.arange(len(chunk)),Y[chunk]],1e-30)).sum();correct+=(p.argmax(axis=1)==Y[chunk]).sum()
 return {'tokens':len(ix),'loss':float(total/len(ix)),'token_accuracy':float(correct/len(ix))}
before={s:metrics(ix) for s,ix in [('validation',valid),('test',test)]};best=(before['validation']['loss'],A.copy(),B.copy());history=[]
for epoch in range(8):
 order=rng.permutation(tr)
 for offset in range(0,len(order),128):
  ix=order[offset:offset+128];x=X[ix];z=x@A;logits=x@W+z@B;logits-=logits.max(axis=1,keepdims=True);p=np.exp(logits);p/=p.sum(axis=1,keepdims=True);p[np.arange(len(ix)),Y[ix]]-=1;p/=len(ix)
  ga=x.T@(p@B.T)+1e-4*A;gb=z.T@p+1e-4*B
  norm=np.sqrt(np.sum(ga*ga)+np.sum(gb*gb));scale=min(1,1/(norm+1e-8));ga*=scale;gb*=scale;step+=1
  for param,grad,m,v in [(A,ga,ma,va),(B,gb,mb,vb)]:
   m*=.9;m+=.1*grad;v*=.999;v+=.001*grad*grad;param-=.001*(m/(1-.9**step))/(np.sqrt(v/(1-.999**step))+1e-8)
 val=metrics(valid,A,B);history.append({'epoch':epoch+1,**val});print('epoch',epoch+1,val,flush=True)
 if val['loss']<best[0]:best=(val['loss'],A.copy(),B.copy())
A,B=best[1:];after={s:metrics(ix,A,B) for s,ix in [('train',tr),('validation',valid),('test',test)]}
bydomain={}
for domain in sorted(set(domains)):
 ix=test[domains[test]==domain];bydomain[domain]={'before':metrics(ix),'after':metrics(ix,A,B)}
passed=after['validation']['loss']<before['validation']['loss'] and all(v['after']['loss']<=v['before']['loss'] for v in bydomain.values())
metadata={'architecture':'frozen_JARVIS_transformer_plus_rank16_output_LoRA','rank':rank,'trainable_parameters':int(A.size+B.size),'base_sha256':hashlib.sha256(base.read_bytes()).hexdigest(),'base_parameters_unchanged':True,'examples_used':len(selected),'example_counts':{str(k):v for k,v in counts.items()},'selected_ids':used,'before':before,'after':after,'test_by_domain':bydomain,'history':history,'gate_passed':bool(passed),'gate_scope':'teacher-forced token loss; not conversational accuracy or intelligence index','runtime_enabled':bool(passed),'elapsed_seconds':time.time()-start}
p=ROOT/'models/language_adapter_v21.npz';np.savez_compressed(p,A=A,B=B,metadata=json.dumps(metadata));metadata['sha256']=hashlib.sha256(p.read_bytes()).hexdigest()
(ROOT/'reports/v21/language_training.json').write_text(json.dumps(metadata,ensure_ascii=False,indent=2))
(ROOT/'models/language_adapter_v21.json').write_text(json.dumps({'enabled':bool(passed),'file':p.name,'base_sha256':metadata['base_sha256'],'sha256':metadata['sha256'],'gate_scope':metadata['gate_scope']},indent=2))
print('complete',passed,before,after,flush=True)
