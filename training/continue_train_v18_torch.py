from __future__ import annotations
import argparse,json,math,sys,time
from datetime import UTC,datetime
from pathlib import Path
import numpy as np,torch
ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from jarvis.neural.torch_model import JarvisTorchTransformer
from jarvis.neural.transformer import JarvisTransformer
from training.continue_train_v12_torch import load_numpy_into_torch,export_torch_into_numpy,make_batch

def pools(path):
 out={'v18':[],'persian_v18':[],'rehearsal':[]}
 for line in path.read_text(encoding='utf-8').splitlines():
  if not line.strip():continue
  r=json.loads(line);item=([int(x) for x in r['tokens']],int(r['prompt_length']),int(r.get('stage',0)))
  if str(r.get('origin','')) in {'jarvis-semantic-v18','jarvis-routing-v18'}:
   out['v18'].append(item)
   if str(r.get('language'))=='fa':out['persian_v18'].append(item)
  else:out['rehearsal'].append(item)
 if not all(out.values()):raise RuntimeError({k:len(v) for k,v in out.items()})
 return out

def probe(m,pool,L,device,limit=18):
 vals=[];m.eval()
 with torch.no_grad():
  step=max(1,len(pool)//limit)
  for toks,pl,_ in pool[::step][:limit]:
   x,y=make_batch((toks,pl),L,device);vals.append(float(m(x,y)['loss']))
 return round(float(np.mean(vals)),6)

def main(a):
 torch.set_num_threads(a.threads);device=torch.device('cpu');src=ROOT/a.source;outp=ROOT/a.output
 nm=JarvisTransformer.load(src,dequantize=True);m=JarvisTorchTransformer(nm.config).to(device);load_numpy_into_torch(nm,m);L=min(a.sequence_length,nm.config.max_seq_len);ps=pools(ROOT/'datasets/tokenized/dataset_v009_train.jsonl')
 before={k:probe(m,v,L,device) for k,v in ps.items()};keys=['v18','persian_v18','rehearsal'];probs=np.array([.48,.22,.30]);rng=np.random.default_rng(18092026);opt=torch.optim.AdamW(m.parameters(),lr=a.learning_rate,weight_decay=.002,betas=(.9,.95));losses=[];counts={k:0 for k in keys};start=time.perf_counter();m.train()
 for s in range(a.steps):
  key=keys[int(rng.choice(len(keys),p=probs))];pool=ps[key];toks,pl,stage=pool[int(rng.integers(0,len(pool)))];x,y=make_batch((toks,pl),L,device);opt.zero_grad(set_to_none=True);loss=m(x,y)['loss'];loss.backward();torch.nn.utils.clip_grad_norm_(m.parameters(),.65);lr=max(a.min_lr,a.learning_rate*(1-(s/max(1,a.steps-1))*.7));[g.update(lr=lr) for g in opt.param_groups];opt.step();losses.append(float(loss.detach()));counts[key]+=1;print(json.dumps({'event':'v18_step','step':s+1,'pool':key,'loss':round(float(loss.detach()),5),'lr':lr}),flush=True)
 after={k:probe(m,v,L,device) for k,v in ps.items()};export_torch_into_numpy(m,nm);meta={**nm.metadata,'model_id':'jarvis_nano_v18_candidate','dataset_version':'dataset_v009','training_complete':True,'continual_training_steps_v18':a.steps,'continual_training':'dataset_v009_semantic_roles_with_rehearsal','backend':'torch_cpu_adamw','trained_at_utc':datetime.now(UTC).isoformat(),'pretrained_source':None};nm.save_quantized(outp,meta)
 rep={'format':'jarvis-continual-training-v18-v1','dataset_version':'dataset_v009','source':a.source,'output':a.output,'parameters':nm.parameter_count,'steps':a.steps,'counts':counts,'probe_before':before,'probe_after':after,'improvement_pct':{k:round(100*(before[k]-after[k])/before[k],3) for k in keys},'mean_loss':round(float(np.mean(losses)),6),'duration_seconds':round(time.perf_counter()-start,3)};(ROOT/'models/continual_training_v18.json').write_text(json.dumps(rep,ensure_ascii=False,indent=2)+'\n',encoding='utf-8');print(json.dumps(rep,ensure_ascii=False,indent=2))
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--source',default='models/jarvis_nano_v13.npz');p.add_argument('--output',default='models/jarvis_nano_v18_candidate.npz');p.add_argument('--steps',type=int,default=12);p.add_argument('--sequence-length',type=int,default=64);p.add_argument('--learning-rate',type=float,default=6e-6);p.add_argument('--min-lr',type=float,default=1.5e-6);p.add_argument('--threads',type=int,default=5);main(p.parse_args())
