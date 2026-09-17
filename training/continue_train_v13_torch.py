from __future__ import annotations
import argparse,json,math,sys,time
from datetime import UTC,datetime
from pathlib import Path
import numpy as np, torch
ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))
from jarvis.neural.torch_model import JarvisTorchTransformer
from jarvis.neural.transformer import JarvisTransformer
from training.continue_train_v12_torch import load_numpy_into_torch, export_torch_into_numpy, make_batch

def load_pools(path:Path):
    pools={'generalization':[],'reasoning':[],'rehearsal':[]}
    for line in path.read_text(encoding='utf-8').splitlines():
        if not line.strip(): continue
        r=json.loads(line); item=([int(x) for x in r['tokens']],int(r['prompt_length']),int(r.get('stage',0)))
        o=str(r.get('origin',''))
        key='generalization' if o=='jarvis-generalization-v13' else ('reasoning' if o=='jarvis-reasoning-v13' else 'rehearsal')
        pools[key].append(item)
    if any(not v for v in pools.values()): raise RuntimeError({k:len(v) for k,v in pools.items()})
    return pools

def loss_probe(model,pool,length,device,limit=24):
    vals=[]; model.eval()
    with torch.no_grad():
        step=max(1,len(pool)//limit)
        for toks,pl,_ in pool[::step][:limit]:
            x,y=make_batch((toks,pl),length,device); vals.append(float(model(x,y)['loss']))
    return round(float(np.mean(vals)),6)

def main(args):
    torch.set_num_threads(max(1,args.threads)); device=torch.device('cpu')
    src=ROOT/args.source; out=ROOT/args.output
    nm=JarvisTransformer.load(src,dequantize=True); model=JarvisTorchTransformer(nm.config).to(device); load_numpy_into_torch(nm,model)
    pools=load_pools(ROOT/'datasets/tokenized/dataset_v005_train.jsonl'); length=min(args.sequence_length,nm.config.max_seq_len)
    before={k:loss_probe(model,v,length,device) for k,v in pools.items()}
    opt=torch.optim.AdamW(model.parameters(),lr=args.learning_rate,weight_decay=args.weight_decay,betas=(0.9,0.95))
    rng=np.random.default_rng(nm.config.seed+1300); probs=np.asarray([args.generalization_ratio,args.reasoning_ratio,args.rehearsal_ratio],dtype=float); probs/=probs.sum(); keys=['generalization','reasoning','rehearsal']
    warm=max(1,min(6,args.steps//10)); losses=[]; counts={k:0 for k in keys}; grads=[]; started=time.perf_counter(); model.train()
    for s in range(args.steps):
        key=keys[int(rng.choice(3,p=probs))]; pool=pools[key]; toks,pl,stage=pool[int(rng.integers(0,len(pool)))]
        x,y=make_batch((toks,pl),length,device); opt.zero_grad(set_to_none=True); loss=model(x,y)['loss']; loss.backward(); grad=float(torch.nn.utils.clip_grad_norm_(model.parameters(),args.gradient_clip))
        if s<warm: lr=args.learning_rate*(s+1)/warm
        else:
            q=(s-warm)/max(1,args.steps-warm); lr=max(args.min_learning_rate,args.min_learning_rate+(args.learning_rate-args.min_learning_rate)*0.5*(1+math.cos(math.pi*q)))
        for g in opt.param_groups:g['lr']=lr
        opt.step(); losses.append(float(loss.detach())); grads.append(grad); counts[key]+=1
        if s in {0,args.steps-1} or (s+1)%10==0: print(json.dumps({'event':'v13_step','step':s+1,'pool':key,'stage':stage,'loss':round(float(loss.detach()),5),'lr':lr},ensure_ascii=False),flush=True)
    after={k:loss_probe(model,v,length,device) for k,v in pools.items()}; export_torch_into_numpy(model,nm)
    metadata={**nm.metadata,'model_id':'jarvis_nano_v13_candidate','training_complete':True,'dataset_version':'dataset_v005','pretrained_source':None,'backend':'torch_cpu_adamw','continual_training':'generalization100_reasoning100_with_v004_rehearsal','continual_steps':args.steps,'step':int(nm.metadata.get('step',0) or 0)+args.steps,'trained_at_utc':datetime.now(UTC).isoformat()}
    nm.save_quantized(out,metadata)
    rep={'format':'jarvis-continual-training-v13-v1','backend':'torch_cpu_adamw','source':args.source,'output':args.output,'parameters':nm.parameter_count,'dataset_version':'dataset_v005','steps':args.steps,'sequence_length':length,'ratios':dict(zip(keys,probs.round(4).tolist())),'learning_rate':args.learning_rate,'min_learning_rate':args.min_learning_rate,'pool_counts':counts,'mean_training_loss':round(float(np.mean(losses)),6),'mean_gradient_norm':round(float(np.mean(grads)),6),'probe_loss_before':before,'probe_loss_after':after,'probe_improvement_pct':{k:round(100*(before[k]-after[k])/max(before[k],1e-9),3) for k in keys},'duration_seconds':round(time.perf_counter()-started,3)}
    (ROOT/'models/continual_training_v13.json').write_text(json.dumps(rep,ensure_ascii=False,indent=2)+'\n',encoding='utf-8'); print(json.dumps(rep,ensure_ascii=False,indent=2))
if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('--source',default='models/jarvis_nano_v12.npz'); p.add_argument('--output',default='models/jarvis_nano_v13_candidate.npz'); p.add_argument('--steps',type=int,default=40); p.add_argument('--sequence-length',type=int,default=64); p.add_argument('--learning-rate',type=float,default=1.5e-5); p.add_argument('--min-learning-rate',type=float,default=3e-6); p.add_argument('--weight-decay',type=float,default=0.003); p.add_argument('--gradient-clip',type=float,default=0.7); p.add_argument('--generalization-ratio',type=float,default=0.38); p.add_argument('--reasoning-ratio',type=float,default=0.42); p.add_argument('--rehearsal-ratio',type=float,default=0.20); p.add_argument('--threads',type=int,default=5); main(p.parse_args())
