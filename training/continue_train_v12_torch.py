from __future__ import annotations

import argparse
import json
import math
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from jarvis.neural.torch_model import JarvisTorchTransformer  # noqa: E402
from jarvis.neural.transformer import JarvisTransformer  # noqa: E402

FOCUS_STAGES = (1, 2, 3, 4, 11)


def load_sequences(path: Path):
    curated = {stage: [] for stage in FOCUS_STAGES}
    rehearsal = {stage: [] for stage in FOCUS_STAGES}
    for line in path.read_text(encoding='utf-8').splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        stage = int(row.get('stage', 0))
        if stage not in curated:
            continue
        item = ([int(x) for x in row['tokens']], int(row['prompt_length']))
        (curated if str(row.get('curated_source_id', '')).strip() else rehearsal)[stage].append(item)
    missing = [s for s in FOCUS_STAGES if not curated[s] or not rehearsal[s]]
    if missing:
        raise RuntimeError(f'Missing curated/rehearsal data for stages: {missing}')
    return curated, rehearsal


def make_batch(item, length: int, device: torch.device):
    tokens, prompt_length = item
    inputs = tokens[:-1]
    labels = [tok if i >= max(0, prompt_length - 2) else -100 for i, tok in enumerate(tokens[1:])]
    if len(inputs) > length:
        max_start = len(inputs) - length
        candidates = [s for s in range(max_start + 1) if any(v >= 0 for v in labels[s:s+length])]
        start = candidates[0] if candidates else max_start
        inputs = inputs[start:start+length]
        labels = labels[start:start+length]
    if len(inputs) < length:
        pad = length - len(inputs)
        inputs += [0] * pad
        labels += [-100] * pad
    return (
        torch.tensor([inputs], dtype=torch.long, device=device),
        torch.tensor([labels], dtype=torch.long, device=device),
    )


def load_numpy_into_torch(source: JarvisTransformer, model: JarvisTorchTransformer) -> None:
    state = model.state_dict()
    state['tok_embeddings.weight'] = torch.from_numpy(source.parameters['tok_embeddings'].copy())
    state['final_norm.weight'] = torch.from_numpy(source.parameters['final_norm'].copy())
    for i in range(source.config.n_layers):
        p = f'layers.{i}'
        state[f'{p}.attn_norm.weight'] = torch.from_numpy(source.parameters[f'{p}.attn_norm'].copy())
        for short, long in [('wq','attention.wq'),('wk','attention.wk'),('wv','attention.wv'),('wo','attention.wo')]:
            state[f'{p}.{long}.weight'] = torch.from_numpy(source.parameters[f'{p}.{short}'].T.copy())
        state[f'{p}.ffn_norm.weight'] = torch.from_numpy(source.parameters[f'{p}.ffn_norm'].copy())
        for short, long in [('w1','feed_forward.w1'),('w2','feed_forward.w2'),('w3','feed_forward.w3')]:
            state[f'{p}.{long}.weight'] = torch.from_numpy(source.parameters[f'{p}.{short}'].T.copy())
    model.load_state_dict(state)


def export_torch_into_numpy(model: JarvisTorchTransformer, target: JarvisTransformer) -> None:
    state = model.state_dict()
    target.parameters['tok_embeddings'][:] = state['tok_embeddings.weight'].detach().cpu().numpy()
    target.parameters['final_norm'][:] = state['final_norm.weight'].detach().cpu().numpy()
    for i in range(target.config.n_layers):
        p = f'layers.{i}'
        target.parameters[f'{p}.attn_norm'][:] = state[f'{p}.attn_norm.weight'].detach().cpu().numpy()
        for short, long in [('wq','attention.wq'),('wk','attention.wk'),('wv','attention.wv'),('wo','attention.wo')]:
            target.parameters[f'{p}.{short}'][:] = state[f'{p}.{long}.weight'].detach().cpu().numpy().T
        target.parameters[f'{p}.ffn_norm'][:] = state[f'{p}.ffn_norm.weight'].detach().cpu().numpy()
        for short, long in [('w1','feed_forward.w1'),('w2','feed_forward.w2'),('w3','feed_forward.w3')]:
            target.parameters[f'{p}.{short}'][:] = state[f'{p}.{long}.weight'].detach().cpu().numpy().T


def probe(model, pools, length, device, limit=8):
    model.eval(); values = {}
    with torch.no_grad():
        for stage in FOCUS_STAGES:
            stage_values=[]
            for item in pools[stage][:min(limit, len(pools[stage]))]:
                x,y=make_batch(item,length,device)
                stage_values.append(float(model(x,y)['loss']))
            values[str(stage)] = round(float(np.mean(stage_values)),6)
    return values


def main(args):
    torch.set_num_threads(max(1, int(args.threads)))
    device=torch.device('cpu')
    source_path=ROOT/args.source
    output_path=ROOT/args.output
    numpy_model=JarvisTransformer.load(source_path,dequantize=True)
    model=JarvisTorchTransformer(numpy_model.config).to(device)
    load_numpy_into_torch(numpy_model,model)
    curated,rehearsal=load_sequences(ROOT/'datasets/tokenized/dataset_v004_train.jsonl')
    length=min(int(args.sequence_length),numpy_model.config.max_seq_len)
    before_curated=probe(model,curated,length,device)
    before_rehearsal=probe(model,rehearsal,length,device)
    optimizer=torch.optim.AdamW(model.parameters(),lr=float(args.learning_rate),weight_decay=float(args.weight_decay),betas=(0.9,0.95))
    rng=np.random.default_rng(numpy_model.config.seed+1200)
    total=int(args.steps_per_stage)*len(FOCUS_STAGES)
    warmup=min(12,max(1,total//12))
    global_step=0; all_losses=[]; stage_metrics=[]; started=time.perf_counter()
    model.train()
    for stage in FOCUS_STAGES:
        losses=[]; grads=[]
        for local in range(1,int(args.steps_per_stage)+1):
            pool=curated[stage] if float(rng.random()) < float(args.curated_ratio) else rehearsal[stage]
            item=pool[int(rng.integers(0,len(pool)))]
            x,y=make_batch(item,length,device)
            optimizer.zero_grad(set_to_none=True)
            loss=model(x,y)['loss']; loss.backward()
            grad=float(torch.nn.utils.clip_grad_norm_(model.parameters(),float(args.gradient_clip)))
            if global_step < warmup:
                rate=float(args.learning_rate)*(global_step+1)/warmup
            else:
                progress=(global_step-warmup)/max(1,total-warmup)
                cosine=0.5*(1+math.cos(math.pi*min(1.0,progress)))
                rate=max(0.000006,float(args.learning_rate)/8)+(float(args.learning_rate)-max(0.000006,float(args.learning_rate)/8))*cosine
            for group in optimizer.param_groups: group['lr']=rate
            optimizer.step(); global_step+=1
            losses.append(float(loss.detach())); all_losses.append(float(loss.detach())); grads.append(grad)
            if local==1 or local%16==0 or local==int(args.steps_per_stage):
                print(json.dumps({'event':'torch_continual_step','stage':stage,'local_step':local,'global_step':global_step,'recent_loss':round(float(np.mean(losses[-16:])),6),'learning_rate':rate,'gradient_norm':round(grad,6)},ensure_ascii=False),flush=True)
        stage_metrics.append({'stage':stage,'steps':len(losses),'mean_loss':round(float(np.mean(losses)),6),'mean_gradient_norm':round(float(np.mean(grads)),6)})
    after_curated=probe(model,curated,length,device)
    after_rehearsal=probe(model,rehearsal,length,device)
    export_torch_into_numpy(model,numpy_model)
    metadata={**numpy_model.metadata,'model_id':'jarvis_nano_v12_candidate','training_complete':True,'dataset_version':'dataset_v004','pretrained_source':None,'backend':'torch_cpu_adamw','continual_training':'curated_100_v12_with_v003_rehearsal','continual_focus_stages':list(FOCUS_STAGES),'continual_steps':total,'step':int(numpy_model.metadata.get('step',0) or 0)+total,'trained_at_utc':datetime.now(UTC).isoformat()}
    numpy_model.save_quantized(output_path,metadata)
    cb=float(np.mean(list(before_curated.values()))); ca=float(np.mean(list(after_curated.values())))
    rb=float(np.mean(list(before_rehearsal.values()))); ra=float(np.mean(list(after_rehearsal.values())))
    report={'format':'jarvis-continual-training-v12-v2','backend':'torch_cpu_adamw','source':args.source,'output':args.output,'parameters':numpy_model.parameter_count,'dataset_version':'dataset_v004','focus_stages':list(FOCUS_STAGES),'steps_per_stage':int(args.steps_per_stage),'total_steps':total,'sequence_length':length,'curated_ratio':float(args.curated_ratio),'learning_rate':float(args.learning_rate),'mean_training_loss':round(float(np.mean(all_losses)),6),'probe_loss_before':{'curated':before_curated,'rehearsal':before_rehearsal},'probe_loss_after':{'curated':after_curated,'rehearsal':after_rehearsal},'probe_summary':{'curated_mean_before':round(cb,6),'curated_mean_after':round(ca,6),'curated_improvement_pct':round(100*(cb-ca)/max(cb,1e-9),3),'rehearsal_mean_before':round(rb,6),'rehearsal_mean_after':round(ra,6),'rehearsal_change_pct':round(100*(ra-rb)/max(rb,1e-9),3)},'stage_metrics':stage_metrics,'duration_seconds':round(time.perf_counter()-started,3)}
    (ROOT/'models/continual_training_v12.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(report,ensure_ascii=False,indent=2))
    return 0

if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('--source',default='models/jarvis_nano_v08.npz'); p.add_argument('--output',default='models/jarvis_nano_v12_candidate.npz'); p.add_argument('--steps-per-stage',type=int,default=48); p.add_argument('--sequence-length',type=int,default=64); p.add_argument('--learning-rate',type=float,default=0.00008); p.add_argument('--weight-decay',type=float,default=0.004); p.add_argument('--gradient-clip',type=float,default=0.8); p.add_argument('--curated-ratio',type=float,default=0.72); p.add_argument('--threads',type=int,default=5)
    raise SystemExit(main(p.parse_args()))
