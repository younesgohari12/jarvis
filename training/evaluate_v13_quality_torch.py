from __future__ import annotations
import argparse,json,math,sys
from pathlib import Path
from collections import defaultdict
import numpy as np, torch
ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))
from jarvis.neural.tokenizer import JarvisTokenizer
from jarvis.neural.transformer import JarvisTransformer
from jarvis.neural.torch_model import JarvisTorchTransformer
from training.continue_train_v12_torch import load_numpy_into_torch

def rows(p): return [json.loads(x) for x in p.read_text(encoding='utf-8').splitlines() if x.strip()]
def encode(tok,r,L):
    ctx='\n'.join(str(x.get('content','')) for x in r.get('context',[]) if isinstance(x,dict)); user=f'{ctx}\n{r["input"]}'.strip() if ctx else str(r['input'])
    prompt=[tok.bos_id,tok.special_to_id['<user>'],*tok.encode(user),tok.special_to_id['<assistant>']]; resp=[*tok.encode(str(r['output'])),tok.eos_id]; seq=prompt+resp; x=seq[:-1]; raw=seq[1:]; boundary=max(0,len(prompt)-2); y=[v if i>=boundary else -100 for i,v in enumerate(raw)];
    if len(x)>L:
        max_start=len(x)-L; candidates=[s for s in range(max_start+1) if any(v>=0 for v in y[s:s+L])]; start=candidates[0] if candidates else max_start; x=x[start:start+L]; y=y[start:start+L]
    if len(x)<L: x=x+[0]*(L-len(x)); y=y+[-100]*(L-len(y))
    return x,y
def score(path,tok,data,L,batch,threads):
    torch.set_num_threads(threads); nm=JarvisTransformer.load(path,dequantize=True); m=JarvisTorchTransformer(nm.config); load_numpy_into_torch(nm,m);m.eval(); losses=[];cor=tot=0; bylang=defaultdict(lambda:[0,0,[]])
    with torch.no_grad():
      for off in range(0,len(data),batch):
        chunk=data[off:off+batch]; enc=[encode(tok,r,L) for r in chunk]; x=torch.tensor([z[0] for z in enc],dtype=torch.long); y=torch.tensor([z[1] for z in enc],dtype=torch.long); logits=m(x)['logits']; flat=torch.nn.functional.cross_entropy(logits.reshape(-1,logits.shape[-1]),y.reshape(-1),ignore_index=-100,reduction='none').reshape(y.shape); pred=logits.argmax(-1)
        for j,r in enumerate(chunk):
          v=y[j]>=0; n=int(v.sum()); cc=int(((pred[j]==y[j])&v).sum()); loss=float(flat[j][v].mean()) if n else float('nan'); losses.append(loss);cor+=cc;tot+=n; z=bylang[str(r.get('language','unknown'))];z[0]+=cc;z[1]+=n;z[2].append(loss)
    del m,nm
    mean=float(np.mean(losses)); return {'cases':len(losses),'cross_entropy':round(mean,6),'perplexity':round(math.exp(min(20,mean)),6),'token_accuracy':round(cor/max(1,tot),6),'by_language_token_accuracy':{k:round(v[0]/max(1,v[1]),6) for k,v in sorted(bylang.items())},'by_language_cross_entropy':{k:round(float(np.mean(v[2])),6) for k,v in sorted(bylang.items())}}
def main(a):
    tok=JarvisTokenizer.load(ROOT/'models/jarvis_tokenizer_v003.json');test=rows(ROOT/'datasets/splits/dataset_v005_test.jsonl');old=rows(ROOT/'datasets/splits/dataset_v004_test.jsonl'); groups={'generalization':sorted([r for r in test if r.get('origin')=='jarvis-generalization-v13'],key=lambda r:r['id'])[:a.max_cases],'reasoning':sorted([r for r in test if r.get('origin')=='jarvis-reasoning-v13'],key=lambda r:r['id'])[:a.max_cases],'v004_regression':sorted(old,key=lambda r:r['id'])[:a.old_cases]};res={'format':'jarvis-v13-quality-gate-v2-torch','baseline':a.baseline,'candidate':a.candidate,'dataset':'dataset_v005','groups':{}}
    for k,data in groups.items():
      b=score(ROOT/a.baseline,tok,data,a.sequence_length,a.batch_size,a.threads); c=score(ROOT/a.candidate,tok,data,a.sequence_length,a.batch_size,a.threads);res['groups'][k]={'baseline':b,'candidate':c,'cross_entropy_improvement_pct':round(100*(b['cross_entropy']-c['cross_entropy'])/max(b['cross_entropy'],1e-9),3),'token_accuracy_delta_points':round(100*(c['token_accuracy']-b['token_accuracy']),3)}
    g=res['groups'];checks={'generalization_ce_improves':g['generalization']['candidate']['cross_entropy']<g['generalization']['baseline']['cross_entropy'],'reasoning_ce_improves':g['reasoning']['candidate']['cross_entropy']<g['reasoning']['baseline']['cross_entropy'],'generalization_token_accuracy_nonworse':g['generalization']['token_accuracy_delta_points']>=-0.25,'reasoning_token_accuracy_nonworse':g['reasoning']['token_accuracy_delta_points']>=-0.25,'v004_ce_regression_under_0_5pct':g['v004_regression']['candidate']['cross_entropy']<=g['v004_regression']['baseline']['cross_entropy']*1.005,'v004_token_accuracy_delta_above_minus_0_5pt':g['v004_regression']['token_accuracy_delta_points']>=-0.5};res['checks']=checks;res['passed']=all(checks.values());(ROOT/'models/evaluation_v13_quality_gate.json').write_text(json.dumps(res,ensure_ascii=False,indent=2)+'\n',encoding='utf-8');print(json.dumps(res,ensure_ascii=False,indent=2))
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--baseline',default='models/jarvis_nano_v12.npz');p.add_argument('--candidate',default='models/jarvis_nano_v13_candidate.npz');p.add_argument('--max-cases',type=int,default=180);p.add_argument('--old-cases',type=int,default=180);p.add_argument('--sequence-length',type=int,default=64);p.add_argument('--batch-size',type=int,default=12);p.add_argument('--threads',type=int,default=5);main(p.parse_args())
