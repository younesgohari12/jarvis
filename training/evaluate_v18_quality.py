from __future__ import annotations
import json,math,sys
from pathlib import Path
from collections import defaultdict
import numpy as np,torch
ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from jarvis.neural.tokenizer import JarvisTokenizer
from jarvis.neural.transformer import JarvisTransformer
from jarvis.neural.torch_model import JarvisTorchTransformer
from training.continue_train_v12_torch import load_numpy_into_torch

def rows(p):return [json.loads(x) for x in p.read_text(encoding='utf-8').splitlines() if x.strip()]
def encode(tok,r,L):
 user=str(r['input']);prompt=[tok.bos_id,tok.special_to_id['<user>'],*tok.encode(user),tok.special_to_id['<assistant>']];resp=[*tok.encode(str(r['output'])),tok.eos_id];seq=prompt+resp;x=seq[:-1];raw=seq[1:];boundary=max(0,len(prompt)-2);y=[v if i>=boundary else -100 for i,v in enumerate(raw)]
 if len(x)>L:x=x[:L];y=y[:L]
 if len(x)<L:x+= [0]*(L-len(x));y += [-100]*(L-len(y))
 return x,y
def score(path,data,tok,L=64,batch=12):
 nm=JarvisTransformer.load(path,dequantize=True);m=JarvisTorchTransformer(nm.config);load_numpy_into_torch(nm,m);m.eval();losses=[];cor=tot=0
 with torch.no_grad():
  for off in range(0,len(data),batch):
   chunk=data[off:off+batch];enc=[encode(tok,r,L) for r in chunk];x=torch.tensor([z[0] for z in enc]);y=torch.tensor([z[1] for z in enc]);logits=m(x)['logits'];flat=torch.nn.functional.cross_entropy(logits.reshape(-1,logits.shape[-1]),y.reshape(-1),ignore_index=-100,reduction='none').reshape(y.shape);pred=logits.argmax(-1)
   for j in range(len(chunk)):
    mask=y[j]>=0;n=int(mask.sum());cor+=int(((pred[j]==y[j])&mask).sum());tot+=n;losses.append(float(flat[j][mask].mean()))
 return {'cases':len(data),'cross_entropy':round(float(np.mean(losses)),6),'token_accuracy':round(cor/max(1,tot),6)}
def main():
 torch.set_num_threads(5);tok=JarvisTokenizer.load(ROOT/'models/jarvis_tokenizer_v003.json');v9=rows(ROOT/'datasets/splits/dataset_v009_test.jsonl');v8=rows(ROOT/'datasets/splits/dataset_v008_test.jsonl')
 new=[r for r in v9 if r.get('origin') in {'jarvis-semantic-v18','jarvis-routing-v18'}][:180];fa=[r for r in new if r.get('language')=='fa'][:120];legacy=v8[:180]
 groups={'v18_new':new,'v18_fa':fa,'legacy_v008':legacy};base=ROOT/'models/jarvis_nano_v13.npz';cand=ROOT/'models/jarvis_nano_v18_candidate.npz';out={'format':'jarvis-v18-quality-gate','dataset_version':'dataset_v009','groups':{}}
 for k,d in groups.items():
  b=score(base,d,tok);c=score(cand,d,tok);out['groups'][k]={'baseline':b,'candidate':c,'ce_delta_pct':round(100*(c['cross_entropy']-b['cross_entropy'])/b['cross_entropy'],3),'token_accuracy_delta_points':round(100*(c['token_accuracy']-b['token_accuracy']),3)}
 g=out['groups'];checks={'v18_ce_improves':g['v18_new']['candidate']['cross_entropy']<g['v18_new']['baseline']['cross_entropy'],'fa_ce_improves':g['v18_fa']['candidate']['cross_entropy']<=g['v18_fa']['baseline']['cross_entropy'],'legacy_ce_regression_under_0_5pct':g['legacy_v008']['ce_delta_pct']<=.5,'legacy_token_delta_above_minus_0_5pt':g['legacy_v008']['token_accuracy_delta_points']>=-.5};out['checks']=checks;out['passed']=all(checks.values());(ROOT/'models/release_gate_v18.json').write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n',encoding='utf-8');print(json.dumps(out,ensure_ascii=False,indent=2));raise SystemExit(0 if out['passed'] else 2)
if __name__=='__main__':main()
