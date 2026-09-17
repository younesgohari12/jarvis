from __future__ import annotations
import argparse,json,math,sys
from collections import defaultdict
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))
from jarvis.neural.tokenizer import JarvisTokenizer
from jarvis.neural.transformer import JarvisTransformer

def rows(p): return [json.loads(x) for x in p.read_text(encoding='utf-8').splitlines() if x.strip()]
def case(tok,r,max_len):
    ctx='\n'.join(str(x.get('content','')) for x in r.get('context',[]) if isinstance(x,dict)); user=f'{ctx}\n{r["input"]}'.strip() if ctx else str(r['input'])
    prompt=[tok.bos_id,tok.special_to_id['<user>'],*tok.encode(user),tok.special_to_id['<assistant>']]; resp=[*tok.encode(str(r['output'])),tok.eos_id]; seq=(prompt+resp)[:max_len+1]
    x=seq[:-1]; y=seq[1:]; boundary=max(0,len(prompt)-2); labels=[v if i>=boundary else -100 for i,v in enumerate(y)]; return x,labels

def score(model,tok,data,max_len):
    losses=[]; cor=tot=0; bylang=defaultdict(lambda:[0,0,[]])
    for r in data:
        x,y=case(tok,r,max_len); logits=model.forward(np.asarray(x,dtype=np.int64))[0]; y=np.asarray(y,dtype=np.int64); valid=y>=0
        if not np.any(valid): continue
        sl=logits[valid]; st=y[valid]; mx=np.max(sl,axis=-1,keepdims=True); ex=np.exp(sl-mx); pr=ex/np.sum(ex,axis=-1,keepdims=True); p=pr[np.arange(st.size),st]; loss=-float(np.mean(np.log(np.maximum(p,1e-12)))); losses.append(loss); pred=np.argmax(sl,axis=-1); cc=int(np.sum(pred==st)); nn=int(st.size); cor+=cc;tot+=nn; z=bylang[str(r.get('language','unknown'))]; z[0]+=cc;z[1]+=nn;z[2].append(loss)
    mean=float(np.mean(losses)) if losses else float('inf')
    return {'cases':len(losses),'cross_entropy':round(mean,6),'perplexity':round(math.exp(min(20,mean)),6),'token_accuracy':round(cor/max(1,tot),6),
            'by_language_token_accuracy':{k:round(v[0]/max(1,v[1]),6) for k,v in sorted(bylang.items())},'by_language_cross_entropy':{k:round(float(np.mean(v[2])),6) for k,v in sorted(bylang.items())}}

def main(a):
    tok=JarvisTokenizer.load(ROOT/'models/jarvis_tokenizer_v003.json'); base=JarvisTransformer.load(ROOT/a.baseline); cand=JarvisTransformer.load(ROOT/a.candidate)
    test=rows(ROOT/'datasets/splits/dataset_v005_test.jsonl'); old=rows(ROOT/'datasets/splits/dataset_v004_test.jsonl')
    groups={
      'generalization':sorted([r for r in test if r.get('origin')=='jarvis-generalization-v13'],key=lambda r:r['id'])[:a.max_cases],
      'reasoning':sorted([r for r in test if r.get('origin')=='jarvis-reasoning-v13'],key=lambda r:r['id'])[:a.max_cases],
      'v004_regression':sorted(old,key=lambda r:r['id'])[:a.old_cases],
    }
    res={'format':'jarvis-v13-quality-gate-v1','baseline':a.baseline,'candidate':a.candidate,'dataset':'dataset_v005','groups':{}}
    for k,data in groups.items():
        b=score(base,tok,data,a.sequence_length); c=score(cand,tok,data,a.sequence_length)
        res['groups'][k]={'baseline':b,'candidate':c,'cross_entropy_improvement_pct':round(100*(b['cross_entropy']-c['cross_entropy'])/max(b['cross_entropy'],1e-9),3),'token_accuracy_delta_points':round(100*(c['token_accuracy']-b['token_accuracy']),3)}
    g=res['groups']; checks={
      'generalization_ce_improves':g['generalization']['candidate']['cross_entropy']<g['generalization']['baseline']['cross_entropy'],
      'reasoning_ce_improves':g['reasoning']['candidate']['cross_entropy']<g['reasoning']['baseline']['cross_entropy'],
      'generalization_token_accuracy_nonworse':g['generalization']['token_accuracy_delta_points']>=-0.25,
      'reasoning_token_accuracy_nonworse':g['reasoning']['token_accuracy_delta_points']>=-0.25,
      'v004_ce_regression_under_0_5pct':g['v004_regression']['candidate']['cross_entropy']<=g['v004_regression']['baseline']['cross_entropy']*1.005,
      'v004_token_accuracy_delta_above_minus_0_5pt':g['v004_regression']['token_accuracy_delta_points']>=-0.5,
    }; res['checks']=checks; res['passed']=all(checks.values())
    (ROOT/'models/evaluation_v13_quality_gate.json').write_text(json.dumps(res,ensure_ascii=False,indent=2)+'\n',encoding='utf-8'); print(json.dumps(res,ensure_ascii=False,indent=2))
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--baseline',default='models/jarvis_nano_v12.npz');p.add_argument('--candidate',default='models/jarvis_nano_v13_candidate.npz');p.add_argument('--max-cases',type=int,default=140);p.add_argument('--old-cases',type=int,default=140);p.add_argument('--sequence-length',type=int,default=64);main(p.parse_args())
