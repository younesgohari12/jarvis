"""Small trained classifiers. NumPy-only inference; no sklearn runtime dependency."""
from __future__ import annotations
import hashlib
import json
import re
from pathlib import Path
import numpy as np
from jarvis.agent.normalization_v20 import normalize, NUMBER

ROOT=Path(__file__).resolve().parents[2]
SIZE=16384

def features(text: str, span: tuple[int,int]|None=None, family: str=''):
    # Caller passes normalized text for token spans; normalizing again would shift them.
    t=text if span else normalize(text)
    t=NUMBER.sub('<n>',t) if span is None else t
    words=re.findall(r'<n>|[^\W\d_]+|[%:;,+*/=-]',t,re.UNICODE)
    if span:
        a,b=span
        left=t[max(0,a-85):a]; right=t[b:b+85]
        lt=re.findall(r'[^\W\d_]+|[%:;,+*/=-]',left,re.UNICODE)[-6:]
        rt=re.findall(r'[^\W\d_]+|[%:;,+*/=-]',right,re.UNICODE)[:6]
        f=[f'L{i}={w}' for i,w in enumerate(reversed(lt))]+[f'R{i}={w}' for i,w in enumerate(rt)]
        f += [f'Lpair={x}|{y}' for x,y in zip(lt,lt[1:])]+[f'Rpair={x}|{y}' for x,y in zip(rt,rt[1:])]
        context=NUMBER.sub('<n>',left[-32:])+'@'+NUMBER.sub('<n>',right[:32])
        f += [f'c={context[i:i+n]}' for n in (3,4) for i in range(len(context)-n+1)]
        f += ['family='+family]+['G='+w for w in set(words)]
        # Neighbouring number markers distinguish ratio positions without numeric indexes.
        f += ['edgeL='+NUMBER.sub('<n>',left[-15:]),'edgeR='+NUMBER.sub('<n>',right[:15])]
    else:
        f=['w='+w for w in words]
        f += ['b='+a+'|'+b for a,b in zip(words,words[1:])]
        f += ['c='+t[i:i+n] for n in (3,4,5) for i in range(len(t)-n+1)]
    counts={}
    for x in f:
        h=int.from_bytes(hashlib.blake2b(x.encode(),digest_size=8,person=b'jrv20ml').digest(),'little')
        j=h%SIZE; counts[j]=counts.get(j,0)+(1 if h>>63 else -1)
    ix=np.array(list(counts),dtype=np.int32); val=np.array(list(counts.values()),dtype=np.float32)
    val/=max(float(np.linalg.norm(val)),1e-12)
    return ix,val

class LearnedModel:
    def __init__(self,name:str,root:Path=ROOT):
        self.path=root/'models'/f'{name}_v20.npz'; self.calls=0; self.ready=False
        self.labels=(); self.metadata={}; self.sha256=''
        if not self.path.is_file(): return
        try:
            with np.load(self.path,allow_pickle=False) as z:
                if str(z['format'].item())!='jarvis-learned-v20': raise ValueError('Invalid v20 checkpoint')
                self.weights=z['weights'].astype(np.float32); self.bias=z['bias'].astype(np.float32)
                self.labels=tuple(z['labels'].tolist()); self.metadata=json.loads(str(z['metadata'].item()))
            if self.weights.shape!=(len(self.labels),SIZE) or self.bias.shape!=(len(self.labels),): raise ValueError('Model shape mismatch')
            if not np.isfinite(self.weights).all() or not np.isfinite(self.bias).all(): raise ValueError('Nonfinite model')
            self.sha256=hashlib.sha256(self.path.read_bytes()).hexdigest(); self.ready=True
        except (OSError,ValueError,KeyError,TypeError) as error:
            self.ready=False; self.metadata={'load_error':type(error).__name__+': '+str(error)}

    @property
    def parameter_count(self): return self.weights.size+self.bias.size if self.ready else 0

    def predict(self,text,span=None,family=''):
        if not self.ready: return []
        self.calls+=1
        ix,val=features(text,span,family)
        scores=self.weights[:,ix]@val+self.bias
        p=np.exp(scores-np.max(scores)); p/=p.sum()
        return sorted(zip(self.labels,map(float,p)),key=lambda x:-x[1])
