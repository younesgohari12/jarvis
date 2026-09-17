from pathlib import Path
import numpy as np,json,hashlib
from jarvis.agent.semantic_models_v20 import features,SIZE
ROOT=Path(__file__).resolve().parents[2]
def model_features(text):return features(text[:800]+' '+text[-200:] if len(text)>1000 else text)
class ClassifierV21:
    def __init__(self,name):
        self.calls=0;self.ready=False;self.sha256='';self.path=ROOT/'models'/f'{name}_v21.npz'
        if not self.path.exists():return
        with np.load(self.path,allow_pickle=False) as z:
            self.weights=z['weights'];self.bias=z['bias'];self.labels=z['labels'].tolist();self.metadata=json.loads(str(z['metadata']))
        if self.weights.shape!=(len(self.labels),SIZE) or not np.isfinite(self.weights).all():raise ValueError('invalid_v21_model')
        self.sha256=hashlib.sha256(self.path.read_bytes()).hexdigest();self.ready=True
    def predict(self,text):
        if not self.ready:return []
        self.calls+=1;i,v=model_features(text);s=self.weights[:,i]@v+self.bias;p=np.exp(s-s.max());p/=p.sum()
        return sorted(zip(self.labels,map(float,p)),key=lambda x:-x[1])
