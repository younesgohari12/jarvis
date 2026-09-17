"""Rank-limited residual output projection on the frozen JARVIS Transformer."""
import json,hashlib
from pathlib import Path
import numpy as np
class LanguageAdapterV21:
    def __init__(self,path,base_sha):
        self.calls=0;self.path=Path(path)
        with np.load(path,allow_pickle=False) as z:self.A=z['A'];self.B=z['B'];self.metadata=json.loads(str(z['metadata']))
        if self.metadata['base_sha256']!=base_sha:raise ValueError('adapter_base_mismatch')
        if not np.isfinite(self.A).all() or not np.isfinite(self.B).all():raise ValueError('invalid_adapter')
        self.sha256=hashlib.sha256(self.path.read_bytes()).hexdigest()
    def residual(self,hidden):
        self.calls+=1
        return (hidden@self.A)@self.B

def attach_adapter(model,path,base_path):
    path=Path(path)
    if not path.is_file():return model
    adapter=LanguageAdapterV21(path,hashlib.sha256(Path(base_path).read_bytes()).hexdigest())
    if adapter.A.shape[0]!=model.config.d_model or adapter.B.shape[1]!=model.config.vocab_size:raise ValueError('adapter_shape')
    original=model._project_head
    def projected(hidden):return original(hidden)+adapter.residual(hidden)
    model._project_head=projected;model.language_adapter_v21=adapter
    return model
