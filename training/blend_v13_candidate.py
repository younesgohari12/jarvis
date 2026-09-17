from pathlib import Path
from datetime import UTC, datetime
import argparse, sys
ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))
from jarvis.neural.transformer import JarvisTransformer

def main(a):
    base=JarvisTransformer.load(ROOT/a.base,dequantize=True); cand=JarvisTransformer.load(ROOT/a.candidate,dequantize=True)
    if base.parameters.keys()!=cand.parameters.keys(): raise RuntimeError('parameter mismatch')
    alpha=float(a.alpha)
    for k in base.parameters:
        base.parameters[k][:]=(1-alpha)*base.parameters[k]+alpha*cand.parameters[k]
    md={**base.metadata,'model_id':f'jarvis_nano_v13_blend_{alpha:g}','dataset_version':'dataset_v005','pretrained_source':None,'training_complete':True,'blend_base':a.base,'blend_candidate':a.candidate,'blend_alpha':alpha,'trained_at_utc':datetime.now(UTC).isoformat()}
    base.save_quantized(ROOT/a.output,md)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--base',default='models/jarvis_nano_v12.npz');p.add_argument('--candidate',default='models/jarvis_nano_v13_candidate.npz');p.add_argument('--alpha',type=float,default=.7);p.add_argument('--output',default='models/jarvis_nano_v13_blend.npz');main(p.parse_args())
