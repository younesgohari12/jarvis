from __future__ import annotations
import hashlib, json
from collections import Counter
from pathlib import Path
from typing import Any
ROOT=Path(__file__).resolve().parents[1]
VERSION='dataset_v005'; SEED=13092026

def stable(s:str,n:int=12)->str: return hashlib.sha256(s.encode('utf-8')).hexdigest()[:n]
def split_for_group(group:str)->str:
    b=int(stable(group,8),16)%100
    return 'train' if b<80 else ('validation' if b<90 else 'test')
def canonical(r:dict[str,Any])->str:
    return json.dumps({'input':str(r.get('input','')).strip().casefold(),'output':str(r.get('output','')).strip().casefold(),'context':r.get('context',[]) if isinstance(r.get('context',[]),list) else []},ensure_ascii=False,sort_keys=True)
def read_jsonl(p:Path): return [json.loads(x) for x in p.read_text(encoding='utf-8').splitlines() if x.strip()]
def write_jsonl(p:Path, rows):
    p.parent.mkdir(parents=True,exist_ok=True); p.write_text(''.join(json.dumps(r,ensure_ascii=False,sort_keys=True)+'\n' for r in rows),encoding='utf-8')

def load_pack(kind:str):
    mp=ROOT/'datasets'/f'{kind}_100_v13_manifest.json'; m=json.loads(mp.read_text(encoding='utf-8'))
    if m.get('source_dataset_count')!=100: raise RuntimeError(f'{kind} pack must have 100 datasets')
    rows=[]
    for src in m['sources']:
        p=ROOT/src['file']; obs=hashlib.sha256(p.read_bytes()).hexdigest()
        if obs!=src['sha256']: raise RuntimeError(f'checksum mismatch {p}')
        for r in read_jsonl(p):
            md=dict(r.get('metadata',{})); concept=str(md.get('concept_group','')).strip()
            if not concept: raise RuntimeError(f'missing concept group {r.get("id")}')
            r['dataset_version']=VERSION; r['split']=split_for_group(concept); r['curated_source_id']=src['id']; md['source_dataset']=src['id']; md['source_file']=src['file']; r['metadata']=md; rows.append(r)
    return rows,m

def build():
    prior=read_jsonl(ROOT/'datasets'/'cleaned'/'dataset_v004.jsonl')
    for r in prior:
        r['inherited_from']=r.get('inherited_from') or 'dataset_v004'; r['dataset_version']=VERSION
        md=dict(r.get('metadata',{})); concept=str(md.get('concept_group','')).strip() or f'legacy:{stable(canonical(r),16)}'; md['concept_group']=concept; r['metadata']=md
        r['split']=str(r.get('split') or split_for_group(concept))
    g,gm=load_pack('generalization'); rr,rm=load_pack('reasoning')
    seen=set(); combined=[]; dup=0
    for r in [*prior,*g,*rr]:
        fp=hashlib.sha256(canonical(r).encode('utf-8')).hexdigest()
        if fp in seen: dup+=1; continue
        seen.add(fp); combined.append(r)
    combined.sort(key=lambda r:(str(r['split']),str(r['id'])))
    splits={s:[r for r in combined if r['split']==s] for s in ('train','validation','test')}
    groups={s:{str(r.get('metadata',{}).get('concept_group','')) for r in v} for s,v in splits.items()}
    leakage=sum(len(groups[a]&groups[b]) for a,b in (('train','validation'),('train','test'),('validation','test')))
    if leakage: raise RuntimeError(f'concept leakage {leakage}')
    write_jsonl(ROOT/'datasets'/'raw'/'seed_v005.jsonl',combined)
    write_jsonl(ROOT/'datasets'/'cleaned'/'dataset_v005.jsonl',combined)
    write_jsonl(ROOT/'datasets'/'normalized'/'dataset_v005.jsonl',combined)
    for s,v in splits.items(): write_jsonl(ROOT/'datasets'/'splits'/f'dataset_v005_{s}.jsonl',v)
    manifest={'format':'jarvis-dataset-manifest-v5','dataset_version':VERSION,'seed':SEED,'total_examples':len(combined),
      'inherited_v004_examples':len(prior),'generalization_source_dataset_count':gm['source_dataset_count'],'reasoning_source_dataset_count':rm['source_dataset_count'],
      'generalization_examples_added':sum(1 for r in combined if r.get('origin')=='jarvis-generalization-v13'),
      'reasoning_examples_added':sum(1 for r in combined if r.get('origin')=='jarvis-reasoning-v13'),
      'duplicates_removed_during_merge':dup,'split_counts':{k:len(v) for k,v in splits.items()},'concept_group_counts':{k:len(v) for k,v in groups.items()},
      'cross_split_concept_leakage':leakage,'languages':dict(Counter(str(r.get('language','unknown')) for r in combined)),
      'stages':{str(k):v for k,v in sorted(Counter(int(r.get('stage',0)) for r in combined).items())},
      'origins':dict(Counter(str(r.get('origin','unknown')) for r in combined)),
      'quality_notes':['v004 holdouts preserved','v13 concepts split as atomic groups','exact input/output duplicates removed','200 new curated source datasets']}
    (ROOT/'datasets'/'manifest_v005.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    return manifest
if __name__=='__main__': print(json.dumps(build(),ensure_ascii=False,indent=2))
