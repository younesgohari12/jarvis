"""Train only on train, select regularization on validation, report untouched test."""
from pathlib import Path
import sys,json,hashlib,time
from collections import Counter
import numpy as np
from scipy.sparse import csr_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score,f1_score,confusion_matrix,log_loss
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from jarvis.agent.semantic_models_v20 import features,SIZE
from jarvis.agent.normalization_v20 import skeleton

def matrix(samples):
    ids=[]; values=[]; pointers=[0]
    for text,span,family,label,split in samples:
        i,v=features(text,span,family); ids.extend(i); values.extend(v); pointers.append(len(ids))
    return csr_matrix((values,ids,pointers),shape=(len(samples),SIZE),dtype=np.float64)

def train(name,samples):
    classes=sorted({s[3] for s in samples if s[4]=='train'})
    samples=[s for s in samples if s[3] in classes]
    X=matrix(samples); y=np.array([s[3] for s in samples]); splits=np.array([s[4] for s in samples])
    tr,va,te=[np.flatnonzero(splits==s) for s in ('train','validation','test')]
    best=None; choices=[]
    for c in (4.,20.,80.):
        model=LogisticRegression(C=c,max_iter=450,solver='lbfgs',random_state=20,class_weight='balanced' if name=='code_intent' else None)
        model.fit(X[tr],y[tr]); score=accuracy_score(y[va],model.predict(X[va]))
        choices.append({'C':c,'validation_accuracy':float(score)})
        if best is None or score>best[0]: best=(score,model)
    model=best[1]; metrics={}
    for split,ix in [('train',tr),('validation',va),('test',te)]:
        pred=model.predict(X[ix]); prob=model.predict_proba(X[ix])
        metrics[split]={'rows':len(ix),'accuracy':float(accuracy_score(y[ix],pred)),'macro_f1':float(f1_score(y[ix],pred,average='macro',zero_division=0)),
            'loss':float(log_loss(y[ix],prob,labels=model.classes_)),'confusion_matrix':confusion_matrix(y[ix],pred,labels=model.classes_).tolist(),
            'distribution':dict(Counter(y[ix]))}
    weights=model.coef_.astype(np.float32); bias=model.intercept_.astype(np.float32)
    if len(model.classes_)==2: weights=np.concatenate([-weights/2,weights/2]); bias=np.concatenate([-bias/2,bias/2])
    metadata={'dataset_version':'dataset_v020_cognitive','architecture':'hashed_context_multinomial_logistic_regression','parameter_count':int(weights.size+bias.size),
        'train_rows':len(tr),'validation_rows':len(va),'test_rows':len(te),'holdout_strategy':'normalized_numeric_skeleton_disjoint','selection':choices,'selected_C':model.C}
    path=ROOT/'models'/f'{name}_v20.npz'
    np.savez_compressed(path,format='jarvis-learned-v20',weights=weights,bias=bias,labels=model.classes_,metadata=json.dumps(metadata))
    report={**metadata,'metrics':metrics,'labels':model.classes_.tolist(),'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'weights':path.name}
    print(name,json.dumps({s:metrics[s]['accuracy'] for s in metrics}),flush=True)
    return report

def main():
    rows=[json.loads(l) for l in (ROOT/'datasets/v20/cognitive.jsonl').read_text().splitlines()]
    jobs={
      'semantic_frame':[(r['text'],None,'',r['label'],r['split']) for r in rows],
      'numeric_role':[(r['normalized'],(n['start'],n['end']),r['label'],n['role'],r['split']) for r in rows if r['provenance']!='v19_frame_replay_one_per_template' for n in r['numbers']],
      'operation':[(r['text'],None,'',r['operation'],r['split']) for r in rows if r['operation']],
    }
    for name,filename in [('code_intent','code.jsonl'),('repair_stage','repair.jsonl')]:
        path=ROOT/'datasets/v20'/filename
        if path.is_file():
            extra=[json.loads(l) for l in path.read_text().splitlines()]
            jobs[name]=[(r['text'],None,'',r['label'],r['split']) for r in extra]
    report={'models':{}}
    for name,samples in jobs.items(): report['models'][name]=train(name,samples)
    out=ROOT/'reports/v20'; out.mkdir(parents=True,exist_ok=True)
    (out/'training.json').write_text(json.dumps(report,indent=2,ensure_ascii=False))
    # A measured duplicate audit includes conceptual similarity; shared concepts are stated.
    sets={s:{skeleton(r['text']) for r in rows if r['split']==s} for s in ('train','validation','test')}
    audit={'rows':len(rows),'new_authored_rows':sum(r['provenance']=='authored_v20' for r in rows),'unique_inputs':len({r['text'] for r in rows}),
      'unique_outputs':len({json.dumps([r['label'],r['numbers'],r['operation']],sort_keys=True) for r in rows}),'unique_templates':len({r['template_id'] for r in rows}),
      'distribution':dict(Counter(r['label'] for r in rows)),'split_counts':dict(Counter(r['split'] for r in rows)),
      'normalized_template_overlap':{a+'_'+b:len(sets[a]&sets[b]) for a,b in [('train','validation'),('train','test'),('validation','test')]},
      'semantic_duplicate_policy':'Concepts recur across splits to measure paraphrase transfer. Semantic-disjointness is NOT claimed.',
      'augmentation_policy':'At most one row per normalized number-masked input; legacy replay capped at one row per legacy template.'}
    from sklearn.feature_extraction.text import TfidfVectorizer
    text=[skeleton(r['text']) for r in rows]; mat=TfidfVectorizer(analyzer='char',ngram_range=(3,5)).fit_transform(text)
    tr=[i for i,r in enumerate(rows) if r['split']=='train']; te=[i for i,r in enumerate(rows) if r['split']=='test']
    sim=mat[te]@mat[tr].T; maximum=sim.max(axis=1).toarray().ravel()
    audit['near_duplicate_audit']={'metric':'character_3_5gram_tfidf_cosine','threshold':.9,'test_near_train':int((maximum>=.9).sum()),'max_similarity':float(maximum.max())}
    (out/'dataset_audit.json').write_text(json.dumps(audit,indent=2,ensure_ascii=False))
    (ROOT/'datasets/v20/registry.json').write_text(json.dumps({'id':'dataset_v020_cognitive','path':'cognitive.jsonl','audit':'../../reports/v20/dataset_audit.json',**audit},indent=2,ensure_ascii=False))
    registry=json.loads((ROOT/'models/model_registry_v2.json').read_text())
    registry['cognitive_v20']={'runtime':'jarvis.agent.local_intelligence_v20.LocalIntelligenceV20','models':report['models'],'core_neural_changed':False}
    (ROOT/'models/model_registry_v2.json').write_text(json.dumps(registry,indent=2,ensure_ascii=False))

if __name__=='__main__': main()
