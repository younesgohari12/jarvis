from pathlib import Path
import json,sys
from collections import Counter
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from jarvis.agent.normalization_v20 import normalize,skeleton
from sklearn.feature_extraction.text import TfidfVectorizer
reports={}
for name in ('code','repair'):
 rows=[json.loads(l) for l in (ROOT/f'datasets/v20/{name}.jsonl').read_text().splitlines()]
 report={'rows':len(rows),'unique_inputs':len({r['text'] for r in rows}),'unique_outputs':len({r['label'] for r in rows}),'unique_templates':len({r['template_id'] for r in rows}),'class_distribution':dict(Counter(r['label'] for r in rows)),'split_distribution':dict(Counter(r['split'] for r in rows))}
 sets={s:{skeleton(r['text']) for r in rows if r['split']==s} for s in ('train','validation','test')}
 report['template_overlap']={a+'_'+b:len(sets[a]&sets[b]) for a,b in [('train','validation'),('train','test'),('validation','test')]}
 mat=TfidfVectorizer(analyzer='char',ngram_range=(3,5)).fit_transform([skeleton(r['text']) for r in rows]);tr=[i for i,r in enumerate(rows) if r['split']=='train'];te=[i for i,r in enumerate(rows) if r['split']=='test']
 maximum=(mat[te]@mat[tr].T).max(axis=1).toarray().ravel()
 report['near_duplicate_audit']={'metric':'char_3_5gram_tfidf_cosine','threshold':.9,'test_near_train':int((maximum>=.9).sum()),'max_similarity':float(maximum.max())}
 report['semantic_duplicates']={'shared_label_concepts':len({r['label'] for r in rows if r['split']=='train'} & {r['label'] for r in rows if r['split']=='test'}),'claim':'Conceptual/semantic disjointness is not claimed; this is classification over repeated concepts.'}
 reports[name]=report
rows=[json.loads(l) for l in (ROOT/'datasets/v20/verification.jsonl').read_text().splitlines()]
reports['verification']={'rows':len(rows),'usage':'deterministic verifier candidate fixtures; NOT described as supervised training for a truth model','unique_templates':len({r['template_id'] for r in rows}),'split_distribution':dict(Counter(r['split'] for r in rows)),'domain_distribution':dict(Counter(r['concept_group'] for r in rows)),'fault_distribution':dict(Counter(r['fault'] for r in rows))}
(ROOT/'reports/v20/auxiliary_dataset_audit.json').write_text(json.dumps(reports,ensure_ascii=False,indent=2))
registry=json.loads((ROOT/'datasets/v20/registry.json').read_text());registry['auxiliary']=reports
(ROOT/'datasets/v20/registry.json').write_text(json.dumps(registry,ensure_ascii=False,indent=2))
print({k:v['rows'] for k,v in reports.items()})
