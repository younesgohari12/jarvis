from pathlib import Path
from collections import Counter
import sys,json,random,hashlib,ast,re
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from jarvis.agent.code_v21 import generate_pipeline,FILTERS,SafePython,explain,refactor
R=random.Random(21910);rows=[];seen=set();programs=set()
def structure(source):
 t=ast.parse(source)
 for n in ast.walk(t):
  if isinstance(n,ast.Constant) and isinstance(n.value,(int,float)):n.value=1
 return ast.dump(t,include_attributes=False)
while len(programs)<2000:
 filters=R.sample(list(FILTERS),R.randint(0,2));maps=[(R.choice(['add','subtract','multiply','divide','power']),R.randint(2,5)) for _ in range(R.randint(1,4))];reducer=R.choice(['list','sum','sorted','len'])
 source=generate_pipeline(filters,maps,reducer);group=structure(source)
 if group in programs:continue
 try:checks=[{'input':[a],'expected':SafePython().run(source,[a.copy()])} for a in [[],[0],[-3,0,2,5],[8,4,2],[-4,-2]]]
 except (ValueError,ArithmeticError,TypeError):continue
 programs.add(group);n=len(programs);splitnumber=int(hashlib.sha256(group.encode()).hexdigest()[:8],16)%10;split='test' if splitnumber==0 else 'validation' if splitnumber==1 else 'train'
 isfa=n%2==0
 conditions='; '.join('keep '+f+' values' for f in filters)
 ops='; '.join({'add':'add','subtract':'subtract','multiply':'multiply by','divide':'divide by','power':'power'}[op]+' '+str(v) for op,v in maps)
 instruction=("یک تابع پایتون بنویس: " if isfa else 'Write a Python function: ')+conditions+'; '+ops+'; '+reducer+' output.'
 # Refactor sources use explicit append loops for list outputs; other tasks retain the function.
 loop_source=source
 if reducer=='list':
  comp=ast.parse(source).body[0].body[0].value
  lines=['def transform(values):','    result = []','    for x in values:'];indent='        '
  for c in comp.generators[0].ifs:lines.append(indent+'if '+ast.unparse(c)+':');indent+='    '
  lines.append(indent+'result.append('+ast.unparse(comp.elt)+')');lines.append('    return result');loop_source='\n'.join(lines)+'\n'
 for task in ['generate','debug','explain','refactor','test']:
  inp=source;target=source
  if task=='generate':text=instruction
  else:
   if task=='debug':inp=source.replace('def transform(values):','def transform(values)');target=source
   if task=='refactor':inp=loop_source;target=refactor(inp)
   if task=='explain':target=explain(source,'fa' if isfa else 'en')
   if task=='test':target='\n'.join(f"assert transform({c['input'][0]!r}) == {c['expected']!r}" for c in checks)
   verb={'debug':('خطای نحوی این کد را اصلاح کن','Debug the syntax error in this function'),'explain':('این تابع را توضیح بده','Explain this function'),'refactor':('این کد را بدون تغییر رفتار بازنویسی کن','Refactor this code preserving behavior'),'test':('برای این تابع تست بنویس','Write tests for this function')}[task][0 if isfa else 1]
   text=verb+'\n```python\n'+inp+'```'
  if text in seen:raise RuntimeError('duplicate code prompt')
  seen.add(text)
  rows.append({'id':f'code21_{len(rows):05}','section':'code','text':text,'target':target,'label':task,'split':split,'template_id':hashlib.sha256(group.encode()).hexdigest()[:20],'concept_group':hashlib.sha256(group.encode()).hexdigest(),'program':{'source':source,'filters':filters,'maps':maps,'reducer':reducer,'checks':checks},'provenance':'compositional_AST_generator','human_authored':False})
 if n%500==0:print('programs',n,flush=True)
out=ROOT/'datasets/v21/code.jsonl';out.write_text('\n'.join(json.dumps(r,ensure_ascii=False) for r in rows)+'\n')
(ROOT/'datasets/v21/code_manifest.json').write_text(json.dumps({'rows':len(rows),'tasks':dict(Counter(r['label'] for r in rows)),'unique_inputs':len(seen),'structural_program_groups':len(programs),'split_counts':dict(Counter(r['split'] for r in rows)),'sha256':hashlib.sha256(out.read_bytes()).hexdigest(),'limits':'Bounded numeric list transformations and syntax repairs; not a general arbitrary-function corpus.'},indent=2))
print('complete',len(rows),flush=True)
