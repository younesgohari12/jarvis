"""Program-generated, auditable training data. No claim of human-authored 20k.

Compositional event order changes the semantics. At most two numeric variants
per lexical skeleton, and grouped train/validation/test splitting prevents leakage.
"""
from pathlib import Path
from collections import Counter
import random,json,hashlib,re,sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from jarvis.agent.normalization_v20 import skeleton
from jarvis.agent.source_facts_v21 import extract_source,source_answer
R=random.Random(210908);OUT=ROOT/'datasets/v21';OUT.mkdir(parents=True,exist_ok=True)
ALL=[];USED=set();counts=Counter()
WORDS={
'add':(['{v} اضافه کن','{v} تا به آن اضافه کن','{v} واحد افزایش بده'],['add {v}','increase by {v}','then add {v}']),
'subtract':(['{v} کم کن','{v} واحد کاهش بده','{v} تا از آن کم کن'],['subtract {v}','decrease by {v}','remove {v}']),
'multiply':(['در {v} ضرب کن','حاصل را در {v} ضرب کن','آن را {v} برابر کن'],['multiply by {v}','multiply the result by {v}','multiply this by {v}']),
'divide':(['بر {v} تقسیم کن','آن را بر {v} تقسیم کن','حاصل را بر {v} تقسیم کن'],['divide by {v}','divide the result by {v}','divide this by {v}']),
'percentage_add':(['{v} درصد زیاد کن','{v} درصد افزایش بده','{v}% اضافه کن'],['increase by {v} percent','add {v}%','increase by {v}%']),
'percentage_remove':(['{v} درصد کم کن','{v} درصد کاهش بده','{v}% کم کن'],['decrease by {v} percent','remove {v}%','decrease by {v}%']),
}
def split(key):
 n=int(hashlib.sha256(key.encode()).hexdigest()[:8],16)%10
 return 'test' if n==0 else 'validation' if n==1 else 'train'
def emit(section,text,target,label,program,group=None):
 sk=skeleton(text);group=group or sk
 if text in USED or counts[(section,sk)]>=2:return False
 counts[(section,sk)]+=1;USED.add(text)
 row={'id':f'v21_{len(ALL):05}','section':section,'text':text,'target':target,'label':label,'program':program,'template_id':hashlib.sha256(sk.encode()).hexdigest()[:20],'concept_group':group,'split':split(group),'provenance':'compositional_program_generator_v21','human_authored':False}
 ALL.append(row);return True

def program_example(section):
 language=R.choice(['fa','en']);lang=0 if language=='fa' else 1
 domain=R.choice(['graph','finance','inventory']) if section=='world_model' else 'graph'
 initial=R.randint(400,1600);ops=[];clauses=[]
 for _ in range(R.randint(3,6)):
  op=R.choice(list(WORDS));value=R.randint(2,9) if op in ('multiply','divide') else R.randint(2,28)
  ops.append({'op':op,'value':value});clauses.append(R.choice(WORDS[op][lang]).format(v=value))
 prefix={'graph':f'عدد {initial} است' if lang==0 else f'Start with {initial}', 'finance':f'موجودی حساب {initial} تومان است' if lang==0 else f'Account balance starts at {initial} dollars','inventory':f'موجودی انبار {initial} کالا است' if lang==0 else f'Inventory starts with {initial} items'}[domain]
 join=R.choice(['؛ ','، ',' و سپس ']) if lang==0 else R.choice(['; ', ', ', ' and then '])
 text=prefix+join+join.join(clauses)
 if section=='adversarial':
  # Semantically irrelevant numeric identifiers: explicitly marked, not undocumented noise.
  text=(f'کد {R.randint(50,999)} مربوط به محاسبه نیست؛ ' if lang==0 else f'Record id {R.randint(50,999)} is irrelevant; ')+text
 if section=='persian':
  text=f'عدد {initial} است؛ '+'؛ '.join(R.choice(WORDS[o['op']][0]).format(v=o['value']) for o in ops)
  if R.random()<.5:text=text.replace('اضافه کن','اضافه کن برام').replace('کم کن','کم کن لطفا').replace('تقسیم کن','تقسیم کن برام')
 # Oracle uses Decimal operations independently from runtime/source extraction.
 from decimal import Decimal
 x=Decimal(initial)
 for op in ops:
  v=Decimal(op['value']);k=op['op']
  if k=='add':x+=v
  elif k=='subtract':x-=v
  elif k=='multiply':x*=v
  elif k=='divide':x/=v
  elif k=='percentage_add':x*=1+v/100
  elif k=='percentage_remove':x*=1-v/100
 if x<0 or abs(x)>1e8:return None
 graph={'domain':domain,'slots':{'initial':initial},'operations':ops,'answer':float(x),'language':language}
 try:
  extracted=extract_source(text);observed=source_answer(extracted)
  if not abs(observed-float(x))<=max(1,abs(float(x)))*1e-8:return None
 except (ValueError,ArithmeticError):return None
 return text,graph

# Compact single-domain examples are capped by template, not inflated to the target count.
for i in range(400):
 a,b,c,d=R.randint(3,13),R.randint(2,8),R.randint(2,8),R.randint(3,12)
 examples=[
 (f'{a} کارگر در {b} ساعت {a*b*c} قطعه می سازند. {c} کارگر در {d} ساعت چند قطعه می سازند؟','work_rate'),
 (f'مبلغ {a*b*10} را با نسبت {c} به {d} تقسیم کن','ratio'),
 (f'احتمال موفقیت {a*5} درصد است؛ در {b} تلاش دقیقاً {min(c,b)} موفقیت؟','binomial'),
 (f'علی {a*50} دلار داشت، {b*10} دلار خرید کرد و {c*5} درصد تخفیف گرفت','finance'),
 (f'مینا {a+10} ساله است؛ خواهرش {b} سال بزرگتر است؛ {c} سال بعد مجموع سنشان؟','age'),
 (f'در {b} ساعت {a*60} کیلومتر طی شد. سرعت چقدر است؟','speed'),
 (f'کار ساعت {b} شروع می شود؛ مدت کار {c} ساعت است؛ پایان کار؟','scheduling')]
 for text,label in examples:
  try:f=extract_source(text);answer=source_answer(f)
  except (ValueError,ArithmeticError):continue
  emit('semantic_ir',text,{'slots':f.slots,'operations':f.operations,'answer':answer},label,f.to_dict())
for section,needed in [('semantic_ir',5000),('world_model',5000),('adversarial',3000),('persian',3000)]:
 attempts=0
 while sum(x['section']==section for x in ALL)<needed:
  attempts+=1
  if attempts>needed*50:raise RuntimeError('quality cap prevented target count')
  sample=program_example(section)
  if sample is None:continue
  text,g=sample
  if section=='persian':target='نتیجهٔ محاسبه برابر '+format(g['answer'],'.10g')+' است.'
  else:target=g
  emit(section,text,target,g['domain'],g)
 print(section,sum(x['section']==section for x in ALL),flush=True)
# Counterfactuals are derived from immutable source programs. Split follows source.
parents=[r for r in ALL if r['section'] in ('semantic_ir','world_model') and r['program'].get('operations')]
for i in range(4000):
 parent=parents[i];g=parent['program'];bad=json.loads(json.dumps(g));kind=['slot_value','event_direction','event_order','answer_value'][i%4]
 if kind=='slot_value':bad['slots']['initial']+=7
 elif kind=='event_direction':bad['operations'][0]['op']='subtract' if bad['operations'][0]['op']!='subtract' else 'add'
 elif kind=='event_order':bad['operations'][0],bad['operations'][-1]=bad['operations'][-1],bad['operations'][0]
 else:bad['answer']+=11
 text=parent['text']+'\nCandidate: '+json.dumps(bad,ensure_ascii=False,sort_keys=True)
 emit('verification',text,{'failed_check':kind,'source_id':parent['id'],'correct':g},kind,bad,parent['concept_group'])
assert Counter(x['section'] for x in ALL)=={'semantic_ir':5000,'world_model':5000,'verification':4000,'adversarial':3000,'persian':3000}
for section in ['semantic_ir','world_model','verification','adversarial','persian']:
 rows=[r for r in ALL if r['section']==section]
 (OUT/(section+'.jsonl')).write_text('\n'.join(json.dumps(r,ensure_ascii=False) for r in rows)+'\n')
manifest={'rows':len(ALL),'sections':dict(Counter(x['section'] for x in ALL)),'unique_inputs':len(USED),'unique_templates':len({r['template_id'] for r in ALL}),'split_counts':dict(Counter(r['split'] for r in ALL)),'max_numeric_variants_per_section_template':2,'data_kind':'synthetic executable compositional programs, not 20k human-authored questions','coverage_limitations':['Most new examples are multi-event arithmetic; domain and language diversity remain narrower than target.','Persian set focuses on quantitative instructions, not unrestricted dialogue.','Graph sequences and domains recur conceptually across splits.'],'files':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in OUT.glob('*.jsonl')}}
(OUT/'manifest.json').write_text(json.dumps(manifest,indent=2,ensure_ascii=False));print(manifest,flush=True)
