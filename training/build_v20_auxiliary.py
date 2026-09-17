from pathlib import Path
import sys,re,json,hashlib,itertools
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT))
from jarvis.agent.normalization_v20 import skeleton
ROWS=[];SEEN=set()
def add(text,label,group,**extra):
 sk=skeleton(text)
 if sk in SEEN:return
 SEEN.add(sk); tid=hashlib.sha256(sk.encode()).hexdigest(); b=int(tid[:8],16)%10
 ROWS.append(dict(text=text,label=label,concept_group=group,template_id='aux_'+tid[:20],split='train' if b<7 else 'validation' if b<9 else 'test',**extra))
forms=['تابعی بده که {}','واسه من کد {} رو آماده کن','میخوام تابع {} داشته باشم','برام {} رو پیاده سازی کن','میشه کد {} رو بدی؟','یک تابع پایتون برای {} ایجاد کن','{} رو با کد حل کن','تابعی بساز واسه {}','give me code for {}','make a function for {}','implement {} in Python','I need a function for {}','write Python code for {}','create a routine for {}','build an algorithm for {}','can you provide a function for {}']
concepts={
'sum_values':('محاسبه مجموع لیست','summing list values'),'sort_values':('مرتب سازی صعودی لیست','sorting a list in ascending order'),
'filter_even':('جدا کردن عددهای زوج','filtering even integers'),'filter_positive':('انتخاب عددهای مثبت','filtering positive numbers'),
'max_value':('پیدا کردن بیشترین مقدار لیست','finding the maximum list value'),'deduplicate':('حذف تکراری ها با حفظ ترتیب','removing duplicates preserving order'),
'reverse_string':('معکوس کردن رشته','reversing a string'),'palindrome':('تشخیص رشته پالیندروم','checking whether text is a palindrome'),
'factorial':('محاسبه فاکتوریل عدد','computing the factorial of a number'),'average':('محاسبه میانگین لیست','calculating the average of list values')}
for label,(fa,en) in concepts.items():
 for i,f in enumerate(forms):add(f.format(fa if i<8 else en),label,'code_generation')
for line in (ROOT/'datasets/code_v19/code_intelligence_5000_real.jsonl').read_text().splitlines():
 r=json.loads(line)
 if r['split']=='train' and r['intent'] in concepts:add(r['input'],r['intent'],'code_replay')
for text in ['explain why sorting takes time','debug my factorial code','refactor this palindrome function','what is a maximum?','توضیح بده مرتب سازی چطور کار میکند','باگ کد فاکتوریل را پیدا کن','این تابع جمع را بازنویسی کن','میانگین یعنی چی؟','explain the algorithm for deduplication','trace this loop','شرح کد معکوس رشته','کد زیر چه خروجی دارد؟','latest Python documentation','مستندات جدید پایتون']:
 add(text,'other','code_task_negative')
(ROOT/'datasets/v20/code.jsonl').write_text('\n'.join(json.dumps(r,ensure_ascii=False) for r in ROWS)+'\n')
# Verification candidates exercise distinct invariants, preserving typed source slots.
ROWS=[];SEEN=set()
from jarvis.agent.cognitive_ir_v20 import SemanticIR
from jarvis.agent.verification_v20 import Verifier
from jarvis.agent.execution_v20 import GraphExecutor
from jarvis.agent.cognitive_ir_v20 import WorldState
cases=[
 SemanticIR('ratio',slots={'total':731,'ratio_a':4,'ratio_b':3}),
 SemanticIR('binomial',slots={'n':7,'k':2,'p':.37}),
 SemanticIR('inventory',slots={'initial':91},operations=[{'op':'inventory_remove','value':21},{'op':'inventory_add','value':13}]),
 SemanticIR('finance',slots={'initial':430},operations=[{'op':'debit','value':55},{'op':'credit','value':19}]),
 SemanticIR('work_rate',slots={'workers_initial':3,'hours_initial':7,'output_initial':168,'workers_target':5,'hours_target':4},units={'time':'hour','output':'piece'}),
 SemanticIR('age',slots={'age':13,'difference':4,'years':6,'query':'sum'}),
 SemanticIR('sequence',slots={'terms':[4,12,36],'n':6}),
 SemanticIR('speed',slots={'distance':143,'time':2.5}),
 SemanticIR('combination',slots={'n':11,'k':3}),
 SemanticIR('dice',slots={'n':3,'target':11}),
 SemanticIR('independent',slots={'probabilities':[.3,.7]}),
 SemanticIR('comparison',slots={'values':[2,-3,8],'mode':'max'}),
 SemanticIR('scheduling',slots={'start':7,'durations':[3,4]}),
 SemanticIR('graph',slots={'initial':107},operations=[{'op':'percentage_remove','value':13},{'op':'add','value':8},{'op':'divide','value':5}])]
V=Verifier();E=GraphExecutor(); verification=[]
for ir in cases:
 answer,_=E.execute(WorldState.from_ir(ir).graph())
 candidates=[('correct',answer),('wrong_total',[1.7,1.2] if isinstance(answer,list) else answer+11),('wrong_operation',list(reversed(answer)) if isinstance(answer,list) else answer*2),('wrong_sign',[-v for v in answer] if isinstance(answer,list) else -answer),('wrong_output_type','not a number')]
 for fault,candidate in candidates:
  verdict=V.verify(ir,candidate)
  verification.append({'ir':ir.to_dict(),'candidate':candidate,'fault':fault,'target':verdict.to_dict(),'template_id':ir.task+'_'+fault,'concept_group':ir.task,'split':{'correct':'train','wrong_total':'train','wrong_operation':'validation','wrong_sign':'test','wrong_output_type':'train'}[fault]})
# Stage training: diverse failure contexts, not a learned replacement for arithmetic truth.
contexts=['candidate check: {}','failed invariant {}','verification detected {}','failure reason {}','repair required after {}','answer rejected: {}','independent verifier reports {}','cannot accept: {}','check result false for {}','invalid candidate because {}']
for invariant,stage in {
 'ratio_sum':'execution','ratio_proportion':'execution','probability_recompute':'execution','inventory_conservation':'execution','balance_conservation':'execution','productivity_conservation':'execution',
 'timeline_consistency':'execution','sequence_nth':'execution','independent_execution':'execution','distance_conservation':'execution','counting_recompute':'execution','dice_recompute':'execution',
 'probability_domain':'slots','invalid_slots_or_candidate':'slots','dimensional_consistency':'slots','missing_or_ambiguous':'slots','division_by_zero':'slots','negative_age':'slots','work_domain':'slots',
 'python_syntax':'response','sentence_count':'response','required':'response','max_length':'response','forbidden':'response','language':'response'}.items():
 for f in contexts:add(f.format(invariant),stage,'repair_'+invariant)
(ROOT/'datasets/v20/repair.jsonl').write_text('\n'.join(json.dumps(r,ensure_ascii=False) for r in ROWS)+'\n')
(ROOT/'datasets/v20/verification.jsonl').write_text('\n'.join(json.dumps(r,ensure_ascii=False) for r in verification)+'\n')
print('code and repair datasets saved; verification candidates',len(verification))
