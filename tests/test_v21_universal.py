import math,copy
from unittest.mock import patch
import pytest
from jarvis.agent.local_intelligence_v21 import LocalIntelligenceV21
from jarvis.agent.source_facts_v21 import extract_source,source_answer,UnsupportedSource
from jarvis.agent.parser_v21 import facts_to_ir
from jarvis.agent.verification_v21 import UniversalVerifier
from jarvis.agent.world_v21 import WorldStateV21,GraphExecutorV21
from jarvis.agent.execution_v20 import ExecutionError
from jarvis.agent.code_v21 import SafePython,CodeLimit,CodeIntelligenceV21,refactor
from tests.helpers import TemporaryRuntime

CASES=[
 ('525 را با نسبت 4 به 3 تقسیم کن',[300,225],'total'),
 ('احتمال موفقیت 30 درصد است؛ در 4 تلاش دقیقاً 2 موفقیت؟',.2646,'n'),
 ('موجودی حساب 500 تومان است؛ 120 کم کن؛ 30 اضافه کن',410,'initial'),
 ('موجودی انبار 100 کالا است؛ 20 کم کن؛ 5 اضافه کن',85,'initial'),
 ('علی 500 دلار داشت، 120 دلار خرید کرد و 30 درصد تخفیف گرفت',416,'initial'),
 ('مینا 17 ساله است؛ خواهرش 5 سال بزرگتر است؛ 4 سال بعد مجموع سنشان؟',47,'difference'),
 ('در 2 ساعت 120 کیلومتر طی شد. سرعت چقدر است؟',60,'time'),
 ('کار ساعت 9 شروع می شود؛ مدت کار 2 ساعت است؛ پایان کار؟',11,'start'),
 ('دنباله 24، 12، 6؛ جمله پنجم چیست؟',1.5,'n'),
 ('با 3 کارگر و 4 ساعت، خروجی 84 قطعه است. با 8 کارگر و 6 ساعت چه خروجی داریم؟',336,'workers_initial'),
 ('Ali has 100 dollars. Sara has 50 dollars. Ali transfers 30 dollars to Sara. Balance of Sara?',80,'query'),
]
@pytest.mark.parametrize('q,expected,key',CASES)
def test_independent_source_answer(q,expected,key):
 f=extract_source(q);assert source_answer(f)==pytest.approx(expected)
 b=LocalIntelligenceV21();ir=b.parser.parse(q);value,verdict,attempts,world=b.reflection.solve(ir)
 assert value==pytest.approx(expected) and verdict.passed
 assert 'source_answer_consistency' in verdict.checks
 assert isinstance(world,WorldStateV21)

@pytest.mark.parametrize('q,expected,key',CASES)
def test_arithmetically_consistent_corrupt_slot_is_rejected(q,expected,key):
 ir=facts_to_ir(extract_source(q),q,'fa')
 if key=='query':ir.slots[key]='ali'
 else:ir.slots[key]+=1
 value=GraphExecutorV21().execute(WorldStateV21.from_ir(ir).graph())[0]
 verdict=UniversalVerifier().verify(ir,value)
 assert not verdict.passed and ('source_slot:'+key) in verdict.failed_checks

@pytest.mark.parametrize('q,expected,key',CASES)
def test_corrupted_answer_never_passes(q,expected,key):
 ir=facts_to_ir(extract_source(q),q,'fa');wrong=[x+1 for x in expected] if isinstance(expected,list) else expected+1
 assert not UniversalVerifier().verify(ir,wrong).passed

def test_event_order_and_discount_scope():
 q='قیمت 100 است؛ 20 درصد زیاد کن؛ 10 درصد کم کن'
 ir=facts_to_ir(extract_source(q),q,'fa');ir.operations[0]['op']='add';ir.operations[1]['op']='subtract'
 assert not UniversalVerifier().verify(ir,110).passed
 q=CASES[4][0];ir=facts_to_ir(extract_source(q),q,'fa')
 assert UniversalVerifier().verify(ir,416).passed
 assert not UniversalVerifier().verify(ir,(500-120)*.7).passed

def test_verifier_ignores_cached_evidence_and_previous_answers():
 ir=facts_to_ir(extract_source(CASES[0][0]),CASES[0][0],'fa');ir.slots['total']=100
 ir.model_evidence={'verified':True,'source_graph_v21':{'slots':ir.slots,'answer':[400/7,300/7]}}
 assert not UniversalVerifier().verify(ir,[400/7,300/7]).passed

def test_graph_dependencies_are_consumed():
 ir=facts_to_ir(extract_source('عدد 100 است؛ 20 اضافه کن؛ 10 کم کن'),'عدد 100 است؛ 20 اضافه کن؛ 10 کم کن','fa')
 world=WorldStateV21.from_ir(ir);world.relations.append({'subject':'event_1','predicate':'before','object':'event_0'})
 with pytest.raises(ExecutionError,match='dependency'):GraphExecutorV21().execute(world.graph())

def test_new_graph_operations_and_constraints():
 graph={'version':2,'initial':10,'state':{},'nodes':[{'id':'growth','op':'change_over_time','rate':2,'duration':3},{'id':'cmp','op':'compare','left':16,'right':15},{'id':'effect','op':'cause_effect','cause':'cmp','effect':{'op':'multiply','value':9}},{'id':'restore','op':'dependency','event':'growth'},{'id':'check','op':'constraint_check','kind':'maximum','value':20}]}
 assert GraphExecutorV21().execute(graph)[0]==16
 graph['nodes'][-1]['value']=15
 with pytest.raises(ExecutionError):GraphExecutorV21().execute(graph)

def test_impossible_inventory_abstains():
 f=extract_source('موجودی انبار 5 کالا است؛ 8 کم کن')
 with pytest.raises(UnsupportedSource):source_answer(f)

def test_ast_isolation_and_boundaries():
 with pytest.raises(CodeLimit):SafePython().run('def f(x):\n    return __import__("os").system("id")',[1])
 with pytest.raises(CodeLimit):SafePython().run('def f(x):\n    return list(range(10000000))',[1])
 assert SafePython().run('def f(xs):\n    total = 0\n    for x in xs:\n        if x > 0:\n            total += x*x\n    return total',[[2,-3,4]])==20

def test_refactor_preserves_bounded_behavior():
 source='def custom(xs):\n    result = []\n    for item in xs:\n        if item > 2:\n            result.append(item * 3 + 1)\n    return result'
 revised=refactor(source);assert revised!=source and 'for item in xs' in revised
 for values in [[],[2],[3,4,-1],[5,2,9]]:assert SafePython().run(source,[values])==SafePython().run(revised,[values])

def test_code_model_loaded_called_affects_output():
 b=CodeIntelligenceV21();q='Write a Python function: keep positive values; multiply by 3; add 2; sum output.'
 result=b.solve(q);assert 'x * 3' in result and b.model.calls>0
 with patch.object(b.model,'predict',return_value=[('explain',.99)]):assert b.solve(q) is None

def test_real_runtime_uses_v21_and_memory_respects_disable():
 with TemporaryRuntime() as rt:
  assert isinstance(rt.agent.reasoning.local_intelligence,LocalIntelligenceV21)
  rt.memory.set_setting('memory_enabled',True)
  rt.agent.respond('من با ESP32 کار می کنم')
  result=rt.agent.respond('برای پروژه جدیدم چه کار کنم؟')
  assert 'ESP32' in result.text and result.data['reasoning_source']=='episodic_memory_v21'
  rt.memory.set_setting('memory_enabled',False)
  rt.agent.respond('من با STM32 کار می کنم')
  assert 'STM32' not in rt.memory.semantic_memories().get('v21_project_topics','')

def test_original_request_overrules_corrupt_ir_source():
 b=LocalIntelligenceV21();q='525 را با نسبت 4 به 3 تقسیم کن';ir=b.parser.parse(q)
 ir.slots['total']=700;ir.source_text='700 را با نسبت 4 به 3 تقسیم کن'
 with patch.object(b.parser,'parse',return_value=ir):answer=b.solve(q)
 assert '300' in answer.text and '225' in answer.text
 assert not b.last_trace['attempts'][0]['verification']['passed']

def test_universal_verifier_requires_source():
 from jarvis.agent.cognitive_ir_v20 import SemanticIR
 verdict=UniversalVerifier().verify(SemanticIR('ratio',slots={'total':7,'ratio_a':4,'ratio_b':3}),[4,3])
 assert not verdict.passed and 'source_missing' in verdict.failed_checks

def test_language_adapter_loaded_called_and_changes_logits():
 import numpy as np
 from pathlib import Path
 from jarvis.neural.conversation import _load_assets
 root=Path(__file__).resolve().parents[1]
 model,tok=_load_assets(str(root/'models/jarvis_nano_v18.npz'),str(root/'models/jarvis_tokenizer_v003.json'))
 assert hasattr(model,'language_adapter_v21')
 adapter=model.language_adapter_v21;before=adapter.calls
 tokens=[tok.bos_id,*tok.encode('سلام، نتیجه را توضیح بده')]
 adapted=model.forward(tokens)
 with patch.object(adapter,'residual',side_effect=lambda x:np.zeros((*x.shape[:-1],model.config.vocab_size),dtype=np.float32)):
  base=model.forward(tokens)
 assert adapter.calls>before and np.max(np.abs(adapted-base))>.01
 assert np.isfinite(adapted).all()

@pytest.mark.parametrize('question,expected',[
 ('50 ta ro 20 darsad ziad kon',60),
 ('adad 100 ast; dar 3 zarb kon; bar 2 taghsim kon',150),
 ('سه برابر نصف 20',30),
 ('عدد 100 است؛ اگر بیشتر از 80 بود آنگاه 20 اضافه کن وگرنه 10 کم کن',120),
 ('Start with 50; if less than 40 then add 7 otherwise subtract 3',47),
])
def test_finglish_scope_and_conditional_source(question,expected):
 b=LocalIntelligenceV21();answer=b.solve(question)
 assert answer is not None and str(expected) in answer.text
 assert b.last_trace['verification']['passed']

def test_semantic_memory_is_attributed_and_forgettable():
 from jarvis.memory.context_v21 import MemoryContextV21
 with TemporaryRuntime() as rt:
  memory=MemoryContextV21(rt.memory)
  memory.observe('ESP32 یک میکروکنترلر است',rt.agent.session_id)
  assert 'طبق توضیح قبلی خودت' in memory.recall('ESP32 چیست؟')
  rt.memory.forget('semantic','v21_user_isa:esp32')
  assert memory.recall('ESP32 چیست؟') is None

def test_explicit_number_of_operations_is_a_constraint():
 q='عدد 100 است؛ 20 درصد زیاد کن؛ 10 درصد کم کن. منظور دو تغییر پیاپی است.'
 b=LocalIntelligenceV21();assert '108' in b.solve(q).text
 with pytest.raises(UnsupportedSource):extract_source(q.replace('دو تغییر','سه تغییر'))

def test_rendered_response_is_independently_verified():
 v=UniversalVerifier();q='525 را با نسبت 4 به 3 تقسیم کن'
 assert v.verify_rendered(q,'300 و 225').passed
 assert not v.verify_rendered(q,'225 و 300').passed
 assert not v.verify_rendered(q,'300 و 225 و 999').passed
 q='احتمال موفقیت 30 درصد است؛ در 4 تلاش دقیقاً 2 موفقیت؟'
 assert v.verify_rendered(q,'26.46%').passed
 assert not v.verify_rendered(q,'0.2646%').passed

@pytest.mark.parametrize('source',[
 'Ali has 100 dollars. Sara has 50 items. Ali transfers 30 dollars to Sara. Balance of Sara?',
 'Ali has 100 dollars. Ali has 200 dollars. Sara has 50 dollars. Ali transfers 30 dollars to Sara. Balance of Sara?',
])
def test_ownership_rejects_incompatible_source_assertions(source):
 with pytest.raises(UnsupportedSource):extract_source(source)

def test_age_question_target_is_not_assumed_to_be_relative():
 q='Mina is 17 years old. Her sister is 5 years older. In 4 years how old will Mina be?'
 f=extract_source(q);assert f.slots['query']=='base' and source_answer(f)==21
 ir=facts_to_ir(f,q,'en');ir.slots['query']='relative'
 assert not UniversalVerifier().verify(ir,26).passed
