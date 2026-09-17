from copy import deepcopy
from pathlib import Path
import json,math,re
from unittest.mock import patch
import pytest
from jarvis.agent.cognitive_ir_v20 import SemanticIR,WorldState
from jarvis.agent.execution_v20 import GraphExecutor,ExecutionError
from jarvis.agent.verification_v20 import Verifier
from jarvis.agent.local_intelligence_v20 import LocalIntelligenceV20
from jarvis.agent.normalization_v20 import normalize
from jarvis.agent.parser_v20 import get_parser
from jarvis.agent.code_intelligence_v20 import verified_recipe,RECIPES
from tests.helpers import TemporaryRuntime,ROOT

@pytest.mark.parametrize('text,expected',[
 ('دقیقاً یک موفقیت','دقیقا 1 موفقیت'),('هیچ موفقیتی','0 موفقیتی'),('چهار کارگر','4 کارگر'),
 ('بیست و سه ساعت','23 ساعت'),('one hundred and twenty three','123'),('یک\u200cچهارم','0.25'),
 ('half','0.5'),('twice','multiply by 2'),('دو برابر','2 برابر'),('all four','all 4')])
def test_word_numbers(text,expected): assert normalize(text)==expected

def test_runtime_all_models_loaded_registered_called():
    with TemporaryRuntime() as rt:
        brain=rt.agent.reasoning.local_intelligence
        assert isinstance(brain,LocalIntelligenceV20)
        models=[brain.parser.frame,brain.parser.numeric,brain.parser.operation,brain.code_v20.model,brain.reflection.stage_model]
        registry=json.loads((ROOT/'models/model_registry_v2.json').read_text())['cognitive_v20']['models']
        for model in models:
            assert model.ready and model.parameter_count>1000
            assert any(v['sha256']==model.sha256 for v in registry.values())
        before=[m.calls for m in models[:3]]
        with patch.object(rt.tools,'invoke',side_effect=AssertionError('unexpected tool')):
            result=rt.agent.respond('500 را 20 درصد کم کن، 30 اضافه کن و حاصل را بر 5 تقسیم کن')
        assert '86' in result.text
        assert all(m.calls>n for m,n in zip(models,before))
        assert brain.verifier.calls>0 and brain.executor.calls>0
        assert result.data['reasoning_source']=='semantic_ir_v21'

def test_trained_operation_changes_final_runtime_answer():
    with TemporaryRuntime() as rt:
        b=rt.agent.reasoning.local_intelligence; original=b.parser.operation.predict
        q='500 را 20 درصد کم کن، 30 اضافه کن و حاصل را بر 5 تقسیم کن'
        assert '86' in rt.agent.respond(q).text
        def altered(text,*args,**kwargs):
            pred=original(text,*args,**kwargs)
            return [('subtract',.999)] if pred and pred[0][0]=='add' else pred
        with patch.object(b.parser.operation,'predict',side_effect=altered):
            # v21: the model still changes the initial graph, but semantic verification repairs it.
            assert '86' in rt.agent.respond(q).text
            attempts=b.last_trace['attempts']
            assert attempts[0]['answer']==74
            assert 'source_event_order' in attempts[0]['verification']['failed_checks']
            assert attempts[-1]['answer']==86

def test_world_state_is_actual_graph_input_and_task_local():
    a=SemanticIR('inventory',slots={'initial':100},operations=[{'op':'inventory_remove','value':30},{'op':'inventory_add','value':20}])
    w=WorldState.from_ir(a); w.state['initial']=150
    e=GraphExecutor()
    assert e.execute(w.graph())[0]==140
    assert e.execute(WorldState.from_ir(a).graph())[0]==90
    assert a.slots['initial']==100

def test_reflection_discards_failed_candidate_and_reexecutes():
    b=LocalIntelligenceV20(); original=b.executor.execute; count=[0]
    def corrupt_once(graph):
        value,trace=original(graph); count[0]+=1
        return ([1.7,1.2] if count[0]==1 else value),trace
    with patch.object(b.executor,'execute',side_effect=corrupt_once):
        result=b.solve('525 را با نسبت 4 به 3 تقسیم کن')
    assert '300' in result.text and '225' in result.text
    trace=b.last_trace['attempts']
    assert len(trace)==2 and not trace[0]['verification']['passed'] and trace[1]['verification']['passed']
    assert trace[0]['answer']!=trace[1]['answer'] and b.reflection.stage_model.calls>0

def test_reflection_repairs_slots_from_source():
    b=LocalIntelligenceV20(); ir=b.parser.parse('شانس موفقیت 30 درصد است؛ در چهار تلاش دقیقاً یک موفقیت؟')
    assert ir is not None
    ir.slots['p']=30
    value,verdict,attempts,world=b.reflection.solve(ir)
    assert verdict.passed and math.isclose(value,.4116)
    assert attempts[-1]['changed_slots']['p']==.3

@pytest.mark.parametrize('op,value,expected',[
 ('add',3,13),('subtract',3,7),('multiply',3,30),('divide',4,2.5),('percentage_add',20,12),('percentage_remove',20,8),
 ('exponent',3,1000),('modulo',3,1),('round',0,10),('min',4,4),('max',13,13),('inventory_add',3,13),('inventory_remove',3,7),('credit',3,13),('debit',3,7)])
def test_graph_operator_independent_verifier(op,value,expected):
    ir=SemanticIR('graph',slots={'initial':10},operations=[{'op':op,'value':value}])
    actual,_=GraphExecutor().execute(WorldState.from_ir(ir).graph())
    assert actual==expected and Verifier().verify(ir,actual).passed
    assert not Verifier().verify(ir,actual+1).passed

def test_conditional_graph_and_independent_verification():
    for x,expected in [(12,15),(7,3)]:
        ir=SemanticIR('graph',slots={'initial':x},operations=[{'op':'conditional','comparison':'gt','threshold':10,'then':[{'op':'add','value':3}],'else':[{'op':'subtract','value':4}]}])
        result,_=GraphExecutor().execute(WorldState.from_ir(ir).graph())
        assert result==expected and Verifier().verify(ir,result).passed
        assert not Verifier().verify(ir,result+2).passed

@pytest.mark.parametrize('terms,n,expected',[
 ([10,5,2.5],4,1.25),([2,6,18],5,162),([2,-4,8],5,32),([1,1,2,3,5],7,13),([2,5,6,9,10],7,14),([3,7,11],8,31)])
def test_sequence_families(terms,n,expected):
    ir=SemanticIR('sequence',slots={'terms':terms,'n':n})
    result,_=GraphExecutor().execute(WorldState.from_ir(ir).graph())
    assert result==expected and Verifier().verify(ir,result).passed
    assert not Verifier().verify(ir,result+.1).passed

@pytest.mark.parametrize('label',list(RECIPES))
def test_generated_code_syntax_and_safe_behavior(label): assert verified_recipe(label) is not None

def test_new_code_model_is_causal():
    from jarvis.agent.code_intelligence_v20 import CodeIntelligenceV20
    code=CodeIntelligenceV20()
    q='give me code for reversing a string'
    assert 'text[::-1]' in code.solve(q)
    with patch.object(code.model,'predict',return_value=[('other',.99)]): assert code.solve(q) is None

def test_rag_avoids_transforms_and_deduplicates():
    with TemporaryRuntime() as rt:
        text='Gravity is the attraction between masses. گرانش نیروی جاذبه بین جرم ها است.'
        rt.rag.index_text(Path('a.txt'),text); rt.rag.index_text(Path('b.txt'),text)
        assert len(rt.rag.search('gravity'))==1
        assert rt.rag.search('7!')==()
        assert rt.rag.search('ترجمه کن: gravity')==()
        assert rt.rag.search('unrelated zephyr cactus')==()

def test_runtime_constraints_new_verifier_called():
    with TemporaryRuntime() as rt:
        result=rt.agent.respond('Write exactly two sentences about learning. Include the word "focus".')
        assert result.data['constraint_verification_v20']['passed']

def test_division_and_probability_invalid_no_unverified_answer():
    b=LocalIntelligenceV20()
    q='start with 240; divide by 0'
    answer=b.solve(q,'en')
    assert b.last_trace and not b.last_trace['verification']['passed']
    assert 'could not verify' in answer.text.casefold()

def test_corrupt_model_disables_head_without_crashing_runtime(tmp_path):
    from jarvis.agent.semantic_models_v20 import LearnedModel
    (tmp_path/'models').mkdir()
    (tmp_path/'models/semantic_frame_v20.npz').write_bytes(b'not a checkpoint')
    model=LearnedModel('semantic_frame',tmp_path)
    assert not model.ready and model.predict('question')==[] and 'load_error' in model.metadata

def test_decimal_rounding_is_verified():
    ir=SemanticIR('graph',slots={'initial':2.675},operations=[{'op':'round','value':2}])
    value,_=GraphExecutor().execute(WorldState.from_ir(ir).graph())
    assert value==2.68 and Verifier().verify(ir,value).passed

def test_mixed_distance_units_are_canonicalized_before_solving():
    b=LocalIntelligenceV20()
    q='distance 3 kilometers, time 10 seconds; speed in meters per second?'
    # Force only the learned labels to isolate and test unit conversion, with real spans.
    original=b.parser.tag
    def roles(text,family):
        return [{'role':'distance','confidence':.99,'value':3,'start':9,'end':10},
                {'role':'time','confidence':.99,'value':10,'start':29,'end':31}]
    from jarvis.agent.normalization_v20 import NUMBER
    def measured_roles(text,family):
        return [{'role':role,'confidence':.99,'value':float(m.group()),'start':m.start(),'end':m.end()} for role,m in zip(['distance','time'],NUMBER.finditer(text))]
    with patch.object(b.parser.frame,'predict',return_value=[('speed',.99)]),patch.object(b.parser,'tag',side_effect=measured_roles):
        ir=b.parser.parse(q,'en')
    assert ir.slots=={'distance':3000.0,'time':10.0}
    value,_,_,_=b.reflection.solve(ir)
    assert value==300

def test_conditional_language_reaches_active_runtime():
    b=LocalIntelligenceV20()
    q='start with 240; if the result is greater than 40 then subtract 9 otherwise add 17'
    answer=b.solve(q,'en')
    assert answer.text=='231' and b.last_trace['ir']['operations'][0]['op']=='conditional'

# 120 adversarial candidate tests with different domains, faults, decimal and negative operands.
# They assert actual invariants, rather than defining PASS as merely having an answer.
ADVERSARIAL=[]
for i in range(20):
    total=101+i*13; a,b=2+i%4,3+i%5
    ir=SemanticIR('ratio',slots={'total':total,'ratio_a':a,'ratio_b':b})
    correct=[total*a/(a+b),total*b/(a+b)]
    ADVERSARIAL.append((f'ratio_wrong_sum_{i}',ir,[correct[0]+.125,correct[1]]))
    ADVERSARIAL.append((f'ratio_wrong_proportion_{i}',ir,[correct[0]+1,correct[1]-1]))
    ir=SemanticIR('binomial',slots={'n':4+i%5,'k':i%3,'p':.13+i*.025})
    right=GraphExecutor().execute(WorldState.from_ir(ir).graph())[0]
    ADVERSARIAL.append((f'probability_wrong_recompute_{i}',ir,min(.999,right+.01)))
    ir=SemanticIR('inventory',slots={'initial':91+i},operations=[{'op':'inventory_add','value':12.5},{'op':'inventory_remove','value':7+i%3}])
    ADVERSARIAL.append((f'inventory_wrong_state_{i}',ir,ir.slots['initial']+12.5))
    ir=SemanticIR('finance',slots={'initial':-20+i},operations=[{'op':'credit','value':32},{'op':'debit','value':13.7}])
    ADVERSARIAL.append((f'finance_sign_error_{i}',ir,ir.slots['initial']+32+13.7))
    ir=SemanticIR('age',slots={'age':12+i,'difference':3,'years':4,'query':'sum'})
    ADVERSARIAL.append((f'age_one_person_advanced_{i}',ir,2*(12+i)+3+4))
@pytest.mark.parametrize('case_id,ir,candidate',ADVERSARIAL,ids=[x[0] for x in ADVERSARIAL])
def test_adversarial_corruptions(case_id,ir,candidate):
    assert not Verifier().verify(ir,candidate).passed
