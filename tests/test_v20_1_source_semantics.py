from copy import deepcopy
from unittest.mock import patch
import math
import pytest
from jarvis.agent.local_intelligence_v20 import LocalIntelligenceV20
from jarvis.agent.cognitive_ir_v20 import WorldState
from jarvis.agent.verification_v20 import Verifier
from jarvis.agent.source_semantics_v20_1 import source_work_facts,sequence_source
from tests.helpers import TemporaryRuntime

WORK='با 3 کارگر و 4 ساعت، خروجی 84 قطعه است. با 8 کارگر و 6 ساعت چه خروجی داریم؟'
SEQ='دنباله 24، 12، 6؛ جمله پنجم چیست؟'

def test_reported_work_wrong_source_but_right_calculation_is_rejected():
    b=LocalIntelligenceV20(); ir=b.parser.parse(WORK)
    ir.slots.update(workers_initial=8,workers_target=3)
    verdict=b.verifier.verify(ir,47.25)
    assert 'productivity_conservation' in verdict.checks
    assert 'productivity_conservation' not in verdict.failed_checks
    assert not verdict.passed and verdict.repair_stage=='slots'
    assert 'source_slot:workers_initial' in verdict.failed_checks
    assert 'source_slot:workers_target' in verdict.failed_checks

def test_semantic_verification_does_not_trust_fabricated_model_evidence():
    b=LocalIntelligenceV20(); ir=b.parser.parse(WORK)
    ir.slots.update(workers_initial=8,workers_target=3)
    ir.model_evidence={'numbers':[],'source_bindings':{'workers_initial':{'source_value':8}}}
    assert not b.verifier.verify(ir,47.25).passed

def test_semantic_failure_repairs_even_if_stage_model_says_execution():
    b=LocalIntelligenceV20(); ir=b.parser.parse(WORK)
    ir.slots.update(workers_initial=8,workers_target=3)
    with patch.object(b.reflection.stage_model,'predict',return_value=[('execution',.999)]):
        value,verdict,attempts,world=b.reflection.solve(ir)
    assert value==336 and verdict.passed and len(attempts)==2
    assert attempts[0]['answer']==47.25
    assert attempts[1]['changed_slots']['workers_initial']==3
    assert attempts[0]['repair_stage_constraint']=='source_slot_consistency'

def test_unrepairable_source_is_not_reported_verified():
    b=LocalIntelligenceV20(); ir=b.parser.parse(WORK)
    ir.source_text='There is not enough information about the workshop.'
    answer,verdict,attempts,_=b.reflection.solve(ir)
    assert answer is None and not verdict.passed
    assert any(x.startswith('source_') for x in verdict.failed_checks)
    with patch.object(b.parser,'parse',return_value=ir):
        reply=b.solve(ir.source_text,'en')
    assert 'v20_verification_failed' in reply.checks
    assert 'result_verified' not in reply.checks

def test_source_target_verification_rejects_next_term_substitution():
    b=LocalIntelligenceV20(); ir=b.parser.parse(SEQ)
    assert ir.slots['n']==5
    ir.slots['n']=4
    verdict=b.verifier.verify(ir,3)
    assert not verdict.passed and 'source_sequence_target' in verdict.failed_checks
    assert 'sequence_nth' not in verdict.failed_checks
    answer,verdict,attempts,_=b.reflection.solve(ir)
    assert answer==1.5 and verdict.passed and attempts[-1]['changed_slots']['n']==5

def test_list_terms_and_order_are_verified_against_source():
    b=LocalIntelligenceV20(); ir=b.parser.parse(SEQ)
    ir.slots['terms']=[12,6,3]
    assert not b.verifier.verify(ir,.75).passed
    assert 'source_sequence_terms' in b.verifier.verify(ir,.75).failed_checks

@pytest.mark.parametrize('q,expected',[
 (WORK,336),
 ('با 8 کارگر و 6 ساعت چه خروجی داریم؟ با 3 کارگر و 4 ساعت، خروجی 84 قطعه است.',336),
 ('در 4 ساعت 3 کارگر 84 قطعه میسازند؛ حالا در 6 ساعت 8 کارگر چقدر میسازند؟',336),
 ('3 workers make 84 pieces in 4 hours. What will 8 workers make in 6 hours?',336),
 ('What will 8 workers make in 6 hours? 3 workers make 84 pieces in 4 hours.',336),
 ('3 کارگر در 4 ساعت 84 قطعه تولید می کنند. 8 کارگر در 6 ساعت چقدر تولید می کنند؟',336),
 ('3 workers produce 84 units in 4 hours; now 8 workers have 6 hours.',336),
])
def test_work_relationship_variants(q,expected):
    b=LocalIntelligenceV20();ir=b.parser.parse(q)
    assert ir is not None
    result,verdict,_,_=b.reflection.solve(ir)
    assert math.isclose(result,expected) and verdict.passed

@pytest.mark.parametrize('q,n',[
 ('دنباله 24، 12، 6؛ جمله پنجم چیست؟',5),
 ('دنباله ۲۴،۱۲،۶؛ جمله پنجم؟',5),
 ('دنباله ٢٤،١٢،٦؛ جمله پنجم؟',5),
 ('شماره پرونده 908 است؛ دنباله 24، 12، 6؛ جمله پنجم چیست؟',5),
 ('جمله پنجم دنباله 24، 12، 6 را بده.',5),
 ('Find term five of the sequence 24, 12, 6.',5),
 ('For the sequence 24, 12, 6 give the fifth term.',5),
 ('Sequence 24, 12, 6. Find term 5.',5),
 ('sequence 24, 12, 6; next term?',4),
 ('دنباله 24،12،6؛ عدد بعدی؟',4),
 ('Find term six of the sequence 24, 12, 6.',6),
 ('دنباله 24،12،6؛ جمله هشتم؟',8),
])
def test_explicit_sequence_target_variants(q,n):
    b=LocalIntelligenceV20();ir=b.parser.parse(q)
    assert ir is not None
    assert ir.slots['terms']==[24,12,6] and ir.slots['n']==n
    result,verdict,_,_=b.reflection.solve(ir)
    assert verdict.passed and result==24*.5**(n-1)

def test_runtime_endpoint_produces_both_fixes_without_tools():
    with TemporaryRuntime() as rt,patch.object(rt.tools,'invoke',side_effect=AssertionError('unexpected tool')):
        a=rt.agent.respond(WORK);b=rt.agent.respond(SEQ)
        assert a.text=='336' and b.text=='1.5'
        assert 'source_slot:workers_initial' in a.data['self_checks']
        assert 'source_sequence_target' in b.data['self_checks']

@pytest.mark.parametrize("question,expected", [
 ("What is the fifth term in 112, 56, 28?",7),
 ("Sequence: 168, 84, 42. Give the fifth term, not just the next value.",10.5),
])
def test_explicit_target_survives_moderate_frame_confidence(question,expected):
    brain=LocalIntelligenceV20()
    ir=brain.parser.parse(question,"en")
    assert ir is not None and ir.slots["n"]==5
    value,verdict,_,_=brain.reflection.solve(ir)
    assert math.isclose(value,expected) and verdict.passed
