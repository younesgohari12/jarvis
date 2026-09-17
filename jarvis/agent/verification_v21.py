"""Fresh source facts + source answer + existing arithmetic invariants."""
from jarvis.agent.verification_v20 import Verifier,Verdict,close
from jarvis.agent.source_facts_v21 import extract_source,source_answer,UnsupportedSource

from contextvars import ContextVar
from dataclasses import replace
AUTHORITATIVE_SOURCE=ContextVar('jarvis_v21_user_source',default=None)

COVERED={'ratio','binomial','combination','permutation','dice','independent','finance','inventory','graph','age','speed','scheduling','comparison','sequence','work_rate','ownership'}
def equivalent(a,b):
    if isinstance(a,dict) and isinstance(b,dict):return a.keys()==b.keys() and all(equivalent(a[k],b[k]) for k in a)
    if isinstance(a,(list,tuple)) and isinstance(b,(list,tuple)):return len(a)==len(b) and all(equivalent(x,y) for x,y in zip(a,b))
    if isinstance(a,(int,float)) and isinstance(b,(int,float)):return close(a,b)
    return a==b

def semantic_checks(ir,candidate):
    # Pure solver unit tests have no source and cannot establish semantic validity.
    if not ir.source_text:return {},None
    try:
        source=extract_source(ir.source_text)
        expected=source_answer(source)
        checks={'source_extraction_complete':True,'source_task':source.domain==ir.task or source.domain=='graph' and ir.task in ('finance','inventory')}
        for key,value in source.slots.items():checks['source_slot:'+key]=key in ir.slots and equivalent(value,ir.slots[key])
        for key,value in source.units.items():checks['source_unit:'+key]=ir.units.get(key)==value
        if source.domain=='ownership':
            checks['source_entities']=equivalent(source.entities,ir.entities)
            checks['source_relations']=equivalent(source.relations,ir.relations)
        clean=lambda es:[{k:v for k,v in e.items() if k not in ('id','depends_on','source_span')} for e in es]
        checks['source_event_order']=equivalent(clean(source.operations),clean(ir.operations))
        checks['source_answer_consistency']=equivalent(candidate,expected)
        return checks,source
    except (UnsupportedSource,ArithmeticError,ValueError,TypeError,KeyError,OverflowError):return {'source_extraction_complete':False},None

class UniversalVerifier(Verifier):
    def verify_rendered(self,source_text,text):
        import re
        try:
            expected=source_answer(extract_source(source_text))
            tail=text.rsplit('=',1)[-1].replace('٪','%')
            values=[float(v) for v in re.findall(r'(?<![\w.])-?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?',tail)]
            if '%' in tail:values=[v/100 for v in values]
            observed=values if isinstance(expected,list) else values[0] if len(values)==1 else None
            passed=equivalent(observed,expected)
        except (ValueError,TypeError,ArithmeticError,KeyError):passed=False
        return Verdict(passed,[] if passed else ['source_rendered_answer_consistency'],repair_stage='none' if passed else 'response',checks=['source_rendered_answer_consistency'])
    def verify(self,ir,candidate):
        self.calls+=1
        source_text=AUTHORITATIVE_SOURCE.get()
        if source_text is not None:ir=replace(ir,source_text=source_text)
        if not ir.source_text:return Verdict(False,['source_missing'],'Original user text is required','slots',['source_missing'])
        source_checks,source=semantic_checks(ir,candidate)
        # A source-based witness independently handles new world operations.
        if ir.task=='ownership' or ir.slots.get('query') in ('distance','simplify','combined_time','base') or any(e['op']=='purchase' for e in ir.operations):
            base=Verdict(True,checks=['world_source_reference'])
        else:
            self.calls-=1;base=super().verify(ir,candidate)
        if ir.source_text and ir.task not in COVERED:source_checks['source_supported_domain']=False
        base.checks+=list(source_checks)
        base.failed_checks+=list(k for k,v in source_checks.items() if not v)
        base.failed_checks=list(dict.fromkeys(base.failed_checks));base.checks=list(dict.fromkeys(base.checks))
        base.passed=not base.failed_checks
        if not base.passed and any(k.startswith('source_') for k in base.failed_checks):base.repair_stage='slots';base.repair_hint='Extract fresh facts from immutable user source'
        if not ir.source_text:base.checks.append('arithmetic_only_no_source')
        return base
