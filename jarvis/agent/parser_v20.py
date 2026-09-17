"""Learned frame -> learned numeric spans -> typed, bounded IR.

The learned model selects the task. Source constraints reconcile explicit lists,
indices and observation groups with numeric predictions, retaining raw evidence.
Unsupported or incomplete parses abstain.
"""
from __future__ import annotations
import re
from functools import lru_cache
from jarvis.agent.normalization_v20 import normalize,NUMBER
from jarvis.agent.semantic_models_v20 import LearnedModel
from jarvis.agent.cognitive_ir_v20 import SemanticIR
from jarvis.agent.execution_v20 import sequence_rule

class SemanticParser:
    def __init__(self):
        self.frame=LearnedModel('semantic_frame'); self.numeric=LearnedModel('numeric_role'); self.operation=LearnedModel('operation')
        self.last_prediction=[]

    def tag(self,text,family):
        tagged=[]
        for m in NUMBER.finditer(text):
            scores=self.numeric.predict(text,(m.start(),m.end()),family)
            if not scores: continue
            role,conf=scores[0]
            tagged.append({'role':role,'confidence':conf,'value':float(m.group()),'start':m.start(),'end':m.end(),'alternatives':scores[:3],
                           'unit':'percent' if re.match(r'\s*(%|درصد|percent)',text[m.end():]) else ''})
        return tagged

    @staticmethod
    def values(tags,role): return [x['value'] for x in tags if x['role']==role and x['confidence']>=.24]
    @classmethod
    def one(cls,tags,role):
        v=cls.values(tags,role)
        if len(v)!=1: raise ValueError('missing_or_ambiguous_'+role)
        return v[0]

    def parse(self,text,language='fa'):
        # Code source must retain whitespace/identifiers and is handled by the code subsystem.
        if re.search(r'```|\bprint\s*\(|\b(?:def|class)\s+\w+|\b(?:for|while)\b[^;\n]*:|(?:^|\n)\s*import\s+|\b\w+\s*=',text): return None
        t=normalize(text); pred=self.frame.predict(t); self.last_prediction=pred
        if not pred or pred[0][1]<.60: return None
        family,conf=pred[0]
        if family in ('other','system','external','knowledge','provided_text','code'): return None
        if family=='sequence' and conf<.75:
            # Explicit list + requested index corroborate a learned sequence frame.
            # Without both, preserve the stronger threshold for ambiguous prose.
            from jarvis.agent.source_semantics_v20_1 import sequence_source, SourceAmbiguity
            try: explicit=sequence_source(text)
            except SourceAmbiguity: return None
            if not explicit['terms'] or explicit['target'] is None: return None
        # A request to discuss non-uniqueness is not a request for a numeric extrapolation.
        if family=='sequence' and re.search(r'یکتا|قطعی|unique|ambiguous|explain why|توضیح بده چرا',t): return None
        tags=self.tag(t,family); one=lambda r:self.one(tags,r); values=lambda r:self.values(tags,r)
        ir=SemanticIR(family,language,confidence=conf,source_text=text,
                      model_evidence={'frame':pred[:3],'numbers':tags,'checkpoints':{'frame':self.frame.sha256,'numeric':self.numeric.sha256,'operation':self.operation.sha256}})
        try:
            if family=='ratio':
                ir.slots={k:one(k) for k in ('total','ratio_a','ratio_b')}; ir.answer_type='parts'
            elif family=='binomial':
                ptag=[x for x in tags if x['role']=='p']
                p=one('p')
                # 1 percent is 0.01, not 1.0; a percent unit is mandatory evidence.
                p=p/100 if ptag[0]['unit']=='percent' else p
                mode='at_least' if re.search(r'حداقل|at least',t) else 'at_most' if re.search(r'حداکثر|at most',t) else 'exactly'
                ir.slots={'p':p,'n':one('n'),'k':one('k'),'mode':mode}; ir.answer_type='probability'
            elif family in ('combination','permutation'):
                ir.slots={'n':one('n'),'k':one('k')}
            elif family=='dice': ir.slots={'n':one('n'),'target':one('target')}; ir.answer_type='probability'
            elif family=='independent':
                ps=[x['value']/100 if x['unit']=='percent' else x['value'] for x in tags if x['role']=='p']
                if len(ps)<2: return None
                ir.slots={'probabilities':ps}; ir.answer_type='probability'
            elif family=='sequence':
                from jarvis.agent.source_semantics_v20_1 import sequence_source
                source=sequence_source(text)
                terms=values('term'); ns=values('n')
                if source['terms']:
                    # Constrained binding retains the classifier's raw predictions for audit.
                    # Explicit list/index spans cannot be silently folded into a next-term task.
                    terms=[item['value'] for item in source['terms']]
                    ns=[source['target']['value']] if source['target'] else []
                    ir.model_evidence['sequence_source']=source
                elif source['target']:
                    ns=[source['target']['value']]
                if len(ns)>1: return None
                if len(terms)>=3:
                    ir.slots={'terms':terms,'n':ns[0] if ns else len(terms)+1,'rule':sequence_rule(terms)}
                else:
                    first=one('first'); ratio=values('ratio'); diff=values('difference')
                    rule={'kind':'geometric','r':ratio[0]} if len(ratio)==1 else {'kind':'arithmetic','d':diff[0]} if len(diff)==1 else None
                    if rule is None or not ns: return None
                    ir.slots={'terms':[first],'n':ns[0],'rule':rule}
            elif family=='work_rate':
                from jarvis.agent.source_semantics_v20_1 import bind_work_spans
                ir.slots,evidence=bind_work_spans(t,tags)
                ir.model_evidence['source_bindings']=evidence
                ir.units={'time':'hour','output':'piece'}
                ir.entities=[{'id':'crew_initial','type':'workers'},{'id':'crew_target','type':'workers'}]
            elif family in ('graph','inventory','finance'):
                ir.slots={'initial':one('initial')}
                conditional=re.search(r'\bif\b|اگر',t)
                if conditional:
                    expr=t[conditional.start():]
                    branch_parts=re.split(r'\bthen\b|آنگاه',expr,maxsplit=1)
                    if len(branch_parts)!=2: return None
                    condition,branches=branch_parts
                    pred_condition=self.operation.predict(condition)
                    if not pred_condition or pred_condition[0][0] not in ('conditional_gt','conditional_lt') or pred_condition[0][1]<.50: return None
                    threshold=self.one(self.tag(condition,family),'threshold')
                    arms=re.split(r'\botherwise\b|\belse\b|وگرنه',branches,maxsplit=1)
                    if len(arms)!=2: return None
                    actions=[]
                    for arm in arms:
                        ap=self.operation.predict(arm); at=self.tag(arm,family)
                        if not ap or ap[0][1]<.52 or ap[0][0].startswith('conditional'): return None
                        actions.append({'op':ap[0][0],'value':self.one(at,'value')})
                    ir.operations=[{'op':'conditional','comparison':'gt' if pred_condition[0][0].endswith('gt') else 'lt',
                                    'threshold':threshold,'then':[actions[0]],'else':[actions[1]]}]
                    return ir
                # Split connector grammar, never decimal points. Value/operation still learned.
                parts=re.split(r'\s*(?:[;؛،,]|(?<!\d)\.(?!\d)|\band then\b|\bafter that\b|\bfollowed by\b|\bthen\b|\band\b|و بعدش|و بعد|بعدش|سپس|آنگاه|حاصل را|\bو\b)\s*',t)
                for part in parts:
                    pts=self.tag(part,family); operands=[x for x in pts if x['role']=='value' and x['confidence']>=.24]
                    if not operands: continue
                    if len(operands)!=1: return None
                    op=self.operation.predict(part)
                    if not op or op[0][1]<.52: return None
                    ir.operations.append({'op':op[0][0],'value':operands[0]['value']})
                if not ir.operations or len(ir.operations)>64: return None
                # Don't silently drop a number the numeric model identified as an operand.
                if len(ir.operations)!=len(values('value')): return None
                ir.entities=[{'id':'account' if family=='finance' else 'warehouse' if family=='inventory' else 'accumulator','type':family}]
            elif family=='age':
                d=one('difference')
                if re.search(r'younger|کوچک',t): d=-abs(d)
                ir.slots={'age':one('age'),'difference':d,'years':one('years'),
                          'query':'sum' if re.search(r'مجموع|جمع|sum|combined',t) else 'relative'}
                ir.relations=[{'type':'age_difference','value':d}]; ir.units={'time':'year'}
            elif family=='speed':
                dist=one('distance'); time=one('time')
                per_second=bool(re.search(r'متر بر ثانیه|meters per second|m / s',t))
                tt=next(x for x in tags if x['role']=='time'); tail=t[tt['end']:tt['end']+20]
                dt=next(x for x in tags if x['role']=='distance'); dtail=t[dt['end']:dt['end']+24]
                distance_km=bool(re.match(r'\s*(?:km|kilometers?|کیلومتر)',dtail))
                distance_m=bool(re.match(r'\s*(?:meters?|متر)',dtail))
                if per_second and distance_km: dist*=1000
                elif not per_second and distance_m: dist/=1000
                if not per_second and re.match(r'\s*(minutes?|دقیقه)',tail): time/=60
                elif not per_second and re.match(r'\s*(seconds?|ثانیه)',tail): time/=3600
                elif per_second and re.match(r'\s*(minutes?|دقیقه)',tail): time*=60
                elif per_second and re.match(r'\s*(hours?|ساعت)',tail): time*=3600
                ir.slots={'distance':dist,'time':time}; ir.units={'speed':'m/s' if per_second else 'km/h'}
            elif family=='scheduling':
                durations=values('duration')
                if not durations: return None
                ir.slots={'start':one('start'),'durations':durations}; ir.units={'time':'hour'}
            elif family=='comparison':
                terms=values('term')
                if len(terms)<2: return None
                mode='mean' if re.search(r'میانگین|معدل|متوسط|average|mean',t) else 'min' if re.search(r'کوچک|کمترین|smallest|minimum',t) else 'max'
                ir.slots={'values':terms,'mode':mode}
            else: return None
        except (ValueError,KeyError,IndexError): return None
        return ir

@lru_cache(maxsize=1)
def get_parser(): return SemanticParser()
