from __future__ import annotations
from jarvis.agent.local_intelligence_v19 import LocalIntelligenceV19
from jarvis.agent.parser_v20 import get_parser
from jarvis.agent.execution_v20 import GraphExecutor
from jarvis.agent.verification_v20 import Verifier
from jarvis.agent.reflection_v20 import ReflectionLoop

class LocalIntelligenceV20(LocalIntelligenceV19):
    VERSION='2.0.1'
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self.parser=get_parser(); self.executor=GraphExecutor(); self.verifier=Verifier()
        self.reflection=ReflectionLoop(self.executor,self.verifier,self.parser)
        from jarvis.agent.code_intelligence_v20 import CodeIntelligenceV20
        self.code_v20=CodeIntelligenceV20()
        self.last_trace=None

    @classmethod
    def matches(cls,text): return get_parser().parse(text) is not None or super().matches(text)

    def solve(self,text,language='fa'):
        self.last_trace=None
        ir=self.parser.parse(text,language)
        if ir is not None:
            value,verdict,attempts,world=self.reflection.solve(ir)
            self.last_trace={'ir':ir.to_dict(),'world':world,'attempts':attempts,'verification':verdict.to_dict()}
            if verdict.passed:
                fmt=lambda x:f'{x:.10g}'
                if isinstance(value,list): answer=(' و ' if language=='fa' else ' and ').join(map(fmt,value))
                elif ir.answer_type=='probability':
                    answer=fmt(value*100)+('٪' if language=='fa' else '%')
                    if ir.task=='binomial' and ir.slots.get('mode','exactly')=='exactly':
                        s=ir.slots
                        answer=f'C({int(s["n"])},{int(s["k"])}) × {s["p"]:g}^{int(s["k"])} × (1−{s["p"]:g})^{int(s["n"]-s["k"])} = '+answer
                    elif ir.task=='dice':
                        from fractions import Fraction
                        answer=str(Fraction(value).limit_denominator(6**int(ir.slots['n'])))+' = '+answer
                else: answer=fmt(value)
                intent='probability_answer' if ir.answer_type=='probability' else 'word_problem_answer' if ir.task in ('ratio','finance','inventory','work_rate','age','speed') else 'reasoned_answer'
                if ir.task=='binomial' and ir.slots.get('mode','exactly')!='exactly': intent='reasoned_answer'
                return self._answer(answer,intent,min(ir.confidence,.99),'v20_learned_ir','world_consumed','graph_executed',*verdict.checks)
            # An explicitly failed invariant must never be hidden by an unverified fallback.
            answer='اطلاعات یا محاسبهٔ این مسئله قابل تأیید نیست؛ لطفاً عددها و واحدها را روشن‌تر بنویس.' if language=='fa' else 'I could not verify this calculation. Please clarify the values and units.'
            from jarvis.agent.local_intelligence_v14 import LocalIntelligenceAnswer
            return LocalIntelligenceAnswer(answer,'reasoned_answer',.25,('v20_verification_failed',*verdict.failed_checks))
        legacy=super().solve(text,language)
        if legacy is not None: return legacy
        code=self.code_v20.solve(text)
        if code is not None:
            return self._answer(code,'coding_answer',self.code_v20.last['confidence'],'v20_code_model','python_syntax','isolated_behavior_tests')
        return None
