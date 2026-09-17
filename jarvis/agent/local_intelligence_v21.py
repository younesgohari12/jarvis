from jarvis.agent.local_intelligence_v20 import LocalIntelligenceV20
from jarvis.agent.parser_v21 import get_parser_v21
from jarvis.agent.world_v21 import GraphExecutorV21
from jarvis.agent.verification_v21 import UniversalVerifier
from jarvis.agent.reflection_v21 import ReflectionLoopV21

class LocalIntelligenceV21(LocalIntelligenceV20):
    VERSION='3.0.0-rc1'
    def __init__(self,*args,**kwargs):
        from jarvis.agent.local_intelligence_v19 import LocalIntelligenceV19
        LocalIntelligenceV19.__init__(self,*args,**kwargs)
        from jarvis.agent.code_intelligence_v20 import CodeIntelligenceV20
        self.code_v20=CodeIntelligenceV20();self.last_trace=None
        self.parser=get_parser_v21();self.executor=GraphExecutorV21();self.verifier=UniversalVerifier()
        self.reflection=ReflectionLoopV21(self.executor,self.verifier,self.parser)
        from jarvis.agent.code_v21 import CodeIntelligenceV21
        self.code_v21=CodeIntelligenceV21()
    def _answer(self,text,intent='reasoned_answer',confidence=.995,*checks):
        if 'v20_learned_ir' in checks:checks=(*checks,'v21_universal_verified','world_graph_v2')
        return super()._answer(text,intent,confidence,*checks)
    @classmethod
    def matches(cls,text):
        from jarvis.agent.local_intelligence_v19 import LocalIntelligenceV19
        return get_parser_v21().parse(text) is not None or LocalIntelligenceV19.matches(text)

    def solve(self,text,language='fa'):
        if '```' in text or __import__('re').search(r'تابع|function|کد|code|python|پایتون',text,__import__('re').I):
            code=self.code_v21.solve(text,language)
            if code is not None:
                self.last_trace=None
                from jarvis.agent.local_intelligence_v14 import LocalIntelligenceAnswer
                return LocalIntelligenceAnswer(code,'coding_answer',.85,('v21_code_task_model','bounded_ast_processing'))
        from jarvis.agent.verification_v21 import AUTHORITATIVE_SOURCE
        source_token=AUTHORITATIVE_SOURCE.set(text)
        try:answer=super().solve(text,language)
        finally:AUTHORITATIVE_SOURCE.reset(source_token)
        if answer and self.last_trace and self.last_trace['verification']['passed']:
            from jarvis.agent.local_intelligence_v14 import LocalIntelligenceAnswer
            from fractions import Fraction
            self.last_trace['initial_ir']=self.last_trace['ir']
            ir=self.last_trace['attempts'][-1]['ir'];self.last_trace['ir']=ir
            value=self.last_trace['attempts'][-1]['answer'];slots=ir['slots'];fmt=lambda v:f'{v:.10g}'
            if isinstance(value,list):
                separator=':' if slots.get('query')=='simplify' else ' و ' if language=='fa' else ' and '
                rendered=separator.join(fmt(v) for v in value)
            elif ir['answer_type']=='probability':
                rendered=fmt(value*100)+('٪' if language=='fa' else '%')
                if ir['task']=='binomial':
                    if slots.get('mode','exactly')=='exactly':rendered=f"C({int(slots['n'])},{int(slots['k'])}) × {slots['p']:g}^{int(slots['k'])} × (1−{slots['p']:g})^{int(slots['n']-slots['k'])} = "+rendered
                    if slots['p']==.5:rendered=str(Fraction(value).limit_denominator())+' = '+rendered
                elif ir['task']=='dice':rendered=str(Fraction(value).limit_denominator(6**int(slots['n'])))+' = '+rendered
            else:
                rendered=fmt(value)
                if ir['task'] in ('combination','permutation'):
                    prefix='C' if ir['task']=='combination' else 'P';rendered=f"{prefix}({int(slots['n'])},{int(slots['k'])}) = "+rendered
            final_verdict=self.verifier.verify_rendered(text,rendered)
            self.last_trace['rendered_verification']=final_verdict.to_dict()
            if not final_verdict.passed:
                message='پاسخ نهایی با متن مسئله سازگار نیست و تأیید نشد.' if language=='fa' else 'The final response could not be verified against the source.'
                return LocalIntelligenceAnswer(message,'reasoned_answer',.2,('v21_render_failed',))
            answer=LocalIntelligenceAnswer(rendered,answer.intent,answer.confidence,(*answer.checks,'source_rendered_answer_consistency'))
        if answer and self.last_trace is None and answer.intent in ('reasoned_answer','word_problem_answer','probability_answer'):
            from jarvis.agent.verification_v21 import COVERED
            pred=self.parser.last_prediction
            if pred and pred[0][0] in COVERED and pred[0][1]>=.60 and __import__('re').search(r'نسبت|ratio|احتمال|probability|سکه|coin|کارگر|worker|سال|aged|older|younger|انبار|inventory|حساب|balance|دنباله|sequence|سرعت|speed|شروع',text,__import__('re').I):
                from jarvis.agent.local_intelligence_v14 import LocalIntelligenceAnswer
                message='اطلاعات این مسئله برای تطبیق مستقل با پاسخ کافی نیست؛ لطفاً رابطهٔ عددها را روشن‌تر بنویس.' if language=='fa' else 'I could not independently verify the source relationships. Please clarify the quantities.'
                return LocalIntelligenceAnswer(message,'reasoned_answer',.2,('v21_source_unverified','fallback_verification_blocked'))
        return answer
