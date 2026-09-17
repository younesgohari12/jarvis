from functools import lru_cache
from jarvis.agent.parser_v20 import SemanticParser
from jarvis.agent.cognitive_ir_v20 import SemanticIR
from jarvis.agent.source_facts_v21 import extract_source,source_answer,UnsupportedSource
from jarvis.agent.execution_v20 import sequence_rule
import re

def facts_to_ir(f,source,language,prediction=None):
    slots=dict(f.slots)
    if f.domain=='sequence' and 'rule' not in slots:slots['rule']=sequence_rule(slots['terms'])
    ir=SemanticIR(f.domain,language,slots=slots,operations=f.operations,entities=f.entities,relations=f.relations,constraints=f.constraints,units=f.units,source_text=source,confidence=.91)
    ir.model_evidence={'source_graph_v21':f.to_dict(),'frame':prediction or [],'source_constrained_parse':True}
    if f.domain in ('binomial','dice','independent'):ir.answer_type='probability'
    if f.domain=='ratio':ir.answer_type='parts'
    return ir

class SemanticParserV21(SemanticParser):
    def __init__(self):
        super().__init__()
        from jarvis.agent.models_v21 import ClassifierV21
        self.numeric_v21=ClassifierV21('numeric_binding')
    def tag(self,text,family):
        tags=super().tag(text,family)
        if family in ('graph','finance','inventory') and len(tags)>1:
            from jarvis.agent.numeric_v21 import numeric_context
            for item in tags:
                if item['confidence']>=.60:continue
                pred=self.numeric_v21.predict(numeric_context(text,(item['start'],item['end']),family))
                item['v21_prediction']=pred[:3]
                if pred and pred[0][1]>=.90 and item['confidence']<.60:
                    item['previous_role']=item['role'];item['role'],item['confidence']=pred[0]
        return tags
    def parse(self,text,language='fa'):
        if re.search(r'```|\b(?:def|class)\s+\w+|\bprint\s*\(|\b\w+\s*=',text):return super().parse(text,language)
        if re.search(r'یکتا|قطعی|unique|ambiguous|explain why|توضیح بده چرا|میانگین وزنی|weighted',text):return super().parse(text,language)
        legacy=super().parse(text,language)
        if legacy is not None:return legacy
        try:
            facts=extract_source(text);source_answer(facts)
            # Literal source grammar supplements, and is explicitly labeled apart from learning.
            return facts_to_ir(facts,text,language,self.last_prediction[:3])
        except (UnsupportedSource,ValueError,ArithmeticError,KeyError,TypeError):return None
    def repair(self,text,language='fa'):
        try:return facts_to_ir(extract_source(text),text,language,self.last_prediction[:3])
        except (UnsupportedSource,ValueError,ArithmeticError,KeyError,TypeError):return None

@lru_cache(maxsize=1)
def get_parser_v21():return SemanticParserV21()
