from __future__ import annotations
from jarvis.agent.local_intelligence_v17 import LocalIntelligenceV17
from jarvis.agent.semantic_ir_v18 import SemanticIRParserV18, SemanticSolverV18

class LocalIntelligenceV18(LocalIntelligenceV17):
    VERSION='1.8.0'
    @classmethod
    def matches(cls,text:str)->bool:
        return SemanticIRParserV18.parse(text) is not None or super().matches(text)
    def solve(self,text:str,language:str='fa'):
        ir=SemanticIRParserV18.parse(text)
        if ir is not None:
            solved=SemanticSolverV18.solve(ir,language)
            if solved is not None:
                answer,intent,confidence,checks=solved
                return self._answer(answer,intent,confidence,'v18_semantic_ir',f'task={ir.task}',f'predicate={ir.predicate}',*checks)
        return super().solve(text,language)
__all__=['LocalIntelligenceV18']
