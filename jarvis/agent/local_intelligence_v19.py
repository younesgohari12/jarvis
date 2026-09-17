from __future__ import annotations
from jarvis.agent.local_intelligence_v18 import LocalIntelligenceV18
from jarvis.agent.semantic_ir_v19 import SemanticIRParserV19, SemanticSolverV19

class LocalIntelligenceV19(LocalIntelligenceV18):
    VERSION='1.9.1'
    @classmethod
    def matches(cls,text:str)->bool:
        return SemanticIRParserV19.parse(text) is not None or super().matches(text)
    def solve(self,text:str,language:str='fa'):
        ir=SemanticIRParserV19.parse(text)
        if ir is not None:
            solved=SemanticSolverV19.solve(ir,language)
            if solved is not None:
                answer,intent,confidence,checks=solved
                return self._answer(answer,intent,confidence,'v19_semantic_ir',f'task={ir.task}',f'predicate={ir.predicate}',*checks)
        return super().solve(text,language)

__all__=['LocalIntelligenceV19']
