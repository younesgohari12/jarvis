from __future__ import annotations

from jarvis.agent.local_intelligence_v16 import LocalIntelligenceV16
from jarvis.agent.semantic_ir_v17 import SemanticIRParserV17, SemanticSolverV17

class LocalIntelligenceV17(LocalIntelligenceV16):
    VERSION='1.7.0'

    @classmethod
    def matches(cls,text:str)->bool:
        return SemanticIRParserV17.parse(text) is not None or super().matches(text)

    def solve(self,text:str,language:str='fa'):
        ir=SemanticIRParserV17.parse(text)
        if ir is not None:
            solved=SemanticSolverV17.solve(ir,language)
            if solved is not None:
                answer,intent,confidence,checks=solved
                return self._answer(answer,intent,confidence,'v17_semantic_ir',f'task={ir.task}',*checks)
        return super().solve(text,language)
