from __future__ import annotations
from copy import deepcopy
from jarvis.agent.world_v21 import WorldStateV21 as WorldState
from jarvis.agent.execution_v20 import ExecutionError

class ReflectionLoopV21:
    def __init__(self,executor,verifier,parser,max_retries=2):
        self.executor=executor; self.verifier=verifier; self.parser=parser
        self.max_retries=max(0,min(2,max_retries)); self.calls=0
        from jarvis.agent.semantic_models_v20 import LearnedModel
        self.stage_model=LearnedModel('repair_stage')
        from jarvis.agent.models_v21 import ClassifierV21
        self.failure_model=ClassifierV21('verification_failure')

    def solve(self,ir):
        from jarvis.agent.verification_v21 import AUTHORITATIVE_SOURCE
        ir=deepcopy(ir)
        if AUTHORITATIVE_SOURCE.get() is not None:ir.source_text=AUTHORITATIVE_SOURCE.get()
        original=deepcopy(ir); attempts=[]; world=WorldState.from_ir(ir)
        for attempt in range(self.max_retries+1):
            self.calls+=1; graph=world.graph()
            try:
                candidate,trace=self.executor.execute(graph)
                verdict=self.verifier.verify(ir,candidate)
            except (ExecutionError,ArithmeticError,ValueError,TypeError,KeyError) as error:
                candidate=None; trace=[]
                from jarvis.agent.verification_v20 import Verdict
                verdict=Verdict(False,[str(error)],'Repair invalid slots','slots')
            record={'attempt':attempt,'ir':ir.to_dict(),'graph':graph,'answer':candidate,'verification':verdict.to_dict(),
                    'changed_slots':{k:v for k,v in ir.slots.items() if original.slots.get(k)!=v},'changed_plan':ir.operations!=original.operations}
            attempts.append(record)
            if verdict.passed: return candidate,verdict,attempts,world
            if attempt==self.max_retries: break
            import json
            record['v21_failure_prediction']=self.failure_model.predict(original.source_text+'\nCandidate: '+json.dumps(ir.to_dict(),ensure_ascii=False,sort_keys=True))[:3]
            stage_prediction=self.stage_model.predict('failed invariant '+' '.join(verdict.failed_checks))
            record['repair_prediction']=stage_prediction[:3]
            stage=stage_prediction[0][0] if stage_prediction and stage_prediction[0][1]>=.65 else verdict.repair_stage
            if any(x.startswith('source_') for x in verdict.failed_checks):
                stage='slots'
                record['repair_stage_constraint']='source_slot_consistency'
            if stage=='response': break
            if stage=='slots':
                revised=self.parser.repair(original.source_text,original.language) if any(x.startswith('source_') for x in verdict.failed_checks) else self.parser.parse(original.source_text,original.language)
                if revised is None or revised.to_dict()==ir.to_dict():
                    if any(x.startswith('source_') for x in verdict.failed_checks):
                        from jarvis.agent.source_semantics_v20_1 import repair_source_slots
                        revised=repair_source_slots(ir)
                    if revised is None or revised.to_dict()==ir.to_dict(): break
                ir=revised
            else:
                # Discard corrupted execution state; rebuild from the immutable source IR.
                ir=deepcopy(original)
            world=WorldState.from_ir(ir)
        return None,verdict,attempts,world
