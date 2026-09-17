from __future__ import annotations
from dataclasses import dataclass, field, asdict
from copy import deepcopy
from typing import Any

@dataclass
class SemanticIR:
    task: str
    language: str='fa'
    entities: list[dict]=field(default_factory=list)
    slots: dict[str,Any]=field(default_factory=dict)
    units: dict[str,str]=field(default_factory=dict)
    relations: list[dict]=field(default_factory=list)
    constraints: list[dict]=field(default_factory=list)
    operations: list[dict]=field(default_factory=list)
    answer_type: str='number'
    confidence: float=0.0
    source_text: str=''
    model_evidence: dict=field(default_factory=dict)
    schema_version: str='jarvis-semantic-ir-v2'

    def to_dict(self): return asdict(self)

@dataclass
class WorldState:
    domain: str
    entities: list[dict]
    state: dict
    events: list[dict]
    relations: list[dict]
    units: dict
    constraints: list[dict]

    @classmethod
    def from_ir(cls, ir: SemanticIR):
        return cls(ir.task,deepcopy(ir.entities),deepcopy(ir.slots),deepcopy(ir.operations),
                   deepcopy(ir.relations),deepcopy(ir.units),deepcopy(ir.constraints))

    def graph(self):
        # Graph is compiled solely from this task's world, never from a second text parser.
        if self.events:
            return {'initial':self.state.get('initial',0), 'nodes':deepcopy(self.events), 'domain':self.domain}
        return {'initial':None,'nodes':[{'op':self.domain,'slots':deepcopy(self.state),'units':deepcopy(self.units)}], 'domain':self.domain}
