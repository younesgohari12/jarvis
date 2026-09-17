"""JARVIS v22 — Semantic IR V2.

Upgrades the v21 SemanticIR into the unified typed schema required by the
spec: typed quantities, expressive numeric roles, slot provenance
(source_span per slot), temporal context, requested output and ambiguity
flags. This is an audit/upgrade layer — the v21 IR remains the execution
source of truth and is never mutated.
"""
from __future__ import annotations
from dataclasses import dataclass, field, asdict
from typing import Any

from jarvis.agent.quantity_v22 import Quantity, extract_typed_quantities
from jarvis.agent.numeric_roles_v22 import classify_source_numbers_v2

SCHEMA_VERSION = 'jarvis-semantic-ir-v22'


@dataclass
class SemanticIRV2:
    task: str
    language: str = 'fa'
    source_text: str = ''
    quantities: list[dict] = field(default_factory=list)
    numeric_roles: list[dict] = field(default_factory=list)
    entities: list[dict] = field(default_factory=list)
    events: list[dict] = field(default_factory=list)
    relations: list[dict] = field(default_factory=list)
    constraints: list[dict] = field(default_factory=list)
    operations: list[dict] = field(default_factory=list)
    slots: dict[str, Any] = field(default_factory=dict)
    units: dict[str, str] = field(default_factory=dict)
    source_mapping: dict[str, str] = field(default_factory=dict)  # slot -> source span
    requested_output: str = 'number'
    temporal_context: dict = field(default_factory=dict)
    assumptions: list[str] = field(default_factory=list)
    ambiguity: dict = field(default_factory=dict)
    confidence: float = 0.0
    schema_version: str = SCHEMA_VERSION

    def to_dict(self):
        return asdict(self)

    # ------------------------------------------------------------------
    @classmethod
    def from_v21(cls, ir, text: str) -> 'SemanticIRV2':
        """Build the V2 view from a v21 SemanticIR + the immutable source."""
        ir_dict = ir.to_dict() if hasattr(ir, 'to_dict') else dict(ir)
        quantities = [q.to_dict() for q in extract_typed_quantities(text)]
        roles = classify_source_numbers_v2(text)
        mapping = _slot_spans(ir_dict, roles)
        temporal = _temporal_context(ir_dict)
        requested = _requested_output(ir_dict)
        return cls(
            task=ir_dict.get('task', 'unknown'),
            language=ir_dict.get('language', 'fa'),
            source_text=text,
            quantities=quantities,
            numeric_roles=roles,
            entities=ir_dict.get('entities', []) or [],
            events=list(ir_dict.get('operations', []) or []),
            relations=ir_dict.get('relations', []) or [],
            constraints=ir_dict.get('constraints', []) or [],
            operations=list(ir_dict.get('operations', []) or []),
            slots=ir_dict.get('slots', {}) or {},
            units=ir_dict.get('units', {}) or {},
            source_mapping=mapping,
            requested_output=requested,
            temporal_context=temporal,
            assumptions=_assumptions(ir_dict),
            ambiguity=_ambiguity(ir_dict, quantities),
            confidence=float(ir_dict.get('confidence', 0.0) or 0.0),
        )


# ----------------------------------------------------------------------
def _slot_spans(ir_dict: dict, roles: list[dict]) -> dict[str, str]:
    """Map each numeric slot to the exact source span it came from."""
    mapping: dict[str, str] = {}
    slots = ir_dict.get('slots', {}) or {}
    used: set[int] = set()
    for key, value in slots.items():
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            for i, f in enumerate(roles):
                if i in used:
                    continue
                if abs(float(f['value']) - float(value)) < 1e-12:
                    mapping[str(key)] = f.get('span', '')
                    used.add(i)
                    break
        elif isinstance(value, list):
            spans = []
            pool = [f for i, f in enumerate(roles) if i not in used]
            for v in value:
                if isinstance(v, (int, float)):
                    for f in pool:
                        if abs(float(f['value']) - float(v)) < 1e-12:
                            spans.append(f.get('span', ''))
                            pool.remove(f)
                            break
            if spans:
                mapping[str(key)] = ' | '.join(spans)
    return mapping


def _temporal_context(ir_dict: dict) -> dict:
    task = ir_dict.get('task')
    slots = ir_dict.get('slots', {}) or {}
    units = ir_dict.get('units', {}) or {}
    if task == 'scheduling' or 'start' in slots:
        return {'kind': 'clock_time', 'unit': units.get('time', 'hour'),
                'calendar_wrap': True}
    return {}


def _requested_output(ir_dict: dict) -> str:
    at = ir_dict.get('answer_type', 'number')
    slots = ir_dict.get('slots', {}) or {}
    q = slots.get('query')
    if q in ('distance', 'simplify', 'combined_time', 'base'):
        return str(q)
    return str(at or 'number')


def _assumptions(ir_dict: dict) -> list[str]:
    out = []
    if ir_dict.get('task') == 'work_rate':
        out.append('constant_product_rule: workers*hours output scales linearly')
    if ir_dict.get('units', {}).get('output') == 'piece':
        out.append('output_unit: piece')
    return out


def _ambiguity(ir_dict: dict, quantities: list[dict]) -> dict:
    dims = {q['dimension'] for q in quantities if q['dimension'] != 'dimensionless'}
    mixed = 'currency' in dims and 'count' in dims
    return {'mixed_dimensions': mixed,
            'dimensions': sorted(dims),
            'flag': 'dimension_mix' if mixed else ''}
