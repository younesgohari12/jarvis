"""Source-grounded constraints. No answers, training examples or benchmark keys.

The parser binds learned spans. The verifier independently reads unit-bearing
facts from source text; it never treats cached model labels as source truth.
"""
from __future__ import annotations
import math
import re
from jarvis.agent.normalization_v20 import normalize, NUMBER

BOUNDARY=re.compile(r'(?<!\d)\.(?!\d)|[!?؟]|[;؛]|\b(?:now|then|whereas|if)\b|حالا|اکنون|اگر')

class SourceAmbiguity(ValueError): pass

def _cut_strength(gap):
    if re.search(r'(?<!\d)\.(?!\d)|[!?؟]',gap): return 4
    if re.search(r'\b(?:now|then|whereas|if)\b|حالا|اکنون|اگر',gap): return 3
    if re.search('[;؛]',gap): return 2
    if re.search('[,،]',gap): return 1
    return 0

def bind_work_spans(text,tags):
    """Bind whole observations together; never sort workers and hours separately."""
    facts=sorted([dict(x) for x in tags if x['role'] in ('workers','hours','output') and x['confidence']>=.24],key=lambda x:x['start'])
    if len(facts)!=5: raise SourceAmbiguity('work_fact_count')
    candidates=[]
    for cut in (2,3):
        groups=(facts[:cut],facts[cut:])
        if not all([x['role'] for x in g].count('workers')==1 and [x['role'] for x in g].count('hours')==1 for g in groups): continue
        if sum(x['role']=='output' for x in facts)!=1: continue
        gap=text[facts[cut-1]['end']:facts[cut]['start']]
        candidates.append((_cut_strength(gap),groups))
    if not candidates: raise SourceAmbiguity('work_interleaved_observations')
    candidates.sort(key=lambda x:-x[0])
    if len(candidates)>1 and candidates[0][0]==candidates[1][0]: raise SourceAmbiguity('work_ambiguous_observation_boundary')
    groups=candidates[0][1]; slots={}; evidence={}
    for group in groups:
        known=any(x['role']=='output' for x in group)
        for fact in group:
            role=fact['role']; key='output_initial' if role=='output' else role+('_initial' if known else '_target')
            scale=1.0
            if role=='hours':
                right=text[fact['end']:fact['end']+24]
                if re.match(r'\s*(?:minutes?|دقیقه)',right): scale=1/60
                elif re.match(r'\s*(?:seconds?|ثانیه)',right): scale=1/3600
            slots[key]=fact['value']*scale
            evidence[key]={'start':fact['start'],'end':fact['end'],'source_value':fact['value'],'scale':scale,'observation':'known' if known else 'target'}
    return slots,evidence

def source_work_facts(source):
    """Independent verifier extraction: explicit unit or field label, not neural tags."""
    t=normalize(source); found=[]
    for number in NUMBER.finditer(t):
        left=t[max(0,number.start()-45):number.start()]; right=t[number.end():number.end()+45]
        roles=[]
        if re.match(r'\s*(?:کارگر\w*|نفر(?:\s+(?:نیرو|کارگر))?|نیروی\s+کار|machine\w*|robot\w*|tap\w*|pipe\w*|pump\w*|printer\w*|workers?\b|people\b|operators?\b|technicians?\b|employees?\b|staff\b|crew\w*|developers?\b|programmers?\b|bakers?\b|packers?\b|painters?\b|drivers?\b|assemblers?\b)',right) or re.search(r'(?:(?:کارگر|نیروی)\s*(?:=|:)|(?:تعداد\s+کارگر|worker count|staffing)\s*(?:is|=|:)?)\s*$',left): roles.append('workers')
        if re.match(r'\s*(?:ساعت\w*|دقیقه|ثانیه|hours?\b|minutes?\b|seconds?\b)',right) or re.search(r'(?:مدت|زمان|duration|time)\s*(?:is|=|:)??\s*$',left): roles.append('hours')
        if re.match(r'\s*(?:قطعه|کالا|واحد|جعبه|boxes\b|pieces?\b|units?\b|items?\b|widgets?\b|gadgets?\b|products?\b|components?\b|parts\b|bricks?\b|bottles?\b|cans\b|jars\b|crates\b|trays\b|toys?\b|dolls?\b|balls\b|cakes?\b|cookies\b|cupcakes\b|pizzas?\b|sandwiches\b|meals?\b|chairs?\b|tables?\b|desks\b|shelves\b|shirts?\b|trousers\b|shoes\b|bags?\b|hats\b|cars\b|trucks?\b|bikes\b|bicycles?\b|engines\b|tools\b|laptops?\b|phones\b|cameras?\b|tickets?\b|packages?\b|parcels?\b|orders?\b|reports?\b|documents?\b|invoices\b|signs\b|banners?\b|posters?\b|flyers\b|booklets\b|notebooks?\b|pencils?\b|pens\b|mugs\b|cups\b|plates?\b|bowls\b|spoons\b|frames?\b|gears?\b|valves?\b|bolts?\b|screws\b|nails\b|panels?\b|modules\b|circuits?\b)',right) or re.search(r'(?:خروجی|تولید|output|production)\s*(?:is|was|=|:)??\s*$',left): roles.append('output')
        roles=list(dict.fromkeys(roles))
        if len(roles)>1: raise SourceAmbiguity('conflicting_source_units')
        if not roles: continue
        role=roles[0]; scale=1.0
        if role=='hours' and re.match(r'\s*(?:minutes?|دقیقه)',right): scale=1/60
        if role=='hours' and re.match(r'\s*(?:seconds?|ثانیه)',right): scale=1/3600
        found.append((number.start(),number.end(),role,float(number.group())*scale))
    if len(found)!=5: raise SourceAmbiguity('source_work_coverage')
    # Check contiguous source assertions on either side of each possible relation boundary.
    alternatives=[]
    for split in range(1,len(found)):
        left,right=found[:split],found[split:]
        ltypes=[x[2] for x in left]; rtypes=[x[2] for x in right]
        if sorted(ltypes)==['hours','output','workers'] and sorted(rtypes)==['hours','workers']: known,target=left,right
        elif sorted(rtypes)==['hours','output','workers'] and sorted(ltypes)==['hours','workers']: known,target=right,left
        else: continue
        separator=t[found[split-1][1]:found[split][0]]
        # This decision is independent of the parser's spans and selected group.
        boundaries=list(BOUNDARY.finditer(separator))
        rank=max((4 if m.group() in '.!?؟' else 2 if m.group() in ';؛' else 3 for m in boundaries),default=1 if re.search('[,،]',separator) else 0)
        expect={x[2]+'_initial':x[3] for x in known}
        expect.update({x[2]+'_target':x[3] for x in target})
        alternatives.append((rank,expect))
    if not alternatives: raise SourceAmbiguity('source_work_relation')
    alternatives.sort(key=lambda x:-x[0])
    if len(alternatives)>1 and alternatives[0][0]==alternatives[1][0]: raise SourceAmbiguity('source_work_relation_ambiguous')
    return alternatives[0][1]

def sequence_source(source):
    """Read explicit list and requested index as task constraints with source spans."""
    t=normalize(source)
    lists=[]
    pattern=re.compile(r'(?<![\w.])-?\d+(?:\.\d+)?(?:\s*[,،]\s*-?\d+(?:\.\d+)?){2,}')
    for run in pattern.finditer(t):
        items=[{'value':float(m.group()),'start':run.start()+m.start(),'end':run.start()+m.end()} for m in NUMBER.finditer(run.group())]
        lists.append(items)
    if len(lists)>1: raise SourceAmbiguity('multiple_sequence_lists')
    explicit=[]
    # Normalized ordinal forms preserve numeric spans: fifth -> 5, پنجم -> 5.
    after=re.compile(r'\b(?:جمله|ترم|عضو|term|member|position|index)\s*(?:ی\s+)?(?:شماره\s+|number\s+)?(?P<n>\d+)(?:\s*(?:ام|م))?\b')
    before=re.compile(r'\b(?P<n>\d+)\s+(?:term|member)\b')
    for pat in (after,before):
        for m in pat.finditer(t):
            # An initial term with a separately stated value is an observation.
            left=t[max(0,m.start()-5):m.start()]
            right=t[m.end():]
            if pat is after and re.search(r'\b1\s*$',left): continue
            if int(m.group('n'))==1 and re.match(r'\s*(?:is|=|برابر)?\s*-?\d',right): continue
            if lists and any(x['start']<=m.start('n')<x['end'] for x in lists[0]): continue
            explicit.append({'value':int(m.group('n')),'start':m.start('n'),'end':m.end('n')})
    # Prefer a separately requested nth term to a descriptive first-term mention.
    nonfirst=[x for x in explicit if x['value']!=1]
    if nonfirst: explicit=nonfirst
    if len({x['value'] for x in explicit})>1: raise SourceAmbiguity('conflicting_sequence_targets')
    target=explicit[-1] if explicit else None
    if target and not 1<=target['value']<=1000: raise SourceAmbiguity('sequence_target_range')
    return {'terms':lists[0] if lists else [],'target':target,'normalized':t,
            'request_kind':'nth' if target else 'next'}

def verify_source_slots(ir):
    """Return source checks before arithmetic; source-less solver unit tests are explicit."""
    checks={}
    if not ir.source_text: return checks
    try:
        if ir.task=='work_rate':
            expected=source_work_facts(ir.source_text)
            for slot,value in expected.items():
                checks['source_slot:'+slot]=slot in ir.slots and math.isclose(float(ir.slots[slot]),value,rel_tol=1e-10,abs_tol=1e-10)
            checks['source_work_coverage']=True
        elif ir.task=='sequence':
            source=sequence_source(ir.source_text)
            if source['terms']:
                expected=[x['value'] for x in source['terms']]
                checks['source_sequence_terms']=ir.slots.get('terms')==expected
                if source['target'] is None: checks['source_sequence_next']=ir.slots.get('n')==len(expected)+1
            if source['target'] is not None: checks['source_sequence_target']=ir.slots.get('n')==source['target']['value']
    except (SourceAmbiguity,TypeError,ValueError): checks['source_semantic_coverage']=False
    return checks

def repair_source_slots(ir):
    from copy import deepcopy
    from jarvis.agent.execution_v20 import sequence_rule
    revised=deepcopy(ir)
    try:
        if ir.task=='work_rate': revised.slots.update(source_work_facts(ir.source_text))
        elif ir.task=='sequence':
            source=sequence_source(ir.source_text)
            if source['terms']:
                terms=[x['value'] for x in source['terms']]
                revised.slots.update(terms=terms,rule=sequence_rule(terms),n=source['target']['value'] if source['target'] else len(terms)+1)
            elif source['target']: revised.slots['n']=source['target']['value']
            else: return None
        else: return None
    except (SourceAmbiguity,ValueError,KeyError): return None
    revised.model_evidence['source_constraint_repair']=True
    return revised
