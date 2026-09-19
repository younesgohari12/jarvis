"""Independent invariants, deliberately separate from graph execution."""
from __future__ import annotations
from dataclasses import dataclass,field,asdict
from decimal import Decimal,localcontext,ROUND_HALF_EVEN
from fractions import Fraction
import ast
import math
import re

@dataclass
class Verdict:
    passed: bool
    failed_checks: list[str]=field(default_factory=list)
    repair_hint: str=''
    repair_stage: str='none'
    checks: list[str]=field(default_factory=list)
    def to_dict(self): return asdict(self)

def close(a,b):
    try: return math.isfinite(float(a)) and math.isclose(float(a),float(b),rel_tol=1e-8,abs_tol=1e-9)
    except (ValueError,TypeError,OverflowError): return False

def decimal_graph(value,events,depth=0):
    if depth>3 or len(events)>64: raise ValueError('graph_limit')
    x=Decimal(str(value))
    for e in events:
        op=e['op']; v=Decimal(str(e.get('value',0)))
        if op in ('add','inventory_add','credit'): x+=v
        elif op in ('subtract','inventory_remove','debit'): x-=v
        elif op=='multiply': x*=v
        elif op=='divide': x/=v
        elif op=='percentage_add': x+=x*v/100
        elif op=='percentage_remove': x-=x*v/100
        elif op=='exponent': x=x**v
        elif op=='modulo': x=x-v*(x/v).to_integral_value(rounding='ROUND_FLOOR')
        elif op=='round': x=x.quantize(Decimal(10)**-int(v),rounding=ROUND_HALF_EVEN)
        elif op=='min': x=min(x,v)
        elif op=='max': x=max(x,v)
        elif op=='conditional':
            b=Decimal(str(e['threshold'])); cmp=e.get('comparison','gt')
            truth={'gt':x>b,'lt':x<b,'eq':x==b,'ge':x>=b,'le':x<=b}[cmp]
            x=decimal_graph(x,e.get('then',[]) if truth else e.get('else',[]),depth+1)
        else: raise ValueError('unknown_operation')
    return x

class Verifier:
    def __init__(self): self.calls=0
    def verify(self,ir,candidate):
        self.calls+=1; checks=[]; failures=[]; s=ir.slots; task=ir.task
        def check(name,condition):
            checks.append(name)
            if not condition: failures.append(name)
        from jarvis.agent.source_semantics_v20_1 import verify_source_slots
        source_checks=verify_source_slots(ir)
        for name,passed in source_checks.items(): check(name,passed)
        try:
            if ir.operations:
                with localcontext() as ctx:
                    ctx.prec=45; expected=decimal_graph(s['initial'],ir.operations)
                invariant={'finance':'balance_conservation','inventory':'inventory_conservation'}.get(task,'independent_execution')
                check(invariant,close(candidate,expected))
            elif task=='ratio':
                if s.get('query') in ('first','second'):
                    total=s['ratio_a']+s['ratio_b']
                    expected=s['total']*(s['ratio_a'] if s['query']=='first' else s['ratio_b'])/total
                    check('ratio_valid',s['ratio_a']>0 and s['ratio_b']>0 and total>0)
                    check('ratio_part_recompute',close(candidate,expected))
                else:
                    check('ratio_sum',isinstance(candidate,list) and len(candidate)==2 and close(sum(candidate),s['total']))
                    check('ratio_proportion',len(candidate)==2 and close(candidate[0]*s['ratio_b'],candidate[1]*s['ratio_a']))
                    check('ratio_valid',s['ratio_a']>0 and s['ratio_b']>0)
            elif task=='binomial':
                n,k,p=s['n'],s['k'],s['p']
                valid=int(n)==n and int(k)==k and 0<=k<=n<=200 and 0<=p<=1
                check('probability_domain',valid); check('probability_range',0<=candidate<=1)
                if valid:
                    dp=[1.0]+[0.0]*int(n)
                    for i in range(int(n)):
                        for j in range(i+1,-1,-1): dp[j]=dp[j]*(1-p)+(dp[j-1]*p if j else 0)
                    expected=sum(dp[int(k):]) if s.get('mode')=='at_least' else sum(dp[:int(k)+1]) if s.get('mode')=='at_most' else dp[int(k)]
                    check('probability_recompute',close(candidate,expected))
            elif task in ('combination','permutation'):
                n,k=int(s['n']),int(s['k']); value=Fraction(1)
                for i in range(k): value*=Fraction(n-i,i+1 if task=='combination' else 1)
                check('counting_recompute',close(candidate,value))
            elif task=='dice':
                # Inclusion-exclusion coefficient, independent of the executor's DP.
                n=int(s.get('n',2)); target=int(s['target']); coeff=0
                for j in range(n+1):
                    top=target-6*j-1
                    if top>=n-1: coeff+=(-1)**j*math.comb(n,j)*math.comb(top,n-1)
                check('dice_recompute',close(candidate,coeff/6**n))
            elif task=='independent':
                p=Fraction(1)
                for q in s['probabilities']: p*=Fraction(str(q))
                check('independent_probability',close(candidate,p) and all(0<=x<=1 for x in s['probabilities']))
            elif task=='work_rate':
                denom=s['workers_initial']*s['hours_initial']
                rhs=s['output_initial']*s['workers_target']*s['hours_target']
                check('productivity_conservation',close(candidate*denom,rhs))
                check('dimensional_consistency',ir.units.get('time')=='hour' and ir.units.get('output')=='piece')
            elif task=='age':
                base=s['age']; d=s.get('difference',0); years=s.get('years',0)
                a,b=base+years,base+d+years
                check('relative_age_preserved',close(b-a,d) and min(a,b)>=0)
                check('timeline_consistency',close(candidate,a+b if s.get('query')=='sum' else b))
            elif task=='speed':
                check('distance_conservation',s['time']>0 and close(candidate*s['time'],s['distance']))
            elif task=='scheduling': check('schedule_timeline',close(candidate-s['start'],sum(s['durations'])))
            elif task=='comparison':
                vals=s['values']
                if s.get('mode')=='mean': check('mean_conservation',close(candidate*len(vals),sum(vals)))
                else:
                    check('comparison_member',any(close(candidate,v) for v in vals))
                    check('comparison_order',all(candidate<=v for v in vals) if s.get('mode')=='min' else all(candidate>=v for v in vals))
            elif task=='sequence':
                from jarvis.agent.execution_v20 import sequence_rule
                terms=s['terms']; rule=s.get('rule') or sequence_rule(terms); n=int(s.get('n',len(terms)+1)); kind=rule['kind']
                # Closed forms for arithmetic/geometric, independent recurrence for the rest.
                if kind=='arithmetic':
                    check('observed_sequence_rule',all(close(v,terms[0]+i*rule['d']) for i,v in enumerate(terms)))
                    expected=terms[0]+(n-1)*rule['d']
                elif kind=='geometric':
                    check('observed_sequence_rule',all(close(v,terms[0]*rule['r']**i) for i,v in enumerate(terms)))
                    expected=terms[0]*rule['r']**(n-1)
                else:
                    vals=list(terms)
                    for i in range(len(vals),n): vals.append(vals[-1]+vals[-2] if kind=='recurrence' else vals[-1]+rule['d'][(i-1)%2])
                    expected=vals[n-1]
                    check('observed_sequence_rule',all(close(terms[i],terms[i-1]+terms[i-2]) for i in range(2,len(terms))) if kind=='recurrence' else all(close(terms[i]-terms[i-1],rule['d'][(i-1)%2]) for i in range(1,len(terms))))
                check('sequence_nth',close(candidate,expected))
            else: check('supported_verifier',False)
        except (KeyError,ValueError,TypeError,ArithmeticError,IndexError) as e:
            failures.append('invalid_slots_or_candidate'); checks.append(type(e).__name__)
        stage='slots' if any(x.startswith('source_') or x in ('invalid_slots_or_candidate','probability_domain','dimensional_consistency') for x in failures) else 'execution'
        return Verdict(not failures,failures,'Rebind source slots' if stage=='slots' else 'Rebuild graph from source world state',stage if failures else 'none',checks)

    def verify_text(self,text,constraints):
        failed=[]; checks=[]
        for c in constraints:
            kind=c['kind']; value=c.get('value'); ok=True
            if kind=='required': ok=value.casefold() in text.casefold()
            elif kind=='forbidden': ok=value.casefold() not in text.casefold()
            elif kind=='max_length': ok=len(text)<=value
            elif kind=='max_words': ok=len(text.split())<=value
            elif kind=='sentence_count': ok=len([s for s in re.split(r'[.!?؟]+',text) if s.strip()])==value
            elif kind=='language': ok=bool(re.search('[آ-ی]',text)) if value=='fa' else not bool(re.search('[آ-ی]',text))
            else: ok=False
            checks.append(kind)
            if not ok: failed.append(kind)
        return Verdict(not failed,failed,'Revise text constraints','response' if failed else 'none',checks)

    def verify_code(self,source):
        try: ast.parse(source); return Verdict(True,checks=['python_syntax'])
        except SyntaxError: return Verdict(False,['python_syntax'],'Repair syntax','code',['python_syntax'])
