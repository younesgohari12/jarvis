"""Bounded graph executor; all numeric answers are produced here from world state."""
from __future__ import annotations
import math
from fractions import Fraction

MAX_STEPS=64

class ExecutionError(ValueError): pass

def sequence_rule(terms):
    if len(terms)<3: raise ExecutionError('sequence_needs_three_terms')
    eq=lambda a,b: math.isclose(a,b,rel_tol=1e-8,abs_tol=1e-9)
    diffs=[b-a for a,b in zip(terms,terms[1:])]
    if all(eq(d,diffs[0]) for d in diffs): return {'kind':'arithmetic','d':diffs[0]}
    if all(a!=0 for a in terms[:-1]):
        ratios=[b/a for a,b in zip(terms,terms[1:])]
        if all(eq(r,ratios[0]) for r in ratios): return {'kind':'geometric','r':ratios[0]}
    if len(terms)>=4 and all(eq(terms[i],terms[i-1]+terms[i-2]) for i in range(2,len(terms))): return {'kind':'recurrence'}
    if len(terms)>=5 and all(eq(d,diffs[i%2]) for i,d in enumerate(diffs)): return {'kind':'alternating','d':diffs[:2]}
    raise ExecutionError('sequence_ambiguous_or_unsupported')

def sequence_values(terms,n,rule=None):
    if int(n)!=n or not 1<=n<=1000: raise ExecutionError('sequence_index')
    rule=rule or sequence_rule(terms)
    result=list(terms)
    while len(result)<int(n):
        k=rule['kind']
        if k=='arithmetic': v=result[-1]+rule['d']
        elif k=='geometric': v=result[-1]*rule['r']
        elif k=='recurrence': v=result[-1]+result[-2]
        elif k=='alternating': v=result[-1]+rule['d'][(len(result)-1)%2]
        else: raise ExecutionError('sequence_rule')
        if not math.isfinite(v) or abs(v)>1e100: raise ExecutionError('numeric_limit')
        result.append(v)
    return result[:int(n)]

class GraphExecutor:
    def __init__(self): self.calls=0
    def execute(self,graph):
        self.calls+=1
        nodes=graph['nodes']
        if len(nodes)>MAX_STEPS: raise ExecutionError('graph_limit')
        value=graph['initial']; trace=[]
        for node in nodes:
            before=value
            value=self.step(value,node)
            vals=value if isinstance(value,list) else [value]
            if any(not isinstance(x,(int,float)) or not math.isfinite(x) or abs(x)>1e100 for x in vals):
                raise ExecutionError('nonfinite_or_limit')
            trace.append({'op':node['op'],'before':before,'after':value})
        return value,trace

    def step(self,x,node,depth=0):
        if depth>3: raise ExecutionError('branch_depth_limit')
        op=node['op']; v=node.get('value'); s=node.get('slots',{})
        if op in ('add','inventory_add','credit'): return x+v
        if op in ('subtract','inventory_remove','debit'): return x-v
        if op=='multiply': return x*v
        if op=='divide':
            if v==0: raise ExecutionError('division_by_zero')
            return x/v
        if op=='percentage_add': return x*(1+v/100)
        if op=='percentage_remove': return x*(1-v/100)
        if op=='exponent':
            if abs(v)>100: raise ExecutionError('exponent_limit')
            return x**v
        if op=='modulo':
            if v==0: raise ExecutionError('division_by_zero')
            return x%v
        if op=='round':
            if int(v)!=v or abs(v)>12: raise ExecutionError('round_precision')
            from decimal import Decimal,ROUND_HALF_EVEN
            return float(Decimal(str(x)).quantize(Decimal(10)**-int(v),rounding=ROUND_HALF_EVEN))
        if op=='min': return min(x,v)
        if op=='max': return max(x,v)
        if op=='conditional':
            cmp=node.get('comparison','gt'); boundary=node['threshold']
            pred={'gt':x>boundary,'lt':x<boundary,'eq':x==boundary,'ge':x>=boundary,'le':x<=boundary}.get(cmp)
            if pred is None: raise ExecutionError('invalid_condition')
            branch=node.get('then',[]) if pred else node.get('else',[])
            if len(branch)>MAX_STEPS: raise ExecutionError('graph_limit')
            for action in branch: x=self.step(x,action,depth+1)
            return x
        if op=='ratio':
            a,b=s['ratio_a'],s['ratio_b']; total=s['total']
            if a<=0 or b<=0: raise ExecutionError('invalid_ratio')
            return [total*a/(a+b),total*b/(a+b)]
        if op=='binomial':
            n,k,p=s['n'],s['k'],s['p']
            if not (int(n)==n and int(k)==k and 0<=k<=n<=200 and 0<=p<=1): raise ExecutionError('probability_domain')
            mode=s.get('mode','exactly')
            ks=range(int(k),int(n)+1) if mode=='at_least' else range(int(k)+1) if mode=='at_most' else [int(k)]
            return sum(math.comb(int(n),i)*p**i*(1-p)**(n-i) for i in ks)
        if op in ('combination','permutation'):
            n,k=s['n'],s['k']
            if int(n)!=n or int(k)!=k or not 0<=k<=n<=100: raise ExecutionError('counting_domain')
            return (math.comb if op=='combination' else math.perm)(int(n),int(k))
        if op=='dice':
            n,target=int(s.get('n',2)),int(s['target'])
            if not 1<=n<=12: raise ExecutionError('dice_limit')
            counts={0:1}
            for _ in range(n):
                new={}
                for total,c in counts.items():
                    for face in range(1,7): new[total+face]=new.get(total+face,0)+c
                counts=new
            return counts.get(target,0)/6**n
        if op=='independent':
            ps=s['probabilities']
            if not all(0<=p<=1 for p in ps): raise ExecutionError('probability_domain')
            return math.prod(ps)
        if op=='sequence':
            vals=sequence_values(s['terms'],s.get('n',len(s['terms'])+1),s.get('rule'))
            return vals[-1]
        if op=='work_rate':
            w,h,o,w2,h2=(s[k] for k in ('workers_initial','hours_initial','output_initial','workers_target','hours_target'))
            if min(w,h)<=0 or min(o,w2,h2)<0: raise ExecutionError('work_domain')
            return o/(w*h)*w2*h2
        if op=='age':
            age,diff,years=s['age'],s.get('difference',0),s.get('years',0)
            a=age+years; b=age+diff+years
            if min(a,b)<0: raise ExecutionError('negative_age')
            return a+b if s.get('query')=='sum' else b
        if op=='speed':
            distance,time=s['distance'],s['time']
            if time<=0 or distance<0: raise ExecutionError('speed_domain')
            return distance/time
        if op=='scheduling': return s['start']+sum(s['durations'])
        if op=='comparison':
            if s.get('mode')=='mean': return sum(s['values'])/len(s['values'])
            return (min if s.get('mode')=='min' else max)(s['values'])
        raise ExecutionError('unsupported_operation:'+str(op))
