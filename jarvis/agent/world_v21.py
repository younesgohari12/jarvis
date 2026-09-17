"""Typed world state and bounded causal graph; copied state prevents cross-turn leaks."""
from copy import deepcopy
from dataclasses import dataclass,field,asdict
import math
from jarvis.agent.execution_v20 import GraphExecutor,ExecutionError

@dataclass
class WorldStateV21:
    domain:str
    entities:list
    state:dict
    events:list
    relations:list
    units:dict
    constraints:list
    @classmethod
    def from_ir(cls,ir):
        entities=deepcopy(ir.entities);relations=deepcopy(ir.relations)
        if not entities:
            typ={'finance':'money','inventory':'products','age':'person','speed':'locations','scheduling':'time','work_rate':'workers'}.get(ir.task,'quantity')
            entities=[{'id':'subject','type':typ}]
        constraints=deepcopy(ir.constraints)
        if ir.task=='inventory':constraints.append({'type':'nonnegative','field':'value'})
        if ir.task in ('finance','inventory'):relations.append({'subject':entities[0]['id'],'predicate':'owns','object':'balance'})
        if ir.task=='work_rate':relations.append({'subject':'crew_initial','predicate':'produces','object':'output_initial'})
        events=deepcopy(ir.operations)
        for i,e in enumerate(events):
            e.setdefault('id',f'event_{i}')
            if i:relations.append({'subject':f'event_{i-1}','predicate':'before','object':e['id']})
        return cls(ir.task,entities,deepcopy(ir.slots),events,relations,deepcopy(ir.units),constraints)
    def graph(self):
        nodes=deepcopy(self.events) or [{'id':'solve','op':self.domain,'slots':deepcopy(self.state),'units':deepcopy(self.units)}]
        by_id={n['id']:n for n in nodes}
        for relation in self.relations:
            if relation.get('predicate')=='before' and relation['object'] in by_id:
                by_id[relation['object']].setdefault('depends_on',[]).append(relation['subject'])
        return {'version':2,'domain':self.domain,'initial':self.state.get('initial'),'nodes':nodes,'entities':deepcopy(self.entities),'relations':deepcopy(self.relations),'constraints':deepcopy(self.constraints),'state':deepcopy(self.state)}

class GraphExecutorV21(GraphExecutor):
    def step(self,x,node,depth=0):
        s=node.get('slots',{})
        if node['op']=='age' and s.get('query')=='base':return s['age']+s['years']
        if node['op']=='ratio' and s.get('query')=='simplify':
            g=math.gcd(int(s['ratio_a']),int(s['ratio_b']));return [s['ratio_a']/g,s['ratio_b']/g]
        if node['op']=='work_rate' and s.get('query')=='combined_time':return 1/sum(1/v for v in s['durations'])
        if node['op']=='speed' and node.get('slots',{}).get('query')=='distance':
            return node['slots']['speed']*node['slots']['time']
        return super().step(x,node,depth)
    def execute(self,graph):
        if graph.get('version')!=2:return super().execute(graph)
        self.calls+=1;nodes=graph['nodes']
        if len(nodes)>64:raise ExecutionError('graph_limit')
        value=graph['initial'];state=deepcopy(graph.get('state',{}));done={};trace=[]
        balances=state.get('balances',{})
        for i,node in enumerate(nodes):
            before=deepcopy(value);op=node['op'];nid=node.get('id',str(i))
            if nid in done:raise ExecutionError('duplicate_event')
            if any(dep not in done for dep in node.get('depends_on',[])):raise ExecutionError('unsatisfied_dependency')
            if op=='purchase':
                if not 0<=node['discount']<=100 or node['value']<0:raise ExecutionError('purchase_domain')
                value-=node['value']*(1-node['discount']/100)
            elif op=='compare':
                a=node.get('left',value);b=node['right'];comparison=node.get('comparison','gt')
                if comparison not in ('gt','lt','eq','ge','le'):raise ExecutionError('comparison_kind')
                value=float({'gt':a>b,'lt':a<b,'eq':a==b,'ge':a>=b,'le':a<=b}[comparison])
            elif op=='change_over_time':value+=node['rate']*node['duration']
            elif op=='dependency':
                if node['event'] not in done:raise ExecutionError('unsatisfied_dependency')
                value=done[node['event']]
            elif op=='ownership':
                if node['owner'] in balances:raise ExecutionError('ownership_redefinition')
                balances[node['owner']]=node['value'];value=node['value']
            elif op=='transfer':
                sender,receiver=node['from'],node['to'];amount=node['value'];total=sum(balances.values())
                if sender not in balances or receiver not in balances or amount<0 or balances[sender]<amount:raise ExecutionError('impossible_transfer')
                balances[sender]-=amount;balances[receiver]+=amount
                if not math.isclose(total,sum(balances.values())):raise ExecutionError('conservation_failure')
                value=balances[state.get('query',sender)]
            elif op=='constraint_check':
                if node.get('kind')=='nonnegative' and value<0:raise ExecutionError('impossible_state')
                if node.get('kind')=='maximum' and value>node['value']:raise ExecutionError('capacity_exceeded')
                if node.get('kind') not in ('nonnegative','maximum'):raise ExecutionError('unsupported_constraint')
            elif op=='cause_effect':
                cause=node['cause']
                if cause not in done:raise ExecutionError('unobserved_cause')
                if bool(done[cause]):value=self.step(value,node['effect'])
            else:value=self.step(value,node)
            values=value if isinstance(value,list) else [value]
            if any(not isinstance(x,(int,float)) or not math.isfinite(x) or abs(x)>1e100 for x in values):raise ExecutionError('numeric_limit')
            for c in graph.get('constraints',[]):
                if c['type']=='nonnegative' and any(x<0 for x in values):raise ExecutionError('impossible_state')
            done[nid]=deepcopy(value)
            trace.append({'id':nid,'op':op,'before':before,'after':deepcopy(value),'balances':deepcopy(balances)})
        return value,trace
