"""Composable code generation and bounded Python AST understanding.

No eval/exec, imports, attribute access, filesystem or network execution.
The AST interpreter accepts a documented subset and enforces a step budget.
"""
import ast,operator,re,copy

class CodeLimit(ValueError):pass
OPS={ast.Add:operator.add,ast.Sub:operator.sub,ast.Mult:operator.mul,ast.Div:operator.truediv,ast.FloorDiv:operator.floordiv,ast.Mod:operator.mod,ast.Pow:operator.pow}
CMPS={ast.Eq:operator.eq,ast.NotEq:operator.ne,ast.Lt:operator.lt,ast.LtE:operator.le,ast.Gt:operator.gt,ast.GtE:operator.ge,ast.In:lambda x,y:x in y,ast.NotIn:lambda x,y:x not in y}
class ReturnValue(Exception):
    def __init__(self,value):self.value=value
class SafePython:
    def __init__(self,budget=10000):self.budget=budget
    def tick(self):
        self.budget-=1
        if self.budget<0:raise CodeLimit('step_budget')
    def run(self,source,args):
        tree=ast.parse(source)
        if len(list(ast.walk(tree)))>1000:raise CodeLimit('ast_size')
        if any(isinstance(n,(ast.Import,ast.ImportFrom,ast.Global,ast.Nonlocal,ast.While,ast.Lambda,ast.With,ast.ClassDef,ast.AsyncFunctionDef)) for n in ast.walk(tree)):raise CodeLimit('unsupported_python')
        functions=[n for n in tree.body if isinstance(n,ast.FunctionDef)]
        if len(functions)!=1:raise CodeLimit('single_function_required')
        fn=functions[0]
        if fn.decorator_list or fn.args.vararg or fn.args.kwarg:raise CodeLimit('signature')
        if len(args)!=len(fn.args.args):raise CodeLimit('argument_count')
        env=dict(zip([a.arg for a in fn.args.args],args))
        try:self.block(fn.body,env)
        except ReturnValue as r:return r.value
        return None
    def assign(self,node,value,env):
        if isinstance(node,ast.Name):env[node.id]=value
        elif isinstance(node,(ast.Tuple,ast.List)):
            if len(node.elts)!=len(value):raise CodeLimit('unpack')
            for n,v in zip(node.elts,value):self.assign(n,v,env)
        else:raise CodeLimit('assignment_target')
    def block(self,body,env):
        for n in body:
            self.tick()
            if isinstance(n,ast.Return):raise ReturnValue(self.expr(n.value,env) if n.value else None)
            if isinstance(n,ast.Assign):
                v=self.expr(n.value,env)
                for target in n.targets:self.assign(target,v,env)
            elif isinstance(n,ast.AugAssign):self.assign(n.target,self.binop(n.op,self.expr(n.target,env),self.expr(n.value,env)),env)
            elif isinstance(n,ast.For):
                iterable=self.expr(n.iter,env)
                if len(iterable)>1000:raise CodeLimit('iteration_size')
                for v in iterable:self.assign(n.target,v,env);self.block(n.body,env)
                self.block(n.orelse,env)
            elif isinstance(n,ast.If):self.block(n.body if self.expr(n.test,env) else n.orelse,env)
            elif isinstance(n,ast.Expr):self.expr(n.value,env)
            elif isinstance(n,ast.Pass):pass
            else:raise CodeLimit('statement:'+type(n).__name__)
    def binop(self,op,a,b):
        if type(op) not in OPS:raise CodeLimit('operator')
        if isinstance(op,ast.Pow) and abs(b)>12:raise CodeLimit('power_limit')
        if isinstance(op,ast.Mult) and isinstance(a,(str,list,tuple)) and len(a)*abs(b)>10000:raise CodeLimit('allocation_limit')
        value=OPS[type(op)](a,b)
        if isinstance(value,(int,float)) and abs(value)>1e100:raise CodeLimit('numeric_limit')
        return value
    def expr(self,n,env):
        self.tick()
        if isinstance(n,ast.Constant):
            if not isinstance(n.value,(int,float,str,bool,type(None))) or isinstance(n.value,str) and len(n.value)>10000:raise CodeLimit('constant')
            return n.value
        if isinstance(n,ast.Name):
            if n.id.startswith('_'):raise CodeLimit('private_name')
            return env[n.id]
        if isinstance(n,(ast.List,ast.Tuple,ast.Set)):
            v=[self.expr(x,env) for x in n.elts];return tuple(v) if isinstance(n,ast.Tuple) else set(v) if isinstance(n,ast.Set) else v
        if isinstance(n,ast.Dict):return {self.expr(k,env):self.expr(v,env) for k,v in zip(n.keys,n.values)}
        if isinstance(n,ast.BinOp):return self.binop(n.op,self.expr(n.left,env),self.expr(n.right,env))
        if isinstance(n,ast.UnaryOp):
            v=self.expr(n.operand,env)
            if isinstance(n.op,ast.USub):return -v
            if isinstance(n.op,ast.UAdd):return +v
            if isinstance(n.op,ast.Not):return not v
            raise CodeLimit('unary')
        if isinstance(n,ast.BoolOp):
            result=self.expr(n.values[0],env)
            for x in n.values[1:]:
                if isinstance(n.op,ast.And) and not result or isinstance(n.op,ast.Or) and result:break
                result=self.expr(x,env)
            return result
        if isinstance(n,ast.Compare):
            a=self.expr(n.left,env)
            for op,x in zip(n.ops,n.comparators):
                b=self.expr(x,env)
                if type(op) not in CMPS:raise CodeLimit('comparison')
                if not CMPS[type(op)](a,b):return False
                a=b
            return True
        if isinstance(n,ast.IfExp):return self.expr(n.body if self.expr(n.test,env) else n.orelse,env)
        if isinstance(n,ast.Subscript):return self.expr(n.value,env)[self.expr(n.slice,env)]
        if isinstance(n,ast.Slice):return slice(*(self.expr(x,env) if x else None for x in (n.lower,n.upper,n.step)))
        if isinstance(n,(ast.ListComp,ast.GeneratorExp,ast.SetComp)):
            if len(n.generators)!=1 or n.generators[0].is_async:raise CodeLimit('comprehension')
            g=n.generators[0];out=[];local=dict(env);iterable=self.expr(g.iter,env)
            if len(iterable)>1000:raise CodeLimit('iteration_size')
            for value in iterable:
                self.tick();self.assign(g.target,value,local)
                if all(self.expr(c,local) for c in g.ifs):out.append(self.expr(n.elt,local))
            return set(out) if isinstance(n,ast.SetComp) else out
        if isinstance(n,ast.Call):
            if n.keywords:raise CodeLimit('keyword_call')
            args=[self.expr(x,env) for x in n.args]
            if isinstance(n.func,ast.Attribute):
                if n.func.attr!='append' or not isinstance(n.func.value,ast.Name):raise CodeLimit('attribute_call')
                target=env[n.func.value.id]
                if type(target)!=list or len(target)>1000:raise CodeLimit('append_limit')
                target.append(args[0]);return None
            if not isinstance(n.func,ast.Name):raise CodeLimit('call_target')
            name=n.func.id
            builtins={'sum':sum,'len':len,'min':min,'max':max,'abs':abs,'sorted':sorted,'list':list,'tuple':tuple,'set':set,'all':all,'any':any,'enumerate':lambda x:list(enumerate(x)),'zip':lambda *x:list(zip(*x))}
            if name=='range':
                value=range(*args)
                if len(value)>1000:raise CodeLimit('range_limit')
                return list(value)
            if name not in builtins:raise CodeLimit('call_not_allowed')
            return builtins[name](*args)
        raise CodeLimit('expression:'+type(n).__name__)

FILTERS={'positive':('x > 0','اعداد مثبت'),'negative':('x < 0','اعداد منفی'),'even':('x % 2 == 0','اعداد زوج'),'odd':('x % 2 != 0','اعداد فرد'),'nonzero':('x != 0','اعداد غیرصفر')}
def generate_pipeline(filters,maps,reducer='list',name='transform'):
    if not re.fullmatch(r'[A-Za-z][A-Za-z0-9_]{0,40}',name):raise CodeLimit('function_name')
    expression='x'
    for op,v in maps:
        if op not in ('add','subtract','multiply','divide','power') or not isinstance(v,(int,float)):raise CodeLimit('map')
        if op=='divide' and v==0 or op=='power' and abs(v)>10:raise CodeLimit('map_domain')
        expression=f'({expression} '+{'add':'+','subtract':'-','multiply':'*','divide':'/','power':'**'}[op]+f' {v})'
    conditions=''.join(' if '+FILTERS[f][0] for f in filters)
    comp=f'[{expression} for x in values{conditions}]'
    if reducer not in ('list','sum','min','max','len','sorted'):raise CodeLimit('reducer')
    body=comp if reducer=='list' else f'{reducer}({comp})'
    source=f'def {name}(values):\n    return {body}\n';ast.parse(source);return source

def explain(source,language='fa'):
    tree=ast.parse(source);fn=next((x for x in tree.body if isinstance(x,ast.FunctionDef)),None)
    if fn is None:return None
    details=[]
    for node in ast.walk(fn):
        if isinstance(node,ast.Return):details.append(('خروجی: ' if language=='fa' else 'Returns: ')+ast.unparse(node.value))
        elif isinstance(node,ast.For):details.append(('پیمایش: ' if language=='fa' else 'Iterates: ')+ast.unparse(node.iter))
        elif isinstance(node,ast.If):details.append(('شرط: ' if language=='fa' else 'Condition: ')+ast.unparse(node.test))
    return (f"تابع {fn.name} با ورودی‌های "+', '.join(a.arg for a in fn.args.args) if language=='fa' else f"Function {fn.name}, parameters: "+', '.join(a.arg for a in fn.args.args))+'.\n'+'\n'.join(details[:12])

class ComprehensionRefactor(ast.NodeTransformer):
    def visit_FunctionDef(self,node):
        # Convert the exact empty-list / append loop / return pattern only.
        if len(node.body)!=3:return node
        init,loop,ret=node.body
        if not isinstance(init,ast.Assign) or len(init.targets)!=1 or not isinstance(init.targets[0],ast.Name) or not isinstance(init.value,ast.List) or init.value.elts:return node
        name=init.targets[0].id
        if not isinstance(loop,ast.For) or loop.orelse or not isinstance(ret,ast.Return) or not isinstance(ret.value,ast.Name) or ret.value.id!=name:return node
        body=loop.body;ifs=[]
        while len(body)==1 and isinstance(body[0],ast.If) and not body[0].orelse:ifs.append(body[0].test);body=body[0].body
        if len(body)!=1 or not isinstance(body[0],ast.Expr) or not isinstance(body[0].value,ast.Call):return node
        call=body[0].value
        if not isinstance(call.func,ast.Attribute) or call.func.attr!='append' or not isinstance(call.func.value,ast.Name) or call.func.value.id!=name or len(call.args)!=1:return node
        # Do not rewrite if the accumulator or loop variable is read by a side effect.
        exprs=[call.args[0],loop.iter,*ifs]
        if any(isinstance(x,(ast.Call,ast.NamedExpr,ast.Attribute)) or isinstance(x,ast.Name) and x.id==name for e in exprs for x in ast.walk(e)):return node
        comp=ast.ListComp(call.args[0],[ast.comprehension(loop.target,loop.iter,ifs,0)])
        node.body=[ast.Return(comp)];return node

def refactor(source):return ast.unparse(ast.fix_missing_locations(ComprehensionRefactor().visit(ast.parse(source))))+'\n'
def repair_syntax(source):
    try:ast.parse(source);return None
    except SyntaxError:pass
    lines=source.splitlines()
    for i,line in enumerate(lines):
        if re.match(r'\s*(def |for |if |elif |else\b)',line) and not line.rstrip().endswith(':'):lines[i]=line+':'
    candidate='\n'.join(lines)+'\n'
    try:ast.parse(candidate);return candidate
    except SyntaxError:return None

class CodeIntelligenceV21:
    def __init__(self):self.last=None;self.model=None
    def solve(self,text,language='fa'):
        self.last=None
        if self.model is None:
            from jarvis.agent.models_v21 import ClassifierV21
            self.model=ClassifierV21('code_task')
        pred=self.model.predict(text)
        if not pred or pred[0][1]<.60:return None
        task=pred[0][0];self.last={'task':task,'prediction':pred[:3],'checkpoint':self.model.sha256}
        blocks=re.findall(r'```(?:python)?\s*\n(.*?)```',text,re.S);source=blocks[0] if blocks else ''
        if source:
            if len(source)>12000:return None
            try:
                if task=='explain':return explain(source,language)
                if task=='debug':
                    fixed=repair_syntax(source)
                    return '```python\n'+fixed+'```' if fixed else ('برای تشخیص خطای منطقی، ورودی و خروجی مورد انتظار را بنویس.' if language=='fa' else 'Provide a failing input and the expected result to diagnose a logic error.')
                if task=='refactor':
                    revised=refactor(source)
                    for values in [[],[0],[-3,0,2,5],[8,4,2],[-4,-2]]:
                        try:
                            a=SafePython().run(source,[values.copy()]);b=SafePython().run(revised,[values.copy()])
                            if a!=b:return None
                        except (CodeLimit,ValueError,TypeError,KeyError,ZeroDivisionError):return None
                    return '```python\n'+revised+'```'
                if task=='test':
                    fn=next(x.name for x in ast.parse(source).body if isinstance(x,ast.FunctionDef));tests=[]
                    for values in [[],[0],[-3,0,2,5],[8,4,2],[-4,-2]]:
                        try:result=SafePython().run(source,[values.copy()])
                        except (CodeLimit,ValueError,TypeError,KeyError,ZeroDivisionError):continue
                        tests.append(f'assert {fn}({values!r}) == {result!r}')
                    if tests:return ('این‌ها رفتار فعلی کد را ثبت می‌کنند؛ درستی نیازمندی را اثبات نمی‌کنند.\n' if language=='fa' else 'These characterize current behavior, not specification correctness.\n')+'```python\n'+'\n'.join(tests)+'\n```'
            except (SyntaxError,StopIteration):return None
            return None
        if task!='generate':return None
        filters=[];maps=[];reducer='list';low=text.casefold()
        for key,(expr,fa) in FILTERS.items():
            if re.search(r'\b'+key+r'\b',low) or fa in low:filters.append(key)
        # Collect map operations in textual order, allowing arbitrary compositions.
        pats=[('add',r'(?:add|اضافه کن)\s+(-?\d+(?:\.\d+)?)'),('subtract',r'(?:subtract|کم کن)\s+(-?\d+(?:\.\d+)?)'),('multiply',r'(?:multiply by|ضرب در)\s+(-?\d+(?:\.\d+)?)'),('divide',r'(?:divide by|تقسیم بر)\s+(-?\d+(?:\.\d+)?)'),('power',r'(?:power|توان)\s+(\d+)')]
        found=[]
        for op,pat in pats:
            for m in re.finditer(pat,low):found.append((m.start(),op,float(m.group(1))))
        maps=[(op,int(v) if v.is_integer() else v) for _,op,v in sorted(found)]
        if re.search('sum|مجموع|جمع خروجی',low):reducer='sum'
        elif re.search('sorted|مرتب',low):reducer='sorted'
        elif re.search('count|تعداد',low):reducer='len'
        if not maps and not filters:return None
        name_match=re.search(r'(?:named|called|به نام)\s+([A-Za-z][A-Za-z0-9_]*)',text)
        name=name_match.group(1) if name_match else filters[0]+'_numbers' if len(filters)==1 and not maps and reducer=='list' else 'transform'
        try:return '```python\n'+generate_pipeline(filters,maps,reducer,name)+'```'
        except (CodeLimit,SyntaxError):return None
