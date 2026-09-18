"""Fresh source witnesses. Never consumes neural labels, IR slots or execution traces.

Supported source grammar is deliberately bounded. Unknown/ambiguous facts produce
UNSUPPORTED, never a semantic PASS. Spans refer to normalized source text.
"""
from __future__ import annotations
from dataclasses import dataclass,field,asdict
from fractions import Fraction
import re,math
from jarvis.agent.normalization_v20 import normalize,NUMBER
from jarvis.agent.source_semantics_v20_1 import source_work_facts,sequence_source,SourceAmbiguity

class UnsupportedSource(ValueError): pass
@dataclass
class FactGraph:
    domain:str
    slots:dict
    operations:list=field(default_factory=list)
    entities:list=field(default_factory=list)
    relations:list=field(default_factory=list)
    constraints:list=field(default_factory=list)
    witnesses:list=field(default_factory=list)
    units:dict=field(default_factory=dict)
    source:str=''
    def to_dict(self):return asdict(self)

def normalized(text):
    # Bounded transliteration vocabulary; code and identifiers never pass here.
    text=re.sub(r'\b(adad|darsad|ziad|kam|kon|zarb|taghsim|dar|bar|ta|ro|ast)\b',lambda m:{'adad':'عدد','darsad':'درصد','ziad':'زیاد','kam':'کم','kon':'کن','zarb':'ضرب','taghsim':'تقسیم','dar':'در','bar':'بر','ta':'تا','ro':'رو','ast':'است'}[m.group().lower()],text,flags=re.I)
    t=normalize(text).replace('٪','%').replace('٫','.')
    return t

def extract_source(text):
    t=normalized(text); numbers=list(NUMBER.finditer(t))
    def identifier(m):
        left=t[max(0,m.start()-65):m.start()]
        return bool(re.search(r'(?:کد(?: بسته)?|شماره پرونده|شناسه|record id(?: is)?|package identifier)\s*$',left))
    numbers=[m for m in numbers if not identifier(m)]
    if len(numbers)>80:raise UnsupportedSource('source_size')
    vals=[float(n.group()) for n in numbers]
    def choose(patterns,exclude=()):
        hits=[]
        for i,m in enumerate(numbers):
            if i in exclude:continue
            left=t[max(0,m.start()-75):m.start()];right=t[m.end():m.end()+75]
            if any(re.search(a,left) and re.match(b,right) for a,b in patterns):hits.append(i)
        if len(hits)!=1:raise UnsupportedSource('ambiguous_numeric_fact:'+str(hits))
        return hits[0]
    def graph(domain,slots,operations=None,units=None):
        return FactGraph(domain,slots,operations or [],witnesses=[{'start':n.start(),'end':n.end(),'value':float(n.group()),'text':n.group()} for n in numbers],source=t,units=units or {})
    if not numbers:raise UnsupportedSource('no_quantitative_facts')
    nested=re.search(r'^(\d+(?:\.\d+)?)\s+برابر\s+(0\.5|0\.25)\s+(\d+(?:\.\d+)?)',t)
    if nested:
        factor,fraction,base=map(float,nested.groups())
        return graph('graph',{'initial':base},[{'op':'multiply','value':fraction},{'op':'multiply','value':factor}])
    if re.search(r'transfers?|می دهد|می‌دهد|منتقل',t):
        return ownership_facts(t)
    if re.search('کارگر|workers',t) and re.search('جداگانه|separately',t):
        m=re.search(r'(?:در|in)\s*(\d+(?:\.\d+)?)\s*(?:و|and)\s*(\d+(?:\.\d+)?)\s*(?:ساعت|hours)',t)
        if m:return graph('work_rate',{'durations':[float(m.group(1)),float(m.group(2))],'query':'combined_time'},units={'time':'hour'})
    if re.search(r'کارگر|workers?\b',t) and re.search(r'ساعت|دقیقه|hours?|minutes?',t):
        try:return graph('work_rate',source_work_facts(t),units={'time':'hour','output':'piece'})
        except SourceAmbiguity as e:raise UnsupportedSource(str(e))
    if re.search(r'دنباله|sequence|\bterm\b|جمله',t):
        try:s=sequence_source(t)
        except SourceAmbiguity as e:raise UnsupportedSource(str(e))
        if s['terms']:
            return graph('sequence',{'terms':[x['value'] for x in s['terms']],'n':s['target']['value'] if s['target'] else len(s['terms'])+1})
        # Explicit first-term/rule form; separate the query index from observations.
        first=choose([(r'(?:از|starts? (?:at|with)|first term (?:is|=))\s*$',r'.*')])
        rule=choose([(r'(?:نسبت|ratio|اختلاف|difference)\s*$',r'.*')],(first,))
        if not s['target']:raise UnsupportedSource('sequence_index_missing')
        typ='geometric' if re.search(r'نسبت|ratio|هندسی|geometric',t) else 'arithmetic'
        return graph('sequence',{'terms':[vals[first]],'n':s['target']['value'],'rule':{'kind':typ,'r' if typ=='geometric' else 'd':vals[rule]}})
    if re.search('نسبت|ratio',t) and re.search('ساده|simplify',t) and len(vals)==2:
        return graph('ratio',{'ratio_a':vals[0],'ratio_b':vals[1],'query':'simplify'})
    if re.search(r'نسبت|\bratio\b',t) and not re.search(r'درصد|percent|%',t):
        pair=re.search(r'(?:نسبت(?:شان|های| ها|شان)?|ratio(?:\s+(?:of|must be))?)\s*(-?\d+(?:\.\d+)?)\s*(?:به|:|to)\s*(-?\d+(?:\.\d+)?)',t)
        if not pair:pair=re.search(r'(-?\d+(?:\.\d+)?)\s*:\s*(-?\d+(?:\.\d+)?)\s+ratio',t)
        if not pair:raise UnsupportedSource('ratio_relationship')
        left=[i for i,m in enumerate(numbers) if not(pair.start()<=m.start()<pair.end())]
        # Reference IDs are not arithmetic inputs.
        left=[i for i in left if not re.search(r'(?:کد|شناسه|\bid|record|سؤال)\s*$',t[max(0,numbers[i].start()-20):numbers[i].start()])]
        left=[i for i in left if not(vals[i]==2 and re.match(r'\s*(?:نفر|shares|بخش|سهم|تیم|beneficiaries)',t[numbers[i].end():]))]
        ordinal_pair=re.search(r'1\s+(?:to|و)\s+2\s*(?:ratio|نسبت)',t)
        if ordinal_pair:left=[i for i in left if not ordinal_pair.start()<=numbers[i].start()<ordinal_pair.end()]
        if len(left)!=1:raise UnsupportedSource('ratio_total')
        return graph('ratio',{'total':vals[left[0]],'ratio_a':float(pair.group(1)),'ratio_b':float(pair.group(2))})
    if re.search(r'موفق|شانس|سکه|success|binomial|chance|coin|شیر|برد',t) and re.search(r'احتمال|شانس|probab|chance',t) and not re.search(r'تاس|dice|\bdie\b',t) and not (re.search('همزمان|هم زمان|both|all',t) and re.search('مستقل|independent',t)):
        try:pi=choose([(r'.*',r'\s*(?:%|درصد|percent)'),(r'(?:احتمال|probability|chance|p\s*\(\s*head\s*\)\s*=)\s*$',r'.*')])
        except UnsupportedSource:
            candidates=[i for i,m in enumerate(numbers) if 0<=vals[i]<=1 and re.search(r'احتمال|probability|chance',t[max(0,m.start()-65):m.start()]) and not re.search(r'دقیقا|دقیقاً|exactly|حداقل|حداکثر|at least|at most',t[max(0,m.start()-30):m.start()])]
            if len(candidates)==1:pi=candidates[0]
            elif len(vals)==2 and re.search('coin|سکه',t) and not re.search('biased|نامتقارن',t):pi=None
            else:raise
        ki=choose([(r'(?:دقیقا|دقیقاً|حداقل|حداکثر|exactly|at least|at most)\s*$',r'.*')],(pi,))
        ni=choose([(r'.*',r'\s*(?:بار|آزمایش|تلاش|بازی|پرتاب|trials?|attempts?|games?|times|coin\s+tosses|tosses|flips?|independent\s+(?:trials|attempts)|-trial)'),(r'(?:در|از|across|in|of)\s*$',r'\s*(?:آزمایش|بازی|تلاش|independent)')],(pi,ki))
        p=.5 if pi is None else vals[pi]/100 if re.match(r'\s*(?:%|درصد|percent)',t[numbers[pi].end():]) else vals[pi]
        mode='at_least' if re.search('حداقل|at least',t) else 'at_most' if re.search('حداکثر|at most',t) else 'exactly'
        return graph('binomial',{'p':p,'n':vals[ni],'k':vals[ki],'mode':mode})
    compact_count=re.search(r'از\s+(\d+)\s+.*?ترتیب\s+(\d+)\s*تایی',t)
    if compact_count:return graph('permutation',{'n':int(compact_count.group(1)),'k':int(compact_count.group(2))})
    if re.search(r'ترکیب|جایگشت|انتخاب|combination|permutation|choose|select|arrange|selections|arrangements',t) and len(vals)==2:
        # Selecting k from n; directional words, not descending numeric sort.
        k=choose([(r'(?:انتخاب|choose|select|arrange|selections of|arrangements of)\s*$',r'.*'),(r'.*',r'\s*(?:نفر|items?|people)?\s*(?:از|from|out of)')])
        n=1-k
        return graph('permutation' if re.search('جایگشت|permutation|arrange|ordered|ترتیب مهم|order matters',t) else 'combination',{'n':vals[n],'k':vals[k]})
    if re.search(r'تاس|dice|\bdie\b',t):
        ni=choose([(r'.*',r'\s*(?:تاس|fair dice|dice|dice rolls)')])
        remaining=[i for i in range(len(vals)) if i!=ni]
        if len(remaining)!=1:raise UnsupportedSource('dice_target')
        ti=remaining[0]
        return graph('dice',{'n':vals[ni],'target':vals[ti]})
    if re.search(r'مستقل|independent',t) and re.search(r'همزمان|هم زمان|both|all|همگی',t):
        ps=[float(m.group())/100 for m in numbers if re.match(r'\s*(?:%|درصد|percent)',t[m.end():])]
        if len(ps)<2:raise UnsupportedSource('independent_probabilities')
        return graph('independent',{'probabilities':ps})
    if len(vals)==2 and re.search(r'سال|years? old|aged',t) and not re.search('بزرگ|کوچک|older|younger',t):
        ai=choose([(r'.*',r'\s*(?:ساله|years? old\b)')]);yi=1-ai
        return graph('age',{'age':vals[ai],'difference':0,'years':vals[yi]*(-1 if re.search('قبل|ago',t) else 1),'query':'relative'},units={'time':'year'})
    if re.search(r'سال|years? old|aged|older|younger',t) and re.search(r'بزرگ|کوچک|اختلاف سنی|older|younger',t):
        ai=choose([(r'.*',r'\s*(?:ساله|سالمه|سال دارد|years? old\b)'),(r'(?:سن\s+\S+|aged|age(?: is)?)\s*$',r'.*')])
        di=choose([(r'.*',r'\s*(?:سال|years?)\s*(?:بزرگ|کوچک|اختلاف سنی|older|younger)'),(r'(?:older by|younger by|اختلاف سنی)\s*$',r'.*')],(ai,))
        yi=choose([(r'.*',r'\s*(?:سال|years?)\s*(?:بعد|دیگه|دیگر|قبل|later|from now|ago)'),(r'(?:پس از|بعد از|after|in)\s*$',r'\s*(?:سال|years?)')],(ai,di))
        d=vals[di]*(-1 if re.search('کوچک|younger',t) else 1);y=vals[yi]*(-1 if re.search('سال قبل|years? ago',t) else 1)
        query='sum' if re.search('مجموع|جمع|sum|combined|total age',t) else 'relative'
        first_name=re.match(r'\s*([^\W\d_]+)',t)
        if query!='sum' and first_name:
            name=re.escape(first_name.group(1));tail=t[numbers[ai].end():]
            if re.search(r'(?:سن\s+'+name+r'\b|'+name+r'\s+چند|how old (?:is|will)\s+'+name+r'\b|age of\s+'+name+r'\b)',tail):query='base'
        return graph('age',{'age':vals[ai],'difference':d,'years':y,'query':query},units={'time':'year'})
    if re.search(r'سرعت|speed|velocity',t) and re.search(r'کیلومتر|متر|kilometers?|meters?|\bkm\b',t):
        di=choose([(r'.*',r'\s*(?:کیلومتر|متر|kilometers?|meters?|km\b)')]);ti=choose([(r'.*',r'\s*(?:ساعت|دقیقه|ثانیه|hours?|minutes?|seconds?)')],(di,))
        ds=t[numbers[di].end():];ts=t[numbers[ti].end():];persecond=bool(re.search(r'متر بر ثانیه|meters? per second|m\s*/\s*s',t))
        dist=vals[di]*(1000 if re.match(r'\s*(?:کیلومتر|kilometer|km)',ds) else 1)
        seconds=vals[ti]*(3600 if re.match(r'\s*(?:ساعت|hour)',ts) else 60 if re.match(r'\s*(?:دقیقه|minute)',ts) else 1)
        if re.match(r'\s*(?:کیلومتر|kilometers?|km)\s*(?:بر ساعت|per hour|/h)',ds):
            return graph('speed',{'speed':vals[di],'time':seconds/3600,'query':'distance'},units={'distance':'km'})
        return graph('speed',{'distance':dist if persecond else dist/1000,'time':seconds if persecond else seconds/3600},units={'speed':'m/s' if persecond else 'km/h'})
    if re.search(r'شروع|start|begins?|leaves?|departs?|opens?',t) and re.search(r'تمام|پایان|finish|end|مدت|duration|lasts?|takes?',t) and re.search(r'ساعت|hours?|minutes?|دقیقه',t):
        si=choose([(r'(?:ساعت|at|start(?:s)?(?: at)?)\s*$',r'.*')])
        durations=[]
        for i,m in enumerate(numbers):
            if i==si:continue
            tail=t[m.end():]
            if re.match(r'\s*(?:ساعت|hours?)',tail):durations.append(vals[i])
            elif re.match(r'\s*(?:دقیقه|minutes?)',tail):durations.append(vals[i]/60)
            else:raise UnsupportedSource('duration_unit')
        if not durations:raise UnsupportedSource('missing_duration')
        return graph('scheduling',{'start':vals[si],'durations':durations},units={'time':'hour'})
    if not re.search('وزنی|weighted',t) and re.search(r'میانگین|کمترین|بیشترین|کوچکترین|بزرگترین|average|mean|minimum|maximum|smallest|largest',t) and len(vals)>=2:
        return graph('comparison',{'values':vals,'mode':'mean' if re.search('میانگین|average|mean',t) else 'min' if re.search('کمترین|کوچکترین|minimum|smallest',t) else 'max'})
    # Purchases discount the purchase price, not the entire owner's balance.
    if re.search(r'خرید|purchase|bought|buy',t) and re.search(r'تخفیف|discount',t):
        if len(vals)!=3:raise UnsupportedSource('discounted_purchase_coverage')
        initial=choose([(r'.*',r'\s*(?:دلار|تومان|dollars?)?\s*(?:داشت|دارم|دارد)'),(r'(?:has|had|balance(?: is)?)\s*$',r'.*')])
        cost=choose([(r'.*',r'\s*(?:دلار|تومان|dollars?)?\s*خرید'),(r'(?:purchase(?: of| costing)?|bought(?: for)?|buy(?: for)?)\s*$',r'.*')],(initial,))
        percent=choose([(r'.*',r'\s*(?:%|درصد|percent)')],(initial,cost))
        f=graph('finance',{'initial':vals[initial]},[{'op':'purchase','value':vals[cost],'discount':vals[percent]}],{'amount':'money'})
        owner=t[:numbers[initial].start()].strip().split()[-1] if t[:numbers[initial].start()].strip() else 'owner'
        f.entities=[{'id':owner,'type':'person'},{'id':'cash','type':'money'},{'id':'purchase','type':'products'}]
        f.relations=[{'subject':owner,'predicate':'owns','object':'cash'},{'subject':'purchase','predicate':'discount_applies_to','object':'purchase_price'}]
        return f
    percent_of=re.search(r'(\d+(?:\.\d+)?)\s*(?:درصد از|percent of)\s*(\d+(?:\.\d+)?)',t)
    if percent_of:return graph('graph',{'initial':float(percent_of.group(2))},[{'op':'multiply','value':float(percent_of.group(1))/100}])
    return extract_events(t,graph)

POS=r'اضافه|زیاد|افزایش|واریز|دریافت|وارد|ورودی|رسید|بیشتر|بگیر|جمع|add|increase|deposit|credit|receive|restock|arrive'
NEG=r'کم|کاهش|برداشت|خرج|پرداخت|ارسال|فروخت|فروخته|خارج|خروجی|subtract|decrease|reduce|withdraw|debit|spend|spent|pay|paid|charge|ship|sell|sold|leave|remove'
def operation_from_source(clause):
    if re.search(r'باقی.?مانده.*تقسیم|modulo|remainder',clause):return 'modulo'
    if re.search(r'توان|power|exponent',clause):return 'exponent'
    if re.search(r'گرد|round',clause):return 'round'
    if re.search(r'ضرب|multiply|times|برابر',clause):return 'multiply'
    if re.search(r'تقسیم|divide',clause):return 'divide'
    positive=bool(re.search(POS,clause));negative=bool(re.search(NEG,clause))
    if positive==negative:raise UnsupportedSource('event_direction')
    percentage=bool(re.search(r'%|درصد|percent',clause))
    return ('percentage_add' if positive else 'percentage_remove') if percentage else ('add' if positive else 'subtract')

def extract_events(t,make):
    if re.search(r'\bif\b|اگر',t):
        chunks=re.split(r'\bif\b|اگر',t,maxsplit=1)
        initial_numbers=list(NUMBER.finditer(chunks[0]))
        condition=re.split(r'\bthen\b|آنگاه',chunks[1],maxsplit=1)
        if len(initial_numbers)!=1 or len(condition)!=2:raise UnsupportedSource('conditional_coverage')
        boundary=list(NUMBER.finditer(condition[0]));arms=re.split(r'\botherwise\b|\belse\b|وگرنه',condition[1],maxsplit=1)
        if len(boundary)!=1 or len(arms)!=2:raise UnsupportedSource('conditional_branches')
        comparison='gt' if re.search('بیشتر|بزرگتر|greater|above|>',condition[0]) else 'lt' if re.search('کمتر|کوچکتر|less|below|<',condition[0]) else None
        if comparison is None:raise UnsupportedSource('conditional_comparison')
        actions=[]
        for arm in arms:
            values=list(NUMBER.finditer(arm))
            if len(values)!=1:raise UnsupportedSource('conditional_action')
            actions.append({'op':operation_from_source(arm),'value':float(values[0].group())})
        return make('graph',{'initial':float(initial_numbers[0].group())},[{'op':'conditional','comparison':comparison,'threshold':float(boundary[0].group()),'then':[actions[0]],'else':[actions[1]]}])
    parts=[p.strip() for p in re.split(r'[;؛،,]|(?<!\d)\.(?!\d)|\band then\b|\bafter that\b|\bfollowed by\b|\bthen\b|\band\b|و بعدش|و بعد|بعدش|سپس|آنگاه|حاصل را|\bو\b',t) if p.strip()]
    initial=None;events=[];declared_steps=[]
    for part in parts:
        ns=list(NUMBER.finditer(part))
        if not ns:continue
        if len(ns)==1 and re.match(r'\s*(?:تغییر|مرحله|steps?|operations?)',part[ns[0].end():]):
            declared_steps.append(int(ns[0].group()));continue
        if re.search(r'کد|شناسه|شماره پرونده|record id|question id|package identifier',part):continue
        if initial is None:
            if len(ns)>2:raise UnsupportedSource('initial_event_coverage')
            initial=float(ns[0].group())
            if len(ns)==2:
                events.append({'op':operation_from_source(part[ns[0].end():]),'value':float(ns[1].group())})
            continue
        if len(ns)!=1:raise UnsupportedSource('event_coverage')
        events.append({'op':operation_from_source(part),'value':float(ns[0].group())})
    if initial is None or not events:raise UnsupportedSource('no_event_graph')
    if any(n!=len(events) for n in declared_steps):raise UnsupportedSource('declared_event_count_mismatch')
    domain='inventory' if re.search(r'انبار|کالا|قطعه|inventory|warehouse|stock|items|pieces|units',t) else 'finance' if re.search(r'حساب|پول|دلار|تومان|bank|account|balance|dollar|money',t) else 'graph'
    g=make(domain,{'initial':initial},events)
    if declared_steps:g.constraints.append({'type':'event_count','value':len(events)})
    return g

def source_answer(f):
    """Independent reference interpreter, no GraphExecutor import or IR dependency."""
    s=f.slots;d=f.domain;F=lambda x:Fraction(str(x))
    if d=='ownership':
        balances={k:F(v) for k,v in s['balances'].items()}
        for e in f.operations:
            amount=F(e['value'])
            if amount<0 or balances[e['from']]<amount:raise UnsupportedSource('impossible_transfer')
            balances[e['from']]-=amount;balances[e['to']]+=amount
        return float(balances[s['query']])
    if f.operations:
        x=F(s['initial'])
        for e in f.operations:
            op=e['op'];v=F(e.get('value',0))
            if op=='add':x+=v
            elif op=='subtract':x-=v
            elif op=='multiply':x*=v
            elif op=='divide':x/=v
            elif op=='percentage_add':x=x*(100+v)/100
            elif op=='percentage_remove':x=x*(100-v)/100
            elif op=='purchase':x-=v*(100-F(e['discount']))/100
            elif op=='exponent':x=x**int(v)
            elif op=='modulo':x=x%v
            elif op=='round':x=F(round(float(x),int(v)))
            elif op=='conditional':
                threshold=F(e['threshold']);truth=x>threshold if e['comparison']=='gt' else x<threshold
                branch=FactGraph('graph',{'initial':float(x)},operations=e['then'] if truth else e['else'])
                x=F(source_answer(branch))
            else:raise UnsupportedSource('reference_operation')
            if d=='inventory' and x<0:raise UnsupportedSource('impossible_inventory')
        return float(x)
    if d=='ratio' and s.get('query')=='simplify':
        divisor=math.gcd(int(s['ratio_a']),int(s['ratio_b']));return [s['ratio_a']/divisor,s['ratio_b']/divisor]
    if d=='work_rate' and s.get('query')=='combined_time':return float(1/sum(1/F(v) for v in s['durations']))
    if d=='ratio':return [float(F(s['total'])*F(s[k])/(F(s['ratio_a'])+F(s['ratio_b']))) for k in ('ratio_a','ratio_b')]
    if d=='work_rate':return float(F(s['output_initial'])*F(s['workers_target'])*F(s['hours_target'])/(F(s['workers_initial'])*F(s['hours_initial'])))
    if d=='binomial':
        n,k=int(s['n']),int(s['k']);p=F(s['p'])
        if n!=s['n'] or k!=s['k'] or not 0<=k<=n<=200 or not 0<=p<=1:raise UnsupportedSource('probability_domain')
        row=[Fraction(1)]+[Fraction(0)]*n
        for j in range(n):
            row=[row[0]*(1-p)]+[row[i]*(1-p)+row[i-1]*p for i in range(1,n+1)]
        return float(sum(row[k:]) if s['mode']=='at_least' else sum(row[:k+1]) if s['mode']=='at_most' else row[k])
    if d in ('combination','permutation'):return (math.comb if d=='combination' else math.perm)(int(s['n']),int(s['k']))
    if d=='dice':
        n=int(s['n']);target=int(s['target']);result=0
        if not 1<=n<=12:raise UnsupportedSource('dice_bound')
        for j in range(n+1):
            top=target-6*j-1
            if top>=n-1:result+=(-1)**j*math.comb(n,j)*math.comb(top,n-1)
        return result/6**n
    if d=='independent':return float(math.prod(F(p) for p in s['probabilities']))
    if d=='age':
        a=s['age']+s['years'];b=a+s['difference']
        if min(a,b)<0:raise UnsupportedSource('impossible_age')
        return a+b if s['query']=='sum' else a if s['query']=='base' else b
    if d=='speed':return float(F(s['speed'])*F(s['time'])) if s.get('query')=='distance' else float(F(s['distance'])/F(s['time']))
    if d=='scheduling':return float(F(s['start'])+sum(F(x) for x in s['durations']))
    if d=='comparison':return sum(s['values'])/len(s['values']) if s['mode']=='mean' else (min if s['mode']=='min' else max)(s['values'])
    if d=='sequence':
        v=list(map(F,s['terms']));n=int(s['n']);rule=s.get('rule')
        if not 1<=n<=1000:raise UnsupportedSource('sequence_range')
        if rule:
            return float(v[0]*F(rule['r'])**(n-1) if rule['kind']=='geometric' else v[0]+(n-1)*F(rule['d']))
        dif=[b-a for a,b in zip(v,v[1:])]
        if dif and len(set(dif))==1:return float(v[0]+(n-1)*dif[0])
        if all(v[:-1]):
            ratios=[b/a for a,b in zip(v,v[1:])]
            if len(set(ratios))==1:return float(v[0]*ratios[0]**(n-1))
        if len(v)>=4 and all(v[i]==v[i-1]+v[i-2] for i in range(2,len(v))):
            while len(v)<n:v.append(v[-1]+v[-2])
            return float(v[n-1])
        if len(v)>=5 and all(x==dif[i%2] for i,x in enumerate(dif)):
            while len(v)<n:v.append(v[-1]+dif[(len(v)-1)%2])
            return float(v[n-1])
        raise UnsupportedSource('sequence_ambiguous')
    raise UnsupportedSource('reference_domain')


def ownership_facts(t):
    names=r"[^\W\d_]+"
    balances={};witness=[];units=set()
    def unit_of(text):
        return 'USD' if re.search('دلار|dollars',text) else 'toman' if 'تومان' in text else 'piece'
    for pattern in [rf"({names})\s+(\d+(?:\.\d+)?)\s*(?:دلار|تومان|کالا|قطعه)\s*(?:دارد|داشت)",rf"({names})\s+has\s+(\d+(?:\.\d+)?)\s*(?:dollars|items|pieces)"]:
        for m in re.finditer(pattern,t):
            if m.group(1) in balances:raise UnsupportedSource('duplicate_owner_assertion')
            balances[m.group(1)]=float(m.group(2));units.add(unit_of(m.group()));witness.append({'start':m.start(),'end':m.end(),'kind':'ownership'})
    events=[]
    for pattern in [rf"({names})\s+(\d+(?:\.\d+)?)\s*(?:دلار|تومان|کالا|قطعه)\s+به\s+({names})\s*(?:می دهد|می‌دهد|منتقل می کند)",rf"({names})\s+transfers?\s+(\d+(?:\.\d+)?)\s*(?:dollars|items|pieces)\s+to\s+({names})"]:
        for m in re.finditer(pattern,t):
            units.add(unit_of(m.group()));events.append((m.start(),{'op':'transfer','from':m.group(1),'to':m.group(3),'value':float(m.group(2))}))
    if len(units)!=1:raise UnsupportedSource('ownership_unit_conflict')
    query=re.search(rf"(?:موجودی|سهم|balance of|how much does)\s+({names})",t)
    if not query or query.group(1) not in balances or len(balances)<2 or not events:raise UnsupportedSource('ownership_coverage')
    return FactGraph('ownership',{'balances':balances,'query':query.group(1)},operations=[e for _,e in sorted(events)],entities=[{'id':k,'type':'person'} for k in balances],relations=[{'subject':k,'predicate':'owns','object':k+':balance'} for k in balances],constraints=[{'type':'conservation'},{'type':'nonnegative'}],witnesses=witness,units={'amount':next(iter(units))},source=t)
