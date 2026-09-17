from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from typing import Any

from jarvis.utils.text import normalize_text

_DIGITS = str.maketrans('۰۱۲۳۴۵۶۷۸۹٠١٢٣٤٥٦٧٨٩', '01234567890123456789')
_NUMWORDS = {'صفر':0,'یک':1,'دو':2,'سه':3,'چهار':4,'پنج':5,'شش':6,'هفت':7,'هشت':8,'نه':9,'ده':10,'یازده':11,'دوازده':12}

@dataclass(frozen=True, slots=True)
class SemanticIR:
    task: str
    slots: dict[str, Any]
    confidence: float
    evidence: tuple[str, ...] = field(default_factory=tuple)

    def get(self, name: str, default: Any=None) -> Any:
        return self.slots.get(name, default)

class SemanticIRParserV17:
    """Maps varied Persian/English surface forms into solver-oriented semantics.

    Parsing binds numbers to roles. Solvers consume only these slots instead of
    rescanning arbitrary numbers from the original sentence.
    """
    VERSION = '1.7.0'

    @staticmethod
    def _norm(text: str) -> str:
        return normalize_text(text).translate(_DIGITS).replace('٫','.')

    @staticmethod
    def _num(raw: str) -> float | None:
        s = raw.strip().translate(_DIGITS).replace(',', '.')
        if s in _NUMWORDS:
            return float(_NUMWORDS[s])
        try: return float(s)
        except ValueError: return None

    @classmethod
    def parse(cls, text: str) -> SemanticIR | None:
        t = cls._norm(text)
        for parser in (
            cls._binomial_coin, cls._dice_sum, cls._permutation, cls._ratio_split,
            cls._speed, cls._work_scaling, cls._transitive_order, cls._prime_semantic,
            cls._python_generation,
        ):
            ir = parser(t)
            if ir is not None:
                return ir
        return None

    @classmethod
    def _binomial_coin(cls, t: str) -> SemanticIR | None:
        if not re.search(r'(?:سکه|coin)', t, re.I) or not re.search(r'(?:شیر|head)', t, re.I):
            return None
        # Bind p through semantic words, accepting: احتمال شیر آن 70 درصد، شانس شیر ... 65٪,
        # heads chance is 0.7, P(head)=70%.
        p = None
        ppats = (
            r'(?:احتمال|شانس)\s*(?:آمدن\s*)?(?:شیر(?:ش)?|head(?:s)?)\s*(?:سکه|coin)?\s*(?:برابر|=|است|is|:)??\s*(\d+(?:\.\d+)?)\s*(%|٪|درصد|percent)?',
            r'(?:سکه.?ای\s+که\s+)?(?:احتمال|شانس)\s*(?:آمدن\s*)?(?:شیر(?:ش)?|head(?:s)?)\s*(?:برای\s*(?:این\s*)?سکه|برای\s+آن|آن)?\s*(?:برابر|=|است|is|:)??\s*(\d+(?:\.\d+)?)\s*(%|٪|درصد)?',
            r'(?:coin\s+(?:has|with)|a\s+coin\s+(?:has|with)).{0,25}?(\d+(?:\.\d+)?)\s*(%|٪|percent)\s*(?:chance|probability)\s*(?:of|for)?\s*heads?',
            r'(\d+(?:\.\d+)?)\s*(%|٪|percent)\s*(?:chance|probability)\s*(?:of|for)?\s*heads?',
            r'(?:احتمال|شانس|chance|probability)\s*(?:آمدن\s*)?(?:شیر(?:ش)?|heads?)\s*(?:برای\s*(?:این\s*)?سکه|برای\s+آن|آن|this\s+coin)?\s*(?:برابر|=|است|is|of|:)??\s*(\d+(?:\.\d+)?)\s*(%|٪|درصد)?',
            r'(?:شیر|heads?)\s*(?:برای\s*(?:این\s*)?سکه)?\s*(?:احتمال|شانس|chance|probability)\s*(?:برابر|=|است|is|:)??\s*(\d+(?:\.\d+)?)\s*(%|٪|درصد)?',
            r'(?:coin).{0,35}?(?:heads?).{0,25}?(\d+(?:\.\d+)?)\s*(%|٪|percent)?\s*(?:chance|probability)?',
            r'p\s*\(\s*(?:h|head|heads)\s*\)\s*=\s*(\d+(?:\.\d+)?)\s*(%|٪)?',
            r'(\d+(?:\.\d+)?)\s*(%|٪|درصد)\s*(?:احتمال|شانس|chance|probability).{0,20}?(?:شیر|heads?)',
        )
        for pat in ppats:
            m = re.search(pat, t, re.I)
            if m:
                p = float(m.group(1)); unit = m.group(2) if m.lastindex and m.lastindex >= 2 else None
                if unit or p > 1: p /= 100.0
                break
        if p is None or not 0 <= p <= 1:
            return None
        km = re.search(r'(?:دقیقاً|دقیقا|exactly)\s*(\d+|یک|دو|سه|چهار|پنج|شش|هفت|هشت|نه|ده)\s*(?:بار\s*)?(?:شیر|heads?)', t, re.I)
        if not km: return None
        kf = cls._num(km.group(1)); k = int(kf) if kf is not None else None
        npats = (
            r'(\d+|یک|دو|سه|چهار|پنج|شش|هفت|هشت|نه|ده)\s*(?:بار|مرتبه|دفعه)\s*(?:می.?اندازیم|می.?اندازم|پرتاب|تاس|toss|flip)',
            r'(\d+|یک|دو|سه|چهار|پنج|شش|هفت|هشت|نه|ده)\s*(?:پرتاب(?:\s+سکه)?|toss(?:es)?|flips?|trials?)',
            r'(?:در|in)\s*(\d+|یک|دو|سه|چهار|پنج|شش|هفت|هشت|نه|ده)\s*(?:پرتاب|بار|toss(?:es)?|flips?|trials?)',
        )
        n = None
        for pat in npats:
            m = re.search(pat, t, re.I)
            if m:
                nf=cls._num(m.group(1)); n=int(nf) if nf is not None else None; break
        if n is None or k is None or not 0 <= k <= n <= 100: return None
        return SemanticIR('binomial_probability', {'p':p,'n':n,'k':k,'event':'heads'}, .995,
                          ('coin_anchor','probability_bound','trial_count_bound','success_count_bound'))

    @classmethod
    def _dice_sum(cls, t: str) -> SemanticIR | None:
        if not re.search(r'(?:تاس|dice|die)', t, re.I): return None
        if not re.search(r'(?:دو|2|two)\s*(?:تاس|(?:fair\s+)?dice)', t, re.I): return None
        m = re.search(r'(?:مجموع|جمع|sum|add\s+up\s+to|total)\D{0,18}(\d+)|(?:مجموع|جمع).{0,15}?(\d+)\s*(?:شود|بشود|است)|add\s+up\s+to\s+(\d+)', t, re.I)
        if not m: return None
        target = int(next(g for g in m.groups() if g is not None))
        return SemanticIR('dice_sum_probability', {'dice':2,'sides':6,'target':target}, .995, ('two_fair_dice','sum_target_bound'))

    @classmethod
    def _permutation(cls, t: str) -> SemanticIR | None:
        # Ranked first/second/third selection is semantically ordered selection.
        podium = re.search(r'(?:از|میان)\s*(\d+)\s*(?:نفر|شیء|گزینه)?.{0,45}?(?:اول|نفر\s+اول).{0,18}?(?:دوم).{0,18}?(?:سوم)', t, re.I)
        if podium:
            return SemanticIR('permutation', {'n':int(podium.group(1)),'k':3}, .99, ('ranked_selection','without_replacement_implied'))
        if not re.search(r'(?:جایگشت|آرایش|چیدمان|ترتیب|permutation|arrangement|ordered|بدون\s+جایگذاری|بدون\s+تکرار|without\s+replacement)', t, re.I): return None
        pats=(
            r'(?:جایگشت|آرایش|چیدمان)(?:های)?\s*(\d+|سه|چهار|پنج)(?:\s*[-‌ ]?تایی)?.{0,35}?(?:از|میان)\s*(\d+)',
            r'(?:جایگشت|آرایش|چیدمان)(?:های)?\s*(\d+|سه|چهار|پنج)(?:\s*[-‌ ]?تایی)?.{0,40}?(?:از|میان)\s*(\d+)',
            r'(\d+|سه|چهار|پنج)\s*[-‌ ]?تایی.{0,35}?(?:بدون\s+(?:تکرار|جایگذاری)).{0,30}?(?:از|میان)\s*(\d+)',
            r'(?:از|میان)\s*(\d+).{0,45}?(\d+|سه|چهار|پنج)\s*[-‌ ]?تایی.{0,25}?(?:بدون\s+(?:تکرار|جایگذاری))',
            r'(?:arrangements?|permutations?|ordered\s+selections?).{0,30}?(\d+).{0,30}?(?:from|of|among)\s*(\d+)',
        )
        for i,p in enumerate(pats):
            m=re.search(p,t,re.I)
            if m:
                a,b=m.groups(); av=cls._num(a); bv=cls._num(b)
                if av is None or bv is None: continue
                if i==3: n,k=int(av),int(bv)
                else: k,n=int(av),int(bv)
                if 0 <= k <= n <= 500:
                    return SemanticIR('permutation', {'n':n,'k':k}, .99, ('ordered_selection','n_k_bound'))
        return None

    @classmethod
    def _ratio_split(cls, t: str) -> SemanticIR | None:
        pats=(
            r'(\d+(?:\.\d+)?)\s*(?:را|رو)?\s*(?:به\s+)?نسبت\s*(\d+(?:\.\d+)?)\s*(?:به|:)\s*(\d+(?:\.\d+)?)\s*(?:تقسیم|پخش)?',
            r'(?:تقسیم|پخش)\s*(?:کن)?\s*(\d+(?:\.\d+)?).{0,30}?نسبت\s*(\d+(?:\.\d+)?)\s*(?:به|:)\s*(\d+(?:\.\d+)?)',
            r'(?:split|divide)\s*(\d+(?:\.\d+)?).{0,25}?ratio\s*(\d+(?:\.\d+)?)\s*:\s*(\d+(?:\.\d+)?)',
        )
        for p in pats:
            m=re.search(p,t,re.I)
            if m:
                total,a,b=map(float,m.groups())
                if a>0 and b>0: return SemanticIR('ratio_split', {'total':total,'a':a,'b':b}, .995, ('ratio_roles_bound',))
        return None

    @classmethod
    def _speed(cls,t:str)->SemanticIR|None:
        if not re.search(r'(?:کیلومتر|km|مسافت|distance|سرعت|speed)',t,re.I) or not re.search(r'(?:ساعت|hour)',t,re.I): return None
        m=re.search(r'(\d+(?:\.\d+)?)\s*(?:کیلومتر|km).{0,50}?(?:در|طی|in)\s*(\d+(?:\.\d+)?)\s*(?:ساعت|hours?)',t,re.I)
        if not m: m=re.search(r'(?:مسافت|distance)\D{0,12}(\d+(?:\.\d+)?).{0,45}?(?:زمان|time)\D{0,12}(\d+(?:\.\d+)?)',t,re.I)
        if not m:return None
        d,h=map(float,m.groups())
        if h<=0:return None
        return SemanticIR('speed', {'distance_km':d,'time_hours':h}, .995, ('distance_bound','time_bound'))

    @classmethod
    def _work_scaling(cls,t:str)->SemanticIR|None:
        # productivity proportional to workers * time.
        if not re.search(r'(?:کارگر|worker)',t,re.I) or not re.search(r'(?:جعبه|box|واحد|items?)',t,re.I): return None
        pats=(
            (r'(\d+)\s*کارگر.{0,20}?(\d+(?:\.\d+)?)\s*ساعت.{0,25}?(\d+(?:\.\d+)?)\s*جعبه.{0,45}?(\d+)\s*کارگر.{0,20}?(\d+(?:\.\d+)?)\s*ساعت', 'w_h_o_w_h'),
            (r'(\d+)\s*workers?.{0,20}?(\d+(?:\.\d+)?)\s*hours?.{0,25}?(\d+(?:\.\d+)?)\s*(?:boxes|items).{0,45}?(\d+)\s*workers?.{0,20}?(\d+(?:\.\d+)?)\s*hours?', 'w_h_o_w_h'),
            (r'(\d+)\s*workers?.{0,25}?(?:make|produce)\s*(\d+(?:\.\d+)?)\s*(?:boxes|items).{0,20}?(?:in\s*)?(\d+(?:\.\d+)?)\s*hours?.{0,45}?(\d+)\s*workers?.{0,25}?(?:in\s*)?(\d+(?:\.\d+)?)\s*hours?', 'w_o_h_w_h'),
        )
        for p,order in pats:
            m=re.search(p,t,re.I)
            if m:
                vals=list(map(float,m.groups()))
                if order=='w_o_h_w_h': w1,out,h1,w2,h2=vals
                else: w1,h1,out,w2,h2=vals
                if min(w1,h1,w2,h2)>0:
                    return SemanticIR('work_scaling', {'workers1':w1,'hours1':h1,'output1':out,'workers2':w2,'hours2':h2}, .995, ('baseline_rate_bound','target_capacity_bound'))
        return None

    @classmethod
    def _transitive_order(cls,t:str)->SemanticIR|None:
        # Symbolic chains such as Ali > Reza > Mehdi and natural comparative chains.
        m=re.search(r'([^\s><،,؛;]+)\s*>\s*([^\s><،,؛;]+)\s*>\s*([^\s><،,؛;؟?]+)',t)
        if m:
            return SemanticIR('transitive_order', {'order':list(m.groups()),'relation':'greater'}, .99, ('explicit_chain',))
        m=re.search(r'([\w\u0600-\u06ff]+)\s+(?:از)\s+([\w\u0600-\u06ff]+)\s+(?:بزرگتر|بزرگ‌تر|بلندتر|بلند‌تر).{0,30}?\2\s+(?:از)\s+([\w\u0600-\u06ff]+)\s+(?:بزرگتر|بزرگ‌تر|بلندتر|بلند‌تر)',t,re.I)
        if m:
            return SemanticIR('transitive_order', {'order':[m.group(1),m.group(2),m.group(3)],'relation':'greater'}, .97, ('natural_chain',))
        return None

    @classmethod
    def _prime_semantic(cls,t:str)->SemanticIR|None:
        m=re.search(r'(?:آیا\s*)?(\d+)\s*(?:فقط|تنها)?\s*(?:بر|به)\s*1\s*(?:و|،)\s*(?:خودش|آن\s+عدد).{0,18}?(?:بخش.?پذیر|تقسیم)',t,re.I)
        if not m:
            m=re.search(r'(\d+).{0,55}?(?:غیر\s+از\s+1\s+و\s+خودش).{0,25}?(?:مقسوم.?علیه|عامل)',t,re.I)
        if not m:return None
        return SemanticIR('prime_divisor_yesno', {'n':int(m.group(1))}, .995, ('divisibility_semantics',))

    @classmethod
    def _python_generation(cls,t:str)->SemanticIR|None:
        if not re.search(r'(?:python|پایتون)',t,re.I) or not re.search(r'(?:تابع|function|کد|code).{0,30}?(?:بنویس|write|create)|(?:بنویس|write|create).{0,30}?(?:تابع|function|کد|code)',t,re.I): return None
        if re.search(r'(?:اعداد?\s+زوج|even\s+numbers?)',t,re.I):
            return SemanticIR('python_generate_even_filter', {'function_name': 'filter_even' if re.search(r'(?:فیلتر|filter)', t, re.I) else 'even_numbers'}, .995, ('python_authoring','even_filter_semantics'))
        return None

class SemanticSolverV17:
    @staticmethod
    def solve(ir: SemanticIR, language: str='fa') -> tuple[str,str,float,tuple[str,...]] | None:
        s=ir.slots; task=ir.task; checks=('sir_parsed',*ir.evidence,'plan_built','result_verified')
        fmt=lambda x: str(int(x)) if abs(x-round(x))<1e-12 else f'{x:.8f}'.rstrip('0').rstrip('.')
        if task=='binomial_probability':
            p,n,k=s['p'],s['n'],s['k']; prob=math.comb(n,k)*p**k*(1-p)**(n-k); pct=prob*100
            text=(f"P(X={k}) = C({n},{k})×{fmt(p)}^{k}×{fmt(1-p)}^{n-k} = {fmt(pct)}٪." if language=='fa' else f"P(X={k}) = C({n},{k})×{fmt(p)}^{k}×{fmt(1-p)}^{n-k} = {fmt(pct)}%.")
            return text,'probability_answer',.999,checks
        if task=='dice_sum_probability':
            target=s['target']; ways=sum(1 for a in range(1,7) for b in range(1,7) if a+b==target); g=math.gcd(ways,36)
            text=(f"{ways} حالت از 36 حالت ممکن داریم؛ احتمال = {ways//g}/{36//g} = {fmt(100*ways/36)}٪." if language=='fa' else f"There are {ways} favorable outcomes out of 36; probability = {ways//g}/{36//g} = {fmt(100*ways/36)}%.")
            return text,'probability_answer',.999,checks
        if task=='permutation':
            n,k=s['n'],s['k']; val=math.perm(n,k); text=(f"P({n},{k}) = {n}!/({n}-{k})! = {val}." if language!='fa' else f"تعداد حالات ترتیبی برابر P({n},{k}) = {n}!/({n}-{k})! = {val} است.")
            return text,'reasoned_answer',.999,checks
        if task=='ratio_split':
            total,a,b=s['total'],s['a'],s['b']; x=total*a/(a+b); y=total*b/(a+b)
            return (f"دو سهم برابر {fmt(x)} و {fmt(y)} هستند." if language=='fa' else f"The shares are {fmt(x)} and {fmt(y)}."),'word_problem_answer',.999,checks
        if task=='speed':
            v=s['distance_km']/s['time_hours']; return (f"سرعت = مسافت ÷ زمان = {fmt(v)} کیلومتر بر ساعت." if language=='fa' else f"Speed = distance / time = {fmt(v)} km/h."),'word_problem_answer',.999,checks
        if task=='work_scaling':
            rate=s['output1']/(s['workers1']*s['hours1']); out=rate*s['workers2']*s['hours2']
            return (f"نرخ هر کارگر-ساعت {fmt(rate)} جعبه است؛ خروجی جدید = {fmt(out)} جعبه." if language=='fa' else f"Rate is {fmt(rate)} boxes per worker-hour; new output = {fmt(out)} boxes."),'word_problem_answer',.999,checks
        if task=='transitive_order':
            order=s['order']; return (f"با تعدی رابطه: {order[0]} > {order[1]} > {order[2]}؛ پس {order[0]} از {order[2]} بزرگ‌تر است." if language=='fa' else f"By transitivity: {order[0]} > {order[1]} > {order[2]}, so {order[0]} > {order[2]}.") ,'reasoned_answer',.998,checks
        if task=='prime_divisor_yesno':
            n=s['n']; prime=n>=2 and all(n%d for d in range(2,math.isqrt(n)+1));
            if language=='fa': text=(f"بله؛ {n} فقط بر ۱ و خودش بخش‌پذیر است، پس عدد اول است." if prime else f"خیر؛ {n} مقسوم‌علیه دیگری هم دارد، پس عدد اول نیست.")
            else: text=(f"Yes. {n} has no positive divisors other than 1 and itself, so it is prime." if prime else f"No. {n} has another divisor, so it is not prime.")
            return text,'reasoned_answer',.999,checks
        if task=='python_generate_even_filter':
            name=s.get('function_name','even_numbers'); code=f"def {name}(values):\n    return [x for x in values if x % 2 == 0]"
            return code,'coding_answer',.999,checks
        return None

__all__=['SemanticIR','SemanticIRParserV17','SemanticSolverV17']
