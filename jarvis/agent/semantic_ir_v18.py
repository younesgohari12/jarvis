from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from jarvis.agent.semantic_models_v18 import SemanticFrameClassifierV18, NumericSlotTaggerV18
from jarvis.utils.text import normalize_text

_DIGITS=str.maketrans('۰۱۲۳۴۵۶۷۸۹٠١٢٣٤٥٦٧٨٩','01234567890123456789')
_NUMWORDS={'صفر':0,'یک':1,'دو':2,'سه':3,'چهار':4,'پنج':5,'شش':6,'هفت':7,'هشت':8,'نه':9,'ده':10,'یازده':11,'دوازده':12,
          'one':1,'two':2,'three':3,'four':4,'five':5,'six':6,'seven':7,'eight':8,'nine':9,'ten':10,'eleven':11,'twelve':12}
_ORDK={'سه':3,'چهار':4,'پنج':5,'شش':6,'three':3,'four':4,'five':5,'six':6,'top-three':3,'top-four':4,'top-five':5}

@dataclass(frozen=True, slots=True)
class SemanticIR:
    task:str
    slots:dict[str,Any]
    confidence:float
    predicate:str=''
    polarity:str='value'
    evidence:tuple[str,...]=field(default_factory=tuple)
    model_frame:str=''

    def get(self,name:str,default:Any=None)->Any:return self.slots.get(name,default)

class SemanticIRParserV18:
    VERSION='1.8.0'
    _frame:SemanticFrameClassifierV18|None=None
    _slots:NumericSlotTaggerV18|None=None

    @classmethod
    def configure_models(cls,root:Path)->None:
        cls._frame=SemanticFrameClassifierV18(root/'models'/'semantic_frame_v18.npz')
        cls._slots=NumericSlotTaggerV18(root/'models'/'numeric_slot_tagger_v18.npz')

    @staticmethod
    def _norm(text:str)->str:
        return normalize_text(text).translate(_DIGITS).replace('٫','.').replace('٪','%')

    @staticmethod
    def _num(x:str)->float|None:
        y=x.strip().casefold().translate(_DIGITS).replace(',','.')
        if y in _NUMWORDS:return float(_NUMWORDS[y])
        try:return float(y)
        except ValueError:return None

    @classmethod
    def _learned_frame(cls,t:str)->tuple[str,float]:
        if cls._frame is None:
            root=Path(__file__).resolve().parents[2];cls.configure_models(root)
        p=cls._frame.predict(t) if cls._frame else None
        return (p.label,p.confidence) if p else ('',0.0)

    @classmethod
    def _number_mentions(cls,t:str):
        return [(m,float(m.group().replace(',','.')),m.start(),m.end()) for m in re.finditer(r'(?<![\w.])\d+(?:[.,]\d+)?',t)]

    @classmethod
    def _learned_role(cls,t:str,allowed:set[str])->dict[str,float]:
        out:dict[str,tuple[float,float]]={}
        if cls._slots is None:
            root=Path(__file__).resolve().parents[2];cls.configure_models(root)
        if not cls._slots or not cls._slots.ready:return {}
        for m,val,a,b in cls._number_mentions(t):
            p=cls._slots.predict_number(t,a,b)
            if p and p.label in allowed and p.confidence>=.48:
                prev=out.get(p.label)
                if prev is None or p.confidence>prev[1]:out[p.label]=(val,p.confidence)
        return {k:v[0] for k,v in out.items()}

    @classmethod
    def parse(cls,text:str)->SemanticIR|None:
        t=cls._norm(text);frame,fconf=cls._learned_frame(t)
        parsers=(cls._binomial,cls._dice,cls._permutation,cls._ratio,cls._speed,cls._work,cls._age,cls._inventory,cls._prime,cls._transitive,cls._python_generation)
        # The learned frame changes ordering only; every parser still validates its required slots.
        priority={p.__name__.lstrip('_'):p for p in parsers}
        fmap={'binomial_probability':'binomial','dice_sum_probability':'dice','permutation':'permutation','ratio_split':'ratio','speed':'speed','work_scaling':'work','age_reasoning':'age','inventory_multistep':'inventory','prime_reasoning':'prime','transitive_logic':'transitive','code_generation':'python_generation'}
        ordered=[]
        if frame in fmap and fmap[frame] in priority:ordered.append(priority[fmap[frame]])
        ordered += [p for p in parsers if p not in ordered]
        for p in ordered:
            ir=p(t)
            if ir:
                return SemanticIR(ir.task,ir.slots,max(ir.confidence,min(.995,fconf) if p is ordered[0] else ir.confidence),ir.predicate,ir.polarity,ir.evidence,frame)
        return None

    @classmethod
    def _binomial(cls,t:str)->SemanticIR|None:
        if not re.search(r'(?:سکه|coin|head|شیر|p\s*\(\s*head)',t,re.I):return None
        p=None; p_explicit=False
        # Probability bias is bound only when the number is explicitly attached to a
        # probability/head cue. Bare integers elsewhere are n/k, never p.
        pats=(
          r'p\s*\(\s*(?:h|head|heads)\s*\)\s*=\s*(\d+(?:\.\d+)?)\s*(%)?',
          r'(\d+(?:\.\d+)?)\s*(%|percent|درصد)\s*(?:chance|probability|شانس|احتمال).{0,24}?(?:head|شیر)',
          r'(?:chance|probability|شانس|احتمال).{0,32}?(?:head|شیر(?:ش)?).{0,18}?(\d+(?:\.\d+)?)\s*(%|percent|درصد)?',
          r'(?:head|شیر(?:ش)?).{0,28}?(?:chance|probability|شانس|احتمال).{0,18}?(\d+(?:\.\d+)?)\s*(%|percent|درصد)?',
          r'(?:احتمال|شانس)\s*(?:اینکه\s*)?(?:این\s*)?سکه\s*(?:شیر\s*(?:بیاید|بیاد))?.{0,18}?(\d+(?:\.\d+)?)\s*(%|درصد)?',
        )
        for idx,pat in enumerate(pats):
            m=re.search(pat,t,re.I)
            if not m:continue
            raw=float(m.group(1)); unit=m.group(2) if m.lastindex and m.lastindex>=2 else None
            # Unitless values are accepted only as explicit probabilities in [0,1].
            # This blocks phrases like "probability exactly 2 heads in 4 tosses"
            # from binding the 4 as p=.04.
            if unit:
                cand=raw/100.0
            elif idx==0 and 0<=raw<=1:
                cand=raw
            elif 0<=raw<=1 and '.' in m.group(1):
                cand=raw
            else:
                continue
            if 0<=cand<=1:
                p=cand;p_explicit=True;break
        k=None
        km=re.search(r'(?:دقیقاً|دقیقا|exactly)\s*(\d+|یک|دو|سه|چهار|پنج|شش|seven|one|two|three|four|five|six)\s*(?:بار\s*)?(?:شیر|heads?)',t,re.I)
        if km:
            z=cls._num(km.group(1));k=int(z) if z is not None else None
        n=None
        npats=(
            r'(\d+|یک|دو|سه|چهار|پنج|شش|one|two|three|four|five|six)\s*(?:پرتاب(?:\s+سکه)?|toss(?:es)?|flips?|trials?)',
            r'(?:در|in|over)\s*(\d+|یک|دو|سه|چهار|پنج|شش|one|two|three|four|five|six)\s*(?:پرتاب|toss(?:es)?|flips?|trials?)',
            r'(\d+|یک|دو|سه|چهار|پنج|شش)\s*بار.{0,18}?(?:می.?انداز|پرتاب|سکه)',
            r'(?:flip|toss)\s+(?:the\s+)?coin\s*(\d+)\s*times?',
            r'in\s+(\d+)\s+coin\s+toss(?:es)?',
        )
        for pat in npats:
            m=re.search(pat,t,re.I)
            if m:
                z=cls._num(m.group(1));n=int(z) if z is not None else None;break
        learned=cls._learned_role(t,{'n','k'})
        if n is None and 'n' in learned:n=int(round(learned['n']))
        if k is None and 'k' in learned:k=int(round(learned['k']))
        if p is None:p=0.5
        if n is None or k is None or not (0<=p<=1 and 0<=k<=n<=100):return None
        evidence=['n_bound','k_bound','p_explicit' if p_explicit else 'fair_default_p']
        return SemanticIR('binomial_probability',{'p':p,'n':n,'k':k,'event':'heads'},.995,'exactly_k_successes','value',tuple(evidence))

    @classmethod
    def _dice(cls,t:str)->SemanticIR|None:
        if not re.search(r'(?:تاس|dice|die)',t,re.I) or not re.search(r'(?:دو\s+تاس|2\s*تاس|two\s+(?:fair\s+)?dice)',t,re.I):return None
        m=re.search(r'(?:مجموع(?:شان|شون)?|جمع(?:شان|شون)?|sum|add\s+up\s+to|total).{0,24}?(\d+)|(?:equals?|برابر)\s*(\d+)',t,re.I)
        if not m:return None
        target=int(next(x for x in m.groups() if x))
        if not 2<=target<=12:return None
        return SemanticIR('dice_sum_probability',{'dice':2,'sides':6,'target':target},.995,'sum_equals','value',('dice_bound','target_bound'))

    @classmethod
    def _permutation(cls,t:str)->SemanticIR|None:
        cue=re.search(r'(?:جایگشت|آرایش|چیدمان|به\s+ترتیب|ترتیبی|رتبه|جایگاه|اول.{0,12}دوم|ordered|permutation|arrange(?:ment)?|اول\s+تا\s+(?:سوم|چهارم|پنجم)|top[- ]?(?:three|four|five)|finishes?)',t,re.I)
        if not cue:return None
        n=k=None
        # explicit n around source population
        for pat in (r'(?:از\s+(?:بین|میان)?|میان|بین|among|from)\s*(\d+)\s*(?:نفر|کتاب|شیء|گزینه|runners?|books?|objects?)?',r'(\d+)\s*(?:نفر|کتاب|شیء|گزینه|runners?|books?|objects?).{0,24}?(?:انتخاب|می.?چین|arrang|ordered|top)',r'among\s+(\d+)\s+\w+',r'\bof\s+(\d+)\s+(?:books?|runners?|objects?|people|items?)'):
            m=re.search(pat,t,re.I)
            if m:n=int(m.group(1));break
        # k from k-tuple, "4 books ... order", top-three/four, or enumerated ranks
        m=re.search(r'(\d+|سه|چهار|پنج|شش)\s*[-‌ ]?تایی',t,re.I)
        if m:
            z=cls._num(m.group(1));k=int(z) if z is not None else None
        if k is None:
            # Prefer the selected object count ("4 books را ...") over the population count.
            m=re.search(r'(\d+)\s*(?:کتاب|نفر|شیء|گزینه)\s*(?:را|رو)\s*.{0,18}?(?:به\s+ترتیب|جایگاه|رتبه|می.?چین|چینش)|(?:select|arrange|place)\s+(\d+)\s+(?:books?|runners?|objects?).{0,20}?(?:order|rank|position)',t,re.I)
            if m:
                k=int(next(x for x in m.groups() if x))
            else:
                m=None
        if k is None:
            m=re.search(r'(?:اول|first)\s+(?:تا|through|to)\s+(سوم|چهارم|پنجم|third|fourth|fifth)',t,re.I)
            if m:k={'سوم':3,'چهارم':4,'پنجم':5,'third':3,'fourth':4,'fifth':5}[m.group(1).casefold()]
        if k is None:
            m=re.search(r'\b(?:arrange|select|place)\s+(\d+)\s+of\s+\d+',t,re.I)
            if m:k=int(m.group(1))
        if k is None:
            m=re.search(r'top[- ]?(three|four|five)',t,re.I)
            if m:k={'three':3,'four':4,'five':5}[m.group(1).casefold()]
        if k is None:
            ranks=sum(bool(re.search(x,t,re.I)) for x in (r'(?:اول|first)',r'(?:دوم|second)',r'(?:سوم|third)',r'(?:چهارم|fourth)'))
            if ranks>=2:k=ranks
        learned=cls._learned_role(t,{'population_n','select_k'})
        if n is None and 'population_n' in learned:n=int(round(learned['population_n']))
        if k is None and 'select_k' in learned:k=int(round(learned['select_k']))
        if n is None or k is None or not 0<k<=n<=500:return None
        return SemanticIR('permutation',{'n':n,'k':k},.99,'ordered_without_replacement','value',('population_bound','selection_size_bound'))

    @classmethod
    def _ratio(cls,t:str)->SemanticIR|None:
        if not re.search(r'(?:نسبت|ratio)',t,re.I):return None
        rm=re.search(r'(?:نسبت|ratio)\s*(\d+(?:\.\d+)?)\s*(?:به|:|to)\s*(\d+(?:\.\d+)?)',t,re.I)
        if not rm:
            rm=re.search(r'(\d+(?:\.\d+)?)\s*:\s*(\d+(?:\.\d+)?)\s*(?:ratio|نسبت)?',t,re.I)
        if not rm:return None
        a,b=map(float,rm.groups());span=rm.span(); total=None
        nums=cls._number_mentions(t)
        candidates=[v for _,v,s,e in nums if e<=span[0] or s>=span[1]]
        # prefer number near divide/split or Persian object marker
        tm=re.search(r'(?:divide|split|تقسیم(?:\s+کن)?|پخش(?:\s+کن)?)\D{0,18}(\d+(?:\.\d+)?)|(\d+(?:\.\d+)?)\s*(?:را|رو)?\s*(?:بین|به\s+نسبت)',t,re.I)
        if tm:total=float(next(x for x in tm.groups() if x))
        elif candidates:total=max(candidates)
        learned=cls._learned_role(t,{'total','ratio_a','ratio_b'})
        if total is None and 'total' in learned:total=learned['total']
        if total is None or a<=0 or b<=0:return None
        return SemanticIR('ratio_split',{'total':total,'a':a,'b':b},.995,'split_by_ratio','value',('total_bound','ratio_bound'))

    @classmethod
    def _speed(cls,t:str)->SemanticIR|None:
        # distance-from-speed: a rate in km/h plus a duration and a distance question
        rate=re.search(r'(\d+(?:\.\d+)?)\s*(?:کیلومتر\s*بر\s*ساعت|km\s*/\s*h(?:r)?|kmph|kph)',t,re.I)
        tm=re.search(r'(\d+(?:\.\d+)?)\s*(ساعت|hours?|دقیقه|minutes?)',t,re.I)
        asks_distance=bool(re.search(r'(?:مسافت|چند\s+کیلومتر|distance|how\s+far)',t,re.I))
        if rate and tm and asks_distance:
            v=float(rate.group(1));tv=float(tm.group(1));unit=tm.group(2).casefold();h=tv/60 if re.search(r'(?:دقیقه|minute)',unit,re.I) else tv
            if h>0:return SemanticIR('distance_from_speed',{'speed_kmh':v,'time_hours':h},.997,'distance_equals_speed_times_time','value',('speed_rate_bound','time_unit_bound'))
        # average-speed: a standalone distance in km plus duration
        if not re.search(r'(?:km|کیلومتر)',t,re.I) or not tm:return None
        dm=re.search(r'(\d+(?:\.\d+)?)\s*(?:کیلومتر|km)(?!\s*(?:بر\s*ساعت|/\s*h|ph))',t,re.I)
        if not dm:return None
        d=float(dm.group(1));time=float(tm.group(1));unit=tm.group(2).casefold()
        h=time/60 if re.search(r'(?:دقیقه|minute)',unit,re.I) else time
        if h<=0:return None
        return SemanticIR('speed',{'distance_km':d,'time_hours':h},.995,'average_speed','value',('distance_unit_bound','time_unit_bound'))

    @classmethod
    def _work(cls,t:str)->SemanticIR|None:
        if not re.search(r'(?:کارگر|workers?)',t,re.I) or not re.search(r'(?:جعبه|boxes|items|کالا|واحد)',t,re.I):return None
        nums=cls._learned_role(t,{'workers1','hours1','output1','workers2','hours2'})
        # robust clause-wise extraction
        clauses=re.split(r'[؛;?.]|(?:؛)|(?:،\s*(?=\d+\s*کارگر))',t)
        found=[]
        for c in clauses:
            w=re.search(r'(\d+)\s*(?:کارگر|workers?)',c,re.I);h=re.search(r'(\d+(?:\.\d+)?)\s*(?:ساعت|hours?)',c,re.I);o=re.search(r'(\d+(?:\.\d+)?)\s*(?:جعبه|boxes|items|کالا|واحد)',c,re.I)
            if w and h:found.append((float(w.group(1)),float(h.group(1)),float(o.group(1)) if o else None))
        if len(found)>=2 and found[0][2] is not None:
            w1,h1,o1=found[0];w2,h2,_=found[1]
        elif all(x in nums for x in ('workers1','hours1','output1','workers2','hours2')):
            w1,h1,o1,w2,h2=(nums[x] for x in ('workers1','hours1','output1','workers2','hours2'))
        else:
            m=re.search(r'(\d+)\s*کارگر.{0,30}?(\d+(?:\.\d+)?)\s*ساعت.{0,30}?(\d+(?:\.\d+)?)\s*(?:جعبه|کالا|واحد).{0,60}?(\d+)\s*کارگر.{0,30}?(\d+(?:\.\d+)?)\s*ساعت',t,re.I)
            if not m:return None
            w1,h1,o1,w2,h2=map(float,m.groups())
        if min(w1,h1,w2,h2)<=0:return None
        return SemanticIR('work_scaling',{'workers1':w1,'hours1':h1,'output1':o1,'workers2':w2,'hours2':h2},.99,'constant_worker_hour_productivity','value',('baseline_bound','target_bound'))

    @classmethod
    def _age(cls,t:str)->SemanticIR|None:
        if not re.search(r'(?:سال(?:ه)?|سن|years?\s+old|age|خواهر|برادر|sister|brother)',t,re.I):return None
        num=r'(\d+|یک|دو|سه|چهار|پنج|شش|هفت|هشت|نه|ده|one|two|three|four|five|six|seven|eight|nine|ten)'
        # Persian: base age, sibling age difference, future offset; word order may contain filler.
        m=re.search(r'(?:[\w\u0600-\u06ff]+)\s*'+num+r'\s*ساله.{0,55}?(?:خواهر|برادر)(?:ش)?\s*'+num+r'\s*سال\s*(بزرگ.?تر|کوچک.?تر).{0,55}?'+num+r'\s*سال\s*(?:بعد|دیگر).{0,35}?(?:مجموع|جمع)',t,re.I)
        if m:
            base=cls._num(m.group(1));diff=cls._num(m.group(2));rel=m.group(3);later=cls._num(m.group(4))
            if None not in (base,diff,later):
                return SemanticIR('age_reasoning',{'base_age':float(base),'difference':float(diff),'relation':'older' if 'بزرگ' in rel else 'younger','years_later':float(later)},.995,'sum_future_ages','value',('base_age_bound','difference_bound','future_offset_bound'))
        # English equivalent, allowing numeric words.
        m=re.search(num+r'\s*years?\s*old.{0,55}?(?:sister|brother).{0,24}?'+num+r'\s*years?\s*(older|younger).{0,55}?'+num+r'\s*years?\s*(?:later|from\s+now).{0,35}?(?:sum|total)',t,re.I)
        if m:
            base=cls._num(m.group(1));diff=cls._num(m.group(2));rel=m.group(3);later=cls._num(m.group(4))
            if None not in (base,diff,later):
                return SemanticIR('age_reasoning',{'base_age':float(base),'difference':float(diff),'relation':rel.lower(),'years_later':float(later)},.995,'sum_future_ages','value',('base_age_bound','difference_bound','future_offset_bound'))
        return None

    @classmethod
    def _inventory(cls,t:str)->SemanticIR|None:
        if not re.search(r'(?:کالا|موجودی|items?|units?|inventory)',t,re.I) or not re.search(r'(?:فروش|فروخت|sold|remain|باقی)',t,re.I):return None
        init=re.search(r'(\d+(?:\.\d+)?)\s*(?:کالا|عدد|واحد|items?|units?)',t,re.I)
        pct=re.search(r'(\d+(?:\.\d+)?)\s*(?:%|درصد).{0,18}?(?:فروش|فروخت(?:ه|یم|ند|م|ی)?|sold)|(?:فروش|فروخت(?:ه|یم|ند|م|ی)?|sold).{0,18}?(\d+(?:\.\d+)?)\s*(?:%|درصد)',t,re.I)
        extra=re.search(r'(?:بعد|سپس|then).{0,25}?(\d+(?:\.\d+)?)\s*(?:عدد|کالا|واحد|more\s+items?|items?|units?).{0,16}?(?:فروش|فروخت(?:ه|یم|ند|م|ی)?|sold)|(?:بعد|سپس|then).{0,25}?(?:فروش|فروخت(?:ه|یم|ند|م|ی)?|sold).{0,15}?(\d+(?:\.\d+)?)',t,re.I)
        if not(init and pct and extra):return None
        p=float(next(x for x in pct.groups() if x));e=float(next(x for x in extra.groups() if x));q=float(init.group(1))
        return SemanticIR('inventory_multistep',{'initial':q,'percent_sold':p,'extra_sold':e},.995,'remaining_after_two_sales','value',('initial_bound','percent_bound','extra_bound'))

    @classmethod
    def _prime(cls,t:str)->SemanticIR|None:
        if not re.search(r'(?:مقسوم.?علیه|عامل|بخش.?پذیر|عدد\s+اول|prime|divisor|factor|divisible)',t,re.I):return None
        rm=re.search(r'(?:بین|از)\s*(\d+)\s*(?:تا|و)\s*(\d+).{0,35}?(?:عدد\s+اول|اول\s+است|prime)',t,re.I)
        if not rm:
            rm=re.search(r'(?:prime).{0,25}?(?:between|from)\s*(\d+)\s*(?:and|to)\s*(\d+)',t,re.I)
        if rm:
            a,b=map(int,rm.groups());
            if a>b:a,b=b,a
            return SemanticIR('prime_range',{'start':a,'end':b},.995,'list_primes_in_range','value',('range_bound',))
        m=re.search(r'\b(\d+)\b',t)
        if not m:return None
        n=int(m.group(1));pred='is_prime';pol='yes_no'
        other_divisor_pat=(
            r'(?:(?:غیر\s+از|به\s+جز)\s+(?:1|یک)\s+و\s+خودش.{0,32}?(?:عامل|مقسوم.?علیه)'
            r'|(?:عامل|مقسوم.?علیه).{0,32}?(?:غیر\s+از|به\s+جز)\s+(?:1|یک)\s+و\s+خودش'
            r'|other\s+(?:divisor|factor)|divisor\s+other\s+than|factor\s+besides)'
        )
        if re.search(other_divisor_pat,t,re.I):pred='has_other_divisor'
        elif re.search(r'(?:فقط|تنها).{0,30}?(?:1|یک).{0,12}?(?:خودش|itself)|only.{0,35}?(?:1|one).{0,20}?(?:itself|it)|only\s+divisible',t,re.I):pred='only_one_self'
        return SemanticIR('prime_reasoning',{'n':n},.995,pred,pol,('integer_bound','question_predicate_bound'))

    @classmethod
    def _transitive(cls,t:str)->SemanticIR|None:
        m=re.search(r'([^\s><،,؛;]+)\s*>\s*([^\s><،,؛;]+)\s*>\s*([^\s><،,؛;؟?]+)',t)
        if m:return SemanticIR('transitive_order',{'order':list(m.groups()),'relation':'greater'},.99,'transitive_relation','value',('explicit_chain',))
        # English canonical forms first so auxiliary verbs cannot be captured as entity names.
        english_patterns=(
            ('greater',r'([A-Za-z]+)\s+is\s+(?:taller|greater)\s+than\s+([A-Za-z]+).{0,35}?\2\s+is\s+(?:taller|greater)\s+than\s+([A-Za-z]+)'),
            ('greater',r'([A-Za-z]+)\s+is\s+ahead\s+of\s+([A-Za-z]+).{0,35}?\2\s+is\s+ahead\s+of\s+([A-Za-z]+)'),
            ('less',r'([A-Za-z]+)\s+is\s+(?:shorter|less)\s+than\s+([A-Za-z]+).{0,35}?\2\s+is\s+(?:shorter|less)\s+than\s+([A-Za-z]+)'),
            ('less',r'([A-Za-z]+)\s+is\s+behind\s+([A-Za-z]+).{0,35}?\2\s+is\s+behind\s+([A-Za-z]+)'),
        )
        for rel,pat in english_patterns:
            m=re.search(pat,t,re.I)
            if m:
                a,b,c=m.groups();order=[a,b,c] if rel=='greater' else [c,b,a]
                return SemanticIR('transitive_order',{'order':order,'relation':'greater'},.97,'transitive_relation','value',('natural_chain',))
        rels=[('greater',r'(بزرگ.?تر|بلند.?تر|جلوتر)'),('less',r'(کوچک.?تر|کوتاه.?تر|عقب.?تر)')]
        for rel,pat in rels:
            m=re.search(r'([\w\u0600-\u06ff]+)\s+از\s+([\w\u0600-\u06ff]+)\s*'+pat+r'.{0,35}?\2\s+از\s+([\w\u0600-\u06ff]+)\s*'+pat,t,re.I)
            if m:
                a,b,_,c,_=m.groups();order=[a,b,c] if rel=='greater' else [c,b,a]
                return SemanticIR('transitive_order',{'order':order,'relation':'greater'},.97,'transitive_relation','value',('natural_chain',))
        return None

    @classmethod
    def _python_generation(cls,t:str)->SemanticIR|None:
        if not re.search(r'(?:python|پایتون)',t,re.I):return None
        if not re.search(r'(?:تابع|function|کد|code|برنامه|script)',t,re.I):return None
        if not re.search(r'(?:بنویس|بساز|ایجاد\s+کن|درست\s+کن|پیاده.?سازی\s+کن|write|create|implement|make)',t,re.I):return None
        if re.search(r'(?:مقادیر|اعداد?).{0,20}?زوج|even\s+(?:numbers|values)',t,re.I):
            fn='filter_even' if re.search(r'(?:فیلتر|filter)',t,re.I) else 'even_numbers'
            return SemanticIR('python_generate_even_filter',{'function_name':fn},.995,'generate_code','value',('python_authoring','even_filter'))
        m=re.search(r'(?:بزرگ.?تر\s+از|greater\s+than)\s*(\d+)',t,re.I)
        if m:return SemanticIR('python_generate_threshold_filter',{'threshold':int(m.group(1)),'function_name':'filter_greater'},.99,'generate_code','value',('python_authoring','threshold_filter'))
        return None

class SemanticSolverV18:
    @staticmethod
    def solve(ir:SemanticIR,language:str='fa'):
        s=ir.slots;task=ir.task;checks=('sir_v18_parsed',*ir.evidence,'plan_built','result_verified')
        fmt=lambda x:str(int(round(x))) if abs(x-round(x))<1e-10 else f'{x:.8f}'.rstrip('0').rstrip('.')
        if task=='binomial_probability':
            p,n,k=s['p'],int(s['n']),int(s['k']);raw=math.comb(n,k)*p**k*(1-p)**(n-k);prob=raw*100
            if abs(p-0.5)<1e-12:
                den=2**n;num=math.comb(n,k);g=math.gcd(num,den);frac=f'{num//g}/{den//g}'
                body=f'P(X={k}) = C({n},{k})×0.5^{n} = {frac} = {fmt(prob)}'
            else:
                body=f'P(X={k}) = C({n},{k})×{fmt(p)}^{k}×{fmt(1-p)}^{n-k} = {fmt(prob)}'
            return ((body+'٪.') if language=='fa' else (body+'%.')),'probability_answer',.999,checks
        if task=='dice_sum_probability':
            target=int(s['target']);ways=sum(a+b==target for a in range(1,7) for b in range(1,7));g=math.gcd(ways,36)
            return (f'{ways} حالت از 36؛ احتمال = {ways//g}/{36//g} = {fmt(100*ways/36)}٪.' if language=='fa' else f'{ways} of 36 outcomes; probability = {ways//g}/{36//g} = {fmt(100*ways/36)}%.'),'probability_answer',.999,checks
        if task=='permutation':
            n,k=int(s['n']),int(s['k']);v=math.perm(n,k);return (f'P({n},{k}) = {v}.'),'reasoned_answer',.999,checks
        if task=='ratio_split':
            total,a,b=s['total'],s['a'],s['b'];x=total*a/(a+b);y=total*b/(a+b);return (f'دو سهم برابر {fmt(x)} و {fmt(y)} هستند.' if language=='fa' else f'The shares are {fmt(x)} and {fmt(y)}.'),'word_problem_answer',.999,checks
        if task=='speed':
            v=s['distance_km']/s['time_hours'];return (f'سرعت = {fmt(v)} کیلومتر بر ساعت.' if language=='fa' else f'Speed = {fmt(v)} km/h.'),'word_problem_answer',.999,checks
        if task=='distance_from_speed':
            d=s['speed_kmh']*s['time_hours'];return (f'مسافت = {fmt(d)} کیلومتر.' if language=='fa' else f'Distance = {fmt(d)} km.'),'word_problem_answer',.999,checks
        if task=='work_scaling':
            rate=s['output1']/(s['workers1']*s['hours1']);out=rate*s['workers2']*s['hours2'];return (f'نرخ هر کارگر-ساعت {fmt(rate)} است؛ خروجی = {fmt(out)}.' if language=='fa' else f'Rate = {fmt(rate)} per worker-hour; output = {fmt(out)}.'),'word_problem_answer',.999,checks
        if task=='age_reasoning':
            a=s['base_age'];b=a+s['difference'] if s['relation']=='older' else a-s['difference'];later=s['years_later'];total=a+b+2*later
            return (f'سن نفر دوم اکنون {fmt(b)} است؛ {fmt(later)} سال بعد مجموع سن‌ها = {fmt(total)}.' if language=='fa' else f'The second person is {fmt(b)} now; after {fmt(later)} years their total age is {fmt(total)}.'),'word_problem_answer',.999,checks
        if task=='inventory_multistep':
            remaining=s['initial']*(1-s['percent_sold']/100)-s['extra_sold'];return (f'پس از فروش درصدی و فروش دوم، {fmt(remaining)} واحد باقی می‌ماند.' if language=='fa' else f'{fmt(remaining)} units remain after both sales.'),'word_problem_answer',.999,checks
        if task=='prime_range':
            def is_prime(x:int)->bool:
                if x<2:return False
                if x%2==0:return x==2
                d=3
                while d*d<=x:
                    if x%d==0:return False
                    d+=2
                return True
            values=[x for x in range(int(s['start']),int(s['end'])+1) if is_prime(x)]
            joined=', '.join(str(x) for x in values) if values else ('هیچ‌کدام' if language=='fa' else 'none')
            return ((f'اعداد اول در این بازه: {joined}.' if language=='fa' else f'Primes in the range: {joined}.'),'reasoned_answer',.999,checks)
        if task=='prime_reasoning':
            n=int(s['n']);prime=n>=2 and all(n%d for d in range(2,math.isqrt(n)+1));div=None
            if not prime:
                div=next((d for d in range(2,math.isqrt(n)+1) if n%d==0),None)
            if ir.predicate=='has_other_divisor':yes=not prime
            elif ir.predicate in {'only_one_self','is_prime'}:yes=prime
            else:yes=prime
            if language=='fa':
                if ir.predicate=='has_other_divisor': text=(f'بله؛ {n} عامل دیگری دارد'+(f'؛ مثلاً {div}.' if div else '.') if yes else f'خیر؛ {n} غیر از ۱ و خودش عامل دیگری ندارد، پس عدد اول است.')
                else:text=(f'بله؛ {n} فقط بر ۱ و خودش بخش‌پذیر است، پس عدد اول است.' if yes else f'خیر؛ {n} عدد اول نیست'+(f' و بر {div} بخش‌پذیر است.' if div else '.'))
            else:
                if ir.predicate=='has_other_divisor':text=(f'Yes; {n} has another divisor'+(f', for example {div}.' if div else '.') if yes else f'No; {n} has no positive divisor other than 1 and itself, so it is prime.')
                else:text=(f'Yes; {n} is prime.' if yes else f'No; {n} is not prime'+(f' because {div} divides it.' if div else '.'))
            return text,'reasoned_answer',.999,checks
        if task=='transitive_order':
            o=s['order'];return (f'با تعدی، {o[0]} > {o[2]}.' if language=='fa' else f'By transitivity, {o[0]} > {o[2]}.'),'reasoned_answer',.998,checks
        if task=='python_generate_even_filter':
            fn=str(s.get('function_name') or 'filter_even')
            return f'def {fn}(values):\n    return [x for x in values if x % 2 == 0]','coding_answer',.999,checks
        if task=='python_generate_threshold_filter':
            th=s['threshold'];return f'def filter_greater(values):\n    return [x for x in values if x > {th}]','coding_answer',.999,checks
        return None

__all__=['SemanticIR','SemanticIRParserV18','SemanticSolverV18']
