"""Freeze 300 diagnostic cases before evaluation; 30 shared template families.
Not a claim of 300 independent linguistic templates or a general intelligence test.
"""
from pathlib import Path
from fractions import Fraction
import json,hashlib,random
ROOT=Path(__file__).resolve().parents[1]
work=[
'با {w} کارگر و {h} ساعت، خروجی {o} قطعه است. با {v} کارگر و {t} ساعت چه خروجی داریم؟',
'{w} کارگر در {h} ساعت {o} قطعه می سازند. {v} کارگر در {t} ساعت چند قطعه می سازند؟',
'در {h} ساعت، {w} کارگر {o} قطعه تولید می کنند؛ حالا در {t} ساعت، {v} کارگر چند قطعه تولید می کنند؟',
'با {v} کارگر و {t} ساعت چه خروجی داریم؟ با {w} کارگر و {h} ساعت، خروجی {o} قطعه است.',
'{w} workers make {o} pieces in {h} hours. What will {v} workers make in {t} hours?',
'What will {v} workers make in {t} hours? {w} workers make {o} pieces in {h} hours.',
'In {h} hours, {w} workers produce {o} units; now {v} workers have {t} hours. Find output.',
'{w} کارگر در {h} ساعت {o} جعبه می سازند؛ {v} کارگر در {t} ساعت چند جعبه؟',
'{w} workers produce {o} boxes in {h} hours; {v} workers in {t} hours produce how many boxes?',
'برای {w} کارگر، زمان {h} ساعت و خروجی {o} قطعه است. برای {v} کارگر، زمان {t} ساعت است؛ خروجی چقدر است؟',
'{w} workers need {h} hours for {o} items. How many items can {v} workers finish in {t} hours?',
'ابتدا {w} کارگر طی {h} ساعت {o} قطعه ساختند. اکنون {v} کارگر طی {t} ساعت چند قطعه می سازند؟',
'{v} کارگر طی {t} ساعت چند قطعه می سازند؟ ابتدا {w} کارگر طی {h} ساعت {o} قطعه ساختند.',
'{w} کارگر در {minutes} دقیقه {o} قطعه تولید می کنند. {v} کارگر در {t} ساعت چقدر تولید می کنند؟',
'{w} workers make {o} pieces in {minutes} minutes. What will {v} workers make in {t} hours?',
]
seq=[
'دنباله {terms}؛ جمله {n} چیست؟',
'دنباله {terms}؛ جمله {ordinal} را پیدا کن.',
'جمله {ordinal} در دنباله {terms} چیست؟',
'Find term {n} of the sequence {english}.',
'What is the {enordinal} term in {english}?',
'The sequence is {english}. Find term number {n}.',
'دنباله {terms} است. عدد بعدی چیست؟',
'What comes next in the sequence {english}?',
'دنباله {terms}؛ جمله پنجم چیست؟',
'جمله هشتم دنباله {terms} را حساب کن.',
'In the sequence {english}, find the term at position {n}.',
'در دنباله {terms}، عضو شماره {n} چقدر است؟',
'دنباله {terms} است؛ جمله {ordinal} را می خواهم، نه صرفاً عدد بعدی.',
'Sequence: {english}. Give the {enordinal} term, not just the next value.',
'سؤال 91: دنباله {terms}؛ جمله {ordinal} را بیاب.',
]
fa={5:'پنجم',6:'ششم',7:'هفتم',8:'هشتم',9:'نهم'}
en={5:'fifth',6:'sixth',7:'seventh',8:'eighth',9:'ninth'}
rows=[]; rng=random.Random(20260907)
for family,template in enumerate(work):
 for variant in range(10):
  w=rng.randint(2,12);h=rng.randint(2,9);v=rng.randint(2,15);t=rng.randint(2,10);o=w*h*rng.randint(2,13)
  q=template.format(w=w,h=h,v=v,t=t,o=o,minutes=h*60)
  rows.append(dict(id=f'work-{family:02}-{variant:02}',family=f'work-{family:02}',category='work_rate',kind='number',question=q,expected=o*v*t/(w*h)))
for family,template in enumerate(seq):
 for variant in range(10):
  n=5+variant%5
  if variant%2==0:
   a=Fraction(rng.randint(5,30)*8); ratio=Fraction(1,2);values=[a,a*ratio,a*ratio**2];typ='geometric'
  else:
   a=Fraction(rng.randint(-20,30));d=Fraction(rng.randint(2,11));values=[a,a+d,a+2*d];typ='arithmetic'
  requested=4 if family in (6,7) else 5 if family==8 else 8 if family==9 else n
  answer=a*ratio**(requested-1) if typ=='geometric' else a+(requested-1)*d
  fmt=lambda x:str(int(x)) if x.denominator==1 else str(float(x))
  q=template.format(terms='، '.join(map(fmt,values)),english=', '.join(map(fmt,values)),n=n,ordinal=fa[n],enordinal=en[n])
  rows.append(dict(id=f'sequence-{family:02}-{variant:02}',family=f'sequence-{family:02}',category='sequence_'+typ,kind='number',question=q,expected=float(answer)))
assert len(rows)==300 and len({r['question'] for r in rows})==300
p=ROOT/'benchmarks/diagnostic_v20_1_300.jsonl'
p.write_text('\n'.join(json.dumps(r,ensure_ascii=False) for r in rows)+'\n')
(ROOT/'reports/v20_1/diagnostic_protocol.json').write_text(json.dumps({'cases':300,'template_families':30,'seed':20260907,'sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'scope':'150 work-rate, 150 sequence; synthetic diagnostic, not a broad external holdout','frozen_before_first_evaluation':True,'no_training_on_these_cases':True,'score_claim':'No overall intelligence score; shared templates limit independence.'},indent=2))
