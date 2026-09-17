"""Frozen 500-case mixed diagnostic/holdout. Built before first evaluation.
Shared mathematical families are disclosed; no 500 independent-template claim.
"""
from pathlib import Path
import json,random,math,hashlib
from fractions import Fraction
R=Path(__file__).resolve().parents[1];rng=random.Random(21500);rows=[]
def add(cat,q,expected,kind='number',**extra):rows.append(dict(id=f'b21_{len(rows):03}',category=cat,question=q,expected=expected,kind=kind,**extra))
# 100 reasoning: unseen wording and independently calculated numerical oracles.
for i in range(20):
 a,b=rng.randint(2,11),rng.randint(2,9);total=(a+b)*rng.randint(20,80)
 add('Reasoning',f'برای دو بخش به ترتیب نسبت {a} به {b} تعیین شده؛ کل بودجه {total} است. سهم هر بخش را بده.',[total*a/(a+b),total*b/(a+b)],'parts',family='ratio')
for i in range(20):
 n=rng.randint(4,9);k=rng.randint(0,n);p=(15+i*3)/100
 add('Reasoning',f'Each of {n} trials independently succeeds with probability {p}. Calculate the chance of exactly {k} successes.',100*math.comb(n,k)*p**k*(1-p)**(n-k),family='binomial')
for i in range(20):
 a=rng.randint(8,90)*4;n=rng.randint(5,9)
 add('Reasoning',f'جمله شماره {n} را می خواهم: دنباله {a}، {a/2:g}، {a/4:g}.',a*.5**(n-1),family='sequence')
for i in range(20):
 a=rng.randint(200,900);b=rng.randint(5,25);c=rng.randint(3,17)
 add('Reasoning',f'Begin at {a}; increase by {b} percent, then subtract {c}. Report the final value.',a*(1+b/100)-c,family='multistep')
for i in range(20):
 a=rng.randint(15,40);d=rng.randint(2,8);y=rng.randint(2,10)
 add('Reasoning',f'Nima is {a} years old. His sister is {d} years younger. What is their combined age after {y} years?',2*a-d+2*y,family='age')
# 100 world-state cases, including abstention on physically impossible stock.
for i in range(25):
 a,b,c=rng.randint(120,400),rng.randint(30,110),rng.randint(10,90)
 add('World Model',f'Reza has {a} dollars; Mina has {b} dollars. Reza transfers {c} dollars to Mina. Balance of Mina?',b+c,family='ownership')
for i in range(25):
 a,c,p=rng.randint(600,1300),rng.randint(100,400),rng.randint(10,40)
 add('World Model',f'رضا {a} دلار داشت؛ {c} دلار خرید کرد و {p} درصد تخفیف گرفت؛ چقدر پول باقی است؟',a-c*(1-p/100),family='purchase')
for i in range(20):
 a=rng.randint(30,100);b=rng.randint(2,20);c=rng.randint(1,10)
 add('World Model',f'موجودی انبار {a} کالا است؛ {b} کم کن؛ {c} اضافه کن. موجودی آخر؟',a-b+c,family='inventory')
for i in range(20):
 start=6+i%7;d=1+i//7;m=15+(i%3)*15
 add('World Model',f'کار ساعت {start} شروع می شود؛ مدت کار {d} ساعت است؛ مرحله بعد {m} دقیقه طول دارد؛ پایان کار؟',start+d+m/60,family='scheduling')
for i in range(10):
 d=rng.randint(60,240);t=rng.choice([30,45,90])
 add('World Model',f'مسافت {d} کیلومتر را در {t} دقیقه طی کردیم؛ سرعت بر حسب کیلومتر بر ساعت؟',d/(t/60),family='speed')
# 100 adversarial cases: scope, ordering, ambiguity, ID numbers, and conversion.
for i in range(20):
 x=(30+i*7)*2
 add('Adversarial',f'سه برابر نصف {x} چقدر است؟',x*1.5,family='nested_fraction')
for i in range(20):
 x=30+i*7
 add('Adversarial',f'عدد {x} است؛ 20 درصد زیاد کن؛ 10 درصد کم کن. منظور دو تغییر پیاپی است.',x*1.08,family='percent_compounding')
for i in range(20):
 a=(20+i*3)*4
 add('Adversarial',f'شناسه 731 را نادیده بگیر. دنباله {a}، {a/2:g}، {a/4:g}؛ جمله هفتم چیست، نه فقط عدد بعدی؟',a/64,family='irrelevant_number')
for i in range(20):
 a=rng.randint(1,30);b=a+rng.randint(1,30)
 add('Adversarial',f'موجودی انبار {a} کالا است؛ {b} کم کن. آیا موجودی نهایی قابل قبول است؟',None,'abstain',family='impossible_stock')
for i in range(20):
 w,h,o,v,t=rng.randint(2,9),rng.randint(2,6),rng.randint(2,12),rng.randint(3,11),rng.randint(2,7)
 add('Adversarial',f'{v} workers in {t} hours produce how many pieces? {w} workers make {w*h*o} pieces in {h*60} minutes.',v*t*o,family='work_reversed_units')
# 100 code tasks; behavior scored with independent bounded input/output cases.
filters=['positive','negative','even','odd','nonzero'];test_inputs=[[],[0],[-3,0,2,5],[8,4,2],[-4,-2]]
for i in range(50):
 filt=filters[i%5];m=rng.randint(2,7);a=rng.randint(1,9);reduce=i%2==0
 pred={'positive':lambda x:x>0,'negative':lambda x:x<0,'even':lambda x:x%2==0,'odd':lambda x:x%2!=0,'nonzero':lambda x:x!=0}[filt]
 checks=[]
 for xs in test_inputs:
  ys=[x*m+a for x in xs if pred(x)];checks.append({'args':[xs],'expected':sum(ys) if reduce else ys})
 q=f'Create a Python function named calculate_{i}: keep {filt} values; multiply by {m}; add {a}; '+('sum output.' if reduce else 'return the resulting list.')
 add('Code',q,None,'code',checks=checks,family='generate')
for i in range(20):
 m=rng.randint(2,9);a=rng.randint(1,11)
 src=f'def process_{i}(values)\n    return [x * {m} - {a} for x in values]\n'
 add('Code','Fix the syntax while keeping the operation:\n```python\n'+src+'```',None,'code',checks=[{'args':[xs],'expected':[x*m-a for x in xs]} for xs in test_inputs],family='debug')
for i in range(15):
 m=rng.randint(2,9)
 src=f'def collect_{i}(values):\n    result = []\n    for x in values:\n        if x > {i}:\n            result.append(x * {m})\n    return result\n'
 add('Code','Refactor this function without changing its behavior:\n```python\n'+src+'```',None,'code',checks=[{'args':[xs],'expected':[x*m for x in xs if x>i]} for xs in test_inputs],family='refactor')
for i in range(15):
 src=f'def pick_{i}(items):\n    return [n + {i+2} for n in items if n < {i+1}]\n'
 add('Code','Explain what this function returns and what condition it uses:\n```python\n'+src+'```',[f'n + {i+2}',f'n < {i+1}'],'contains',family='explain')
# 50 Persian and 50 English instructions with explicit, reviewable semantic keys.
pairs=[('کتاب روی میز است.','The book is on the table.'),('من فردا به مدرسه می روم.','I will go to school tomorrow.'),('پنجره را باز کن.','Open the window.'),('او هر روز ورزش می کند.','She exercises every day.'),('هوا امروز سرد است.','The weather is cold today.'),('ما به کمک نیاز داریم.','We need help.'),('این تصمیم مهم است.','This decision is important.'),('جلسه ساعت نه شروع می شود.','The meeting starts at nine.'),('لطفاً دوباره توضیح بده.','Please explain again.'),('آنها دیروز رسیدند.','They arrived yesterday.'),('من کلیدم را پیدا کردم.','I found my key.'),('این خانه دو اتاق دارد.','This house has two rooms.'),('صدای تو را می شنوم.','I can hear your voice.'),('دوستم در تهران زندگی می کند.','My friend lives in Tehran.'),('پروژه هنوز تمام نشده است.','The project is not finished yet.'),('می توانیم فردا صحبت کنیم.','We can talk tomorrow.'),('لطفاً کمی صبر کن.','Please wait a moment.'),('کودک در حال خواب است.','The child is sleeping.'),('این راه کوتاه تر است.','This route is shorter.'),('من با تو موافق نیستم.','I do not agree with you.'),('به وقت بیشتری نیاز دارم.','I need more time.'),('در را آرام ببند.','Close the door gently.'),('پاسخ تو درست است.','Your answer is correct.'),('این فایل را ذخیره کن.','Save this file.'),('از کمکت ممنونم.','Thank you for your help.')]
for fa,en in pairs:
 add('Persian','این جمله را به فارسی ترجمه کن: '+en,[fa],'translation',family='translation_en_fa')
 add('English','Translate this Persian sentence into English: '+fa,[en],'translation',family='translation_fa_en')
fa_rewrites=[('میخوام این کارو انجام بدی','می‌خواهم این کار را انجام دهید'),('چرا جواب ندادی','چرا پاسخ ندادید'),('فایل رو واسم بفرست','فایل را برای من ارسال کنید'),('این مشکل رو درست کن','این مشکل را برطرف کنید'),('یه کم صبر کن','کمی صبر کنید')]
for i in range(25):
 raw,target=fa_rewrites[i%5];context=['در ایمیل اداری','با لحن محترمانه','برای همکار','در پیام کاری','برای نامه رسمی'][i//5]
 add('Persian',f'{context}، این جمله را رسمی بازنویسی کن: «{raw}»',[target],'translation',family='formal_rewrite')
english_grammar=[('She go to work every day.','She goes to work every day.'),('They was late yesterday.','They were late yesterday.'),('I has two books.','I have two books.'),('He do not know.','He does not know.'),('We is ready.','We are ready.')]
for i in range(25):
 raw,target=english_grammar[i%5];prefix=['Correct the grammar','Fix the grammatical error','Rewrite in grammatical English','Provide the corrected sentence','Correct this sentence without changing its meaning'][i//5]
 add('English',prefix+': '+raw,[target],'translation',family='grammar')
assert len(rows)==500
assert len({r['question'] for r in rows})==500, [q for q in {r['question'] for r in rows} if sum(r['question']==q for r in rows)>1]
from collections import Counter
assert Counter(r['category'] for r in rows)=={'Reasoning':100,'Persian':50,'English':50,'Code':100,'World Model':100,'Adversarial':100}
p=R/'benchmarks/broad_v21_500.jsonl';p.write_text('\n'.join(json.dumps(r,ensure_ascii=False) for r in rows)+'\n')
(R/'reports/v21/broad_protocol.json').write_text(json.dumps({'count':500,'categories':dict(Counter(r['category'] for r in rows)),'sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'seed':21500,'frozen_before_evaluation':True,'planned_runs':'v20.1 baseline and v21 candidate, same machine and scoring','use_for_training':False,'limitations':['Synthetic numerical variants and repeated language instructions are present; 500 cases do not mean 500 independent templates.','Translation score uses normalized reference sentences and is strict; valid paraphrases may be underscored.','No conversion to the user intelligence index.']},indent=2))
print('frozen',len(rows))
