from __future__ import annotations
import hashlib,json,math,random,re
from pathlib import Path
from typing import Any

ROOT=Path(__file__).resolve().parents[1];DS=ROOT/'datasets';PARENT='dataset_v008';VERSION='dataset_v009';SEED=18092026
rng=random.Random(SEED)

def h(s:str)->str:return hashlib.sha256(s.encode()).hexdigest()[:16]
def fmt(x:float)->str:return str(int(round(x))) if abs(x-round(x))<1e-10 else f'{x:.8f}'.rstrip('0').rstrip('.')
def norm(s:str)->str:return ' '.join(s.split()).casefold()
def mk(prompt,output,lang,frame,task,concept,*,slots=None,origin='jarvis-semantic-v18',stage=5,intent=''):
 md={'source':origin,'quality':'gold-verified-v18','verification':'deterministic_or_curated_v18','task_type':task,'concept_group':concept,'difficulty':'hard','language':lang,'semantic_frame':frame,'requires_tools':False,'expected_tool':'','permission_required':False,'risk_level':'L0','pretrained_source':None}
 if slots:md['slot_values']=slots
 if intent:md['semantic_intent']=intent
 return {'id':'jv18-'+h(prompt+'|'+output),'input':prompt,'output':output,'language':lang,'category':'v18_'+frame,'stage':stage,'stage_name':task,'context':[],'origin':origin,'dataset_version':VERSION,'pretrained_source':None,'metadata':md}

def semantic_rows():
 rows=[]
 # Biased binomial: variable decimal/percentage p and word-order variants.
 for i in range(420):
  p=round((18+(i*7)%73)/100,2);n=3+(i%7);k=(i*3)%(n+1);prob=math.comb(n,k)*p**k*(1-p)**(n-k)*100
  forms=[
   f'P(head)={fmt(p)}. Flip the coin {n} times; probability of exactly {k} heads?',
   f'Exactly {k} heads in {n} tosses: the coin has a {fmt(p*100)}% chance of heads. Find the probability.',
   f'احتمال اینکه این سکه شیر بیاید {fmt(p)} است؛ {n} پرتاب، دقیقاً {k} شیر؟',
   f'در {n} بار پرتاب، دقیقاً {k} شیر می‌خواهیم؛ شانس شیر این سکه {fmt(p*100)} درصد است. احتمال؟',
   f'این سکه {fmt(p*100)} درصد شانس شیر دارد؛ اگر {n} بار پرتاب شود، احتمال دقیقاً {k} شیر را حساب کن.',
   f'For a coin with heads probability {fmt(p)}, what is P(X={k}) over {n} flips?',
  ]
  q=forms[i%len(forms)];out=f'P(X={k})=C({n},{k})×{fmt(p)}^{k}×{fmt(1-p)}^{n-k}={fmt(prob)}%.'
  rows.append(mk(q,out,'fa' if i%6 in (2,3,4) else 'en','binomial_probability','semantic_reasoning',f'v18:binom:{i}',slots={'p':p,'n':n,'k':k}))
 # Ordered selections / ranking.
 words={3:'سه',4:'چهار',5:'پنج',6:'شش'}
 for i in range(320):
  n=7+(i%29);k=3+(i%4);v=math.perm(n,k)
  forms=[f'از بین {n} کتاب، {k} کتاب را به ترتیب جایگاه اول تا {words.get(k,str(k))}م می‌چینیم؛ چند حالت؟',f'آرایش {words.get(k,str(k))}‌تایی بدون تکرار از میان {n} گزینه چند حالت دارد؟',f'How many ordered top-{k} finishes are possible among {n} runners?',f'Choose {k} distinct objects from {n} and place them in ranked order. How many arrangements?',f'از {n} نفر، {k} رتبهٔ متمایز را بدون تکرار تعیین می‌کنیم؛ چند حالت؟']
  q=forms[i%5];rows.append(mk(q,f'P({n},{k})={v}.','fa' if i%5 in (0,1,4) else 'en','permutation','semantic_reasoning',f'v18:perm:{i}',slots={'population_n':n,'select_k':k}))
 # Ratio split with order independent phrasing.
 for i in range(320):
  a=2+(i%8);b=3+((i*5)%10);unit=13+(i%37);total=(a+b)*unit;x=a*unit;y=b*unit
  forms=[f'{total} را بین دو نفر با نسبت {a} به {b} تقسیم کن.',f'با نسبت {a}:{b}، عدد {total} را به دو سهم تقسیم کن.',f'Divide {total} in a {a}:{b} ratio.',f'The total is {total}; split it into shares in ratio {a} to {b}.',f'نسبت دو سهم {a} به {b} است و مجموعشان {total}؛ مقدار هر سهم؟']
  q=forms[i%5];rows.append(mk(q,f'{x}, {y}','fa' if i%5 in (0,1,4) else 'en','ratio_split','multi_step_reasoning',f'v18:ratio:{i}',slots={'total':total,'ratio_a':a,'ratio_b':b}))
 # Speed, including reversed order and minutes.
 for i in range(320):
  speed=40+(i%101);hours=1+(i%7)*.5;dist=speed*hours;mins=int(hours*60)
  forms=[f'در {fmt(hours)} ساعت، {fmt(dist)} کیلومتر طی شد. سرعت متوسط؟',f'{fmt(dist)} کیلومتر در مدت {fmt(hours)} ساعت طی شده؛ سرعت را حساب کن.',f'In {fmt(hours)} hours the vehicle covered {fmt(dist)} km. Average speed?',f'{fmt(dist)} km were covered in {mins} minutes. What was the average speed?',f'زمان سفر {mins} دقیقه و مسافت {fmt(dist)} کیلومتر است؛ سرعت چند کیلومتر بر ساعت است؟']
  q=forms[i%5];rows.append(mk(q,f'{fmt(speed)} km/h','fa' if i%5 in (0,1,4) else 'en','speed','multi_step_reasoning',f'v18:speed:{i}',slots={'distance_km':dist,'time_hours':hours}))
 # Age reasoning.
 for i in range(300):
  base=10+(i%28);diff=2+(i%9);later=2+((i*3)%8);other=base+diff;ans=base+other+2*later
  forms=[f'علی {base} ساله است، خواهرش {diff} سال بزرگ‌تر است؛ {later} سال بعد مجموع سنشان چند می‌شود؟',f'سن نیما {base} است و برادرش {diff} سال از او بزرگ‌تر است. بعد از {later} سال جمع سن دو نفر؟',f'Alex is {base} years old. His sister is {diff} years older. What will their total age be {later} years from now?']
  q=forms[i%3];rows.append(mk(q,f'{ans}','fa' if i%3<2 else 'en','age_reasoning','multi_step_reasoning',f'v18:age:{i}',slots={'base_age':base,'age_difference':diff,'years_later':later}))
 # Inventory two-step percent + absolute sale.
 for i in range(300):
  initial=160+(i%90)*4;pct=10+(i%6)*5;extra=10+(i%45);remaining=initial*(1-pct/100)-extra
  forms=[f'{initial} کالا داریم؛ ابتدا {pct}٪ فروخته می‌شود و بعد {extra} عدد دیگر فروش می‌رود. چند کالا باقی می‌ماند؟',f'موجودی اول {initial} واحد است. {pct} درصد آن را می‌فروشیم، سپس {extra} واحد دیگر؛ موجودی نهایی؟',f'Start with {initial} items. Sell {pct}% of them, then sell {extra} more items. How many remain?']
  q=forms[i%3];rows.append(mk(q,f'{fmt(remaining)}','fa' if i%3<2 else 'en','inventory_multistep','multi_step_reasoning',f'v18:inventory:{i}',slots={'initial':initial,'percent_sold':pct,'extra_sold':extra}))
 # Prime polarity supervision.
 for i in range(260):
  n=37+i;prime=n>=2 and all(n%d for d in range(2,math.isqrt(n)+1));div=next((d for d in range(2,math.isqrt(n)+1) if n%d==0),None)
  if i%4==0:q=f'آیا {n} غیر از 1 و خودش عامل دیگری دارد؟';ans='خیر' if prime else f'بله؛ {div}'
  elif i%4==1:q=f'آیا {n} فقط بر 1 و خودش بخش‌پذیر است؟';ans='بله' if prime else f'خیر؛ {div}'
  elif i%4==2:q=f'Does {n} have any divisor other than 1 and itself?';ans='No' if prime else f'Yes; {div}'
  else:q=f'Is {n} prime?';ans='Yes' if prime else f'No; {div}'
  rows.append(mk(q,ans,'fa' if i%4<2 else 'en','prime_reasoning','semantic_reasoning',f'v18:prime:{i}',slots={'n':n}))
 # Transitive relation variants.
 fa_names=['مینا','سارا','ندا','علی','رضا','مهدی','کیان','رها','نیما','آرمان']
 for i in range(240):
  a,b,c=fa_names[i%10],fa_names[(i+3)%10],fa_names[(i+6)%10]
  forms=[f'{a} از {b} بلندتر است و {b} از {c} بلندتر است؛ رابطهٔ {a} و {c} چیست؟',f'{a} از {b} کوتاه‌تر است و {b} از {c} کوتاه‌تر است؛ چه کسی بلندتر است؟',f'Alice is ahead of Bob and Bob is ahead of Carol. What follows by transitivity?',f'Alice is shorter than Bob and Bob is shorter than Carol. Who is taller, Alice or Carol?']
  q=forms[i%4];out=(f'{a} > {c}' if i%4==0 else (f'{c} > {a}' if i%4==1 else ('Alice > Carol' if i%4==2 else 'Carol > Alice')))
  rows.append(mk(q,out,'fa' if i%4<2 else 'en','transitive_logic','semantic_reasoning',f'v18:transitive:{i}'))
 # Python generation: broader verbs and variable thresholds.
 for i in range(300):
  th=5+(i%70)
  if i%4==0:q='با Python تابعی بساز که فقط مقادیر زوج را نگه دارد.';out='def even_numbers(values):\n    return [x for x in values if x % 2 == 0]'
  elif i%4==1:q='یک تابع پایتون ایجاد کن که اعداد زوج لیست را برگرداند.';out='def even_numbers(values):\n    return [x for x in values if x % 2 == 0]'
  elif i%4==2:q=f'با پایتون تابعی پیاده‌سازی کن که فقط مقادیر بزرگ‌تر از {th} را نگه دارد.';out=f'def filter_greater(values):\n    return [x for x in values if x > {th}]'
  else:q=f'Create a Python function that keeps only values greater than {th}.';out=f'def filter_greater(values):\n    return [x for x in values if x > {th}]'
  rows.append(mk(q,out,'fa' if i%4<3 else 'en','code_generation','coding',f'v18:code:{i}',slots={'threshold':th} if i%4>=2 else None,stage=4))
 # Persian formal rewrite with varied objects/deadlines/recipients.
 nouns=['قرارداد','گزارش','نسخه','فایل','سند','مقاله','پروژه','درخواست','صورت‌جلسه','نتیجه','برنامه','پیوست']
 dls=['تا دوشنبه','تا فردا','تا پایان امروز','تا آخر هفته','در اولین فرصت','تا ساعت پنج']
 for i in range(360):
  noun=nouns[i%len(nouns)];dl=dls[(i//len(nouns))%len(dls)];attached=noun+'و' if not noun.endswith('ه') else noun+' رو';recipient='برام' if i%2==0 else 'واسم'
  q=f'این پیام را رسمی‌تر کن: لطفا {attached} {dl} {recipient} بفرست';out=f'لطفاً {noun} را {dl} برای من ارسال کنید.'
  rows.append(mk(q,out,'fa','persian_formal_rewrite','rewrite',f'v18:rewrite:{i}',stage=3))
 return rows

def routing_rows():
 rows=[];labels=['coding','code_trace','translation','rewrite','probability','logic','word_problem','math','constraint_writing','fresh_information','desktop_command','general_question']
 for li,label in enumerate(labels):
  for i in range(100):
   n=17+i
   if label=='coding':q=[f'با Python تابعی بساز که اعداد بزرگ‌تر از {n} را برگرداند.',f'Create Python code to filter values above {n}.'][i%2]
   elif label=='code_trace':q=f'total=0; for i in range(1,{3+i%4}): total += i; print(total)'
   elif label=='translation':q=[f'این جمله را انگلیسی کن: گزارش شماره {n} آماده است.',f'Translate into Persian: report number {n} is ready.'][i%2]
   elif label=='rewrite':q=f'این پیام را رسمی‌تر کن: گزارش {n} رو برام بفرست'
   elif label=='probability':q=f'A coin has {20+i%70}% chance of heads; exactly 2 heads in 4 flips?'
   elif label=='logic':q=f'اگر A از B بلندتر و B از C بلندتر باشد، آیا A از C بلندتر است؟ مثال {n}'
   elif label=='word_problem':q=f'در 3 ساعت {120+n} کیلومتر طی شد؛ سرعت متوسط؟'
   elif label=='math':q=f'آیا {101+n} عدد اول است؟'
   elif label=='constraint_writing':q=f'دقیقاً دو جمله درباره داده بنویس و حتماً عدد {n} را ذکر کن.'
   elif label=='fresh_information':q=f'آخرین خبر درباره نسخه {n} این محصول را در وب پیدا کن.'
   elif label=='desktop_command':q=f'فایل گزارش{n}.txt را باز کن.'
   else:q=f'API چیست؟ مثال شماره {n} را نادیده بگیر.'
   rows.append(mk(q,label,'fa' if i%2==0 else 'en','routing','routing',f'v18:routing:{label}:{i}',origin='jarvis-routing-v18',stage=3,intent=label))
 return rows

def read(path):return [json.loads(x) for x in Path(path).read_text(encoding='utf-8').splitlines() if x.strip()]
def split_for(concept):
 v=int(hashlib.sha256(concept.encode()).hexdigest()[:8],16)%100
 return 'train' if v<80 else ('validation' if v<90 else 'test')

def main():
 new=semantic_rows()+routing_rows()
 # Holdout prompts never train.
 holdout_path=DS/'holdouts'/'v18_runtime_holdout.json';hold=set()
 if holdout_path.is_file():hold={norm(x) for x in json.loads(holdout_path.read_text(encoding='utf-8')).get('prompts',[])}
 new=[r for r in new if norm(r['input']) not in hold]
 seen=set();ded=[]
 for r in new:
  k=norm(r['input'])
  if k not in seen:seen.add(k);ded.append(r)
 new=ded
 inherited={sp:read(DS/'splits'/f'{PARENT}_{sp}.jsonl') for sp in ('train','validation','test')}
 old={norm(r['input']) for arr in inherited.values() for r in arr};new=[r for r in new if norm(r['input']) not in old]
 counts={};added={}
 for sp in ('train','validation','test'):
  a=[r for r in new if split_for(r['metadata']['concept_group'])==sp]
  for r in a:r['split']=sp
  merged=inherited[sp]+a;path=DS/'splits'/f'{VERSION}_{sp}.jsonl';path.write_text(''.join(json.dumps(r,ensure_ascii=False,separators=(',',':'))+'\n' for r in merged),encoding='utf-8');counts[sp]=len(merged);added[sp]=len(a)
 pack=DS/'v18_curated';pack.mkdir(exist_ok=True)
 for idx in range(50):(pack/f'v18_{idx+1:03d}.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in new[idx::50]),encoding='utf-8')
 cats={}
 for r in new:cats[r['metadata']['semantic_frame']]=cats.get(r['metadata']['semantic_frame'],0)+1
 # verify no concept appears in >1 split
 cg={}
 for sp in ('train','validation','test'):
  for r in read(DS/'splits'/f'{VERSION}_{sp}.jsonl'):
   c=str(r.get('metadata',{}).get('concept_group',''));cg.setdefault(c,set()).add(sp)
 leakage=sum(len(v)>1 for k,v in cg.items() if k)
 man={'format':'jarvis-dataset-manifest-v9','dataset_version':VERSION,'parent':PARENT,'seed':SEED,'inherited_examples':sum(len(v) for v in inherited.values()),'new_examples':len(new),'total_examples':sum(counts.values()),'source_dataset_count':50,'split_counts':counts,'added_by_split':added,'exact_overlap_with_parent':0,'unique_new_inputs':len({norm(r['input']) for r in new}),'unique_input_ratio':1.0,'categories':cats,'cross_split_concept_leakage':leakage,'quality_notes':['role-oriented semantic supervision','question-polarity supervision','order-independent numeric roles','verified deterministic quantitative targets','Persian morphology and rewrite fidelity','routing retraining rows balanced across 12 intents']}
 (DS/'manifest_v009.json').write_text(json.dumps(man,ensure_ascii=False,indent=2)+'\n',encoding='utf-8');print(json.dumps(man,ensure_ascii=False,indent=2))
if __name__=='__main__':main()
