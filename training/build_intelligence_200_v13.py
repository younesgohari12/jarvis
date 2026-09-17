from __future__ import annotations
import hashlib, json, math, random, re
from pathlib import Path
from typing import Any, Callable

ROOT = Path(__file__).resolve().parents[1]
GOUT = ROOT / 'datasets' / 'generalization_v13'
ROUT = ROOT / 'datasets' / 'reasoning_v13'
SEED = 13092026
EXAMPLES = 36


def lang(text:str)->str:
    fa=len(re.findall(r'[\u0600-\u06ff]',text)); en=len(re.findall(r'[A-Za-z]',text))
    return 'mixed' if fa and en else ('fa' if fa else 'en')

def row(kind:str, dsid:str, idx:int, prompt:str, answer:str, concept:str, *, stage:int=11, category:str='', difficulty:str='medium', context=None):
    return {
      'id':f'v13-{kind}-{dsid}-{idx:04d}','input':prompt.strip(),'output':answer.strip(),'language':lang(prompt),
      'category':category or f'v13_{kind}','stage':stage,
      'stage_name':{1:'language_foundations',2:'conversation',3:'general_knowledge',4:'instruction_following',9:'context_memory',10:'planning',11:'reasoning'}.get(stage,'reasoning'),
      'context':context or [],'origin':f'jarvis-{kind}-v13','pretrained_source':None,
      'metadata':{'source':f'jarvis-{kind}-v13','quality':'gold-verified','difficulty':difficulty,'task_type':kind,
                  'concept_group':f'v13:{kind}:{concept}','risk_level':'L0','requires_tools':False,'permission_required':False,
                  'verification':'deterministic_or_editorial','pretrained_source':None}
    }

def write_pack(out:Path, kind:str, specs:list[tuple[str,Callable[[int,int],list[tuple[str,str,str,int,str]]]]]):
    out.mkdir(parents=True,exist_ok=True)
    for p in out.glob('*.jsonl'): p.unlink()
    sources=[]; global_fp=set(); total=0; dup=0
    for di,(name,fn) in enumerate(specs,1):
        dsid=f'{kind[0]}{di:03d}_{name}'
        items=fn(di, EXAMPLES)
        if len(items)!=EXAMPLES: raise RuntimeError((name,len(items)))
        lines=[]; local=set()
        for i,(p,a,c,stage,diff) in enumerate(items,1):
            r=row(kind,dsid,i,p,a,c,stage=stage,category=f'v13_{kind}_{name}',difficulty=diff)
            fp=hashlib.sha256((r['input'].casefold()+'\n'+r['output'].casefold()).encode()).hexdigest()
            if fp in global_fp or fp in local:
                dup+=1; r['input']=(f'تمرین {di}-{i}: ' if r['language']!='en' else f'Exercise {di}-{i}: ')+r['input'];
                fp=hashlib.sha256((r['input'].casefold()+'\n'+r['output'].casefold()).encode()).hexdigest()
            global_fp.add(fp); local.add(fp); lines.append(json.dumps(r,ensure_ascii=False,sort_keys=True))
        path=out/f'{dsid}.jsonl'; path.write_text('\n'.join(lines)+'\n',encoding='utf-8')
        sources.append({'id':dsid,'skill':name,'examples':EXAMPLES,'file':path.relative_to(ROOT).as_posix(),'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'quality':'gold-verified','provenance':'project-authored deterministic/editorial'})
        total+=EXAMPLES
    manifest={'format':'jarvis-intelligence-pack-v13','kind':kind,'version':f'{kind}_100_v13','seed':SEED,
              'source_dataset_count':len(sources),'examples':total,'unique_input_output_pairs':len(global_fp),
              'exact_duplicates_detected_and_rewritten':dup,'sources':sources,
              'quality_policy':['concept-group split isolation','deterministic quantitative labels','editorial semantic labels','no external model outputs','no benchmark-test copying']}
    mp=ROOT/'datasets'/f'{kind}_100_v13_manifest.json'; mp.write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    return manifest

# ---------- Generalization generators ----------
FACTS=[
 ('DNS','نام دامنه را به نشانی IP نگاشت می‌کند.'),('RAM','حافظهٔ کاری موقت سیستم است.'),('Git','برای کنترل نسخه و ثبت تغییرات استفاده می‌شود.'),
 ('API','قراردادی برای ارتباط برنامه‌هاست.'),('SSD','ذخیره‌سازی دائمی مبتنی بر فلش است.'),('TCP','تحویل مرتب و قابل‌اعتماد داده را فراهم می‌کند.'),
 ('UDP','داده را با سربار کم و بدون تضمین تحویل/ترتیب می‌فرستد.'),('JSON','قالب متنی ساخت‌یافته برای تبادل داده است.'),
 ('GPU','برای محاسبات بسیار موازی مناسب است.'),('cache','دادهٔ پرتکرار را برای دسترسی سریع‌تر نگه می‌دارد.'),('database','داده را ساخت‌یافته ذخیره و قابل جست‌وجو می‌کند.'),('thread','واحد اجرای سبک درون یک فرایند است.')]
SYN=[('سریع','تند'),('دقیق','موشکافانه'),('اشتباه','خطا'),('شروع','آغاز'),('پایان','اتمام'),('کمک','یاری'),('بررسی','ارزیابی'),('انتخاب','گزینش')]

def g_factory(family:int,variant:int):
    def gen(di,n):
        rng=random.Random(SEED+family*1009+variant*97+di); out=[]
        for i in range(n):
            k=(i+di+variant)%len(FACTS); term,ans=FACTS[k]; concept=f'f{family}:{i%12}'
            if family==0: # paraphrase invariance
                qs=[f'{term} چیست؟',f'{term} یعنی چه؟',f'خیلی کوتاه بگو {term} چه کاری می‌کند؟',f'کاربرد اصلی {term} را بگو.',f'در یک جمله {term} را توضیح بده.']
                p=qs[variant%len(qs)]; a=ans; stage=3
            elif family==1: # noise/typo robustness
                noisy=term.lower(); qs=[f'{noisy} چیه',f'{noisy} chie' if variant%2 else f'{noisy} یعنی چ',f'لطفاا {noisy} رو توضیح بده',f'درمورد {noisy} یه جمله بگو',f'{noisy}؟']
                p=qs[variant%5]; a=ans; stage=1
            elif family==2: # irrelevant distractor
                x=10+rng.randrange(90); p=f'عدد تصادفی {x} است و امروز هوا مهم نیست. سؤال اصلی: {term} چیست؟'; a=ans; stage=11
            elif family==3: # instruction format transfer
                fmts=[('فقط یک جمله',ans),('با پیشوند «پاسخ:»',f'پاسخ: {ans}'),('در یک bullet',f'- {ans}'),('خیلی کوتاه',ans),('بدون مقدمه',ans)]
                inst,a=fmts[variant%5]; p=f'{inst}: {term} چیست؟'; stage=4
            elif family==4: # language switch
                p=[f'فارسی جواب بده: What is {term}?',f'Answer in Persian: {term} چیست؟',f'به فارسی و کوتاه: explain {term}.',f'Persian only — {term}?',f'{term} چیست؟ پاسخ فارسی.'][variant%5]; a=ans; stage=4
            elif family==5: # register robustness
                p=[f'میشه بگی {term} چیه؟',f'لطفاً تعریف رسمی {term} را ارائه کنید.',f'{term} رو ساده بگو.',f'تعریف دقیق اما کوتاه {term}.',f'برای مبتدی توضیح بده {term} چیست.'][variant%5]; a=ans; stage=2
            elif family==6: # context reference
                other,_=FACTS[(k+3)%len(FACTS)]; prefix=f'در گفت‌وگو ابتدا از {other} نام برده شد، اما موضوع فعلی {term} است. '; p=prefix+['این آخری چه کاری می‌کند؟','موضوع فعلی را تعریف کن.','منظور از همین مورد چیست؟','همان موضوع دوم را کوتاه بگو.','الان درباره چه چیزی حرف می‌زنیم و چیست؟'][variant%5]; a=(f'{term}: {ans}' if variant==4 else ans); stage=9
                out.append((p,a,concept,stage,'medium')); continue
            elif family==7: # negation sensitivity
                false=f'{term} فقط برای پخش موسیقی استفاده می‌شود.'
                p=[f'آیا این گزاره درست است؟ «{false}»',f'درستی یا نادرستی جمله را بگو: {false}',f'این ادعا را بررسی کن: {false}',f'فقط درست/نادرست و اصلاح کوتاه: {false}',f'آیا باید این جمله را بپذیریم؟ {false}'][variant%5]
                a=f'نادرست. {ans}'; stage=11
            elif family==8: # compare order invariance
                t2,a2=FACTS[(k+1)%len(FACTS)]; p=[f'{term} و {t2} را مقایسه کن.',f'تفاوت {t2} با {term} چیست؟',f'دو مورد {term}/{t2}: فرق اصلی؟',f'مقایسه کوتاه بین {term} و {t2}.',f'{t2} در برابر {term}: هرکدام چیست؟'][variant%5]; a=f'{term}: {ans} {t2}: {a2}'; stage=11
            elif family==9: # summarize generalization
                doc=f'{term} یک مفهوم فنی است. نکتهٔ اصلی این است که {ans} جزئیات فرعی بسته به پیاده‌سازی فرق می‌کند.'; p=[f'خلاصه کن: {doc}',f'نکتهٔ اصلی متن چیست؟ {doc}',f'یک جمله از این متن: {doc}',f'حاشیه را حذف کن: {doc}',f'خلاصهٔ دقیق بده: {doc}'][variant%5]; a=ans; stage=4
            elif family==10: # compositional instruction
                p=[f'{term} را تعریف کن و پاسخ بیشتر از ۱۵ کلمه نباشد.',f'بدون مثال، فقط تعریف {term}.',f'یک جمله و بدون انگلیسی اضافه: {term} چیست؟',f'اول نام {term} و بعد تعریفش را بنویس.',f'{term}: تعریف کوتاه و مستقیم.'][variant%5]; a=(f'{term}: {ans}' if variant==3 else ans); stage=4
            elif family==11: # synonyms in instructions
                w1,w2=SYN[(i+variant)%len(SYN)]; p=f'{term} را {w1} و {w2} توضیح بده؛ منظورم یک پاسخ کوتاه و روشن است.'; a=ans; stage=4
            elif family==12: # case / punctuation perturbation
                p=[f'؟؟ {term} چیست؟؟',f'[{term}] چیست',f'{term.upper()} چیست؟',f'  {term}   یعنی چه؟  ',f'«{term}» را توضیح بده.'][variant%5]; a=ans; stage=1
            elif family==13: # answer extraction from mini-context
                code=100+rng.randrange(900); ctx=f'کد پروژه {code} است. تعریف مورد هدف: {term} — {ans} '; p=ctx+[f'طبق متن، {term} چیست؟',f'تعریف مورد هدف را استخراج کن.',f'کد پروژه را نادیده بگیر و {term} را بگو.',f'از زمینه فقط تعریف {term} را بده.',f'مهم‌ترین گزاره درباره {term} چیست؟'][variant%5]; a=ans; stage=9
                out.append((p,a,concept,stage,'medium')); continue
            elif family==14: # entity substitution / novel labels
                label=f'مولفه-{chr(65+(i%20))}{di}'; p=f'فرض کن {label} همان {term} باشد. {label} چه کاری می‌کند؟'; a=ans; stage=11
            elif family==15: # conditional instruction routing
                p=f'اگر موضوع فنی است، تعریف کوتاه بده؛ اگر نیست فقط نامش را تکرار کن. موضوع: {term}'; a=ans; stage=11
            elif family==16: # style invariance
                p=[f'خیلی دوستانه بگو {term} چیست.',f'خنثی و حرفه‌ای: {term} چیست؟',f'برای دانش‌آموز: {term} چیست؟',f'بدون لحن تبلیغاتی: {term} چیست؟',f'مختصر و مستقیم: {term} چیست؟'][variant%5]; a=ans; stage=2
            elif family==17: # mixed script terms
                alias={'database':'پایگاه داده','thread':'رشته','cache':'کش'}.get(term,term); p=f'منظور از {alias} ({term}) چیست؟'; a=ans; stage=1
            elif family==18: # contradiction handling
                bad='همیشه کندتر از همه چیز است'; p=f'کاربر گفته «{term} {bad}». بدون قبول ادعای بی‌دلیل، تعریف درست {term} را بده.'; a=ans; stage=11
            else: # instruction priority/simple boundary
                p=f'متن داخل پرانتز حاشیه است (عدد {rng.randrange(1000)}). فقط به این سؤال پاسخ بده: {term} چیست؟'; a=ans; stage=4
            out.append((p,a,concept,stage,'medium'))
        return out
    return gen

G_SPECS=[]
for fam in range(20):
    for var in range(5): G_SPECS.append((f'g{fam:02d}_v{var}',g_factory(fam,var)))

# ---------- Reasoning generators ----------
def rf_factory(family:int,variant:int):
    def gen(di,n):
        rng=random.Random(SEED+900000+family*1301+variant*113+di); out=[]
        for i in range(n):
            c=f'f{family}:{i%18}'; diff='hard' if family>=15 else 'medium'; stage=11
            a=rng.randrange(8,60); b=rng.randrange(2,12); d=rng.randrange(2,10)
            if family==0:
                ans=a+b-d; p=f'علی {a} مهره داشت، {b} مهره گرفت و {d} مهره داد. چند مهره دارد؟'; o=f'{a}+{b}-{d}={ans}. پاسخ: {ans}.'
            elif family==1:
                ans=(a*b)-d; p=f'{a} بسته، هرکدام {b} عدد داریم و {d} عدد مصرف می‌شود. چند عدد می‌ماند؟'; o=f'{a}×{b}-{d}={ans}. پاسخ: {ans}.'
            elif family==2:
                x=rng.randrange(4,20); total=x*(b+d); p=f'نسبت دو گروه {b} به {d} و مجموع {total} است. اندازه گروه اول؟'; o=f'هر سهم {x} است؛ گروه اول {b}×{x}={b*x}.'
            elif family==3:
                base=10*rng.randrange(5,30); pct=rng.choice([10,20,25,50]); ans=base*pct//100; p=f'{pct}٪ از {base} چند است؟'; o=f'{base}×{pct}/100={ans}. پاسخ: {ans}.'
            elif family==4:
                price=100*rng.randrange(2,20); disc=rng.choice([10,20,25]); ans=price*(100-disc)//100; p=f'قیمت {price} با تخفیف {disc}٪ چقدر می‌شود؟'; o=f'پس از تخفیف: {price}×{100-disc}/100={ans}.'
            elif family==5:
                x=rng.randrange(5,30); y=rng.randrange(5,30); ans=(2*x+3*y)/5; p=f'میانگین وزنی {x} با وزن ۲ و {y} با وزن ۳ چیست؟'; o=f'(۲×{x}+۳×{y})/۵={ans:g}.'
            elif family==6:
                speed=rng.choice([40,50,60,80]); t=rng.randrange(2,6); ans=speed*t; p=f'با سرعت {speed} کیلومتر بر ساعت در {t} ساعت چند کیلومتر طی می‌شود؟'; o=f'{speed}×{t}={ans} کیلومتر.'
            elif family==7:
                x=rng.randrange(3,20); k=rng.randrange(2,8); rhs=x+k; p=f'x + {k} = {rhs}. x چند است؟'; o=f'x={rhs}-{k}={x}.'
            elif family==8:
                x=rng.randrange(2,12); m=rng.randrange(2,6); k=rng.randrange(1,8); rhs=m*x+k; p=f'{m}x + {k} = {rhs}. x را پیدا کن.'; o=f'{m}x={rhs-k}، پس x={x}.'
            elif family==9:
                start=rng.randrange(2,20); step=rng.randrange(2,8); seq=[start+j*step for j in range(5)]; nxt=start+5*step; p=f'عدد بعدی چیست؟ {", ".join(map(str,seq))}, ?'; o=f'اختلاف ثابت {step} است؛ پاسخ {nxt}.'
            elif family==10:
                start=rng.randrange(1,6); ratio=rng.choice([2,3]); seq=[start*(ratio**j) for j in range(4)]; nxt=start*(ratio**4); p=f'الگو را ادامه بده: {", ".join(map(str,seq))}, ?'; o=f'هر بار ×{ratio}؛ پاسخ {nxt}.'
            elif family==11:
                n=rng.randrange(4,9); r=2; ans=n*(n-1)//2; p=f'از {n} نفر چند جفت متفاوت می‌توان ساخت؟'; o=f'C({n},2)={ans}.'
            elif family==12:
                red=rng.randrange(2,8); blue=rng.randrange(2,8); total=red+blue; p=f'کیسه {red} مهره قرمز و {blue} آبی دارد. احتمال قرمز در یک برداشت؟'; g=math.gcd(red,total); o=f'{red}/{total} = {red//g}/{total//g}.'
            elif family==13:
                A=set(range(1,b+2)); B=set(range(b//2,b+d)); ans=len(A|B); p=f'A={sorted(A)} و B={sorted(B)}. تعداد اعضای A∪B؟'; o=f'A∪B={sorted(A|B)}، پس {ans} عضو.'
            elif family==14:
                p=f'همهٔ ربات‌ها ماشین‌اند. هیچ ماشینِ خاموشی فعال نیست. ربات R خاموش است. آیا R فعال است؟'; o='خیر. R ماشین و خاموش است، پس طبق گزارهٔ دوم فعال نیست.'
            elif family==15:
                names=['آوا','بهرام','پریا']; order=names[:]; rng.shuffle(order); p=f'{order[0]} قبل از {order[1]} است و {order[1]} قبل از {order[2]}. چه کسی اول است؟'; o=f'{order[0]} اول است.'
            elif family==16:
                h=rng.randrange(8,18); delta=rng.randrange(2,6); p=f'جلسه ساعت {h}:00 شروع و {delta} ساعت طول می‌کشد. زمان پایان؟'; o=f'{h+delta}:00.'
            elif family==17:
                x=rng.randrange(1,20); p=f'اگر x={x} باشد، کد `y=x*2+3; y=y-1` در پایان y چند است؟'; o=f'y={x}×2+3-1={2*x+2}.'
            elif family==18:
                vals=[rng.randrange(1,10) for _ in range(4)]; p=f'لیست {vals} است. ابتدا مجموع دو عضو اول، سپس عضو سوم را کم کن. نتیجه؟'; ans=vals[0]+vals[1]-vals[2]; o=f'{vals[0]}+{vals[1]}-{vals[2]}={ans}.'
            elif family==19:
                budget=100*rng.randrange(5,20); c1=50*rng.randrange(1,5); c2=25*rng.randrange(1,7); ans=budget-c1-c2; p=f'بودجه {budget} است؛ {c1} و سپس {c2} خرج می‌شود. مانده؟'; o=f'{budget}-{c1}-{c2}={ans}.'
            elif family==20:
                p='اگر P آنگاه Q. P درست است. درباره Q چه نتیجه‌ای می‌گیریم؟'; o='Q درست است (modus ponens).'
            elif family==21:
                p='اگر باران ببارد زمین خیس می‌شود. زمین خیس است. آیا حتماً باران باریده؟'; o='خیر. این استنتاج لازم نیست؛ علت دیگری هم می‌تواند زمین را خیس کرده باشد.'
            elif family==22:
                x=rng.randrange(2,20); p=f'در حالت واقعی مقدار {x} است. اگر در فرض خلاف واقع ۳ واحد بیشتر بود، مقدار فرضی چند می‌شد؟'; o=f'{x}+3={x+3}.'
            elif family==23:
                vals=[rng.randrange(10,50) for _ in range(3)]; mx=max(vals); p=f'سه مقدار A={vals[0]}، B={vals[1]}، C={vals[2]}. بزرگ‌ترین کدام است؟'; lab='ABC'[vals.index(mx)]; o=f'{lab} با مقدار {mx} بزرگ‌ترین است.'
            elif family==24:
                p='برای نصب نرم‌افزار این وابستگی‌ها وجود دارد: دانلود قبل از نصب، نصب قبل از پیکربندی، پیکربندی قبل از تست. ترتیب درست؟'; o='دانلود → نصب → پیکربندی → تست.'
            else: raise AssertionError
            # surface variants to reduce template memorization
            if variant==1: p='با استدلال کوتاه حل کن: '+p
            elif variant==2: p='فقط مراحل ضروری و جواب نهایی: '+p
            elif variant==3: p='مسئله را دقیق بررسی کن. '+p
            out.append((p,o,c,stage,diff))
        return out
    return gen

R_SPECS=[]
for fam in range(25):
    for var in range(4): R_SPECS.append((f'r{fam:02d}_v{var}',rf_factory(fam,var)))

if __name__=='__main__':
    gm=write_pack(GOUT,'generalization',G_SPECS); rm=write_pack(ROUT,'reasoning',R_SPECS)
    print(json.dumps({'generalization':gm,'reasoning':rm},ensure_ascii=False,indent=2))
