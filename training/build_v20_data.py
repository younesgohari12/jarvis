"""Reproducible annotated compositional corpus, capped at one input per skeleton.
No benchmark is imported. Train/validation/test are grouped by lexical skeleton.
"""
from pathlib import Path
import sys,re,json,hashlib,random,itertools
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from jarvis.agent.normalization_v20 import normalize,NUMBER,skeleton
R=random.Random(20619)
ROWS=[]; SEEN=set()
def add(frame,marked,operation=None,provenance='authored_v20'):
    annotations=re.findall(r'\{(\w+)=([^{}]+)\}',marked)
    text=re.sub(r'\{\w+=([^{}]+)\}',r'\1',marked)
    idx=iter(range(len(annotations)))
    markers={}
    def mark(m):
        i=next(idx); key='qzmark'+chr(97+i)+'qz'
        markers[key]=(m.group(1),normalize(m.group(2))); return key
    marked_norm=normalize(re.sub(r'\{(\w+)=([^{}]+)\}',mark,marked))
    positions={}; norm=''; last=0
    for m in re.finditer(r'qzmark[a-z]qz',marked_norm):
        norm+=marked_norm[last:m.start()]; role,value=markers[m.group()]
        positions[len(norm)]=role; norm+=value; last=m.end()
    norm+=marked_norm[last:]
    matches=list(NUMBER.finditer(norm)); sk=skeleton(text)
    if sk in SEEN: return
    SEEN.add(sk)
    tid='v20_'+hashlib.sha256(sk.encode()).hexdigest()[:20]
    bucket=int(hashlib.sha256(('split:'+sk).encode()).hexdigest()[:8],16)%10
    split='train' if bucket<7 else 'validation' if bucket<9 else 'test'
    spans=[{'start':m.start(),'end':m.end(),'value':float(m.group()),'role':positions.get(m.start(),'other')} for m in matches]
    ROWS.append(dict(text=text,normalized=norm,label=frame,numbers=spans,operation=operation,template_id=tid,concept_group=frame,split=split,provenance=provenance))

# Independent lexical expressions; combined with different states and graph structures.
OPS={
'add':['{value=17} اضافه کن','{value=17} بیفزا','حاصل را با {value=17} جمع کن','به نتیجه {value=17} افزوده شود','add {value=17}','increase the result by {value=17}','plus {value=17}','include an extra {value=17}'],
'subtract':['{value=9} کم کن','از حاصل {value=9} کسر کن','{value=9} بردار','نتیجه منهای {value=9}','subtract {value=9}','take away {value=9}','reduce the result by {value=9}','minus {value=9}'],
'multiply':['در {value=3} ضرب کن','حاصل را {value=3} برابر کن','ضریب {value=3} را اعمال کن','نتیجه ضرب در {value=3}','multiply by {value=3}','scale the result by {value=3}','times {value=3}','apply a multiplier of {value=3}'],
'divide':['بر {value=4} تقسیم کن','حاصل تقسیم بر {value=4}','نتیجه را به {value=4} بخش مساوی تقسیم کن','حاصل را تقسیم بر {value=4} کن','divide by {value=4}','divide the result into {value=4} equal parts','take the quotient by {value=4}','split the result equally among {value=4}'],
'percentage_add':['{value=12} درصد اضافه کن','{value=12}٪ افزایش بده','حاصل را {value=12} درصد بیشتر کن','رشد {value=12} درصد اعمال کن','increase by {value=12}%','add {value=12} percent','apply a {value=12} percent increase','grow the result by {value=12} percent'],
'percentage_remove':['{value=15} درصد کم کن','{value=15}٪ تخفیف بده','حاصل را {value=15} درصد کاهش بده','کاهش {value=15} درصد اعمال کن','reduce by {value=15}%','subtract {value=15} percent','apply a {value=15} percent discount','remove {value=15} percent'],
'exponent':['به توان {value=2} برسان','حاصل را به توان {value=2} ببر','توان {value=2} را حساب کن','نتیجه به توان {value=2}','raise to power {value=2}','exponentiate by {value=2}','take the {value=2} power','apply exponent {value=2}'],
'modulo':['باقیمانده تقسیم بر {value=7} را بده','حاصل را پیمانه {value=7} کن','نتیجه مدولوی {value=7}','باقیمانده را بر {value=7} حساب کن','modulo {value=7}','take the remainder after division by {value=7}','compute mod {value=7}','return the remainder modulo {value=7}'],
'round':['تا {value=2} رقم اعشار گرد کن','نتیجه را به {value=2} رقم اعشار برسان','گرد کردن تا {value=2} رقم اعشار','حاصل تا {value=2} رقم اعشار گرد شود','round to {value=2} decimal places','round the result with {value=2} decimals','round using {value=2} fractional digits','apply rounding to {value=2} places'],
'min':['کوچکتر بین حاصل و {value=31} را بگیر','حداقل حاصل و {value=31} را بده','کمینه نتیجه با {value=31}','نتیجه را حداکثر {value=31} کن','take the minimum of the result and {value=31}','cap the result at {value=31}','min with {value=31}','choose the smaller of this and {value=31}'],
'max':['بزرگتر بین حاصل و {value=31} را بگیر','حداکثر حاصل و {value=31} را بده','بیشینه نتیجه با {value=31}','نتیجه را حداقل {value=31} کن','take the maximum of the result and {value=31}','floor the result at {value=31}','max with {value=31}','choose the larger of this and {value=31}'],
'inventory_add':['{value=18} کالا وارد انبار شد','{value=18} قطعه به موجودی اضافه شد','{value=18} واحد دریافت کردیم','ورودی انبار {value=18} واحد بود','receive {value=18} items','restock {value=18} units','{value=18} pieces arrive','add {value=18} items to stock'],
'inventory_remove':['{value=13} کالا از انبار خارج شد','{value=13} قطعه فروخته شد','{value=13} واحد ارسال کردیم','خروجی انبار {value=13} واحد بود','ship {value=13} items','sell {value=13} units','{value=13} pieces leave','remove {value=13} items from stock'],
'credit':['{value=23} تومان واریز شد','{value=23} تومان به حساب اضافه شد','واریزی {value=23} تومان است','{value=23} تومان دریافت کردم','deposit {value=23} dollars','credit the account with {value=23}','receive {value=23} dollars','a payment of {value=23} was received'],
'debit':['{value=11} تومان برداشت شد','{value=11} تومان خرج کردم','پرداختی {value=11} تومان است','{value=11} تومان از حساب کم شد','withdraw {value=11} dollars','debit the account by {value=11}','spend {value=11} dollars','pay a charge of {value=11}'],
}
STATES=['از {initial=240} شروع کن','مقدار اولیه {initial=240} است','{initial=240} را','حساب را با عدد {initial=240} آغاز کن','start with {initial=240}','the initial value is {initial=240}','take {initial=240}','starting from {initial=240}']
# Each operation receives standalone, different initial-state and composed context examples.
for op,phrases in OPS.items():
    family='inventory' if op.startswith('inventory') else 'finance' if op in ('debit','credit') else 'graph'
    for j,phrase in enumerate(phrases):
        add(family,phrase,op)
        for k in range(4):
            en=j>=4; state=STATES[(4 if en else 0)+k]
            if family=='inventory': state=('inventory starts with {initial=240} items' if en else 'موجودی اولیه انبار {initial=240} کالا است')
            if family=='finance': state=('the account balance is {initial=240} dollars' if en else 'موجودی حساب {initial=240} تومان است')
            sep= ['؛ ',' و ',' سپس ','، '][k] if not en else ['; ',' and then ','. ',' followed by '][k]
            add(family,state+sep+phrase,op)
# Genuinely different operation orders and lengths; no 1000 numeric copies of a template.
math_ops=list(OPS)[:11]
for a,b in itertools.permutations(math_ops,2):
    for lang in (0,1):
        j=R.randrange(4)+4*lang
        sep=' and then ' if lang else ' و '
        text=STATES[4*lang+R.randrange(4)]+'; '+OPS[a][j]+sep+OPS[b][4*lang+R.randrange(4)]
        if R.random()<.45:
            c=R.choice(['add','subtract','divide','multiply'])
            text+=('; ' if lang else '، حاصل را ')+OPS[c][4*lang+R.randrange(4)]
        add('graph',text)
for family,inop,outop in [('inventory','inventory_add','inventory_remove'),('finance','credit','debit')]:
    for j,k in itertools.product(range(8),range(8)):
        if (j<4)!=(k<4): continue
        state=('inventory starts at {initial=240}' if family=='inventory' else 'account balance is {initial=240}') if j>=4 else ('انبار {initial=240} کالا دارد' if family=='inventory' else 'موجودی حساب {initial=240} تومان است')
        add(family,state+'؛ '+OPS[inop][j]+'؛ '+OPS[outop][k])

PATTERNS={
'ratio':[
'مجموع سهم ها {total=616} است و نسبت سهم اول به دوم {ratio_a=5} به {ratio_b=3}؛ سهم هرکدام؟',
'نسبت {ratio_a=5}:{ratio_b=3} برای تخصیص بودجه {total=616} در نظر گرفته شده؛ سهم ها را حساب کن',
'{total=616} واحد را به نسبت {ratio_a=5} به {ratio_b=3} تقسیم کن',
'برای نسبت {ratio_a=5} به {ratio_b=3} کل مبلغ {total=616} را پخش کن',
'از کل {total=616} سهم ها باید نسبت {ratio_a=5}:{ratio_b=3} داشته باشند',
'بودجه {total=616} داریم؛ سهم الف نسبت به ب {ratio_a=5} به {ratio_b=3} است',
'برام {total=616} رو با نسبت {ratio_a=5} به {ratio_b=3} سهم بندی کن',
'واسه تقسیم {total=616} نسبت {ratio_a=5}:{ratio_b=3} رو بگیر',
'share a total of {total=616} in ratio {ratio_a=5} to {ratio_b=3}',
'ratio {ratio_a=5}:{ratio_b=3} applies to an allocation of {total=616}',
'the budget totals {total=616}; allocate in proportion {ratio_a=5} to {ratio_b=3}',
'with share weights {ratio_a=5} and {ratio_b=3}, divide total {total=616}',
'find both portions of {total=616} given a {ratio_a=5}:{ratio_b=3} split',
'first to second is {ratio_a=5} to {ratio_b=3}; the combined amount is {total=616}',
'allocate {total=616}; the share ratio equals {ratio_a=5}:{ratio_b=3}',
'distribute a fund of {total=616} using ratio {ratio_a=5} to {ratio_b=3}'],
'binomial':[
'شانس موفقیت {p=35} درصد است؛ در {n=6} تلاش دقیقاً {k=2} موفقیت چقدر احتمال دارد؟',
'از {n=6} آزمون مستقل دقیقاً {k=2} موفقیت با احتمال هر بار {p=35}٪ میخواهیم',
'برای {n=6} بار تکرار، نرخ موفقیت {p=35} درصد و موفقیت مطلوب {k=2} است',
'اگه در هر تلاش {p=35}٪ شانس داشته باشم، از {n=6} بار دقیقاً {k=2} بار ببرم چقدر میشه؟',
'دقیقاً {k=2} موفقیت در {n=6} آزمایش مستقل؛ شانس هر آزمایش {p=35} درصد',
'احتمال هر موفقیت {p=0.35} است و تعداد تلاش ها {n=6}؛ هدف دقیقاً {k=2} برد',
'با {p=35} درصد احتمال برد، {n=شش} نوبت و دقیقاً {k=دو} برد را حساب کن',
'نرخ قبولی {p=35} درصد است؛ از {n=6} نفر دقیقاً {k=هیچ} قبولی چه احتمالی دارد؟',
'exactly {k=2} successes among {n=6} independent trials, each with probability {p=0.35}',
'success chance is {p=35}%; in {n=6} attempts what is the chance of exactly {k=2} successes?',
'with {n=6} trials and per-trial probability {p=35} percent, find exactly {k=2} successes',
'compute a binomial probability for n={n=6}, k={k=2}, p={p=0.35}',
'out of {n=six} attempts exactly {k=two} should succeed; success rate is {p=35} percent',
'probability per attempt {p=0.35}; target successes {k=2}; trials {n=6}',
'what are the odds of exactly {k=none} successes in {n=6} trials with a {p=35}% success rate?',
'a biased coin has heads chance {p=35} percent; exactly {k=2} heads in {n=6} flips?'],
'work_rate':[
'در {hours=6} ساعت {workers=4} کارگر {output=120} قطعه میسازند؛ حالا {workers=8} کارگر در {hours=3} ساعت چقدر تولید میکنند؟',
'{workers=4} نفر نیرو ظرف {hours=6} ساعت {output=120} واحد تولید میکنند؛ برای {workers=8} نفر در {hours=3} ساعت خروجی چیست؟',
'ساخت {output=120} کالا با {workers=4} کارگر {hours=6} ساعت طول کشید؛ اگر {hours=3} ساعت و {workers=8} کارگر داشته باشیم چه؟',
'در بازه {hours=شش} ساعته {workers=چهار} نفر نیرو {output=120} قطعه ساختند؛ حالا {workers=هشت} نفر در {hours=سه} ساعت؟',
'تولید اولیه {output=120} قطعه است با نیروی {workers=4} نفر و مدت {hours=6} ساعت؛ نیروی جدید {workers=8} نفر و مدت جدید {hours=3} ساعت',
'کارگاه با {workers=4} کارگر طی {hours=6} ساعت {output=120} واحد تحویل داد؛ با {workers=8} کارگر طی {hours=3} ساعت چند واحد؟',
'{output=120} قطعه را {workers=4} کارگر در {hours=6} ساعت ساختند؛ حالا مدت {hours=3} ساعت و کارگر {workers=8} نفر است',
'{workers=4} workers produce {output=120} pieces in {hours=6} hours; now {workers=8} workers work for {hours=3} hours',
'in {hours=6} hours, {workers=4} workers make {output=120} units; what can {workers=8} workers make in {hours=3} hours?',
'output was {output=120} items using {workers=4} workers for {hours=6} hours; now use {hours=3} hours and {workers=8} workers',
'a crew of {workers=4} people made {output=120} pieces over {hours=6} hours; if {workers=8} people have {hours=3} hours, calculate production',
'the {hours=6} hour shift employed {workers=4} workers to produce {output=120} pieces; the new shift has {workers=8} workers for {hours=3} hours',
'production {output=120} units; staffing {workers=4} workers; duration {hours=6} hours; now staffing {workers=8} workers and duration {hours=3} hours'],
'sequence':[
'دنباله {term=12}، {term=6}، {term=3} است؛ جمله {n=چهارم} را پیدا کن',
'اعداد {term=5}، {term=10}، {term=20} را داریم؛ عدد بعدی چیست؟',
'جمله {n=پنجم} دنباله {term=2}، {term=6}، {term=18} چیست؟',
'دنباله {term=4}، {term=7}، {term=10}؛ عضو {n=9}؟',
'الگوی {term=8}، {term=-4}، {term=2} را ادامه بده',
'سری اعداد {term=1}، {term=1}، {term=2}، {term=3}، {term=5}؛ جمله بعدی؟',
'دنباله {term=2}، {term=5}، {term=6}، {term=9}، {term=10}؛ جمله {n=هفتم}',
'در تصاعد هندسی جمله اول {first=7} و نسبت {ratio=0.5} است؛ جمله {n=6}؟',
'جمله اول دنباله حسابی {first=7} و اختلاف {difference=4} است؛ جمله {n=8} را بده',
'what is term {n=5} of the sequence {term=3}, {term=9}, {term=27}?',
'continue the sequence {term=16}, {term=8}, {term=4}',
'find the {n=fourth} member of {term=9}, {term=6}, {term=3}',
'next number in {term=2}, {term=3}, {term=5}, {term=8}, {term=13}?',
'sequence {term=1}, {term=4}, {term=5}, {term=8}, {term=9}; find term {n=7}',
'geometric progression with first term {first=3} and common ratio {ratio=-2}; find term {n=6}',
'arithmetic sequence starts at {first=3}, common difference {difference=5}, requested index {n=8}'],
'age':[
'علی {age=12} ساله است و خواهرش {difference=3} سال بزرگتر است؛ {years=5} سال بعد مجموع سنشان چقدر است؟',
'سن مینا {age=18} است؛ برادرش {difference=4} سال کوچکتر است؛ پس از {years=6} سال مجموع سن ها؟',
'الان {age=15} سالمه؛ خواهرم {difference=2} سال بزرگتره؛ {years=7} سال دیگه چند سالشه؟',
'عمر رضا {age=21} سال است و فاصله سنی با خواهر بزرگتر {difference=6} سال؛ بعد از {years=3} سال سن خواهر؟',
'Ali is {age=12} years old and his sister is {difference=3} years older; what is their age sum in {years=5} years?',
'Mina is aged {age=18}; her brother is {difference=4} years younger; find their combined ages {years=6} years later',
'my age is {age=15}; my sister is older by {difference=2}; how old will she be after {years=7} years?',
'current age {age=21}, older sibling age difference {difference=6}; sibling age in {years=3} years?'],
'speed':[
'مسافت {distance=150} کیلومتر در {time=3} ساعت طی شد؛ سرعت متوسط؟',
'طی {time=2} ساعت {distance=90} کیلومتر رفت؛ سرعت چقدر است؟',
'ماشین در {time=120} دقیقه مسافت {distance=80} کیلومتر را رفت؛ سرعت کیلومتر بر ساعت؟',
'فاصله {distance=1200} متر و مدت {time=60} ثانیه؛ سرعت متر بر ثانیه؟',
'average speed for {distance=150} km covered in {time=3} hours?',
'in {time=2} hours a vehicle travelled {distance=90} kilometers; find speed',
'distance {distance=80} km, duration {time=120} minutes; speed in km per hour?',
'{distance=1200} meters over {time=60} seconds; velocity in meters per second?'],
'combination':[
'از {n=9} نفر {k=3} نفر بدون اهمیت ترتیب انتخاب کنیم چند حالت؟',
'تعداد ترکیب {n=12} انتخاب {k=4} چقدر است؟',
'کمیته {k=3} نفره از {n=9} نفر چند روش دارد؟',
'choose {k=3} from {n=9} without regard to order',
'combinations of {n=12} items taken {k=4} at a time',
'how many unordered committees of {k=3} from {n=9} people?'],
'permutation':[
'جایگشت {n=8} انتخاب {k=3} را حساب کن',
'از {n=9} نفر {k=4} نفر با اهمیت ترتیب چند حالت دارد؟',
'تعداد چیدمان {k=3} عضو از {n=8} عضو بدون تکرار',
'permutations of {n=8} taking {k=3}',
'ordered selections of {k=4} out of {n=9} distinct people',
'arrange {k=3} different items chosen from {n=8} without repetition'],
'dice':[
'احتمال مجموع {target=9} برای {n=2} تاس منصفانه چقدر است؟',
'{n=سه} تاس سالم بیندازیم احتمال جمع {target=10} چیست؟',
'برای {n=2} تاس، شانس اینکه مجموع {target=7} بشود؟',
'roll {n=2} fair dice; probability of sum {target=9}?',
'what is the chance that {n=three} dice total {target=10}?',
'sum target {target=7} using {n=2} fair dice; find probability'],
'independent':[
'رخدادهای مستقل با احتمال {p=20} درصد و {p=40} درصد؛ احتمال هر دو؟',
'احتمال مشترک رخدادهای مستقل {p=0.3} و {p=0.7} را بده',
'independent events have probabilities {p=0.2} and {p=0.4}; chance both happen?',
'joint probability of independent events with {p=30}% and {p=70}% chances'],
'scheduling':[
'شروع برنامه ساعت {start=9} است؛ کار اول {duration=2} ساعت و کار بعدی {duration=3} ساعت طول میکشد؛ پایان چه ساعتی است؟',
'از ساعت {start=8} جلسه {duration=1} ساعته و سپس کار {duration=2} ساعته داریم؛ زمان پایان؟',
'start at hour {start=9}; a task takes {duration=2} hours followed by {duration=3} hours; finishing time?',
'schedule begins at {start=8}, with sequential durations {duration=1} hours and {duration=2} hours; end time?'],
'comparison':[
'کوچکترین عدد بین {term=12} و {term=-3} و {term=8} کدام است؟',
'بیشترین مقدار از {term=5}، {term=14}، {term=7} را بده',
'find the smallest among {term=12}, {term=-3}, {term=8}',
'which is largest: {term=5}, {term=14}, {term=7}?'],
'system':[
'صدا را روی {other=30} درصد بگذار','روشنایی را {other=40} درصد کن','ولوم رو ببر روی {other=20} درصد','set volume to {other=30}%','change brightness to {other=40}%','turn the volume down to {other=20} percent'],
'external':[
'امروز قیمت بیت کوین چقدر است؟','آخرین اخبار هوش مصنوعی چیست؟','هوای امروز برلین چطور است؟','جدیدترین نسخه پایتون را در اینترنت پیدا کن','current bitcoin price today?','what is the latest AI news?','weather in Berlin today','find the current Python documentation online'],
'knowledge':[
'فتوسنتز چیست؟','چرا آسمان آبی است؟','تفاوت رم و حافظه ذخیره سازی چیست؟','what is photosynthesis?','why is the sky blue?','explain the difference between memory and storage'],
'other':[
'سلام خوبی؟','ممنون از کمکت','یادت باشد اسم من یونس است','یک داستان کوتاه تعریف کن','hello how are you?','thanks for your help','remember my name is Younes','tell me a short story'],
'provided_text':[
'بر اساس متن زیر جواب بده: نیما به بازار رفت. نیما کجا رفت؟','خلاصه کن: باران بارید و خیابان خیس شد.','according to this passage answer: Mina went home. Where did she go?','summarize the supplied text: Rain fell and the street became wet.'],
'code':[
'تابع پایتون برای مرتب سازی لیست بده','برام کدی بساز که عددهای مثبت رو جدا کنه','یه تابع بده مجموع لیست رو حساب کنه','کد حذف تکراری ها رو پیاده سازی کن','explain this Python code','debug the function below','refactor this code to reduce duplication','design an algorithm for finding the maximum','give me code for filtering even numbers','make a function that returns sorted values'],
}
PATTERNS['comparison'] += [
 'میانگین این مقدارها را بده: {term=7}، {term=13}، {term=22}',
 'معدل اعداد {term=8} و {term=16} و {term=24} را حساب کن',
 'از داده های {term=3}، {term=9}، {term=15} میانگین بگیر',
 'حساب کن میانگین عددهای {term=4} و {term=11} و {term=18} چقدر میشود',
 'عددها {term=11}، {term=17} و {term=23} هستند؛ میانگینشان چیست؟',
 'برای لیست {term=6}، {term=8}، {term=13} متوسط را بده',
 'find the average of these values: {term=7}, {term=13}, {term=22}',
 'calculate the arithmetic mean for {term=8}, {term=16}, {term=24}',
 'the data are {term=3}, {term=9}, {term=15}; what is their mean?',
 'average these numbers: {term=4}, {term=11}, {term=18}',
 'return the mean value from {term=11}, {term=17}, {term=23}',
 'compute the average using {term=6}, {term=8}, {term=13}'
]
# Conditional clauses are labelled with a comparison operator; branch grammar is separate.
for cmp,forms in {
 'conditional_gt':['اگر حاصل بزرگتر از {threshold=40} بود','اگر نتیجه از {threshold=40} بیشتر شد','if the result is greater than {threshold=40}','if the result exceeds {threshold=40}'],
 'conditional_lt':['اگر حاصل کوچکتر از {threshold=40} بود','اگر نتیجه از {threshold=40} کمتر شد','if the result is less than {threshold=40}','if the result falls below {threshold=40}']}.items():
 for i,f in enumerate(forms):
  add('graph',f,cmp)
  for op in ('add','subtract','multiply','divide'):
   en=i>=2; j=(4 if en else 0)+i%2
   conditional=f+(' then ' if en else ' آنگاه ')+OPS[op][j]+(' otherwise ' if en else ' وگرنه ')+OPS['add'][j]
   add('graph',STATES[4 if en else 0]+'; '+conditional)
for family,patterns in PATTERNS.items():
    for t in patterns: add(family,t)
# Context distractors and clause permutation introduce new syntax, not merely new numbers.
for family in ['ratio','binomial','work_rate','age','speed']:
    originals=list(PATTERNS[family])
    for i,t in enumerate(originals):
        en=bool(re.search('[a-zA-Z]',re.sub(r'\{.*?\}','',t)))
        distractions=['record id {other=81}; ','ignore shipment number {other=81}; ','the label reads {other=81}. '] if en else ['شماره پرونده {other=81} است؛ ','شماره بسته {other=81} ربطی به محاسبه ندارد؛ ','روی برچسب {other=81} نوشته شده؛ ']
        add(family,distractions[i%3]+t)
        if ';' in t or '؛' in t:
            parts=re.split('[;؛]',t)
            if len(parts)==2 and family in ('ratio','binomial','speed'): add(family,parts[1]+'; '+parts[0])
# Replay a maximum of ONE row per legacy template. Legacy examples remain provenance-tagged.
paths=[ROOT/'datasets/numeric_roles_v19/numeric_role_24000_real.jsonl',ROOT/'datasets/semantic_ir_v19/semantic_frame_18000_real.jsonl']
seen_legacy=set()
family_map={'work':'work_rate','ratio':'ratio','probability':'binomial','age':'age','money_percent':'graph','speed':'speed','generic':'other','multistep':'graph','ratio_split':'ratio','probability_binomial':'binomial','age_reasoning':'age','code_generation':'code','work_scaling':'work_rate','system_control':'system'}
role_map={'probability':'p','money':'initial','percentage':'value','years_later':'years','hours':'hours','count':'other','index':'other'}
for path in paths:
    for line in path.read_text().splitlines():
        r=json.loads(line)
        if r['split']!='train': continue
        key=(path.name,r['template_id'])
        if key in seen_legacy: continue
        seen_legacy.add(key)
        text=r.get('text',r.get('input','')); norm=normalize(text)
        # Replay is optional frame support only for already supported families.
        family=family_map.get(r.get('family',r.get('label','')))
        if family is None: continue
        if 'slots' in r:
            roles=[]; valid=True
            for m in NUMBER.finditer(norm):
                candidates=[role_map.get(k,k) for k,v in r['slots'].items() if float(v)==float(m.group())]
                roles.append(candidates[0] if len(candidates)==1 else 'other')
            marked=norm
            for m,role in reversed(list(zip(list(NUMBER.finditer(norm)),roles))): marked=marked[:m.start()]+'{'+role+'='+m.group()+'}'+marked[m.end():]
            add(family,marked,provenance='v19_train_replay_one_per_template')
        else:
            # Unlabelled numbers cannot teach roles: mark other and exclude numeric loss.
            marked=NUMBER.sub(lambda m:'{other='+m.group()+'}',norm)
            before=len(ROWS); add(family,marked,provenance='v19_frame_replay_one_per_template')
            if len(ROWS)>before: ROWS[-1]['numbers']=[]

# Cluster near duplicates across partitions before training (cosine >= .90).
from sklearn.feature_extraction.text import TfidfVectorizer
mat=TfidfVectorizer(analyzer='char',ngram_range=(3,5)).fit_transform([skeleton(r['text']) for r in ROWS])
links=(mat@mat.T).tocoo(); parents=list(range(len(ROWS)))
def find(i):
 while parents[i]!=i: parents[i]=parents[parents[i]]; i=parents[i]
 return i
for i,j,v in zip(links.row,links.col,links.data):
 if i<j and v>=.90:
  a,b=find(int(i)),find(int(j))
  if a!=b: parents[max(a,b)]=min(a,b)
groups={}
for i,r in enumerate(ROWS): groups.setdefault(find(i),[]).append(r)
for group in groups.values():
 key=min(r['template_id'] for r in group); bucket=int(hashlib.sha256(key.encode()).hexdigest()[:8],16)%10
 for r in group:
  r['split']='train' if bucket<7 else 'validation' if bucket<9 else 'test'
  r['near_duplicate_group']=key
out=ROOT/'datasets/v20'; out.mkdir(parents=True,exist_ok=True)
(out/'cognitive.jsonl').write_text('\n'.join(json.dumps(r,ensure_ascii=False) for r in ROWS)+'\n')
print('Cognitive rows',len(ROWS),'new',sum(r['provenance']=='authored_v20' for r in ROWS),'unique skeletons',len(SEEN))
