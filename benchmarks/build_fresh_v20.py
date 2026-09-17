"""Fresh evaluation only. This module is never imported by training or runtime.
Prompts and expected results are frozen before either version is evaluated.
"""
from pathlib import Path
import json,math,hashlib,sys,re
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from jarvis.agent.normalization_v20 import normalize,skeleton
ROWS=[]
def add(cat,q,expected,kind='number',**extra):
 ROWS.append({'id':f'fresh_{len(ROWS)+1:03}','category':cat,'question':q,'expected':expected,'kind':kind,'language':'fa' if re.search('[آ-ی]',q) else 'en','template_id':'fresh_'+hashlib.sha256(skeleton(q).encode()).hexdigest()[:20],**extra})
# Independently composed wording, varied target positions, units and mathematical structure.
ratio=[
('قرارداد تقسیم جایزه 847 تومانی میگوید سهم نادر به بهرام 6 به 5 باشد؛ هر سهم چقدر است؟',847,6,5),
('برای دو تیم با نسبت 7:4، بودجه 935 واحدی را سهم بندی کن.',935,7,4),
('کل پول 702 است. سهم ها را چنان جدا کن که نسبتشان 8 به 5 باشد.',702,8,5),
('مبلغ ۱۱۴۰ رو بین سهم های نسبت ۵ به ۷ پخش میکنی؟',1140,5,7),
('اگر سهم اول و دوم نسبت 9 به 2 داشته باشند و کل 913 باشد، دو سهم چیست؟',913,9,2),
('Allocate a grant worth 897 with the two beneficiaries in ratio 8 to 5.',897,8,5),
('A pool of 1044 is shared; the first-to-second ratio must be 7:5. Find both amounts.',1044,7,5),
('The allocations follow ratio 3:8 and add up to 979; calculate the allocations.',979,3,8),
('For a total prize of 756, how much does each side get at ratio 4 to 5?',756,4,5),
('Divide the 1265 fund according to shares weighted 6 to 5.',1265,6,5)]
for q,t,a,b in ratio:add('ratio',q,[t*a/(a+b),t*b/(a+b)],'parts')
prob=[
('در هفت آزمایش مستقل با شانس موفقیت 25٪، احتمال دقیقاً سه موفقیت را میخواهم.',7,3,.25),
('شانس برد 45 درصد است. از پنج بازی مستقل دقیقاً دو برد چقدر محتمل است؟',5,2,.45),
('با احتمال 0.2 برای هر تلاش، دقیقاً هیچ موفقیتی در شش تلاش چه احتمالی دارد؟',6,0,.2),
('از هشت تلاش دقیقاً یک موفقیت میخواهیم؛ احتمال موفقیت هر تلاش 15 درصد است.',8,1,.15),
('سکه نامتقارن با شانس شیر 70٪ را چهار بار میاندازیم؛ دقیقاً سه شیر چه احتمالی دارد؟',4,3,.7),
('Across nine independent attempts with a 40% success chance, calculate exactly four successes.',9,4,.4),
('Exactly zero successes in five independent trials, each successful with chance 12 percent?',5,0,.12),
('A six-trial binomial experiment has probability 0.65 per success. Find exactly two successes.',6,2,.65),
('With a success rate of one percent over four attempts, what is the chance of exactly one success?',4,1,.01),
('An unfair coin has heads probability 0.8. Toss it seven times; exactly five heads?',7,5,.8)]
for q,n,k,p in prob:add('probability',q,100*math.comb(n,k)*p**k*(1-p)**(n-k))
seq=[
('از الگوی 14، 7، 3.5، ... جمله ششم را حساب کن.',.4375),
('جمله هفتم برای دنباله 3، 9، 27، ... چه میشود؟',2187),
('دنباله 5، -10، 20 ادامه دارد؛ جمله ششم را بگو.',-160),
('چه عددی پس از دنباله 1، 4، 5، 9، 14 می آید؟',23),
('برای دنباله 4، 9، 11، 16، 18، عدد بعدی را تعیین کن.',23),
('Determine term eight of the sequence 7, 11, 15, ...',35),
('The terms 18, 6, 2 form a geometric pattern. What comes next?',2/3),
('Sequence 3, -1.5, 0.75; give the fifth term.',.1875),
('What follows 2, 2, 4, 6, 10 in this recurrence?',16),
('Starting with 2, 6, 7, 11, 12, find term seven using alternating increments.',17)]
for q,e in seq:add('sequence',q,e)
work=[
('شش کارگر در هشت ساعت 192 قطعه تولید کردند؛ چهار کارگر در سه ساعت چقدر میسازند؟',48),
('در 5 ساعت 7 کارگر 210 قطعه میسازند؛ حالا 9 کارگر برای 2 ساعت داریم. تولید؟',108),
('با 3 کارگر و 4 ساعت، خروجی 84 قطعه است. با 8 کارگر و 6 ساعت چه خروجی داریم؟',336),
('کارگاه طی 9 ساعت با 5 کارگر 360 قطعه تحویل داد؛ حالا طی 3 ساعت با 10 کارگر چند قطعه؟',240),
('برای ساخت 156 قطعه، 4 کارگر 3 ساعت کار کردند. 6 کارگر در 2 ساعت چند قطعه میسازند؟',156),
('A workshop makes 275 pieces with 5 workers in 5 hours. Predict output for 3 workers over 7 hours.',231),
('For 168 units the factory used 4 workers and 6 hours; now 7 workers have 2 hours. Output?',98),
('In 8 hours, 3 workers finish 144 pieces. What will 5 workers finish in 4 hours at that rate?',120),
('Known output: 336 items from 8 workers over 7 hours. New team: 6 workers for 5 hours. Find output.',180),
('Ten workers build 450 units in five hours. How many units can six workers build in three hours?',162)]
for q,e in work:add('work_rate',q,e)
for q,e in [
('انبار اول 137 کالا دارد؛ 28 کالا ارسال شد، 19 کالا دریافت شد. موجودی نهایی چیست؟',128),
('موجودی اولیه 208 قطعه است. 43 قطعه فروخته شد و سپس 17 قطعه وارد انبار شد.',182),
('داخل انبار 96 واحد داریم؛ 12 واحد دریافت کردیم و 31 واحد ارسال کردیم. چند واحد مانده؟',77),
('اول انبار 175 کالا داشت، ورودی انبار 23 واحد بود؛ خروجی انبار 49 واحد بود؛ موجودی؟',149),
('موجودی انبار 114 است؛ 16 کالا از انبار خارج شد؛ 35 کالا وارد انبار شد. باقی؟',133),
('Stock begins with 163 items; ship 27 items, receive 16 items. How many remain?',152),
('The warehouse holds 219 pieces. Sell 38 units and restock 12 units; final stock?',193),
('Inventory originally contains 87 units; 24 pieces arrive and 19 pieces leave. Remaining units?',92),
('Start inventory at 142 items, receive 31 items, then ship 54 items. Find the balance.',119),
('The stock count is 183; remove 29 items from stock and add 14 items to stock. Result?',168)]:add('inventory',q,e)
for q,e in [
('موجودی حساب من 735 تومان است؛ 68 تومان خرج کردم و 24 تومان واریز شد. مانده؟',691),
('حساب با موجودی 480 شروع میشود. 35 تومان برداشت شد؛ 62 تومان دریافت کردم؛ موجودی آخر؟',507),
('اول حساب 920 تومان دارد؛ واریزی 75 تومان است؛ پرداختی 128 تومان است. چقدر داریم؟',867),
('موجودی حساب 310 تومان است؛ 27 تومان از حساب کم شد و 53 تومان به حساب اضافه شد.',336),
('موجودی حساب 605 تومان است. 19 تومان پرداخت کردم و 41 تومان دریافت کردم؛ مانده چیست؟',627),
('Account balance starts at 845 dollars; withdraw 63 dollars and deposit 28 dollars. Final balance?',810),
('The account contains 560 dollars, receives a credit of 39 and a debit of 72. What remains?',527),
('Start with a bank balance of 925, pay a charge of 47, receive 16 dollars. What is left?',894),
('The balance is 315 dollars; spend 28 dollars and credit the account with 67. New balance?',354),
('An account begins with 720 dollars. A payment of 35 was received, then withdraw 81 dollars.',674)]:add('finance',q,e)
for q,e in [
('مینا 17 ساله است و خواهرش 5 سال بزرگتر است؛ 4 سال بعد مجموع سنشان چند است؟',47),
('رضا 23 سال دارد؛ برادرش 6 سال کوچکتر است. 7 سال بعد سن برادر چقدر میشود؟',24),
('من 14 سالمه و خواهرم 3 سال بزرگتره؛ 8 سال دیگه مجموع سن ما چند میشه؟',47),
('سن لیلا 19 سال است؛ خواهر بزرگتر 4 سال اختلاف سنی دارد؛ پس از 6 سال مجموع سن ها؟',54),
('علی 16 ساله است؛ برادرش 2 سال کوچکتر است؛ 9 سال بعد مجموع سن ها را بده.',46),
('Sara is 18 years old, and her sister is 7 years older. Find their combined ages in 5 years.',53),
('Omar is aged 24; his brother is 8 years younger. How old will the brother be 3 years later?',19),
('I am 13 years old and my sister is older by 6 years; our age sum after 9 years?',50),
('Current age is 21; the sibling is 4 years younger. Find the sibling age in 8 years.',25),
('Lena is 15 years old, her brother 5 years older; calculate their combined ages 6 years from now.',47)]:add('age',q,e)
for q,e in [
('735 را 15 درصد کم کن، 17 اضافه کن، حاصل را بر 7 تقسیم کن.',(735*.85+17)/7),
('از 84 شروع کن؛ در 3 ضرب کن و بعد 19 کم کن؛ بر 5 تقسیم کن.',(84*3-19)/5),
('مقدار اولیه 260 است؛ 12 درصد اضافه کن؛ 23 کم کن؛ حاصل را در 2 ضرب کن.',(260*1.12-23)*2),
('از 47 شروع کن، به توان 2 برسان و باقیمانده تقسیم بر 13 را بده.',47**2%13),
('از 92 شروع کن؛ بر 7 تقسیم کن و تا 3 رقم اعشار گرد کن.',round(92/7,3)),
('Starting from 325, reduce by 18 percent, add 14, then divide by 9.',(325*.82+14)/9),
('Take 73; multiply by 4, subtract 29, followed by divide by 3.',(73*4-29)/3),
('The initial value is 180; increase by 16%, subtract 27, and then multiply by 5.',(180*1.16-27)*5),
('Start with 19; raise to power 3; modulo 17.',19**3%17),
('Starting from 87, divide by 11, round to 2 decimal places.',round(87/11,2))]:add('multistep',q,e)
for q,e in [
('قطار مسافت 238 کیلومتر را در 3.5 ساعت پیمود. سرعت متوسط؟',68),
('برای سفر 156 کیلومتری 120 دقیقه وقت صرف شد؛ سرعت به کیلومتر بر ساعت چیست؟',78),
('در زمان 45 ثانیه فاصله 315 متر طی شد. سرعت متر بر ثانیه؟',7),
('سرعت متوسط برای مسافت 99 کیلومتر با زمان 1.5 ساعت را حساب کن.',66),
('خودرو طی 4 ساعت فاصله 284 کیلومتر رفت. سرعت؟',71),
('A vehicle covers 258 km over 3 hours; calculate average speed.',86),
('The trip distance is 135 km and duration is 90 minutes. Speed in km per hour?',90),
('A runner covers 420 meters in 70 seconds. Speed in meters per second?',6),
('Travel distance 234 kilometers with elapsed time 2.6 hours; mean speed?',90),
('In 4.5 hours, a bus goes 315 km. Find average speed.',70)]:add('distance_time',q,e)
for q,e in [
('از 11 نفر یک گروه 4 نفره بدون ترتیب چند جور میتوان انتخاب کرد؟',math.comb(11,4)),
('ترکیب 13 انتخاب 5 را بده.',math.comb(13,5)),
('جایگشت 9 انتخاب 4 چقدر میشود؟',math.perm(9,4)),
('احتمال جمع 8 در پرتاب 2 تاس سالم را بگو.',5/36*100),
('احتمال همزمان رخدادهای مستقل با شانس 35 درصد و 60 درصد چیست؟',21),
('How many unordered selections of 5 can be made from 14 objects?',math.comb(14,5)),
('Compute permutations for 10 items taken 4 at a time.',math.perm(10,4)),
('Probability that two fair dice sum to 10?',3/36*100),
('Independent events have chances 0.25 and 0.36; what is their joint probability?',9),
('How many ways to choose a committee of 4 from 12 people?',math.comb(12,4))]:add('counting',q,e)
# Existing language/code/routing capabilities must participate in the comparison too.
for q,parts in [
('به انگلیسی ترجمه کن: من فردا به مدرسه می روم',['i','school']),('ترجمه کن به فارسی: The door is open.',['در','باز']),
('معادل انگلیسی «کتابخانه» را بگو',['library']),('به فارسی برگردان: I need more time.',['زمان']),
('این را انگلیسی کن: من آب می خواهم',['water']),('Translate into Persian: I am tired.',['خسته']),
('Translate to English: هوا سرد است',['cold']),('How do you say "پنجره" in English?',['window']),
('به انگلیسی بگو: او یک معلم است',['teacher']),('ترجمه به فارسی: Thank you for your help.',['کمک'])]:add('translation',q,parts,'contains')
for q,parts in [
('رسمی تر کن: میخوام فردا بیام اداره',['فردا']),('بازنویسی رسمی: لطفا این فایل رو برام بفرست',['فایل']),
('این جمله رو رسمی کن: واسه من وقت بذار',['زمان']),('روان تر بنویس: من به علت خستگی به خانه برگشتم',['خانه']),
('بازنویسی کن: این برنامه سریع و ساده است',['برنامه']),('Rewrite formally: I wanna ask for more time.',['time']),
('Make this formal: Can you send me the report?',['report']),('Rephrase: The system is fast and simple.',['system']),
('Rewrite politely: Give me the file.',['file']),('Rewrite formally: I gotta leave now.',['leave'])]:add('rewrite',q,parts,'contains')
for q,parts in [
('یه تابع بده واسه معکوس کردن رشته در پایتون',['def','[::-1]']),('کد تشخیص پالیندروم رو بهم بده',['def','[::-1]']),
('تابعی پیاده سازی کن برای میانگین عناصر لیست',['def','sum(']),('یک تابع برای حذف تکراری ها با حفظ ترتیب بده',['def']),
('تابع جمع لیست را با پایتون ایجاد کن',['def','sum(']),('Give me a Python function for reversing text.',['def','[::-1]']),
('Implement a palindrome checker function.',['def','[::-1]']),('Make a function to calculate the average of list entries.',['def','sum(']),
('Create Python code that keeps only positive values.',['def','> 0']),('Write a function that finds the largest list element.',['def','max('])]:add('code_generation',q,parts,'contains')
for q,e in [
('خروجی این کد چیست؟ x=7; print(x*3+2)',23),('خروجی کد: x=18; x-=5; print(x)',13),
('کد زیر چه چاپ میکند؟ print(sum([4,7,9]))',20),('خروجی چیست؟ print(len([3,6,9,12]))',4),
('این کد را ردگیری کن: x=5; y=4; print(x*y)',20),('Trace this Python: x=9; print(x*4-3)',33),
('What is printed? print(17 % 5)',2),('Find output: print(3 ** 4)',81),
('Run mentally: x=14; x+=6; print(x)',20),('What does print(max([8,3,11])) output?',11)]:add('code_trace',q,e)
for i,topic in enumerate(['باران','یادگیری','دوستی','کار','کتاب','rain','learning','friendship','work','books']):
 q=f'دقیقاً دو جمله فقط فارسی درباره {topic} بنویس و حتماً شامل کلمه «امید» باشد.' if i<5 else f'Write exactly two sentences only in English about {topic}; include the word "hope".'
 add('constraints',q,{'sentences':2,'required':'امید' if i<5 else 'hope','language':'fa' if i<5 else 'en'},'constraints')
for q,e in [('فاکتوریل 9 را حساب کن',362880),('بزرگترین مقسوم علیه مشترک 84 و 126 چیست؟',42),('کمترین مضرب مشترک 12 و 35 را بده',420),('میانگین 7، 15 و 23 را حساب کن',15),('درصد 17 از 460 چقدر است؟',78.2),('Calculate 8 factorial.',40320),('What is the greatest common divisor of 63 and 105?',21),('Find the least common multiple of 14 and 25.',350),('Average the numbers 13, 21 and 29.',21),('What is 23 percent of 340?',78.2)]:add('local_math',q,e)
for q,needed in [('قیمت فعلی اتریوم امروز چقدر است؟',True),('آخرین خبرهای امروز آلمان را پیدا کن',True),('آب و هوای فردا در هامبورگ چطور است؟',True),('جدیدترین مستندات رسمی پایتون را جستجو کن',True),('نرخ ارز امروز چیست؟',True),('What is the current Ethereum price?',True),('Find the latest news about Germany today.',True),('Tomorrow weather in Hamburg?',True),('Search the current official Python documentation.',True),('What is today\'s exchange rate?',True)]:add('web_needed',q,needed,'freshness')
for q,intent in [('صدا رو روی 37 درصد تنظیم کن','set_volume'),('روشنایی صفحه را روی 63 درصد بگذار','set_brightness'),('ولوم را روی 42 درصد بگذار','set_volume'),('صدا را روی 55 درصد بگذار','set_volume'),('روشنایی را روی 27 درصد تنظیم کن','set_brightness'),('Set volume to 37 percent.','set_volume'),('Set brightness to 63 percent.','set_brightness'),('Change volume to 42 percent.','set_volume'),('Set the volume to 55%.','set_volume'),('Set screen brightness to 27 percent.','set_brightness')]:add('routing',q,intent,'route')
# 120 input perturbations separate from the invariant corruption pytest suite.
# Ten semantic base cases, twelve orthographic/context variants each. Numbers remain fixed,
# so this measures robustness rather than multiplying nominal examples by numeric substitution.
bases=[('ratio','با نسبت 7 به 4، مبلغ 825 را تقسیم کن',[525,300],'parts'),
 ('probability','شانس موفقیت 24 درصد است؛ در 5 تلاش دقیقاً 2 موفقیت؟',100*math.comb(5,2)*.24**2*.76**3,'number'),
 ('sequence','دنباله 24، 12، 6؛ جمله پنجم چیست؟',1.5,'number'),
 ('multistep','از 175 شروع کن؛ 16 درصد کم کن؛ 13 اضافه کن؛ بر 4 تقسیم کن',40,'number'),
 ('inventory','موجودی انبار 129 کالا است؛ 23 کالا خارج شد؛ 18 کالا وارد انبار شد',124,'number'),
 ('ratio','Split a total of 936 in ratio 5 to 7.',[390,546],'parts'),
 ('probability','Success probability is 28 percent; in 6 attempts exactly 3 successes?',100*math.comb(6,3)*.28**3*.72**3,'number'),
 ('sequence','Find term six of the sequence 8, 4, 2.',.25,'number'),
 ('multistep','Start with 215; subtract 19; multiply by 3; divide by 7.',84,'number'),
 ('finance','Account balance is 650 dollars; deposit 43 dollars; withdraw 29 dollars.',664,'number')]
for cat,q,e,k in bases:
 fa=bool(re.search('[آ-ی]',q))
 variants=[q,q.translate(str.maketrans('0123456789','۰۱۲۳۴۵۶۷۸۹')),q.translate(str.maketrans('0123456789','٠١٢٣٤٥٦٧٨٩')),q.replace(' ','  '),q.replace('درصد','٪').replace('percent','%'),q.replace(';',' then ').replace('؛',' سپس '),q.replace(';',',').replace('؛','،'),q.replace('؟','?'),('شماره پرونده 908 است؛ ' if fa else 'Record id is 908; ')+q,('کد بسته 731 مربوط به این محاسبه نیست؛ ' if fa else 'Package identifier 731 is irrelevant to this calculation; ')+q,q+(' جواب را با دقت حساب کن.' if fa else ' Please calculate carefully.'),q.replace('را','رو') if fa else q.upper()]
 for j,v in enumerate(variants): add('adversarial_'+cat,v,e,k,perturbation=j,adversarial=True)
# The frozen set includes duplicates in normalized spelling by design, not in exact input.
# Drop exact duplicates inside robustness groups without inspecting model outputs.
seen=set(); unique=[]
for r in ROWS:
 if r['question'] in seen: continue
 seen.add(r['question']);unique.append(r)
ROWS=unique
out=ROOT/'benchmarks/fresh_v20.jsonl';out.write_text('\n'.join(json.dumps(r,ensure_ascii=False) for r in ROWS)+'\n')
manifest={'rows':len(ROWS),'sha256':hashlib.sha256(out.read_bytes()).hexdigest(),'frozen_before_evaluation':True,'scoring':'numeric equality; exact route/freshness; required fragments for language/code; explicit constraints','note':'Contains a separately tagged robustness subset; normalized variants within holdout are intentional.'}
(ROOT/'benchmarks/fresh_v20_manifest.json').write_text(json.dumps(manifest,indent=2))
print(manifest)
