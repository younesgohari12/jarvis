"""Assemble measured release evidence. Never substitutes benchmark % for INDEX-v1."""
from pathlib import Path
import sys,json,re,hashlib,statistics
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT)); R=ROOT/'reports/v20'
from jarvis.agent.semantic_models_v20 import LearnedModel
read=lambda name:json.loads((R/name).read_text())
base=read('fresh_baseline.json');first=read('fresh_new.json');new=read('fresh_new_final.json');train=read('training.json');overlap=read('fresh_overlap.json');preserved=read('checkpoint_preservation.json')
old={r['id']:r for r in base['results']}
regressed=[r['id'] for r in new['results'] if old[r['id']]['passed'] and not r['passed']]
improved=[r['id'] for r in new['results'] if not old[r['id']]['passed'] and r['passed']]
log=(R/'final_pytest.log').read_text();match=re.search(r'(\d+) passed, (\d+) subtests passed in ([\d.]+)s',log)
assert match and int(match[1])>=696 and int(match[2])>=585 and not re.search(r'\d+ failed|ERRORS|FAILURES',log)
assert base['benchmark_sha256']==new['benchmark_sha256']==overlap['clean_sha256']
assert base['total']==new['total']>=200 and new['correct']>base['correct']
assert not preserved['changed_existing_weight_files']
for name,report in train['models'].items():
 model=LearnedModel(name)
 assert model.ready and model.sha256==report['sha256']
 assert model.parameter_count==report['parameter_count']
gate={'engineering_gate_passed':True,'mission_fully_completed':False,'release_label':'TESTED_COGNITIVE_UPGRADE_WITH_LIMITATIONS',
 'regression':{'primary_passed':int(match[1]),'subtests_passed':int(match[2]),'failures':0,'duration_seconds':float(match[3])},
 'fresh':{'baseline_correct':base['correct'],'first_v20_correct':first['correct'],'final_v20_correct':new['correct'],'total':new['total'],'first_evaluation_unseen':True,'final_evaluation_is_retest':True,'net_gain':new['correct']-base['correct'],'improved_case_ids':improved,'regressed_case_ids':regressed,'sha256':new['benchmark_sha256']},
 'all_original_checkpoints_unchanged':True,'new_model_count':len(train['models']),'new_parameter_count':sum(v['parameter_count'] for v in train['models'].values()),
 'official_index':{'baseline':80.19,'new':None,'delta':None,'reason':'Missing executable scoring rubric; no invented score.'},
 'limits':['Code and Persian language upgrades are partial.','Core transformer not retrained.','60k-120k sample target not claimed.','One fresh case regresses despite net improvement.','Live web and Windows GUI not tested.','Auxiliary code classifier holdout contains near-duplicates; full audit supplied.']}
(R/'release_gate.json').write_text(json.dumps(gate,ensure_ascii=False,indent=2))
dimensions={'Core Neural':52,'NLU / Routing':92,'Persian':81,'English':81,'Reasoning':88,'Context / Memory':80,'Knowledge / RAG':62,'Web Research':78,'Planning / Tools':90,'Safety / Recovery':95,'Generalization':81,'Efficiency':88,'Testing / Reliability':99}
lines=['# گزارش واقعی ارتقای JARVIS v20','',
'کد ارتقایافته، پنج مدل آموزش‌دیده و متصل به Runtime، داده‌ها، اسکریپت‌های آموزش، بنچمارک و گزارش اجرای تست‌ها در بسته قرار دارند. این یک ارتقای افزایشی آزموده‌شده است؛ تحقق کامل تمام هدف‌های متن، آموزش مدل زبانی اصلی یا رسیدن به امتیاز ۸۵ ادعا نمی‌شود.','',
'## Baseline','', 'v19 REAL: **80.19 / 100** روی شاخص ثابت `JARVIS-INTELLIGENCE-INDEX-v1`. این عدد مطابق دستور کاربر دست‌نخورده مانده است.','',
'## New version','', 'Actual measured score روی همان شاخص: **قابل محاسبه نیست**؛ فرمول و ارزیاب اصلی در ZIP موجود نبود. درصد پاسخ درست بنچمارک زیر، جایگزین آن شاخص نیست.','',
'## Delta','', '**نامشخص برای شاخص رسمی**. هیچ امتیاز تخمینی ساخته نشده است.','',
'| Dimension | v19 | New | Delta |','|---|---:|---:|---:|']
for name,value in dimensions.items():lines.append(f'| {name} | {value} | اندازه‌گیری هم‌معیار موجود نیست | — |')
lines += ['', '## تست و مقایسهٔ واقعی','',f'- Baseline اجراشده: **696 تست اصلی + 585 زیرتست؛ صفر خطا**.',f'- بستهٔ نهایی: **{match[1]} تست اصلی + {match[2]} زیرتست؛ صفر خطا** در {match[3]} ثانیه.',
'- ۱۷۵ تست اصلی جدید اضافه شده؛ ۱۲۰ مورد آن کاندیدهای خراب نسبت، احتمال، موجودی، مالی و سن را به Verifier می‌دهند و ردشدن واقعی را بررسی می‌کنند.',
'- تست‌های معماری، بارگذاری و فراخوانی مدل، تغییر پاسخ نهایی با تغییر خروجی مدل، ورود World State به Graph، فراخوانی Verifier و تغییر پاسخ/Slot در Reflection را ثابت می‌کنند.','',
'| بنچمارک ثابت | پاسخ درست | درصد |','|---|---:|---:|',f'| v19 | {base["correct"]}/{base["total"]} | {100*base["accuracy"]:.2f}% |',f'| اولین ارزیابی تازهٔ v20 | {first["correct"]}/{first["total"]} | {100*first["accuracy"]:.2f}% |',f'| بازآزمایی نسخهٔ نهایی | {new["correct"]}/{new["total"]} | {100*new["accuracy"]:.2f}% |','',
f'نتیجهٔ نهایی نسبت به v19، **{new["correct"]-base["correct"]} پاسخ درست بیشتر** و **{100*(new["accuracy"]-base["accuracy"]):.2f} واحد درصد** بهبود دارد. {len(improved)} مورد بهتر و {len(regressed)} مورد بدتر شد؛ شناسهٔ مورد پس‌رفت: {", ".join(regressed)}. هنوز {new["total"]-new["correct"]} مورد موفق نیست.',
'اولین آزمون ۲۰۱ پاسخ درست داشت؛ پس از اصلاح عمومی واحد زمان و بازآموزی سازگار با نرمال‌سازی، نتیجهٔ بستهٔ نهایی ۱۹۷ شد. عدد ۲۰۱ به‌عنوان نتیجهٔ بستهٔ نهایی ارائه نمی‌شود. نتیجهٔ نهایی بازآزمایی همان مجموعه است، نه یک Holdout تازهٔ دوم.',
'آزمون شامل ۱۷۹ سؤال پایه و ۹۵ نمونهٔ تغییر نگارش/زمینه است؛ هم‌بستگی این تغییرها در تفسیر امتیاز باید لحاظ شود. ارزیابی ترجمه/بازنویسی مبتنی بر قطعات مورد انتظار است و ارزیابی کامل روانی زبان نیست؛ کد تولیدشده علاوه بر قطعات مورد انتظار، آزمون رفتاری محدود دارد.','',
'| دسته | تعداد | v19 درست | نسخهٔ نهایی درست |','|---|---:|---:|---:|']
for cat,v in new['category'].items():lines.append(f'| {cat} | {v["n"]} | {base["category"][cat]["correct"]} | {v["correct"]} |')
lines += ['', '## داده‌ها و آموزش','',
'دادهٔ اصلی **۱۶۴۳ ردیف با ۱۶۴۳ الگوی عددزدایی‌شدهٔ یکتا** دارد: ۱۱۶۶ ردیف جدید و ۴۷۷ بازپخش محدود از Train قدیمی. داده‌های کمکی: ۶۷۴ ردیف کد، ۲۵۰ ردیف تشخیص مرحلهٔ تعمیر و ۷۰ کاندیدِ کنترل پاسخ. ۷۰ کاندید برای وارسی Verifier قطعی است؛ ادعای آموزش یک مدل تشخیص حقیقت با این ۷۰ مورد وجود ندارد.',
'در دادهٔ اصلی، الگوهای متنی بین Splitها مشترک نیستند؛ تعداد مشابه‌های Test با Train با آستانهٔ cosine≥0.90 برابر صفر است. در دادهٔ کد، ۴۶ نمونه از ۶۳ نمونهٔ Test به Train شباهت بالاتر از این آستانه دارند؛ بنابراین دقت داخلی مدل کد خوش‌بینانه است و نباید با توان کدنویسی عمومی یکی گرفته شود.',
f'برای Holdout، {overlap["training_texts_scanned"]:,} متن از فایل‌های داده اسکن شد؛ از ۲۷۵ کاندید، یک موردِ دارای هم‌پوشانی کنار گذاشته شد و **۲۷۴** مورد باقی ماند. هم‌پوشانی دقیق، نرمال‌شده و الگوی متنی باقی‌مانده **۰** است. تعریف الگو در گزارش آمده؛ هم‌پوشانی مفهومی صفر ادعا نمی‌شود.',
'مدل‌ها طبقه‌بندهای کوچک خطی روی ویژگی‌های متنی/بافت هستند، نه ترنسفورمر جدید. وزن‌های قبلی دست‌نخورده‌اند. حجم پیشنهادی ۶۰ تا ۱۲۰ هزار داده تولید نشده است؛ تعداد با تغییر دادن صرف عددها بالا برده نشده است.','',
'| مدل فعال | پارامتر | Train | Validation | Test | دقت Test | F1 ماکرو | Loss تست |','|---|---:|---:|---:|---:|---:|---:|---:|']
for name,v in train['models'].items():
 m=v['metrics'];lines.append(f'| {name} | {v["parameter_count"]:,} | {m["train"]["rows"]} | {m["validation"]["rows"]} | {m["test"]["rows"]} | {100*m["test"]["accuracy"]:.2f}% | {m["test"]["macro_f1"]:.4f} | {m["test"]["loss"]:.4f} |')
lines += ['',f'مجموع پارامترهای جدید: **{gate["new_parameter_count"]:,}**. ماتریس خطا، Loss هر Split، تنظیمات انتخاب‌شده و SHA256 هر مدل در `training.json` ثبت شده است. مقایسهٔ قبل/بعدِ مستقیم در سطح Runtime انجام شده؛ برای Head جدید با نقش‌ها/کلاس‌های متفاوت، دقت v19 با عنوان «همان مدل» جعل نشده است.','', '## سرعت و حافظه','',
'CPU-only، Python 3.12.13، Linux، یک Thread برای BLAS. پنج سؤال محلی ثابت، ۱۲۰ درخواست گرم و سه ساخت Runtime؛ زمان Import در Startup زیر نیست. این اعداد مربوط به همین محیط‌اند، نه تضمین سرعت روی ویندوز یا GPU کاربر.','',
'| معیار محدود عملکرد | v19 | v20 |','|---|---:|---:|']
pa=read('performance_baseline.json');pn=read('performance_new.json')
for name,key,unit in [('Startup میانه','startup_median_ms','ms'),('تأخیر P50','latency_p50_ms','ms'),('تأخیر P95','latency_p95_ms','ms'),('بیشینه RSS','peak_rss_mb','MiB')]:lines.append(f'| {name} | {pa[key]:.2f} {unit} | {pn[key]:.2f} {unit} |')
lines += ['',f'در مجموعهٔ متنوع ۲۷۴تایی، بیشینهٔ RSS برابر {new["peak_rss_mb"]:.2f} MiB بود. کاهش/افزایش کوچک عملکرد را نباید نتیجهٔ قطعی بهینه‌سازی دانست؛ اندازه‌گیری‌ها نویز دارند. GPU و رابط گرافیکی ویندوز در این محیط آزمایش نشده‌اند.','',
'## محدودیت‌های حل‌نشده','',
'- یک مسئلهٔ Work Rate که v19 درست جواب می‌داد، در نسخهٔ نهایی غلط است؛ نسبت‌دهی مشاهدهٔ اولیه/هدف در جمله‌های آزاد هنوز محدودیت دارد.',
'- ترجمه و بازنویسی در این بنچمارک پیشرفت نکردند؛ مدل زبانی اصلی بازآموزی نشد.',
'- مدل تولید کد عمومی، دیباگر عمومی یا Refactor عمومی جدید ساخته نشده؛ ده دستورالعمل کدنویسی آزموده‌شده و قابلیت‌های قبلی وجود دارند.',
'- RAG دوزبانه به واژگان محدود و کنترل شواهد متکی است؛ ارتقای بزرگ پایگاه دانش یا بازیابی چندزبانهٔ عمومی انجام نشده.',
'- وب زنده تست نشده؛ تصمیم نیاز به اطلاعات تازه و اصلاح امتیازدهی شواهد آینه‌شده بررسی شده‌اند.',
'- World Model و Graph برای دامنه‌های مشخص‌اند. Verifier صحت داخلی محاسبه را می‌سنجد و تضمین فهم بی‌خطای متن نیست.',
'- امتیاز ۸۴ یا ۸۵–۸۷ در شاخص رسمی اثبات نشده است.',
'', '## فایل‌های شواهد','',
'`final_pytest.log`، `fresh_baseline.json`، `fresh_new_final.json`، `fresh_overlap.json`، `training.json`، `dataset_audit.json`، `auxiliary_dataset_audit.json`، `checkpoint_preservation.json`، `release_gate.json` و `phase_status.json` در همین پوشه قرار دارند. گزارش معماری در `ARCHITECTURE.md` است. همهٔ پاسخ‌های غلط نیز نگه داشته شده‌اند.']
(R/'REPORT_FA.md').write_text('\n'.join(lines)+'\n')
print(json.dumps({k:gate[k] for k in ('engineering_gate_passed','mission_fully_completed','regression','new_parameter_count')},ensure_ascii=False))
