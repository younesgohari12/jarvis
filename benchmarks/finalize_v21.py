"""Build release evidence from measured results; refuses a failing regression gate."""
from pathlib import Path
import json,re,hashlib,zipfile,sys
ROOT=Path(__file__).resolve().parents[1];D=ROOT/'reports/v21'
BASE=Path(sys.argv[1]);OUT=Path(sys.argv[2]);OUT.mkdir(parents=True,exist_ok=True)
def read(n):return json.loads((D/n).read_text())
def write(n,d):(D/n).write_text(json.dumps(d,ensure_ascii=False,indent=2)+'\n')
def compare(a,b):
 assert a['benchmark_sha256']==b['benchmark_sha256']
 x={r['id']:r for r in a['results']};y={r['id']:r for r in b['results']};assert x.keys()==y.keys()
 return {'total':len(x),'baseline_correct':a['correct'],'candidate_correct':b['correct'],'improved':[k for k in x if not x[k]['passed'] and y[k]['passed']],'regressed':[k for k in x if x[k]['passed'] and not y[k]['passed']],'benchmark_sha256':b['benchmark_sha256']}
with zipfile.ZipFile(BASE) as z:
 prefix=z.namelist()[0].split('/')[0]+'/'
 old=json.loads(z.read(prefix+'reports/v20_1/previous_274_final.json'))
 checkpoints=[];changed=[]
 for n in z.namelist():
  if n.endswith('/'):continue
  rel=n[len(prefix):];p=ROOT/rel;sha=hashlib.sha256(z.read(n)).hexdigest()
  same=p.is_file() and hashlib.sha256(p.read_bytes()).hexdigest()==sha
  if rel.endswith('.npz'):checkpoints.append({'file':rel,'sha256':sha,'unchanged':same})
  if not same:changed.append(rel)
preservation={'baseline_archive':BASE.name,'baseline_sha256':hashlib.sha256(BASE.read_bytes()).hexdigest(),'original_checkpoint_count':len(checkpoints),'all_original_checkpoints_unchanged':all(c['unchanged'] for c in checkpoints),'checkpoints':checkpoints,'modified_existing_files':changed,'new_release_is_separate_copy':True}
write('preservation.json',preservation)
regtext=(D/'regression_final.log').read_text();match=re.search(r'(\d+) passed, (\d+) subtests passed',regtext)
assert match and not re.search(r'\d+ failed',regtext),'Regression gate is not green'
regression={'primary_passed':int(match.group(1)),'subtests_passed':int(match.group(2)),'failures':0,'log':'regression_final.log','legacy_test_change':'One old corruption test now requires source rejection and repair instead of displaying the intentionally wrong answer. Runtime metadata assertion reflects v21. No tests removed.'}
previous=compare(old,read('previous_274_release.json'));broad=compare(read('broad_baseline.json'),read('broad_final.json'))
assert not previous['regressed'],'Previously correct 274-case answers regressed'
assert preservation['all_original_checkpoints_unchanged']
write('comparison.json',{'previous_274':previous,'broad_500_final_retest':broad,'first_unseen_candidate_correct':read('broad_candidate.json')['correct'],'first_unseen_baseline_correct':read('broad_baseline.json')['correct']})
audit=read('dataset_audit.json');assert not audit['verification_parent_or_split_errors'] and not audit['verification_identity_mutations'] and not audit['code_oracle_errors']
generation=read('language_generation.json');generation_accepted=sum(x['adapter']['heuristic_quality_gate'] for x in generation['examples'])
heads=read('training_heads.json')['models'];numeric=read('training_numeric.json');language=read('language_training.json');perf0=read('performance_baseline.json');perf1=read('performance_final.json')
model_entries={
 'numeric_binding':{'parameters':numeric['parameters'],'test_accuracy':numeric['metrics']['test']['accuracy'],'status':'active_low_confidence_numeric_spans','sha256':numeric['sha256']},
 'code_task':{'parameters':heads['code_task']['parameters'],'test_accuracy':heads['code_task']['metrics']['test']['accuracy'],'status':'active_code_task_selection','sha256':heads['code_task']['sha256']},
 'language_adapter':{'parameters':language['trainable_parameters'],'status':'active_neural_generation_path' if language['runtime_enabled'] else 'disabled_by_gate','sha256':language['sha256'],'test_loss_before':language['before']['test']['loss'],'test_loss_after':language['after']['test']['loss']},
 'verification_failure':{'parameters':heads['verification_failure']['parameters'],'test_accuracy':heads['verification_failure']['metrics']['test']['accuracy'],'status':'diagnostic_log_only_not_verification_authority','sha256':heads['verification_failure']['sha256']},
 'operation_language':{'parameters':heads['operation_language']['parameters'],'test_accuracy':heads['operation_language']['metrics']['test']['accuracy'],'status':'not_enabled_due_to_weak_holdout','sha256':heads['operation_language']['sha256']},
}
for name, entry in model_entries.items():
 assert hashlib.sha256((ROOT/'models'/f'{name}_v21.npz').read_bytes()).hexdigest()==entry['sha256'], name
for rel,digest in read('broad_final.json')['source_hashes'].items():
 assert hashlib.sha256((ROOT/rel).read_bytes()).hexdigest()==digest, 'Source changed after final benchmark: '+rel
registry=json.loads((ROOT/'models/model_registry_v2.json').read_text());registry['architecture_v21']={'runtime':'jarvis.agent.local_intelligence_v21.LocalIntelligenceV21','world':'jarvis.agent.world_v21.WorldStateV21','verifier':'jarvis.agent.verification_v21.UniversalVerifier','models':model_entries,'base_transformer_weights_preserved':True,'effective_neural_projection_changed':True}
(ROOT/'models/model_registry_v2.json').write_text(json.dumps(registry,ensure_ascii=False,indent=2)+'\n')
write('runtime_models.json',model_entries)
phases={
 '1_universal_verification':{'implemented':True,'limits':'A common source-to-slots/events/units/candidate/rendered-answer protocol with bounded extractors. Not universal natural-language understanding.'},
 '2_world_model':{'implemented':True,'limits':'Typed entities, ownership ledger, event order, conservation, dependencies. General free-text causal worlds remain unsupported.'},
 '3_execution_graph_v2':{'implemented':True,'operations':['compare','change_over_time','dependency','transfer','ownership','constraint_check','cause_effect'],'limits':'All execute through GraphExecutorV21; some currently require structured graphs rather than arbitrary natural language.'},
 '4_reasoning_dataset':{'rows':20000,'count_target_met':True,'semantic_program_groups':json.loads((ROOT/'datasets/v21/manifest.json').read_text())['semantic_program_groups'],'broad_linguistic_quality_target_met':False},
 '5_core_adapter':{'trained':True,'active':language['runtime_enabled'],'trainable_parameters':language['trainable_parameters'],'core_index_improvement_proven':False},
 '6_code':{'rows':10000,'tasks':['generate','debug','explain','refactor','test'],'implemented':True,'general_arbitrary_code_model':False},
 '7_memory':{'implemented':True,'limits':'Explicit episodic topic recall, attributed user semantic assertions, ambiguity questions, disable/forget support on existing SQLite tables.'},
 '8_benchmark':{'cases':500,'first_unseen_v21':read('broad_candidate.json')['correct'],'final_retest_v21':read('broad_final.json')['correct'],'unique_numeric_masked_templates':read('broad_overlap.json')['unique_numeric_masked_templates'],'external_human_benchmark':False},
}
write('phase_status.json',phases)
gate={'release':'v21','runtime_version':'3.0.0-rc1','label':'ARCHITECTURE_TRAINED_TESTED_WITH_LIMITATIONS','engineering_gate_passed':True,'mission_fully_completed':False,'regression':regression,'previous_274':previous,'broad_500':broad,'first_500_score':read('broad_candidate.json')['correct'],'original_checkpoints_preserved':True,'new_reasoning_rows':20000,'new_code_rows':10000,'official_index':None,'user_reported_v20_1_index':82.08,'reasons_mission_incomplete':['Language benchmark remains weak; lower token loss does not prove a general language upgrade.','Synthetic corpus has high lexical near-duplicate rates despite structural split separation.','General debug/refactor/arbitrary algorithm generation is not achieved.','500 benchmark cases share templates and the final run is a retest.','Some graph operations need structured inputs.']}
write('release_gate.json',gate);(ROOT/'RELEASE_V21.json').write_text(json.dumps(gate,ensure_ascii=False,indent=2)+'\n')
cats=read('broad_final.json')['category'];oldcats=read('broad_baseline.json')['category'];initial=read('broad_candidate.json')['category']
category_table='\n'.join(f"| {c} | {oldcats[c]['correct']}/{oldcats[c]['total']} | {initial[c]['correct']}/{initial[c]['total']} | {cats[c]['correct']}/{cats[c]['total']} |" for c in ['Reasoning','Persian','English','Code','World Model','Adversarial'])
near=audit['near_duplicates'];total_active=sum(model_entries[k]['parameters'] for k in ['numeric_binding','code_task','language_adapter'])
report=f'''# JARVIS v21 — نسخهٔ معماری و آموزش با محدودیت‌های ثبت‌شده

این بسته از v20.1 ساخته شده است. نسخه‌های قبلی برای بازگشت حفظ شده‌اند. امتیاز 82.08 برای v20.1، عدد گزارش‌شدهٔ شماست؛ برای v21 امتیاز JARVIS-INTELLIGENCE-INDEX-v1 محاسبه یا حدس زده نشده است. این نسخه ادعای رسیدن به 84–85 یا تکمیل همهٔ اهداف زبانی را ندارد.

## تغییر واقعی در Runtime

`reasoning.py` و `intent_router.py` اکنون `LocalIntelligenceV21` را بارگذاری می‌کنند و پاسخ‌های مسیر جدید با `semantic_ir_v21` مشخص می‌شوند. مسیر فعال شامل Parser، WorldStateV21، GraphExecutorV21، UniversalVerifier و ReflectionLoopV21 است.

Verifier متن اصلی همان درخواست را مستقل از متن ذخیره‌شده در IR نگه می‌دارد؛ برچسب عدد، Slot، Graph و پاسخ قبلی را شاهد حقیقت نمی‌گیرد. استخراج تازهٔ واقعیت‌ها با Slot، ترتیب رویداد، واحد و جواب محاسبه‌شده مقایسه می‌شود. بعد از ترمیم نیز قالب پاسخ از **IR نهایی** ساخته و متن عددی نهایی دوباره با منبع مقایسه می‌شود. پاسخ ردشده برچسب `result_verified` نمی‌گیرد. متن مبهم یا پوشش‌داده‌نشده به تأیید معنایی جعلی تبدیل نمی‌شود.

این پروتکل Ratio، Probability، Finance، Inventory، Age، Distance/Speed، Scheduling، Work Rate، Sequence، Comparison و Ownership را در دستور زبان پشتیبانی‌شده پوشش می‌دهد. «Universal» نام پروتکل مشترک است؛ به معنی فهم بی‌خطای هر جمله نیست. استخراج مستقل از دادهٔ میانی است، اما بعضی مسیرهای ترمیم و اعتبارسنجی از همان کتابخانهٔ قواعد منبع استفاده می‌کنند؛ خطای سیستماتیک زبانی همچنان ممکن است.

در World Model، موجودی، مالکیت، اشخاص، کالا، کارگر و زمان ثبت می‌شوند. انتقال بین دو مالک مجموع دارایی را حفظ می‌کند؛ اختلاط دلار و کالا، مالکیت متعارض، موجودی ناممکن و وابستگی اجرا‌نشده رد می‌شوند. رابطهٔ `before` در ساخت Graph مصرف می‌شود. عملیات جدید `compare`، `change_over_time`، `dependency`، `transfer`، `ownership`، `constraint_check` و `cause_effect` اجرا و آزموده شده‌اند؛ تبدیل متن آزاد به همهٔ آن‌ها هنوز عمومی نیست.

| مثال | نتیجه |
|---|---:|
| ۳ کارگر، ۴ ساعت، ۸۴ قطعه → ۸ کارگر، ۶ ساعت | 336 |
| دنبالهٔ 24، 12، 6؛ جملهٔ پنجم | 1.5 |
| ۵۰۰ دلار موجودی، خرید ۱۲۰ دلاری با ۳۰٪ تخفیف | 416 |
| افزایش ۲۰٪ و سپس کاهش ۱۰٪ روی ۱۰۰ | 108 |
| انتقال ۳۰ از مالکِ دارای ۱۰۰ به مالکِ دارای ۵۰ | 80 برای گیرنده |
| سه برابر نصف ۲۰ | 30 |

محدودیت مهم: مدل زبان یا قواعد هنوز نمی‌توانند هر صورت‌بندی پیچیده، ارجاع مبهم یا رابطهٔ علّی دلخواه را بفهمند. ابزارهای محلی قدیمی خارج از دامنهٔ این پروتکل نیز همچنان وجود دارند؛ آن‌ها به‌عنوان تأیید معنایی v21 معرفی نمی‌شوند.

## داده و آموزش

| بخش | نمونه |
|---|---:|
| Semantic IR | 5000 |
| World Model | 5000 |
| Verification | 4000 |
| Adversarial | 3000 |
| Persian Intelligence | 3000 |
| Code، پنج وظیفهٔ ۲۰۰۰‌تایی | 10000 |
| جمع | 30000 |

این‌ها برنامه‌ها و مسئله‌های ترکیبیِ مصنوعی با جواب قابل‌محاسبه‌اند؛ ۳۰هزار نمونهٔ دست‌نویس انسانی نیستند. عمدهٔ ۲۰هزار نمونهٔ اول زنجیرهٔ عملیات عددی است؛ دادهٔ فارسی هم بیشتر دستورهای کمی است و جای یک پیکرهٔ مکالمهٔ متنوع را نمی‌گیرد. تنوع دامنه‌های World Model نیز به اندازهٔ هدف اولیه متوازن نیست.

۳۰۰۰۰ ورودی یکتا و {audit['unique_template_ids']} شناسهٔ قالب ثبت شده است. تقسیم ۲۰هزار نمونه بر اساس ساختار معناییِ عملیات و دامنه انجام شد؛ {phases['4_reasoning_dataset']['semantic_program_groups']} گروه ساختاری دارد. دادهٔ کد ۲۰۰۰ گروه AST دارد و پنج وظیفهٔ یک برنامه در یک Split می‌مانند. هم‌پوشانی گروه و شناسهٔ قالب بین Splitها صفر است. ۱۴ نمونهٔ منفی که عملاً چیزی را عوض نکرده بودند اصلاح شدند؛ آزمون مرجع دادهٔ کد و ارتباط نمونه‌های Verification با منبع، خطایی گزارش نکرد.

با این حال **{near['test_near_train_at_0_9']} از {near['test_samples']} نمونهٔ آزمون** در معیار شباهت واژگانی TF-IDF کاراکتریِ متنِ ماسک‌شده، شباهت حداقل 0.9 به Train دارند. اشتراک واژه‌ها و اجزای زنجیره زیاد است؛ یکتایی متن و جدایی ساختار، جای استقلال زبانی و مفهومی را نمی‌گیرد. بنابراین هدف «۲۰هزار دادهٔ عمومی با تنوع زبانی بالا» تکمیل‌شده اعلام نشده است.

سه جزء آموخته‌شدهٔ تولیدی جدید، مجموعاً {total_active:,} پارامتر دارند:

- Numeric Binding: {numeric['parameters']:,} پارامتر؛ آموزش روی {numeric['numeric_samples']:,} جایگاه عدد از {numeric['parent_programs_used']:,} برنامه. فقط به پیش‌بینی‌های کم‌اطمینان قبلی کمک می‌کند. دقت آزمونِ این زمینه‌های محدود {numeric['metrics']['test']['accuracy']:.1%} است؛ این عدد، دقت نقش‌یابی عمومی فارسی نیست. نمونهٔ تغییر واقعی برچسب در `numeric_model_effect.json` ثبت شده است.
- Code Task: {heads['code_task']['parameters']:,} پارامتر؛ وظیفهٔ Generate/Debug/Explain/Refactor/Test را انتخاب می‌کند. تولید کد از ترکیب Filter/Map/Reduce انجام می‌شود؛ تحلیل و بازنویسی از AST استفاده می‌کنند. اجرای آزمون‌ها با مفسر محدود، بدون `eval` یا `exec` و با سقف گام انجام می‌شود. Debug عمدتاً خطای نحوی را تعمیر می‌کند؛ برای خطای منطقی بدون مشخصات، ورودی خراب و خروجی مورد انتظار می‌خواهد. این یک Code Model عمومی نیست.
- Adapter زبان: {language['trainable_parameters']:,} پارامتر با رتبهٔ ۱۶ روی خروجی Transformer اصلی JARVIS. ماتریس‌های A و B واقعاً آموزش دیده‌اند؛ Qwen یا مدل خارجی اضافه نشده است. {language['examples_used']} مثال برای آموزش/اعتبارسنجی/آزمون انتخاب شد. خطای توکن آزمون از {language['before']['test']['loss']:.4f} به {language['after']['test']['loss']:.4f} و دقت توکن از {language['before']['test']['token_accuracy']:.2%} به {language['after']['test']['token_accuracy']:.2%} تغییر کرد. معیار Teacher Forcing است؛ به معنی بهبود اثبات‌شدهٔ ترجمه یا مکالمه نیست. Adapter در مسیر تولید عصبی و برای checkpoint سازگار فعال می‌شود؛ مسیرهای پاسخ آماده و کنترل کیفیت قبلی باقی‌اند.

در آزمون مستقیم تولید عصبی روی {len(generation["examples"])} پرسش، آداپتر خروجی {generation["changed_outputs"]} پرسش را تغییر داد، اما فقط **{generation_accepted}/{len(generation["examples"])} خروجی** از کنترل کیفیت موجود عبور کرد. این شاهد اثر واقعی آداپتر است، ولی بهبود کیفیت مکالمه را ثابت نمی‌کند؛ خروجی خام و علت رد در `language_generation.json` ثبت شده‌اند.

دو مدل دیگر نیز آموزش و ارزیابی شده‌اند: تشخیص نوع خرابی با دقت آزمون {heads['verification_failure']['metrics']['test']['accuracy']:.2%} فقط در گزارش تشخیصی استفاده می‌شود و اختیار PASS/FAIL ندارد؛ مدل Operation Language با دقت {heads['operation_language']['metrics']['test']['accuracy']:.2%} فعال نشده است. از وجود فایل آن‌ها، ارتقای تولیدی ادعا نشده است.

تمام {len(checkpoints)} checkpoint قبلی بایت‌به‌بایت حفظ شده‌اند. وزن پایهٔ Transformer تغییر نکرده، اما projection مؤثر آن هنگام استفاده از Adapter تغییر می‌کند. جزئیات Loss، Split، Confusion Matrix، شناسه‌های نمونه و SHA256 در `reports/v21` و Registry آمده است.

## حافظه

بیان صریحی مثل «من با ESP32 کار می‌کنم» به‌صورت Episode و موضوع پروژه ذخیره می‌شود. در «برای پروژهٔ جدیدم»، JARVIS موضوع قبلی را با نسبت‌دادن به گفتهٔ کاربر یادآوری می‌کند و زمینه را قطعی فرض نمی‌کند. اگر چند زمینه وجود داشته باشد، سؤال روشن‌کننده می‌پرسد. گزاره‌های معناییِ گفته‌شده توسط کاربر نیز با انتساب به همان گفته بازیابی می‌شوند؛ دانستهٔ تأییدشدهٔ خارجی جا زده نمی‌شوند. خاموش‌کردن حافظه و فراموش‌کردن اطلاعات آزموده شده‌اند؛ از جداول موجود SQLite استفاده می‌شود.

## ارزیابی

تست کامل: **{regression['primary_passed']} تست و {regression['subtests_passed']} زیرآزمون پاس، صفر شکست**. {regression['primary_passed']-899} تست اصلی به v20.1 اضافه شده است. هیچ تستی حذف نشده است. یک تست قدیمی که نمایش جواب عمداً خراب‌شده توسط مدل را انتظار داشت، اکنون تغییر Graph اولیه، رد جواب خراب و ترمیم جواب را بررسی می‌کند؛ assertion نام مسیر Runtime نیز به v21 به‌روز شد.

بازآزمون ۲۷۴ سؤال قبلی: **{previous['baseline_correct']}/274 → {previous['candidate_correct']}/274**، با **{len(previous['regressed'])} پاسخ درست ازدست‌رفته**.

بنچمارک جدید دقیقاً ۵۰۰ سؤال با تقسیم خواسته‌شده دارد. ممیزی {read('broad_overlap.json')['training_inputs_scanned']:,} ورودی آموزشی، هم‌پوشانی دقیق، نرمال‌شده و قالب عددماسک‌شده را صفر گزارش کرد. با این حال، این مجموعه فقط {read('broad_overlap.json')['unique_numeric_masked_templates']} قالب عددماسک‌شده دارد و بسیاری از سؤال‌ها خانواده‌های مشترک دارند. یک ارزیابی بیرونیِ انسانی یا شاخص جامع هوش نیست.

| بخش | v20.1 | اولین اجرای v21 | بازآزمون نهایی v21 |
|---|---:|---:|---:|
{category_table}
| جمع | {read('broad_baseline.json')['correct']}/500 | {read('broad_candidate.json')['correct']}/500 | {read('broad_final.json')['correct']}/500 |

اولین نتیجهٔ v21 پیش از استفاده از پاسخ‌های این مجموعه برای اصلاح، {read('broad_candidate.json')['correct']}/500 بود. پس از آن، عبارت مربوط به تعداد عملیات یک باگ تشخیصی را آشکار کرد و اصلاح شد؛ تقسیم ساختاری داده و آموزش Adapter نیز نهایی شد. نتیجهٔ ستون آخر **Retest** است و Holdout کورِ دست‌نخورده نیست. سؤال‌ها، کل جواب‌ها، شکست‌ها و پروتکل اولیه حفظ شده‌اند.

امتیاز کامل بخش کد مربوط به وظایف محدود همین بنچمارک است؛ برای Debug و Refactor دلخواه یا تولید هر الگوریتمی تعمیم داده نمی‌شود. بخش زبان عملاً بهبود عمومی نشان نداده است. ارزیابی ترجمه، پاسخ مرجع سخت‌گیرانه دارد و ممکن است بازنویسی‌های معتبر را کمتر بشمارد؛ نمونه‌های خروجی برای بازبینی موجودند.

## کارایی

| معیار | v20.1 | v21 |
|---|---:|---:|
| ساخت Runtime، میانهٔ ms | {perf0['startup_median_ms']:.2f} | {perf1['startup_median_ms']:.2f} |
| پاسخ P50، ms | {perf0['latency_p50_ms']:.3f} | {perf1['latency_p50_ms']:.3f} |
| پاسخ P95، ms | {perf0['latency_p95_ms']:.3f} | {perf1['latency_p95_ms']:.3f} |
| Peak RSS، MiB | {perf0['peak_rss_mb']:.2f} | {perf1['peak_rss_mb']:.2f} |

این اندازه‌گیری روی Linux، CPU، پنج سؤال ثابت، سه ساخت Runtime و ۱۲۰ درخواست انجام شده؛ import اولیه در این Startup نیست. اعداد، تضمین سرعت روی Windows یا GPU نیستند. آزمون زندهٔ وب، رابط گرافیکی Windows و GPU انجام نشده است. هزینهٔ مسیر تولید عصبی را نباید از این پنج سؤال کوتاه استنتاج کرد.

## اجرا و بازتولید

```bash
python -m pip install -r requirements-runtime.txt
python main.py
```

برای تست:

```bash
python -m pip install pytest pytest-subtests
python -m pytest -q
python benchmarks/run_broad_v21.py . benchmarks/broad_v21_500.jsonl reports/v21/reproduced_500.json
python benchmarks/run_fresh_v20.py . benchmarks/fresh_v20_clean.jsonl reports/v21/reproduced_274.json
```

برای بازسازی داده و آموزش، وابستگی‌های `requirements-training.txt` را نصب و اسکریپت‌ها را به ترتیب اجرا کنید:

```bash
python training/build_v21_data.py
python training/build_code_v21.py
python training/strengthen_splits_v21.py
python training/train_heads_v21.py
python training/train_numeric_v21.py
python training/train_language_adapter_v21.py
```

فایل‌های بنچمارک را وارد آموزش نکنید. اطلاعات نسخه‌های قبلی در بسته تاریخی‌اند؛ وضعیت فعلی در `RELEASE_V21.json` و `reports/v21/release_gate.json` است. `engineering_gate_passed=true` و **`mission_fully_completed=false`** به‌صورت صریح ثبت شده‌اند.
'''
(D/'REPORT_FA.md').write_text(report);(ROOT/'V21_README_FA.md').write_text(report);(OUT/'Jarvis_v21_Report_FA.md').write_text(report)
readme=ROOT/'README.md';original=readme.read_text()
if not original.startswith('> **نسخهٔ فعلی: v21**'):
 readme.write_text('> **نسخهٔ فعلی: v21** — مسیر Runtime، نتایج آموزش و محدودیت‌های این بسته در [V21_README_FA.md](V21_README_FA.md) است. توضیحات نسخه‌های قبلی در ادامه تاریخی‌اند.\n\n'+original)
print('release evidence ready',regression,previous['candidate_correct'],broad['candidate_correct'])
