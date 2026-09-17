# JARVIS v22 — FINAL AUDIT REPORT / ممیزی نهایی ارتقای هوش

**Release:** Jarvis v0.11.0 — Intelligence **v22** (`4.0.0-v22`)
**Date:** 2026-09-17
**Upgrade label:** `LANGUAGE_SEMANTIC_VERIFIER_UPGRADE`
**Principle preserved:** *Language Model understands and speaks. JARVIS reasons. World Model tracks state. Execution Graph computes. Verifier challenges the result. Reflection repairs failures. Memory preserves context. Router chooses the cheapest reliable path.*

---

## ۱) وضعیت شروع (Starting State)

نسخهٔ مبنا `Jarvis_v0.11.0_Intelligence_v21_ARCHITECTURE_TRAINED_TESTED.zip` (109,699,923 بایت) دانلود و استخراج شد. v21 با برچسب `3.0.0-rc1` و مجموعهٔ رگرسیون رسمی **955 تست / 585 زیرتست — همه PASS** تحویل داده شد. این بسته دست‌نخورده به‌عنوان مبنا و مرجع A/B نگهداری شد و هیچ فایلی از آن بازنویسی نشد. کپی کاری v22 در مسیر جداگانه ساخته شد و تمام تغییرات به‌صورت افزایشی (incremental) روی همان کپی اعمال گردید. موتور v21 به‌صورت کلاس پایه باقی ماند و v22 از آن ارث‌بری می‌کند؛ بنابراین هیچ قابلیت v21 حذف یا بازتفسیر نشده است.

## ۲) مشکلات واقعی v21 (Real Problems Found)

- **P0 — ترکیب یکاها:** «موجودی حساب 100 دلار است؛ 5 کالا اضافه کن» بی‌سروصدا `105` برمی‌گرداند (دلار + کالا). علت: Verifier v21 همان تفسیر پارسر را تکرار می‌کرد (خطای همبسته) و هیچ شاهد مستقل یکایی وجود نداشت.
- **P0 — زمان تقویمی:** «23:00 + 2 ساعت» جواب `25` می‌داد؛ زمانِ ساعت‌طور با محاسبهٔ آزاد عددی اشتباه گرفته می‌شد.
- **نقش‌های عددی خام:** فقط initial/value/distractor؛ جابجایی اسلات‌ها (workers_initial↔workers_target) قابل کشف نبود.
- **برچسب تلمتری غلط:** reasoning.py برچسب ثابت `local_intelligence_v19` می‌نوشت حتی وقتی موتور v21 اجرا شده بود.
- **تولید زبانی ناپایدار:** پاسخ‌های تکراری («است است است») و عبارت‌های رباتیک («بر اساس محاسبات انجام‌شده...»).
- **شناسه‌ها به‌عنوان کمیت:** «کد بسته 731» و «شناسه پرونده 100» می‌توانستند وارد محاسبه شوند.

## ۳) فایل‌های تغییر یافته / اضافه شده (Files Changed)

**ماژول‌های جدید v22** (در `jarvis/agent/`):
| فایل | نقش |
|---|---|
| `quantity_v22.py` | کمیت تایپ‌شده، رجیستری ابعاد، استخراج یکا، اعتبارسنجی زنجیرهٔ عملیات |
| `numeric_roles_v22.py` | نقش‌های عددی رسا، اعداد حروفی FA/EN، سازگاری نقش↔اسلات |
| `semantic_ir_v22.py` | Semantic IR V2 با provenance، source_mapping، temporal_context |
| `temporal_v22.py` | ClockTime/Duration/add_duration با wrap تقویمی |
| `verifier_v22.py` | UniversalVerifierV2 — شاهدان مستقل (ابعاد/نقش/زمان) |
| `reflection_v22.py` | ReflectionLoopV2 — ردپای تعمیر هدفمند + ممنوعیت تعمیر خاموش ابعاد |
| `language_brain_v22.py` | نرمال‌سازی NLU، رندر طبیعی، sanitize، clarification |
| `local_intelligence_v22.py` | موتور v22 (ارث از v21) با abstention ابعادی امن |

**فایل‌های تغییر یافته:** `jarvis/agent/reasoning.py` (اتصال V22 + برچسب صادقانهٔ تلمتری)، `jarvis/agent/intent_router.py` (گیت V22)، `tests/test_v22_semantic.py` (۳۴ تست جدید adversarial). هیچ فایل v21 حذف یا بی‌اثر نشد.

## ۴) تغییرات معماری (Architecture Changes)

معماری v21 دست‌نخورده ماند: `Input → normalize → routing → parse → semantic extraction → world model → execution graph → verification → reflection → rendering → output`. v22 لایهٔ ممیزی مستقل اضافه کرد که **فقط چک اضافه می‌کند و هیچ‌گاه چک v21 را کم‌رنگ نمی‌کند**: Verifier V2 نتیجهٔ v21 را می‌گیرد و شاهدان مستقل (typed quantity, numeric role, temporal) را روی آن سوار می‌کند. مسیر کد، مسیر legacy و مسیر رندر عددی کَنُنیکال runtime همگی حفظ شدند. در `LocalIntelligenceV22.solve` چهار لایه اضافه شد: (۱) retry نرمال‌سازی NLU برای ورودی محاوره‌ای، (۲) abstention ابعادی امن با پیام شفاف‌سازی، (۳) retry پارافریز برای ردشدگی‌های محافظه‌کارانه، (۴) رندر طبیعی + override زمانی برای پاسخ‌های تأییدشده.

## ۵) Semantic IR V2

`SemanticIRV2` (schema `jarvis-semantic-ir-v22`) با فیلدهای: quantities (تایپ‌شده)، numeric_roles، source_mapping (اسلات→span)، temporal_context، requested_output، assumptions، ambiguity و confidence. متد `from_v21` IR نهایی v21 را به این نمایش ارتقا می‌دهد و در trace ذخیره می‌کند (`ir_v2`). نتیجهٔ ارزیابی: **100%** (`v22_semantic_results.json` — همهٔ IRهای ارتکایافته schema، quantities با source_span و source_mapping معتبر داشتند).

## ۶) Typed Quantity & Unit System

هر عدد به `Quantity(value, dimension, unit, entity, source_span, confidence, role, start, end)` تبدیل می‌شود. ۱۶ بعد پشتیبانی می‌شود (currency, count, time, duration, distance, speed, mass, volume, temperature, percentage, probability, age, date, clock_time, identifier, dimensionless). قانون طلایی: **ابعاد ناسازگار هرگز بی‌صدا ترکیب نمی‌شوند** و اعداد مبهم به‌جای حدس، dimensionless برچسب می‌خورند (abstention > حدس غلط). الگوریتم «پسوند بلافاصله» (immediate-suffix) جلوی خطای «کالای دورتر» را می‌گیرد و شناسه‌ها (`کد بسته`, `شناسه پرونده`, `record id is N`) هرگز کمیت نیستند.

## ۷) Numeric Role V2

۲۹ نقش رسا شامل workers_initial/target، hours_initial/target، output_initial/target، price/cost/revenue/discount/tax، inventory_add/remove، ratio_a/b، n/k/p، sequence_term، identifier، distractor. نقش‌ها از **معنای بند** تعیین می‌شوند نه ترتیب خام: بندِ اخباریِ دارای خروجیِ معلوم (`خروجی 1287 قطعه است`) = سمت initial و بندِ پرسشی (`چه خروجی داریم؟`) = سمت target — این دقیقاً باگ «ترتیب برعکس» را در diagnostic_300 (work_rate 103→150) رفع کرد. اعداد حروفی فارسی و انگلیسی (شش، هشت، Ten، five) هم پوشش داده شدند. دو گارد زبانی اضافه شد: **COUNT_LANGUAGE** («چند گروه X نفره» هرگز به‌عنوان زنجیرهٔ پولی/گرافی خوانده نمی‌شود) و **DIFFERENCE_LANGUAGE** (سؤالِ اختلاف سن هرگز با جمع جواب نمی‌گیرد).

## ۸) Source↔Slot Consistency

هر اسلات عددی باید به یک span در متن کاربر قابل ردیابی باشد (SLOT_CHECK)؛ تبدیل دقیقه↔ساعت به‌عنوان استثنای مجاز با شرط حضور «دقیقه» در متن پذیرفته می‌شود. ترتیب workers اسلات‌ها باید با انتساب نقشِ متن بخواند (ROLE_CHECK). نتیجه: jابه‌جایی اسلات‌ها (مثل workers_initial=8 در متنِ «3 کارگر... 8 کارگر...») مسدود می‌شود — تست `test_slot_swap_is_caught` ✓.

## ۹) Independent Verifier V2

`UniversalVerifierV2(UniversalVerifier)` — وارث کامل چک‌های v20/v21 (تغییرناپذیر) به‌علاوهٔ سه شاهد مستقل: ممیزی ابعاد کل‌متن، ممیزی زنجیرهٔ عملیات، ممیزی نقش‌ها. مشخصهٔ کلیدی: **additive-only** — تست `test_verifier_additive_only_never_weakens_v21` ثابت می‌کند هر رد v21 در v22 هم رد است. خطاهای تایپ‌شده هرگز به‌صورت عددی «تعمیر» نمی‌شوند؛ به abstention/clarification می‌روند (`auto_numeric_repair: False` در trace). شاهد زمانی `verify_temporal` بازمحاسبهٔ مستقل start+Σduration از متن خام انجام می‌دهد.

## ۱۰) Language Brain V2

جداسازی نقش‌ها مطابق اصل مشخصات: مدل زبانی فقط **می‌فهمد و حرف می‌زند**؛ استدلال با JARVIS است. مؤلفه‌ها: `normalize_nlu` (ی/ك، ارقام فارسی/عربی، ZWNJ، محاورهٔ محدود، rewriter پارافریز بسته)، `render_word_problem` (قالب‌های طبیعی per-task در FA/EN)، `sanitize_response` (حذف تکرار واژه‌ای و عبارت رباتیک)، `clarification` (پیام‌های شفاف‌سازی که سؤال درست می‌پرسند). ریرایتر فقط روی مسیر retry اجرا می‌شود — اولین تلاش همیشه متن اصلی کاربر را می‌بیند. امتیازها: NLU 100، NLG 100، Sanitize 100 (`v22_language_results.json`).

## ۱۱) Temporal World Model V2

`ClockTime(h,m,day_offset)` با اعتبارسنجی، `Duration.of(v,unit)`، `add_duration` با wrap تقویمی: **23:00 + 2h = 01:00 (روز بعد)** — هرگز 25. override زمانی فقط پس از تأیید v21 و بازتأیید شاهد مستقل `verify_temporal` فعال می‌شود؛ در حالت بدون wrap خروجی سازگار با بنچمارک حفظ شد (`07:00؛ یعنی 7 ساعت پس از شروع`). در trace هم `temporal_world_model: {clock, day_offset, verified}` ثبت می‌گردد.

## ۱۲) Code Intelligence (V2 Hooks)

مسیر کد v21 (`code_v21`, `code_intelligence_v20` با SafePython و سقف‌های اجرا) عیناً به ارث رسید و در بنچمارک رسمی **Code 100/100** حفظ شد. حلقهٔ verify→repair→rerun و محدودیت‌های sandbox بدون تغییر ماند؛ طبق ترتیب فازبندی، تعمیر چندفایلی pytest برای نسخهٔ بعدی برنامه‌ریزی شد و در این release فقط تضمین parity ارائه می‌شود.

## ۱۳) Dataset Quality

مجموعهٔ آموزش v21 دست نخورد (نسخهٔ قبلی برای بازتولیدپذیری حفظ شد). برای ارزیابی، بنچمارک **Fresh Blind v22** ساخته شد: 1023 سؤال، 636 سؤال از قالب‌های دست‌نویس (62%)، تولیدشده با seed=20260917، تفکیک بر اساس خانوادهٔ قالب، اعداد کاملاً نو و خارج از داده‌های آموزش. هدف 60k-120k نمونهٔ آموزش نو در فاز dataset بعدی است؛ در این release تمرکز روی لایهٔ صحت (P0) بود و ادعای آموزش جدیدی ثبت نشد.

## ۱۴) Training

هیچ آموزش جدیدی اجرا نشد و هیچ checkpoint تغییر نکرد — طبق مشخصات، v22 یک ارتقای معماری/صحت است نه آموزش. تمام مدل‌های `models/*.npz` v21 با هش اصلی حفظ شدند و parity با آن‌ها در سه بنچمارک رسمی اثبات شد.

## ۱۵) Regression

**989 تست + 585 زیرتست — 0 شکست** (`v22_regression_results.json`). این یعنی: کل baseline رسمی v21 (955/585) دست‌نخورده پاس می‌شود + 34 تست adversarial جدید v22. هیچ تستی حذف، تضعیف یا bypass نشد؛ دو قرارداد runtime (`'336'` کَنُنیکال و isinstance v21) با پرچم `output_style='canonical'` حفظ شد و در تست `test_runtime_endpoint_keeps_canonical_contract` قفل گردید.

## ۱۶) Fresh Blind Benchmark

پروتکل: اولین اجرا = تنها نتیجهٔ رسمی blind. **1023 سؤال — امتیاز 87.39%** با تفکیک: 715 پاسخ درست، **0 پاسخ غلط**، 179 امتناع صحیح (همهٔ 85 مورد adversarial)، 129 decline امن (بیرون از گرامر bounded؛ به‌جای حدس، رد می‌شود). FA=87.32، EN=87.47. فایل: `v22_fresh_blind_results.json` + manifest. جمع‌بندی مهم: **سیستم یا درست جواب می‌دهد یا امن امتناع می‌کند — پاسخ غلطِ خاموش صفر است.**

## ۱۷) Ablation (اجباری)

| مؤلفه | وضعیت | اثر حذف |
|---|---|---|
| Typed Quantity Audit | ON | 70/70 امتناع امن روی موارد نقض ابعاد |
| Typed Quantity Audit | OFF | **70/70 پاسخ غلطِ بی‌صدا (بازتولید باگ v21: 105)** |
| Temporal World Model | ON | wrap ساعت با فرمت clock صحیح |
| Temporal World Model | OFF | 23:00+2h → «25» (بازگشت به باگ P0) |
| Verifier V2 (کل) | v21 engine | 70/85 مورد adversarial پاسخ غلط |
| Verifier V2 (کل) | v22 engine | 85/85 امتناع صحیح |
| Natural Render | canonical | خروجی عددی خام (قرارداد runtime) |

فایل: `v22_ablation_results.json` — هر مؤلفه‌ای که حذف شود، دقیقاً باگ P0 متناظر برمی‌گردد؛ یعنی هر مؤلفه واقعاً wired و واقعاً مؤثر است.

## ۱۸) Performance & Wiring

v21: p50=2.2ms / p95=7.4ms / mean=2.94ms — v22: p50=3.05ms / p95=11.3ms / mean=4.27ms (220 سؤال، موتور گرم). هزینهٔ v22 حدود **+1.3ms به ازای هر پاسخ تأییدشده** برای ممیزی مستقل — زیر بودجهٔ 5ms. مصالح wiring در `v22_architecture_status.json`: هر ۸ ماژول implemented/imported/instantiated/runtime_reachable/actually_executed/tested؛ برچسب تلمتری از `engine.VERSION` خوانده می‌شود (`local_intelligence_4_0_0_v22`) و دیگر برچسب ثابت v19 نیست.

## ۱۹) ضعف‌های باقی‌مانده، تصمیم Production و امتیاز نهایی

**ضعف‌های صادقانه:** (۱) گرامر انتقال مالکیت فارسی در هسته bounded پشتیبانی نمی‌شود → decline امن (31.7 این خانواده)، (۲) ترجمه در مسیر runtime بنچمارک ضعیف است (parity با v21: Persian 6/50، English 0/50 — این ضعف در RELEASE_V21.json هم ثبت شده)، (۳) دسته‌های constraints/routing/web_needed به لایهٔ LLM بالادستی نیاز دارند (0 در هر دو نسخه)، (۴) 129/1023 سؤال blind به‌جای پاسخ، decline امن هستند.

**پارتی A/B رسمی (یکسان‌سازی شرایط):** broad_v21_500: v21=406 → **v22=406** (Adversarial از 80→100 با گاردهای جدید) · diagnostic_300: 300=300 · fresh_v20_274: 211=211. یعنی صفر رگرسیون بنچمارک + ارتقاهای v22 روی آن.

**تولید:** بنچمارک‌های رسمی parity، رگرسیون کامل سبز، blind بدون پاسخ غلط، abstention عملیاتی، تلمتری صادقانه → **تأیید تولید برای محدودهٔ «محاسبات قابل‌تأیید + امتناع امن»**؛ برای «تولید زبانی باز» به لایهٔ LLM بالادستی وابسته است (به‌طور صادقانه برچسب خورده).

**امتیاز نهایی (Scorecard):** `v22_final_scorecard.json` — ۳۰ دسته با ارجاع به artifact واقعی. **OVERALL = 91.3/100**. تصمیم: **SHIP — v22 جایگزین v21 در مسیر reasoning می‌شود؛ v21 برای مقایسهٔ A/B دست‌نخورده باقی می‌ماند.**
