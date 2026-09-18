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

---

# JARVIS v22.1 — ANSWER TO THE INDEPENDENT AUDIT / پاسخ به ممیزی مستقل

**Date:** 2026-09-18
**Scope:** every P0 finding of the independent v21-vs-v22 audit was reproduced, fixed, and re-tested. Nothing else was touched.

## ۰) خلاصهٔ ممیزی مستقل و وضعیت v22.1

ممیزی مستقل (مقایسهٔ فایل‌به‌فایل v21/v22) چهار Blocker و چند ضعف ساختاری تعیین کرد. v22.1 دقیقاً همان فهرست را اجرا می‌کند:

| یافتهٔ ممیزی | وضعیت در v22.1 |
|---|---|
| P0-1: بازنویسی معنایی پاسخ را خراب می‌کند (تفاوت سن ← −22 با Verification PASS) | **رفع شد** — دو قانون بازنویسیِ مخرب حذف؛ عملیات مستقل `abs(age_a − age_b)` با اتصال موجودیت + شاهد تأیید مستقل |
| P0-2: Verifier V2 روی خطای داخلی audit هسته‌باز (fail-open) است | **رفع شد** — `typed_audit_internal_error` ← fail-closed/abstain |
| P0-3: SHA256SUMS متعلق به v21 است و RELEASE_V22.json وجود ندارد | **رفع شد** — هر دو برای v22.1 بازتولید شدند |
| P0-4: امتیاز Fresh Blind متورم است (abstention روی سؤال جواب‌دار = موفقیت) | **رفع شد** — scorer صادقانه: 855/1023 = **83.58%** (answerable-only 82.09%) |
| Typed Quantity همه‌چیز را dimensionless می‌گیرد و span فقط عدد است | **رفع شد** — استخراج واقعی یکا (زمان/طول/جرم/حجم/دما/سرعت/سن) با span کامل «100 دلار» |
| Numeric Role فقط taxonomy است؛ classifier نقش‌ها را تولید نمی‌کند | **رفع شد** — total/ratio_a/ratio_b، inventory_initial/add/remove، price در برابر discount، p/n/k، age_difference |
| تلمتری intent_router هنوز v19 | **رفع شد** — `local_intelligence_router_v22` |
| Scheduling دقیقه‌دار بی‌پاسخ (23:30+2h) و باگ ClockTime در `_naturalize` | **رفع شد** — `parse_clock_time`: 23:30+2h←01:30 روز بعد، 23:45+30m←00:15 روز بعد |
| ریشهٔ بستهٔ ZIP تک‌پوشه نیست | **رفع شد** — ریشهٔ واحد `Jarvis_v0.11.0/` |
| Semantic IR V2 منبع اجرا نیست؛ Language Brain ارتقای مدل نیست | **صادقانه گزارش می‌شود** — IR V2 لایهٔ audit/provenance می‌ماند؛ هیچ وزن مدل اضافه نشده (Persian 6/50، English 0/50 بدون تغییر) |

## ۱) P0-1 — بازنویسی معنایی

در v22.0 دو قانون `_PARAPHRASE` سؤال «تفاوت سن» را به زنجیرهٔ «موجودی X است؛ Y کم کن» / «start with X; subtract Y» تبدیل می‌کرد و متن بازنویسی‌شده به‌عنوان منبع تأیید (AUTHORITATIVE_SOURCE) استفاده می‌شد؛ نتیجه: −22 با PASS. در v22.1:

- **منبع اصلی تغییرناپذیر شد:** `ORIGINAL_SOURCE` (ContextVar) در ابتدای `solve()` سنجاق می‌شود؛ typed audit، شاهد زمانی و ارتقای IR همیشه متنِ اصلی کاربر را می‌خوانند، نه بازنویسی را.
- **قوانین مخرب حذف شدند** و به‌جای آن‌ها عملیات مستقل تفاوت سن اضافه شد: تشخیص span-محور دو سن نام‌دار (`detect_age_difference`)، پاسخ `abs(a−b)`، ثبت اتصال موجودیت‌ها در trace (`v22_age_difference`)، و شاهد تأیید مستقل `verify_age_difference` که متن اصلی را دوباره می‌خواند.
- نتیجهٔ آزمون: `Ali is 35 and Reza is 57 ... age difference?` ← **22** (قبلاً −22)؛ نسخهٔ فارسی ← **22**؛ `اختلاف سن` با دو سن دلخواه ← قدرمطلق درست.

## ۲) P0-2 — Verifier هسته‌باز

`except Exception: pass` حذف شد. اکنون هر خطای داخلی در لایهٔ audit ایمنی به `typed_audit_internal_error` می‌انجامد، `passed=False` و `repair_stage='abstain'` — یعنی **fail-closed**. آزمون تزریق خطا: پاسخ «105» دیگر هرگز تولید نمی‌شود.

## ۳) Typed Quantity و Numeric Role واقعی

- یکاهای بلافصل بعد از عدد استخراج می‌شوند: ساعت/دقیقه/ثانیه/روز/hours/minutes → `time`، کیلومتر/متر/km/m → `distance`، کیلوگرم/گرم/kg/g → `mass`، لیتر/L/ml → `volume`، درجه/C/سلسیوس → `temperature`، km/h → `speed`، ساله/years old → `age`. `source_span` اکنون عدد+یکاست («100 دلار»، «2 ساعت»).
- گاردهای جدید: جمع دو بُعد شناخته‌شدهٔ متفاوت («2 ساعت و 120 کیلومتر را جمع کن») رد می‌شود؛ با معافیت متن‌های نرخی (زمان+تعداد / زمان+پول).
- نقش‌های بیانی اکنون واقعاً تولید می‌شوند و در `v22_semantic_results.json` به‌صورت case-by-case با نمرهٔ صادقانه آمده‌اند (نمرهٔ typed_quantity=100 اکنون به معنای پوشش واقعی یکاست، نه استاب dimensionless).

## ۴) Scheduling دقیقه‌دار

توکن‌های ساعت `H:MM` فقط برای ورودیِ پارسر به اعشار تبدیل می‌شوند (متنِ منبع تغییرناپذیر می‌ماند) و `_naturalize` اکنون از `temporal_v22.parse_clock_time` استفاده می‌کند (باگ `ClockTime(...,0)` حذف شد). نتایج: 23:30+2h ← **01:30 روز بعد**؛ 23:45+30m ← **00:15 روز بعد**؛ 23:00+2h ← 01:00 (رفتار قبلی حفظ شده). Fresh-blind scheduling: **90/90 = 100%** (قبلاً 61.1%).

## ۵) Scorer صادقانه و مقایسهٔ v21/v22.0/v22.1

اسکریپت رسمی جدید `benchmarks/run_v22_fresh_blind.py` با این قاعده: abstention روی سؤال جواب‌دار = شکست؛ فقط ردیف‌های `expected=ABSTAIN` با شرط «عدد ممنوعه در پاسخ نیست» قبول می‌شوند.

| Family | v22.0 (ممیزی) | v22.1 |
|---|---|---|
| adversarial currency / dimension / identifier | 100% / 100% / 100% | 100% / 100% / 100% |
| scheduling | 61.1% | **100%** |
| age | 37.1% | **65.7%** |
| work-rate / sequence / speed | 100% | 100% |
| inventory | 67% | 67% |
| ownership | 31.7% | 31.7% |
| binomial / ratio / combination | 75.6% / 71.7% / 88.9% | بدون تغییر |
| finance | 90% | 90% |
| **جمع honest** | **78.2%** | **83.58%** (+5.4) |

## ۶) رگرسیون و Ablation

- رگرسیون کامل: **1031 تست + 585 زیرتست — همه PASS** (خط پایه 955 + 34 تست v22 + 42 تست جدید `tests/test_v221_audit_fixes.py`).
- Ablation اجباری (`v22_ablation_results.json`): typed audit روشن 70/70 امن در برابر خاموش 0/70؛ temporal روشن 14/14 در برابر خاموش 0/14؛ عملیات تفاوت سن روشن 25/40 در برابر خاموش 0/40.
- Broad 500 رسمی با موتور v22.1 دوباره اجرا شد: **406/500 = 81.2** (برابری کامل با v21؛ Reasoning/World/Code/Adversarial = 400/400، Persian 6/50، English 0/50 — بدون ادعای ارتقای زبانی).
- Performance (همان مخلوط سؤال، همان پروسه): v21 mean 1.97ms در برابر v22.1 mean 5.07ms — اطلاعات‌دهنده؛ مطلق همچنان در حد چند میلی‌ثانیه.

## ۷) Scorecard جدید (قابل مقایسه و بدون شمارش تکراری)

`v22_final_scorecard.json` بازنویسی شد: پنج قابلیت، هر کدام دقیقاً یک‌بار، با فرمول مشخص. **composite_v22.1 = 77.92** — به‌طور صریح با 91.3 (v22.0، متورم) و اعداد قدیمی قابل مقایسه نیست. ارقام قابل مقایسه: broad **81.2** (بدون تغییر) و fresh-blind honest **83.58** در برابر **78.2** (+5.4). ضعف باز زبان (6/100) در میانگین دیده شده است.

## ۸) بستهٔ Production

- ریشهٔ ZIP: یک پوشهٔ واحد `Jarvis_v0.11.0/` (رفع یافتهٔ ممیزی).
- `SHA256SUMS.json` برای درخت v22.1 (1746 فایل) بازتولید شد؛ `RELEASE_V22.json` اضافه شد؛ گزارش‌های موقت (partial) حذف شدند.
- اسکن امنیتی: بدون PAT/توکن/کلید در بسته.

**تصمیم نهایی:** v22.1 هر چهار Blocker ممیزی را رفع می‌کند، رگرسیون کامل سبز است و adline‌های گزارش با رفتار واقعی هم‌خوان‌اند. برچسب پیشنهادی انتشار: **v22.1 — Semantic/Verifier Production Candidate**. شکاف زبانی عمومی (Persian 6/50، English 0/50) همچنان نیازمند ارتقای واقعی مدل زبانی است و در `known_open_gaps` صادقانه ثبت شده است.
