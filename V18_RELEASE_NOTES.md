# JARVIS Intelligence v18 — Final Release Notes

## هدف نسخه

v18 روی کاهش حساسیت JARVIS به جمله‌بندی، فعال‌کردن واقعی داده‌های جدید در وزن‌های یادگرفتنی، و استدلال چندمرحله‌ای ساختاریافته تمرکز دارد. مسیر اصلی حل مسائل معنایی اکنون به‌صورت Parse → Role Binding → Plan → Solve → Verify → Answer کار می‌کند.

## Semantic Intermediate Representation v18

`SemanticIR` اکنون علاوه بر `task` و `slots`، اطلاعات `predicate` و `polarity` را نیز نگه می‌دارد. این تغییر برای سؤال‌های بله/خیر مانند Prime ضروری است تا پاسخ «بله/خیر» با خود گزارهٔ پرسیده‌شده هماهنگ بماند.

Role binding برای probability، permutation، ratio، speed، work scaling، age و inventory از ترتیب واژه‌ها مستقل‌تر شده و در صورت کافی نبودن قواعد قطعی می‌تواند از مدل Numeric Slot Tagger کمک بگیرد.

## مدل‌های یادگرفتنی فعال

- Transformer: `jarvis_nano_v18.npz` — 23,077,376 پارامتر — dataset_v009
- Semantic Router: `semantic_router_v18.npz` — 3,145,740 پارامتر — dataset_v009
- Semantic Frame Classifier: `semantic_frame_v18.npz` — 1,310,730 پارامتر — dataset_v009
- Numeric Slot Tagger: `numeric_slot_tagger_v18.npz` — 1,179,666 پارامتر — dataset_v009
- مجموع model-oriented parameters: **28,713,512**

Transformer v18 با continual training محافظه‌کارانه و rehearsal ساخته شده و فقط پس از عبور از quality gate فعال شده است.

## Dataset v009

- Parent: dataset_v008
- کل نمونه‌ها: 41,295
- نمونه‌های جدید v18: 3,195
- ورودی جدید یکتا: 3,195 / 3,195
- Exact overlap با parent: 0
- Cross-split concept leakage: 0
- Train: 33,176
- Validation: 3,953
- Test: 4,166
- Tokenizer normalized round-trip accuracy: 100%
- Unknown token rate: 0%

داده‌های جدید روی semantic roles، question polarity، order-independent numerical roles، probability، permutation، ratio، speed، age reasoning، inventory multi-step، prime reasoning، code generation، Persian formal rewrite و routing تمرکز دارند.

## Neural Quality Gate

در مقایسه با baseline release قبلی:

- v18-new cross entropy: 2.321% بهتر
- v18-new token accuracy: +0.145 واحد درصد
- Persian-v18 cross entropy: 2.354% بهتر
- Legacy-v008 cross entropy regression: +0.364% (زیر سقف release 0.5%)
- Legacy token accuracy delta: -0.388 واحد درصد (داخل حد release)

این release به‌خاطر پاس‌کردن تمام شروط gate فعال شده است.

## Semantic model metrics

روی split ساختاریافتهٔ داخلی:

- Frame Classifier test accuracy: 100% روی کلاس‌های حاضر در split
- Numeric Slot Tagger test accuracy: 95.80%
- Semantic Router test accuracy: 100% روی کلاس‌های حاضر در split

این اعداد معیار داخلی مدل‌ها هستند و به معنی 100% هوش عمومی نیستند.

## Holdout End-to-End

Benchmark رسمی v18 شامل 50 سؤال تازه است:

- Exact overlap با dataset_v009: 0 / 50
- Web: غیرفعال
- Tool invocation: غیرفعال
- نتیجه: 50 / 50 پاس

این benchmark روی خانواده‌هایی مانند biased-coin probability، dice probability، permutation phrasing، ratio، speed با ترتیب‌های متفاوت، age، inventory multi-step، worker scaling، prime polarity، transitive reasoning، code generation و Persian formal rewrite تمرکز دارد.

## Regression

پس از آخرین patch، کل test suite در سه batch مستقل اجرا شد:

- Base through v07: 604 passed + 64 subtests
- v08 through v13: 59 passed + 45 subtests
- v14 through v18: 23 passed + 340 subtests

مجموع: **686 تست اصلی + 449 subtest، صفر Failure**.

تست‌های تاریخی v12/v13 برای حفظ validation وزن‌های قدیمی نگه داشته شده‌اند، اما assertion قدیمی «v13 باید مدل پیش‌فرض باشد» به انتظار صحیح release v18 مهاجرت داده شده است.

## تغییرات رفتاری مهم

- Prime question polarity در IR ذخیره و در پاسخ رعایت می‌شود.
- Probability برای `p` اعشاری/درصدی و ترتیب‌های مختلف جمله عمومی‌تر شده است.
- Probability سکهٔ سالم در پاسخ، کسر دقیق و درصد را هم‌زمان ارائه می‌کند.
- Ratio، Permutation، Speed، Age، Inventory و Work Scaling solverهای ساختاریافته دارند.
- Code Generation قبل از Tool/Web فرصت حل محلی دارد.
- Code generation دیگر به یک نام تابع ثابت در benchmark وابسته نیست؛ semantics کد معیار اصلی است.
- Explicit arithmetic مانند `125 * 43` قبل از learned semantic routing وارد Calculator قطعی می‌شود.
- Learned semantic router تنها با شواهد معنایی کافی اجازهٔ override مسیرهای قدیمی را دارد.

## نکته

v18 یک مدل زبانی بزرگ جدید نیست. جهش این نسخه از ترکیب Transformer بازآموزی‌شده، مدل‌های semantic کمکی، SIR/role binding، solverهای چندمرحله‌ای و verification به‌دست آمده است.
