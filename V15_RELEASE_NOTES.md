# JARVIS Intelligence v15 — Final Release Notes

این نسخه برای رفع گلوگاه واقعی v14 ساخته شده است: سؤال قبل از رسیدن به Reasoning/Smart Brain نباید به‌اشتباه به `unknown`، `web_search`، Desktop action یا یک regex نامرتبط فرستاده شود.

## تغییرات Runtime

- Semantic Router یادگرفتنی با **3,145,740 پارامتر** و 12 intent مستقل: math، probability، logic، translation، rewrite، coding، code_trace، word_problem، constraint_writing، fresh_information، desktop_command و general_question.
- ترتیب Dispatch بازطراحی شده تا Code Trace قبل از Math، reasoning guardهای قطعی قبل از actionهای مبهم، و fallback هوشمند قبل از `unknown/web` قرار گیرد.
- جلوگیری از Regex Collision: prime/combinatorics/percentage/rate و سایر solverها فقط با شواهد مفهومی کافی فعال می‌شوند.
- Local Intelligence v15 برای Probability دوجمله‌ای با نقش معنایی `k` و `n`، ترکیب و جایگشت، سن، تناسب، نرخ کار، سرعت/زمان/مسافت، دنباله و محاسبات چندمرحله‌ای.
- Code Understanding امن بر پایه AST برای trace کدهای ساده، بدون اجرای arbitrary code.
- Translation/Rewrite مسیر مستقل دارند و متن inline داخل همان Prompt را استخراج می‌کنند.
- Constraint Following با `ConstraintExtractor -> Generate -> ResponseVerifier -> Repair` برای تعداد جمله/کلمه، زبان، include/exclude و محدودیت خروجی.

## داده و پارامتر

- `dataset_v007`: **36,530 نمونه**.
- دادهٔ جدید v15: **21,800 نمونه** در **500 shard/dataset**.
- Generalization: 5,000 نمونه، 5,000 ورودی یکتا، 4,996 خروجی یکتا.
- Reasoning: 5,000 نمونه، 5,000 ورودی و 5,000 خروجی یکتا، 50 خانوادهٔ معنایی.
- Persian: 4,000 نمونه، 4,000 ورودی/خروجی یکتا با گیت زبانی فارسی.
- Routing: 4,800 نمونه، 12 کلاس کاملاً متوازن، 400 نمونه برای هر intent و بدون conflicting label.
- Constraint: 3,000 نمونه، 3,000 ورودی/خروجی یکتا.
- Concept leakage بین train/validation/test: **0**.
- Tokenizer round-trip: **100%**؛ unknown-token rate: **0%**.
- هستهٔ Neural فعلی: 23,077,376 پارامتر؛ Semantic Router جدید: 3,145,740 پارامتر؛ مجموع اجزای model-oriented: **26,223,116 پارامتر**.

## گیت رفتاری مستقل

روی 22 سؤال End-to-End که exact-match آن‌ها در `dataset_v007` صفر است و Web/Tool برای پاسخ‌دهی ممنوع شده:

- v14: **4/22 = 18.2%**
- v15: **22/22 = 100.0%**

این benchmark مخصوص حوزه‌های اصلاح‌شدهٔ v15 است و نباید به‌عنوان امتیاز هوش عمومی تفسیر شود.

## Regression Gate

تمام 27 ماژول تست پروژه در batchهای مستقل اجرا شدند:

- **672 تست اصلی پاس**
- **231 subtest پاس**
- **0 failure**

اجرای یک‌جای suite در محیط ساخت به سقف زمانی فرمان رسید، اما همان مجموعه به‌طور کامل در batchهای مستقل اجرا و پاس شده است.

## شفافیت درباره Neural Core

`jarvis_nano_v13.npz` عمداً rename نشده است. وزن 23M قبلی حفظ شده، چون آموزش یک core جدید در CPU این محیط به release gate قابل‌اعتماد نرسید. بنابراین جهش v15 واقعی است، اما عمدتاً از **Semantic Routing + Runtime Reasoning + Code Understanding + Constraint Following + Dataset Quality** می‌آید، نه از ادعای ساختگیِ وزن Transformer جدید.
