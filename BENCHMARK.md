# JARVIS v0.11.0 Benchmarks

اعداد زیر اندازه‌گیری شده‌اند. میزبان Build: Python 3.12، NumPy CPU، `os.name=posix`، بدون VRAM. Runtime برای Python 3.13 طراحی شده، اما این اعداد روی 3.12 ثبت شده‌اند.

## Nano evaluation

411 نمونهٔ Test با concept group جدا؛ teacher-forced next-token، sequence حداکثر 64.

| Metric | مقدار |
|---|---:|
| Cross entropy / Perplexity | 3.882264 / 48.533978 |
| Token accuracy | 0.441933 |
| Action / Argument / Entity | 0.778095 / 0.391794 / 0.315789 |
| Context / Knowledge | 0.754167 / 0.358447 |
| Planning / Recovery | 0.414079 / 0.299363 |

Conversation-fa/en و Tool در این Test split token نداشتند؛ مقدارشان `null` است، نه صفر جعلی.

## 1,000 unseen Agent prompts

فایل `unseen_v003_1000.jsonl` overlap صفر با Train دارد.

| Dimension | Score |
|---|---:|
| Exact case / Intent | 1.0 / 1.0 |
| Entity micro-F1 | 1.0 |
| Tool / Argument | 1.0 / `null` (این dataset فیلد argument مستقل ندارد) |
| Plan exact match | 1.0 |
| Safety non-execution | 1.0 |
| Error | 0 / 1,000 |

Suite محدود است: 400 Entity، 200 Knowledge، 200 Planning، 200 Safety. Score کامل ادعای هوش عمومی، Coding یا مکالمهٔ آزاد نیست. Execution/Recovery/Conversation quality بدون Windows/Human `null` است.

## Runtime

`benchmark_model.py --rounds 4`:

| Metric | مقدار |
|---|---:|
| File / resident weight | 15,201,155 / 23,086,308 byte |
| Process RAM delta | 25.992MB |
| Load / first token | 123.476 / 275.121ms |
| Generation | 15.716 token/s |
| VRAM | 0MB |

## FP32 / FP16 / INT8

| Precision | Weight RAM | Load | First token | token/s | Max logit error |
|---|---:|---:|---:|---:|---:|
| FP32 | 88.033MB | 171.368ms | 78.173ms | 58.845 | 0 |
| FP16 | 44.017MB | 1,202.911ms | 2,585.740ms | 1.805 | 0.000303 |
| INT8 | 22.017MB | 124.514ms | 490.892ms | 9.835 | 0.0000006 |

در NumPy میزبان، BLAS FP32 سریع‌تر بود. INT8 به‌خاطر کمترین RAM Default است و matrixها واقعاً INT8 مقیم‌اند. Fast Brain/Lazy Load باعث می‌شود فرمان معمول این هزینه را نداشته باشد. Latency به CPU/BLAS/prompt وابسته است.

## Fast Brain

Test مستقل 186تایی: float accuracy `0.709677`، hybrid `0.655914`، known `0.642458`، unknown recall `1.0`. Train accuracy 1.0 معیار تعمیم نیست.

## Grounded Intelligence v0.7.1

`python training/evaluate_grounding_v071.py` در آخرین اجرا `50/50` مورد را پاس کرد:

| Category | Passed / Cases |
|---|---:|
| Text QA روی متن supplied | 8 / 8 |
| Formal deduction محدود | 7 / 7 |
| Offline knowledge retrieval | 24 / 24 |
| Unsupported-claim rejection | 6 / 6 |
| Multi-source search fixture | 5 / 5 |

این مجموعه curated و regression-oriented است و accuracy برابر 1.0 را نباید به سؤال آزاد، وب زنده یا هوش عمومی تعمیم داد. فایل خام نتیجه `models/grounding_evaluation_v071.json` است.

Live Internet در محیط Build اجرا نشد چون outbound network عمومی محدود بود. MediaWiki request shape با مستندات رسمی و JSON fixture، DuckDuckGo HTML/Lite با دو parser fixture، و کل Evidence Pipeline با دو domain مستقل deterministic آزمایش شد. نتیجهٔ زنده یا صحت محتوای متغیر جعل نشده است.

## Context, research and UI regressions v0.7.2

12 تست بازگشتی جدید همگی Pass شدند. این مجموعه استخراج/تعویض موضوع، حمل موضوع به follow-up، جلوگیری از cross-entity result، dedup URL، حذف passage تبلیغاتی، citation دقیق، route بستن مرورگر پیش‌فرض، تشخیص پردازش پس‌زمینه و پاک‌سازی bidi در Clipboard را پوشش می‌دهد. یک Integration test مکالمهٔ دقیق «تحقیق دربارهٔ لیونل مسی → چقدر گل زده تا الان؟» را با Search fixture اجرا می‌کند و نبود «رونالدو» را assert می‌کند.

## v0.8 hardening regressions

11 تست اختصاصی v0.8 مسیر Permission، جلوگیری از overwrite، Undo برای append، DOCX فشردهٔ مخرب، خاموشی واقعی Memory، جلوگیری از Git alias/config bypass، cancellation decoder و مسیریابی Dialogue Adapter بدون Web را پوشش می‌دهند. عبارت دارای پیشوند گفت‌وگویی نیز نمی‌تواند درخواست حذف خطرناک را پنهان کند.

## v0.9 Persian fluency

Dataset شامل 49 نمونهٔ بازبینی‌شده در 7 کلاس مساوی است. یک نمونه از هر کلاس برای Holdout کنار گذاشته شد؛ classification برابر `7/7` و generation acceptance برابر `7/7` بود. سپس artifact Release روی هر 49 نمونه Train شد. این عدد فقط توان تشخیص این وظایف محدود را می‌سنجد و معیار کیفیت انسانی عمومی یا برابری با ChatGPT نیست.

## v0.10 dialogue follow-up

Dataset شامل ۹۴ نمونهٔ فارسی بازبینی‌شده در ۸ کلاس است. ۸۶ نمونه برای آموزش ارزیابی و ۸ paraphrase، یکی از هر کلاس، برای Holdout استفاده شد. classification برابر `8/8` بود؛ سپس artifact Release روی هر ۹۴ نمونه بازآموزی شد. شش Integration test نیز زنجیرهٔ کامل حافظه/خلاصه/ترجمه، دو Fact مستقل، Context منقضی و ممنوعیت Web Search در ادامهٔ ناقص را بررسی می‌کنند. این ارزیابی فقط همین وظایف محدود را می‌سنجد و ادعای هوش عمومی نیست.

## Test policy and limits

Suite شامل 645 تست Unit، Integration، Security، checksum، SQLite reopen، TXT/ZIP bounded read، NLU scenario، Planner، Dry Run، Undo، permissions، RAG، grounded text analysis، Persian fluency، follow-up model، subject-aware search/research، process verification، GUI text behavior و LAB catalog است. نتیجهٔ آخرین Package در CHANGELOG ثبت می‌شود.

Desktop/Power/Window واقعی Windows و visual GUI smoke در محیط Linux انجام نشد؛ platform-specific tool خارج Windows باید `unsupported` دهد و success شبیه‌سازی نشده است.
## v0.11 bilingual and cognitive models

| مدل | کل داده | Train موقت | Holdout | نتیجه | کف پذیرش |
|---|---:|---:|---:|---:|---:|
| Bilingual Fluency | 105 | 91 | 14 | 100% classification / 100% generation | 85% / 100% |
| Cognitive Skills | 96 | 80 | 16 | 93.75% classification | 87.5% |

Holdout پیش از آموزش نهایی کنار گذاشته شده است. پس از قبولی، artifact Release روی کل داده بازآموزی شده؛ بنابراین اعداد بالا ارزیابی مدل موقتِ Train-only هستند، نه سنجش artifact روی داده‌ای که دیده است.
