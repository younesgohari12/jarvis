# CHANGELOG — JARVIS v0.9.5

## Release 0.9.5 — Generalization, turn ordering and Persian Paste

- قیاس «همه/هیچ» به‌صورت مجموعه‌ای و محلی حل می‌شود و دیگر برای واژه‌های ساختگی به وب نمی‌رود.
- نام روزهای مرکب با تطبیق طولانی‌ترین نام تشخیص داده می‌شوند؛ «چهارشنبه» دیگر به‌اشتباه «شنبه» نیست.
- حافظهٔ پروژه هر دو عبارت «کد پروژه» و «رمز پروژه» را می‌پذیرد و مقدار را پس از سؤال‌های میانی بازیابی می‌کند.
- پیگیری CPU/GPU برای مثال و خلاصه و زنجیرهٔ ترجمهٔ بدون واژهٔ «کلمه» مانند `کودک → child → children` اضافه شد.
- Paste در چیدمان فارسی ویندوز، رویداد استاندارد `<<Paste>>`، `Shift+Insert` و منوی راست‌کلیک «چسباندن» پشتیبانی می‌شود.
- هنگام پردازش پاسخ، Composer موقتاً قفل می‌شود تا پیام دوم پیش از پایان پاسخ اول وارد صف و باعث جابه‌جایی ظاهری خروجی نشود.
- ۵ تست بازگشتی جدید دقیقاً خروجی‌های گزارش‌شده و رفتار Clipboard/صف را پوشش می‌دهند؛ مجموعهٔ کامل ۶۳۹ تست موفق است.

تاریخ Build: 2026-08-28

## Release 0.9.4 — Local challenge reasoning and route isolation

- حل‌گر محلی قابل‌تعمیم برای زاویهٔ ساعت، صفرهای انتهای فاکتوریل، احتمال حداقل یک موفقیت، مسئلهٔ مجموع/اختلاف و چند خانوادهٔ دنباله اضافه شد.
- معماهای جعبه‌های با برچسب غلط، طناب‌های غیریکنواخت، کلید/لامپ، روز هفتهٔ فرضی و قیاس مجموعه‌ای همراه با استدلال و کنترل جواب حل می‌شوند.
- گارد استدلالی پیش از Date/Time، فرمان‌های Desktop و Search اجرا می‌شود؛ واژه‌های «بسته» و «پیدا کن» در متن معما دیگر ابزار نامرتبط را فعال نمی‌کنند.
- چند سؤال نقل‌قول‌شده به اجزای مستقل تقسیم می‌شود و اگر متن در واقع اسکریپت آزمون حافظه باشد، از نشت موضوع ترجمهٔ قبلی جلوگیری و ارسال نوبت‌به‌نوبت درخواست می‌شود.
- ۷ تست بازگشتی جدید، همهٔ خطاهای گزارش‌شده، زنجیرهٔ حافظه/ترجمه و مقادیر دیده‌نشدهٔ فرمول‌ها را پوشش می‌دهند؛ مجموعهٔ کامل ۶۳۴ تست با موفقیت اجرا شد.

تاریخ Build: 2026-08-28

## Release 0.9.3 — Context and medical-source quality

- Greeting ابتدای یک سؤال واقعی از متن سؤال جدا می‌شود؛ «سلام بیضه چیست» دیگر به پاسخ احوال‌پرسی ختم نمی‌شود.
- حالت ترجمه و واژهٔ موضوع در Context نگه‌داری می‌شود؛ ادامه‌هایی مانند «و کلمهٔ … چطور؟» و «انگلیسیش چی میشه؟» موضوع قبلی را حفظ می‌کنند.
- تعریف خنثی و محلی برای بیضه، آلت مردانه و واژن و معادل انگلیسی واژه‌های مرتبط افزوده شد تا سؤال‌های پایه بی‌دلیل وارد وب نشوند.
- نتایج پزشکی تیترزرد/تبلیغاتی حذف و URLهای یکسان با شکل encoded و Unicode پیش از رتبه‌بندی ادغام می‌شوند.

## Release 0.9.2 — Conversation copy controls

- انتخاب‌گر تعداد پیام با گزینه‌های ۱، ۵، ۱۰، ۲۰، ۵۰ و همهٔ پیام‌ها به نوار کنترل گفتگو اضافه شد.
- دکمهٔ «کپی گفتگو» پیام‌های انتخاب‌شده را به‌ترتیب و همراه با برچسب «شما» و `JARVIS` در Clipboard قرار می‌دهد.
- `Ctrl+C` برای متن انتخاب‌شده در Bubbleها و کادر ورودی و `Ctrl+V` برای کادر ورودی به‌صورت صریح فعال شد.
- متن Paste و خروجی Copy از کنترل‌های bidi پاک می‌شود تا مشکل کادرهای Unicode دوباره ایجاد نشود.

## Release 0.9.1 — RTL display and language-question patch

- نویسه‌های کنترلی bidi پیش از نمایش پاک می‌شوند تا در برخی نسخه‌های Tk/Windows به‌شکل کادرهای `RLI` و `PDI` دیده نشوند.
- Code، URL و Path بدون افزودن نویسهٔ مخفی و با tag چپ‌چین نمایش داده می‌شوند؛ کپی نیز همهٔ کنترل‌های bidi شناخته‌شده را حذف می‌کند.
- پرسش‌هایی مانند «مخفف انگلیسی کلمه سلام چیست؟» دیگر Greeting محسوب نمی‌شوند و از دانش محلی پاسخ دقیق می‌گیرند.
- نسخهٔ Seed دانش به 9 افزایش یافت تا نصب‌های قبلی هم مورد جدید را به پایگاه دادهٔ محلی خود دریافت کنند.

تاریخ Build: 2026-08-27

## Release 0.9.0 — Persian fluency and writing quality

- مدل مکمل Persian Fluency روی 49 نمونهٔ فارسی بازبینی‌شده و 7 کلاس آموزش داده شد.
- درخواست متن، پیام رسمی، کپشن، داستان، توصیه و بازنویسی دیگر به UNKNOWN یا Web Search اشتباه نمی‌رود.
- داستان بر اساس موضوع تغییر می‌کند و پیام رسمی، دادهٔ صریح کاربر مانند ارسال یا تحویل را حفظ می‌کند.
- توضیحی که «ساده و روان» خواسته شود، مقدمهٔ طبیعی می‌گیرد بدون افزودن ادعای تازه.
- پاسخ حمایتی همدلانه‌تر شد و تشخیص پزشکی ساختگی ارائه نمی‌دهد.
- Quality Gate حروف فارسی، طول، token کنترلی و تکرار جمله را بررسی می‌کند.
- ارزیابی Holdout مدل فارسی `7/7` و مجموعهٔ کامل Release شامل 618 تست موفق است.

## Release 0.8.0 — Safety, cancellation, search and dialogue quality

- Permission مرکزی اصلاح شد: UI، Browser DOM، Clipboard و Process close/restart به تأیید صریح نیاز دارند.
- Git به‌جز version query نیازمند تأیید است و alias/config execution override مسدود شده است.
- File write/copy/move/rename مقصد موجود را overwrite نمی‌کند و append دارای Undo است.
- DOCX/XLSX با بودجهٔ byte، compression ratio و سقف cell/string خوانده می‌شوند.
- Memory خاموش دیگر متن Chat، argument ابزار، query تحقیق و failure sample را persist نمی‌کند.
- Stop پاسخ به decoder و CommandRunner متصل شد و shutdown رابط با worker فعال race ندارد.
- Search به DuckDuckGo + MediaWiki + Bing RSS ارتقا یافت و page fetch موازی و محدود است.
- Dialogue Adapter فارسی/انگلیسی اضافه و token leakage/repetition guard سخت‌تر شد.
- metadata مدل Release واقعاً MHA با 8 KV head و FFN 1024 است؛ alignment ضعیف رد شد.
- ارزیابی ساختاری 1,000 ورودی بدون overlap و مجموعهٔ 606 تست Unit/Integration/Security با موفقیت کامل اجرا شد.

## Patch 0.7.2 — Context-safe research and modern RTL chat

- `DialogueSubjectResolver` اضافه شد تا موضوع یک تحقیق در follow-upهای کوتاه حفظ شود و موضوع صریح جدید جای آن را بگیرد.
- Search/Research اکنون `subject` ساختاری می‌پذیرند؛ result یا passage نامرتبط با موضوع حذف می‌شود و کمبود شواهد به‌جای پاسخ دربارهٔ فرد دیگری گزارش می‌شود.
- `ResearchEngine` خلاصهٔ طبیعی و ساختاریافته می‌سازد، URLهای encode‌شده/تکراری را یکی می‌کند، متن تبلیغاتی را کنار می‌گذارد و citationها را دقیقاً به منابع استفاده‌شده وصل می‌کند.
- سناریوی واقعی «تحقیق دربارهٔ لیونل مسی → چقدر گل زده تا الان؟» به تست Integration تبدیل شد و پاسخ رونالدو در این Context ممنوع است.
- Intent مستقل `close_default_browser` اضافه شد. در Windows مرورگر پیش‌فرض از HTTPS `UserChoice` تشخیص داده می‌شود؛ بستن ابتدا با `WM_CLOSE` انجام و بسته‌شدن پنجره بررسی می‌شود. باقی‌ماندن پردازش پس‌زمینهٔ مرورگر دیگر failure جعلی نیست.
- پنل Chat با `CustomTkinter 6.0.0` بازطراحی شد: Bubbleهای گرد و مستقل، Composer مدرن، کارت attachment/status و منوی Copy.
- متن فارسی در Bubble راست‌چین و Code/URL چپ‌چین است؛ علامت‌های bidi هنگام Copy حذف می‌شوند تا نویسه‌های `⁧` و `⁩` وارد Clipboard نشوند.
- UI همچنان Worker-thread است و هنگام Search/Reasoning فریز نمی‌شود؛ Tk کلاسیک fallback باقی مانده است.
- 12 تست بازگشتی جدید اضافه شد؛ مجموع Suite به 594 تست رسید.

وزن‌ها و metricهای مدل در Patch 0.7.2 تغییر نکرده‌اند؛ این ارتقا در Context، Retrieval، Synthesis، Process verification و UI است.

## Patch 0.7.1 — Grounded understanding and research

- `QueryAnalyzer` اضافه شد تا پیش از پاسخ، زبان، نوع سؤال، متن supplied، تازگی، volatility، پیچیدگی و ambiguity را مشخص کند.
- `TextReasoner` متن داخل پیام، TXT پیوست‌شده و ارجاع صریح به پیام قبلی را می‌خواند؛ پاسخ چندسؤالی، خلاصهٔ extractive و استنتاج منطقی محدود ارائه می‌دهد.
- `EvidenceVerifier` پاسخ متنی را در برابر evidence بررسی می‌کند؛ نبود overlap کافی به UNKNOWN/Search می‌رود.
- شاخهٔ hard-coded مقایسهٔ SQLite/JSON از Reasoning حذف شد و پاسخ از Knowledge عمومی بازیابی می‌شود.
- Search تک‌Provider به DuckDuckGo HTML/Lite + MediaWiki فارسی/انگلیسیِ موازی تبدیل شد.
- Search نتیجه را deduplicate، page text را bounded extract، passage را rank و هر جمله را به URL خودش cite می‌کند.
- `SearchReport` و `ResearchReport` دارای confidence، status، providers و errors شدند. Statusهای failure دیگر به پیام عمومی Tool failure تبدیل نمی‌شوند.
- Research query rewrite از جدول ترجمهٔ چهارسؤالی به بازنویسی generic بر پایهٔ term/freshness تغییر کرد.
- cache جست‌وجوی پایدار 5 دقیقه و سؤال تازه 45 ثانیه است؛ Provider errorها در `search.jsonl` ثبت می‌شوند.
- Knowledge آفلاین از 64 به 97 مدخل فارسی/انگلیسی افزایش یافت؛ شبکه، امنیت، برنامه‌نویسی، آمار و علوم پایه پوشش بیشتری دارند.
- تشخیص زبان برای سؤال فارسیِ دارای اصطلاح انگلیسی اصلاح شد.
- false-positiveهای `3-2-1` و `زتا-۹` در Calculator، و «همبستگی یعنی حتما علت؟» در affirmation حل شدند.
- Quality Guard اکنون prompt echo، duplicate phrase، incomplete sentence، language mismatch و invalid-character distribution را هم Reject می‌کند.
- attachment متنی هنگام attach به RAG محلی index می‌شود؛ failure این index کل attachment را خراب نمی‌کند.

مدل و وزن Nano در Patch 0.7.1 دوباره Train نشده‌اند؛ parameter count، model metrics و provenance همان Release 0.7.0 است. ارتقای این Patch در reasoning/retrieval/verification انجام شده و ادعای بهبود مصنوعی weight مطرح نمی‌شود.

## تغییرات اصلی

- نسخهٔ Runtime، GUI، Config، Registry و assetها به `0.7.0` ارتقا یافت.
- معماری Hybrid به Fast Brain eager و Smart Brain lazy تفکیک شد.
- Transformer اختصاصی Nano از random initialization آموزش داده و وزن آماده داخل Release قرار گرفت؛ هیچ pretrained source یا API مدل استفاده نشد.
- Tokenizer نسل v003 برای فارسی/انگلیسی/Path/URL/JSON/code ساخته و benchmark شد.
- Dataset v003 با metadata، dedup و concept-group split ساخته شد.
- NLU ساختاری، Persian number parser، typo correction محافظه‌کارانه و File workflow parser اضافه شد.
- Planner از reference پویا به خروجی مرحلهٔ قبلی پشتیبانی می‌کند.
- Verifier، Recovery، Dry Run، Undo و Permission Levelهای L0 تا L4 یکپارچه شدند.
- Memory به Working/Episodic/Semantic و Action history با consolidation/conflict/forgetting توسعه یافت.
- RAG به BM25 + hashed vector + reranking ارتقا یافت.
- Skill/Plugin manifest registry و Diagnostics read-only اضافه شد.
- GUI اصلی RTL-aware، action/status viewer، Smart lazy label و worker-thread دارد.
- JARVIS LAB دارای Dataset catalog، duplicate/malformed validation، enable/disable/import/delete محدود، Model import/activate/delete guard، evaluation و benchmark است.
- INT8 loader v2 checksum تک‌پارامتر و aggregate دارد و وزن matrix را واقعاً INT8 در RAM نگه می‌دارد.

## باگ‌های حل‌شده

- «داخل درایو D یک فایل به نام تست بساز» دیگر به اشتباه `open_folder` نمی‌شود و `D:\تست.txt` می‌سازد/Preview می‌کند.
- «همونجا فایل notes بساز» از Context فولدر قبلی استفاده می‌کند.
- سؤال آموزشی حذف فایل به‌اشتباه Tool خطرناک اجرا نمی‌کند، حتی با ZWNJ یا شمارهٔ مثال بعد سؤال.
- عبارت «درایو C رو پاک کن» به safety gate می‌رود.
- typoهایی مثل «با زکن»، «کرمو» و «انتغال» با edit-distance محدود فهمیده می‌شوند.
- Case نام Entity انگلیسی مثل `Atlas_0001` هنگام typo normalization حفظ می‌شود.
- پسوند ضمیری «ببندش» حفظ شد و reference به آخرین App از بین نمی‌رود.
- ساخت Folder پسوند `.txt` نمی‌گیرد و عبارت «موسوم به» استخراج می‌شود.
- Calculator دیگر شمارهٔ نسخه یا Path را به‌اشتباه expression نمی‌گیرد.
- Persian word-number برای volume/brightness پشتیبانی شد.
- GUI/LAB undefined profile variable و resume path قدیمی اصلاح شد.
- full-model FP32 dequantization هنگام INT8 runtime حذف شد.

## مدل پایهٔ v0.7 و Tokenizer (اطلاعات تاریخی)

| مورد | مقدار واقعی |
|---|---:|
| Nano parameters | 23,077,376 |
| Architecture | Decoder-only, GQA, RoPE, SwiGLU, RMSNorm |
| Layers / width / FFN | 8 / 512 / 1,280 |
| Context / vocab | 2,048 / 4,096 |
| Weight file | 15,201,068 byte |
| Resident INT8 weights | 23,086,308 byte / 22.017MB |
| Training steps/time | 672 / 176.126s CPU |
| Initial/final-stage val loss | 8.190477 / 3.634206 |
| Pretrained source | ندارد |

Core: 82,201,344 و Pro: 252,228,608 پارامتر، هر دو config-only و بدون وزن bundled.

Tokenizer Candidateها:

- 2,048: tokens/UTF-8-byte `0.355626`
- 4,096: `0.336526`، انتخاب‌شده
- سقف corpus 4,657: `0.335423`
- round-trip `1.0` و OOV `0.0`

## Dataset

- مجموع: 5,529
- Train / Validation / Test: 4,583 / 535 / 411
- زبان: 2,931 فارسی، 1,748 انگلیسی، 850 mixed
- Concept groups: 518 / 63 / 53
- Cross-split concept leakage: 0
- Duplicate حذف‌شده: 146
- Benchmark جدا: 1,000 Prompt، overlap با Train: 0

## Benchmark واقعی

Model teacher-forced روی 411 Test:

- token accuracy: `0.441933`
- cross-entropy: `3.882264`
- perplexity: `48.533978`
- context token accuracy: `0.754167`
- knowledge token accuracy: `0.358447`
- planning token accuracy: `0.414079`

Agent روی suite ساختاری unseen 1,000تایی:

- exact case / intent / entity F1 / tool / argument / plan / safety: `1.0`
- error: `0`
- Execution/Recovery/Conversation quality: `null` چون در محیط Build قابل‌اندازه‌گیری معتبر نبود.

Runtime INT8 روی CPU Build:

- load: `123.476ms`
- first token: `275.121ms`
- generation: `15.716 token/s`
- process RAM delta: `25.992MB`
- VRAM: `0MB`

مقایسهٔ precision نشان داد FP32 در NumPy این CPU سریع‌تر ولی 4 برابر پرحافظه‌تر از INT8 است؛ INT8 برای کمترین RAM انتخاب شد، نه با ادعای سرعت بیشتر.

## Tests

- Baseline 0.7.0: 550 تست
- تست جدید 0.7.1: 32
- تست جدید 0.7.2: 12
- مجموع: 594
- آخرین اجرای source tree: `Ran 594 tests ... OK`
- Package builder دوباره `main.py --self-test` را پس از Extract اجرا می‌کند.

حوزه‌ها: Unit، Integration، Security، scenario regression، model integrity/tamper، KV cache، Dataset split، 1,000 unseen artifact، NLU فارسی، context، Planner، Dry Run، Undo، Memory migration/reopen، RAG، TXT/ZIP و LAB catalog.

Grounded evaluation جداگانه: `50/50`، شامل 8 Text-QA، 7 deduction، 24 Knowledge، 6 unsupported-claim guard و 5 search fixture. این مجموعه curated است و معیار وب آزاد نیست.

## Known limitations

- Nano 23M با Dataset 5.5K هنوز کیفیت مدل مکالمهٔ آزاد بزرگ را ندارد. Quality Guard بخشی از generationها را رد می‌کند و پاسخ پایدار اغلب از Knowledge/RAG Ground می‌شود.
- پاسخ TCP/UDP فعلی از سند محلی Knowledge بازیابی می‌شود؛ مستقل از اینترنت است ولی نمونهٔ تولید دانش بدون retrieval نیست.
- آزمایش Alignment با وجود کاهش loss، به‌خاطر tool-token leakage/repetition Reject شد و وزن آن Release نشد.
- Core/Pro وزن آماده ندارند.
- اجرای واقعی Windows Desktop/Power و visual GUI smoke در محیط Linux Build ممکن نبود؛ هیچ success شبیه‌سازی نشد.
- Benchmark 1,000تایی محدود و ساختاری است؛ Score کامل آن ادعای هوش عمومی یا کیفیت مکالمه نیست.
- اعداد Performance روی Python 3.12 محیط Build اندازه‌گیری شدند؛ سازگاری 3.13 در کد/dependency هدف‌گذاری شده ولی همین میزبان 3.13 نداشت.
- Live Internet در محیط Build به‌علت محدودیت outbound اجرا نشد؛ fixture/parser/API-contract تست شد و نتیجهٔ زنده جعل نشده است.
- Search هیچ‌گاه صحت مطلق اینترنت را تضمین نمی‌کند؛ هدف نسخه citation، چند منبع، confidence و خودداری از پاسخ بی‌مدرک است.
