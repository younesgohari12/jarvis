# BUG_FIX_REPORT_FA — JARVIS v22.4.2

> طبقه‌بندی شواهد (§42):
> **«اندازه‌گیری شده اکنون»** = در همین محیط اجرا شده · **«تأیید از سورس»** = بازرسی کد بدون اجرا · **«ادعای تاریخی ریلیز»** = آرتیفکت‌های نسخه قبل
> تاریخ: 2026-09-20 · محیط: Linux x86_64، Python 3.12.14 (هدف پروژه: Windows + CPython 3.13؛ تفاوت محیط صادقانه ثبت شده است)

خلاصه اجرایی: هر سه باگ تأییدشده‌ی نسخه v22.4.1 (BUG-001/002/003) بازتولید، ریشه‌یابی، رفع، پوشش تست رگرسیون و تست جهش شدند. یک نقص کلاس واژگانی در موتور work-rate نیز (WEAK-W1) در چهار لایه‌ی regex اصلاح و اندازه‌گیری شد. هیچ ماژول هوشی بازنویسی، هیچ مدل بازآموزی و هیچ دیتاست فریز دستکاری نشده است.

---

## BUG-001

Severity: **HIGH** (امنیتِ مجوز — دور زدن تأییدیه برای تغییر جدول مسیریابی ویندوز)
Status: **FIXED — VERIFIED (اندازه‌گیری‌شده: بازتولید قبل از رفع، رفع، رگرسیون، تست جهش)**

Affected files:
- `jarvis/tools/terminal.py` (تابع `CommandPolicy.assess` — خط 107 نسخه v22.4.1)

Problem:
سیاست فرمان‌ها برای `route` فقط توکن دوم را می‌سنجید و `-4`/`-6` (سوییچ‌های سراسری خانواده آدرس) را هم‌ارز `print` می‌گرفت. در نتیجه هر عملیات تغییردهنده‌ای که بعد از سوییچ بیاید از فیلتر عبور می‌کرد.

Reproduction (اندازه‌گیری شده اکنون، قبل از رفع — متن کامل در `evidence` ریلیز):
```text
route -4 add 10.0.0.0 mask 255.0.0.0 1.2.3.4  -> safe (read_only)   ← اشتباه
route -6 add 10.0.0.0 mask 255.0.0.0 1.2.3.4  -> safe (read_only)   ← اشتباه
route -4 delete 10.0.0.0                      -> safe (read_only)   ← اشتباه
route -6 delete 10.0.0.0                      -> safe (read_only)   ← اشتباه
route -4 change 10.0.0.0 mask 255.0.0.0 1.2.3.4 -> safe (read_only) ← اشتباه
CommandRunner.risk_category() آن‌ها -> read_only → PermissionLayer بدون تأیید اجازه می‌داد
```

Actual result: طبقه‌بندی `safe/read_only` برای ۵ فرمان تغییردهنده (تأییدشده با اجرای واقعی policy).
Expected result: `dangerous/network_change_confirmation_required` برای هر عملیات تغییردهنده.

Root cause:
پارس «موقعیتی» به‌جای «گرامر معنایی» فرمان: سوییچ سراسری با «عملیات» اشتباه گرفته شده بود. زنجیره‌ی عبور: planner → run_command → CommandPolicy.assess → طبقه‌بندی نادرست safe → risk_category()=read_only → PermissionLayer.check مجاز → CommandRunner.run → subprocess.Popen (تأیید از سورس برای زنجیره، بازتولید واقعی برای طبقه‌بندی).

Technical fix:
پیاده‌سازی پارسر صریح گرامر Route در `CommandPolicy._assess_route`:
- جداسازی سوییچ‌های سراسری (`ROUTE_GLOBAL_FLAGS = {-4, -6}`) از «عملیات»؛
- `print` (با هر تعداد سوییچ) → `safe/read_only`؛
- `add|delete|change` → `dangerous/network_change_confirmation_required`؛
- عملیات نامعلوم (`route nonsense`، `route -f ...`) → `blocked/route_unknown_operation_blocked`؛
- عملیات غایب (`route`، `route -4`) → `blocked/route_operation_unverified`.
جهت همه‌ی موارد مبهم «بسته‌شدن به خطا» (fail-closed) است؛ هیچ حالت جدیدی safe پیش‌فرض نمی‌شود.

Why this is root-cause and not a patch:
رفع، «گرامر واقعی فرمان route» را مدل کرده است (سوییچ‌های سراسری ≠ عملیات) نه اینکه یک عبارت خاص را به لیست سیاه اضافه کند. هیچ عبارت بنچمارک یا جمله‌ی آزمون در کد hardcode نشده و رفتار برای همه‌ی آرگومان‌های بعد از عملیات یکسان است.

Tests added (همه در `tests/test_v2242_security_and_rootcause.py` — اندازه‌گیری شده اکنون، 65 تست):
- `TestRouteCommandPolicy` — ماتریس کامل §5.5 (4 مورد print، 9 مورد add/delete/change با و بدون -4/-6، 7 مورد fail-closed)، نگاشت risk_category، تست یکپارچگی Registry+PermissionLayer با Mock (`subprocess.Popen` هرگز بدون confirmed=True فراخوانی نمی‌شود) و تست جریان تأیید.
- `TestPermissionFloor` — کف ریسک اعلان‌شده نمی‌تواند ریسک پویا را پایین بیاورد؛ دسته‌ی shell/نامعلوم بدون تأیید مجاز نیست.

Regression result:
`1292 passed, 1 skipped (محیطی: نبود ZIP در درخت), 585 subtests passed, 0 failures` — اندازه‌گیری شده اکنون؛ خط مبنا 1215 (v22.4) و 1228 (v22.4.1) حفظ شد.

Mutation result:
`M8_route_readonly_set_overextended` (بازگرداندن عمدی رفتار آسیب‌پذیر با اضافه‌کردن add/delete/change به عملیات‌های read-only) → `KILLED_BY_ASSERTION` با معیار معتبر (applied ✓, collected≥1 ✓, failed≥1 ✓, errors==0 ✓, source restored ✓).

Remaining risk:
پارسر فقط `-4`/`-6` را سوییچ سراسری می‌شناسد؛ سایر شکل‌های غیرمتعارف (مثل `/p`) به‌عنوان عملیات نامعلوم **مسدود** می‌شوند (جهت ایمن). اجرای واقعی `route` روی ماشین آزمایش انجام نشد (§5.5: فرمان‌های تغییردهنده‌ی شبکه هرگز اجرا نمی‌شوند) — فقط policy و زنجیره‌ی مجوز تست شد.

---

## BUG-002

Severity: **MEDIUM** (شکاف پوششِ تشخیص در اسکنر اسرار — نه نشانه‌ی نشت واقعی)
Status: **FIXED — VERIFIED (بازتولید + رفع + رگرسیون + تست جهش)**

Affected files:
- `benchmarks/v22_4_1_security.py`

Problem:
الگوی قبلی `\b(api[_-]?key|apikey)\b` به‌دلیل عضویت `_` در کلاس کاراکترهای word در regex، هیچ‌گاه نام‌های پیشونددار مثل `OPENAI_API_KEY` را نمی‌گرفت (مرز کلمه قبل از `API` برقرار نمی‌شود). فرم‌های JSON و `.env` و خانواده‌های ACCESS_TOKEN/SECRET_TOKEN/BOT_TOKEN هم پوشش نداشتند.

Reproduction (اندازه‌گیری شده اکنون، قبل از رفع — با مقادیر فیک):
```text
MISSED: OPENAI_API_KEY = "TESTSECRET..."        MISSED: {"service_api_key": "..."}
MISSED: AVALAI_API_KEY = "TESTSECRET..."        MISSED: SERVICE_API_KEY=...  (.env)
MISSED: MY_SERVICE_ACCESS_TOKEN = "TESTTOKEN..."
MISSED: CUSTOM_SECRET_KEY = "TESTSECRET..."
نتیجه: 6/8 مورد مثبت از دست می‌رفت؛ منفی‌ها تمیز بودند.
```
تذکر صادقانه (مطابق §6.2): این نقص «شکاف پوشش تشخیص» است، نه «نشت شناسایی‌شده‌ی اعتبار»؛ اسکن جدید روی درخت v22.4.1 هم 0 مقدار مخفی یافت.

Root cause:
گرامر نامِ متغیر بیش‌ازحد باریک بود و «تشخیص نام»، «تشخیص مقدار» و «تشخیص placeholder» به‌درستی تفکیک نشده بودند.

Technical fix:
- الگوی اعتبارِ نام‌محور مشترک: `\b((prefix_)*<credential-suffix>)["']?\s*[:=]\s*<quoted | .env-value>` با پنج خانواده‌ی `api_key/apikey`، `access_token`، `secret_key`، `secret_token`، `bot_token`؛
- پشتیبانی فرم پایتون (`X = "..."`)، JSON (`"x": "..."`) و `.env` (`X=...`)؛
- فیلتر placeholder روی مقدارِ گرفته‌شده (فقط برای تصمیم؛ مقدار هرگز ثبت/چاپ نمی‌شود): `YOUR_*`، `CHANGE_ME`، `<API_KEY>`، `${API_KEY}`، `os.getenv/os.environ`، خالی/None، پرکننده‌های تکراری؛
- الگوی شکل‌محور تلگرام و بقیه‌ی الگوها (GitHub PAT/classic/OAuth، AWS، کلید خصوصی، Slack، Google) حفظ شد و `ghu_/ghs_/ghr_` اضافه شد؛
- `hardcoded_password` هم به نام‌های پیشونددار گسترش یافت.

Why this is root-cause and not a patch:
کلاس خطا (نام کاملِ متغیر + سه فرم نگارشی + تفکیک placeholder) به‌صورت عمومی مدل شد؛ افزودن «نام‌های بنچمارک» خاص انجام نشد. یافته‌ها فقط `(file, secret_type)` هستند — تستِ «نبود مقدار در خروجی» رگرسیون شده است.

Tests added:
11 مورد مثبت (همه‌ی خانواده‌ها + JSON + .env + بدون فاصله)، 10 مورد منفی (خالی/None/placeholder/getenv/environ/کوتاه)، ایمنی متادیتا (کلیدهای مجاز findings و نبود مقدار در JSON خروجی)، حفظ الگوهای شکل‌محور.

Regression result: 1292 passed / 0 failed (اندازه‌گیری شده اکنون). اسکن منبع نهایی: 1783 فایل، 0 secret، `passed=true`.

Mutation result:
`M9_scanner_prefix_group_dropped` → KILLED_BY_ASSERTION · `M10_placeholder_filter_disabled` → KILLED_BY_ASSERTION.

Remaining risk:
مجموعه کاراکترهای مقدار (`A-Za-z0-9_-/+=:.` با حداقل 16 کاراکتر) مقدارهایی با جداکننده‌های عجیب را نمی‌گیرد؛ اسکنر باینری‌ها را نمی‌خواند. این محدودیت‌ها مستندند و جهت‌گیری کلی fail-closed است.

---

## BUG-003

Severity: **LOW (affecting release-metadata trust)**
Status: **FIXED — VERIFIED (ریشه در پایپ‌لاین اثبات شد؛ اسناد اصلاح؛ گاره‌ی خودکار اضافه شد)**

Affected files:
- `JARVIS_V22_4_1_RELEASE_INTEGRITY_AUDIT.md` (سه ادعای فعلی)
- `V23_TRAINING_HANDOFF.md` (یک ادعا)
- `benchmarks/v22_4_1_metadata_consistency.py` (گاره‌ی سازگاری — فقط JSON را می‌سنجید)

Problem:
دو سند Markdown عدد «1227 tests» را ادعا می‌کردند در حالی که آرتیفکت ساخت‌یافته‌ی مرجع (`reports/v22_4_1/regression.json`) «1228 tests + 585 subtests, 0 failures» اندازه گرفته بود. خط مبنای تاریخی v22.4 (1215) صحیح و دست‌نخورده است.

Root cause (ردیابی واقعی پایپ‌لاین — §7.2):
چکر سازگاریِ `v22_4_1_metadata_consistency.py` همه‌ی **JSON**ها را با هم می‌سنجد اما **Markdownها هرگز اعتبارسنجی نمی‌شوند**؛ اسناد دست‌نویس هستند. دنباله‌ی وقوع: اجرای قبلی pytest → 1227 در Markdown ثبت شد → یک تست دیگر اضافه شد → regression.json بازتولید شد (1228) → Markdown قدیمی ماند. تفسیر §7.2 تأیید شد؛ علاوه بر آن، شکاف ساختاری «نبود گاره برای Markdown» به‌عنوان ریشه‌ی قابل‌تکرار بسته شد.

Technical fix:
1) اصلاح ادعاهای فعلی در دو سند: 1227→1228 و «افزایش +12»→«+13»؛ اعداد تاریخی (1215، 1042) دست نخوردند.
2) افزودن `check_markdown_consistency()` به چکر سازگاری: هر ادعای فعلی «N tests [+ M subtests]» و ستون «جاری» جدول تست‌ها باید با regression.json برابر باشد؛ فقط اعداد تاریخیِ صریح (1042، 1215) مجازند؛ در صورت مغایرت، خروجی `all_consistent=false` و کد خروج 1 (fail-closed). از این پس Markdownها منبع مستقل ندارند — «یک منبع ساخت‌یافته‌ی مرجع» (§7.3).

Tests added:
`TestMarkdownMetadataGate` — تطابق اسناد واقعی با regression.json، رد ادعای کهنه (1227)، مجازبودن تاریخی‌ها، و کنترل ستون جدول.

Regression result: بخشی از 1292/0 — اندازه‌گیری شده اکنون.
Mutation result: `M11_markdown_gate_forced_green` → KILLED_BY_ASSERTION.
Remaining risk: اسناد Markdownی که در `MARKDOWN_DOCS` ثبت نشده باشند پوشش داده نمی‌شوند؛ افزودن اسناد آینده به همین ثابت باید جزو چک‌لیست ریلیز v23 باشد.

---

## WEAK-W1 (نقص هوشی مکمل — کلاس واژگانی work-rate، خارج از سه باگ اصلی)

Severity: LOW (کیفیت هوشی؛ نقص امنیتی نیست — برخلاف §30 بزرگ‌نمایی نشد)
Status: **FIXED (محدود و اندازه‌گیری‌شده) — بخش ساختاری به V23 ارجاع شد**

Affected files (چهار لایه‌ی regex، هم‌کلاس):
- `jarvis/agent/source_semantics_v20_1.py` — W1a: گسترش اسم‌های خروجی (widgets، parts، cakes، …) · W1b: گسترش اسم‌های کارگر (machines، robots، crew، developers، …)
- `jarvis/agent/numeric_roles_v22.py` — W1c: هم‌سطح‌سازی `_WORKERS_PAT`/`_OUTPUT_PAT`/الگوی immediate برای ممیزی نقش‌ها
- `jarvis/agent/source_facts_v21.py` — W1d: گیت ورود به مسیر work-rate برای کلاس اسم‌های کارگر

Problem (کشف با جمله‌های نو — نه ردیف‌های بنچمارک):
«If 5 workers build 90 widgets in 3 hours…» با وجود پاسخ‌پذیری ساختاری، رد می‌شد (`source_semantic_coverage`) زیرا کلاس اسم خروجی در extractor فقط units/pieces/boxes/items را می‌شناخت. «machines/robots» هم به‌عنوان عامل کار شناسایی نمی‌شدند.

Fix و چرا root-cause است:
گسترش «کلاس اسم‌های شمارش‌پذیر محصول» و «کلاس اسم‌های عامل کار» به‌صورت یکنواخت در هر چهار لایه — نه حفظِ جمله‌ی شکست‌خورده. جمله‌های قبلاً قابل‌تأیید تغییر رفتار ندادند (تست‌های canonical قبلی سبز ماندند).

Tests added:
`test_output_noun_generalization_widgets` (120)، `test_worker_noun_generalization_machines_and_robots` (231/180)، `test_slot_binding_is_positional_not_first_two_numbers` (252 — اثبات اتصال صحیح اسلات‌ها §10.3)، تست‌های `TestSourceSlotConsistency` (برچسب‌گذاری initial/target، تشخیص swap با `role_slot_consistency`).

Measured impact (مجموعه تشخیصی فریز §26 — اجرای یک‌بار پس از W1a و یک‌بار پس از W1b/c/d):
- work_rate: 6/20 → 7/20 (ردیف machines)؛ خارج از مجموعه، ردیف‌های robots/frames و machines/parts و widgets/unit همه verified شدند (تست‌های بالا).
- کل تشخیصی: 31.78→32.71 numeric · 24.74→25.77 answerable-only (تفسیر صادقانه: بهبود محدودِ کلاس واژگانی؛ خانواده‌های ساختاری — نرخ ترکیبی، سؤال معکوس ساعت، نرخ هر-کارگر، مانده‌ی کار — باز هستند و در `V23_LANGUAGE_BRAIN_PLAN.md` برنامه دارند).

Remaining risk: طبقه‌بند فریمِ آموخته‌شده و لایه‌های neural دست نخوردند (§2.3/§40 — بازآموزی حواله به V23 شد). این یعنی جمله‌هایی که هیچ کلمه‌ی آشنا برای طبقه‌بند نداشته باشند همچنان abstain می‌کنند (رفتار fail-closed سالم، نه جواب غلط).
