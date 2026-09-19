# CHANGESET — JARVIS v22.4.2

> اصل §29: هر تغییر باید به یک باگ/الزام امنیتی/الزام آزمون/الزام ریلیز نگاشت شود. جدول زیر همان نگاشت است. هیچ حذف یا بازنویسی معماری رخ نداده است.

## جدول تغییرات سورس/اسناد

| فایل | چرا تغییر کرد | شناسه | اثر رفتاری |
|---|---|---|---|
| `jarvis/tools/terminal.py` | پارسر گرامر Route (`_assess_route`) به‌جای شرط موقعیتی؛ سوییچ -4/-6 از «عملیات» جدا شد؛ عملیات نامعلوم/غایب → blocked | BUG-001 | `route [-4/-6] add/delete/change` → dangerous (تأیید لازم)؛ `route print*` → safe؛ `route nonsense`/`route`/`route -f` → blocked (قبلاً dangerous/safe) |
| `benchmarks/v22_4_1_security.py` | گرامر نام کامل اعتبار + خانواده‌های ۵گانه + فرم JSON/.env + فیلتر placeholder + الگوی ghu/ghs/ghr + پسورد پیشونددار | BUG-002 | تشخیص OPENAI_API_KEY و …؛ نرفتن placeholderها؛ همچنان 0 FP روی منفی‌ها |
| `JARVIS_V22_4_1_RELEASE_INTEGRITY_AUDIT.md` | اصلاح ادعاهای «جاری» (1227→1228، +12→+13) | BUG-003 | سند با regression.json سازگار |
| `V23_TRAINING_HANDOFF.md` | اصلاح همان ادعا (1227→1228) | BUG-003 | — |
| `benchmarks/v22_4_1_metadata_consistency.py` | افزودن `check_markdown_consistency` (گاره‌ی Markdown) به چکر سازگاری | BUG-003 | از این پس ادعای Markdown بدون تطابق با regression.json ریلیز را fail می‌کند |
| `jarvis/agent/source_semantics_v20_1.py` | W1a/W1b: گسترش کلاس اسم خروجی/کارگر در `source_work_facts` | WEAK-W1 | جمله‌های work-rate با اسم‌های عمومی‌تر verified می‌شوند |
| `jarvis/agent/numeric_roles_v22.py` | W1c: هم‌سطح‌سازی `_WORKERS_PAT`/`_OUTPUT_PAT`/الگوی immediate + پذیرش output_* در `_mark_inventory_roles` | WEAK-W1 (+رفع تداخل ایجادشده توسط W1c در جمله موجودی — با تست `test_roles_inventory` کنترل شد) | ممیزی نقش‌ها روی واژگان گسترده کار می‌کند |
| `jarvis/agent/source_facts_v21.py` | W1d: گیت ورود work-rate به کلاس اسم کارگر گسترش یافت | WEAK-W1 | مسیر استخراج مستقل برای machines/robots/… باز شد |

## فایل‌های جدید (الزام آزمون/ریلیز)

| فایل | نقش |
|---|---|
| `tests/test_v2242_security_and_rootcause.py` | 65 تست رگرسیون مستقل (BUG-001/002/003 + §16.4 + WEAK-W1 + کف مجوز) |
| `benchmarks/v22_4_2_mutation.py` | جهش‌های M8–M11 با معناشناسی valid-kill بازاستفاده‌شده |
| `benchmarks/v22_4_2_release_pipeline.py` | رگرسیون خام → `reports/v22_4_2/regression.json` |
| `benchmarks/v22_4_2_security_scan.py` | اسکن منبع/بسته → `reports/v22_4_2/security_*.json` |
| `benchmarks/v22_4_2_performance_warm.py` | اندازه‌گیری warm/cold → `reports/v22_4_2/performance_warm.json` |
| `benchmarks/v22_4_2_finalize.py` | `RELEASE_V22_4_2.json` فقط از آرتیفکت‌های اندازه‌گیری‌شده |
| `benchmarks/package_v22_4_2.py` | بسته‌بندی قطعی v22.4.2 (بازاستفاده از v22.4.1 با هویت جدید) |
| `benchmarks/v22_4_2_diagnostic_set.jsonl` + `benchmarks/run_v22_4_2_diagnostic.py` | مجموعه تشخیصی فریز §26 + runner |
| `BUG_FIX_REPORT_FA.md`, `SECURITY_AUDIT_FA.md`, `INDEPENDENT_BENCHMARK_REPORT_FA.md`, `V23_LANGUAGE_BRAIN_PLAN.md`, `TEST_RESULTS.json`, `PACKAGE_INTEGRITY.json`, این سند | الزامات §31 |

## دست‌نخورده‌ها (اثبات‌شده)

- `datasets/` و فایل فریز بنچمارک 2075: **0 فایل تغییر** (SHA در RELEASE_V22_4_2.json ثبت شد).
- مدل‌ها/وزن‌ها: بازآموزی یا جایگزینی نشد (frame classifier و neural دست‌نخورده).
- `shell=False`، verifier fail-closed، confirmها، گرامر مالکیت/انبار: بدون تغییر.
- آرتیفکت‌های تاریخی v22.4.1 (مثل `reports/v22_4_1/mutation_tests.json`): در حین اندازه‌گیری مجدد M1–M7 پشتیبان‌گیری و بایت‌یکسان بازیابی شد.

## اعداد نهایی (اندازه‌گیری شده اکنون)

- pytest: **1292 passed, 585 subtests, 0 failures, 1 skipped (محیطی: نبود ZIP هنگام اجرا)** — پیش از بسته‌بندی؛ خود ZIP پس از آن ساخته شد.
- جهش: 11/11 کشته‌شده (7 قدیمی + 4 جدید)، 0 زنده، همه‌ی منابع بازیابی.
- اسکن اسرار: منبع 1783 فایل/0 secret؛ ZIP نهایی/0 secret.
- ZIP: `Jarvis_v0.11.0_Intelligence_v22.4.2_SECURITY_AUDITED_FIXED.zip` — hash نهایی در `PACKAGE_INTEGRITY.json` (آرتیفکت بیرونی، پس از بسته‌بندی نهایی).
