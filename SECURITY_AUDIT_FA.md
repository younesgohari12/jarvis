# SECURITY_AUDIT_FA — JARVIS v22.4.2

> طبقه‌بندی شواهد: **[اندازه‌گیری]** = اجرا در همین محیط · **[سورس]** = تأیید از بازرسی کد · **[تاریخی]** = آرتیفکت ریلیز قبل
> دامنه: ممیزی متمرکز مرزهای اعتماد مرتبط با باگ‌های رفع‌شده (§8) — بدون بازنویسی گسترده.

## 1. زنجیره‌ی مجوز ابزارها (§8.1) — [سورس] با استخراج برنامه‌ای از `jarvis/runtime/bootstrap.py`

۱۱۰ ابزار ثبت‌شده. نگاشت دسته→سطح در `jarvis/agent/permissions.py`:
`filesystem_read/information/network → READ(L0)` · `external_app/caution → SAFE(L1)` · `write_file/rename/move/sensitive_action/ui_control/browser_interaction/clipboard_access/process_control → MODIFY(L2)` · `destructive/delete/system_control/process_terminate/clipboard_destructive → DESTRUCTIVE(L3)` · `power_control/shell/privileged → CRITICAL(L4)`. هر سطح بالاتر از L1 نیازمند تأیید صریح است.

| خانواده (تعداد) | ابزارهای نمونه | دسته | سطح مؤثر | تأیید لازم | ارزیابی |
|---|---|---|---|---|---|
| خواندن فایل/سیستم (9) | list_folder، file_info | filesystem_read | L0 | خیر | سازگار |
| اطلاعات (24) | system_info، date_time | information | L0 | خیر | سازگار |
| شبکه (9) | web_search، fetch_text | network | L0 | خیر | سازگار (خواندنی) |
| بازکردن اپ/مرورگر/پوشه (22) | open_app، open_url | external_app | L1 | خیر | طراحی مستند v2x |
| کنترل UI (8) | type_text، click، hotkey | ui_control | L2 | **بله** | سازگار — ورودی کیبورد/موس نیازمند تأیید است |
| نوشتن فایل (7) | write_file، append، truncate، create | write_file | L2 | **بله** | سازگار |
| جابجایی/نام‌گذاری (2) | move_file، rename_file | move/rename | L2 | **بله** | سازگار |
| پایان پروسه (7) | close_app، close_browser، restart_app | process_terminate | L3 | **بله** | سازگار — «terminate» درست طبقه‌بندی شده |
| حذف (2) | delete_file، delete_folder | delete + risk=dangerous | L3 | **بله** | سازگار |
|clipboard (2)|read→L2 / clear→L3|clipboard_*|L2/L3|بله|سازگار|
| برق/سیستم (4) | shutdown/restart/sleep/logoff | power_control + dangerous | L4 | **بله** | سازگار |
| فرمان (1) | run_command | caution + **resolver پویا** | پویا: safe→L1 … dangerous→L4 | بله برای هر چیز غیر read_only | **BUG-001 همین‌جا بود — رفع شد** |

**کف ریسک اعلان‌شده** (`ToolSpec.effective_permission_category`): اگر resolver پویا ریسک را پایین بیاورد، `risk="confirm"` کف MODIFY و `risk="dangerous"` کف DESTRUCTIVE اعمال می‌کند. نتیجه‌ی پویای بالاتر هرگز پایین نمی‌آید. [سورس + تست `TestPermissionFloor` — اندازه‌گیری]

**جمع‌بندی §8.1:** پس از رفع BUG-001، هیچ موردی از «اثر واقعی > طبقه‌بندی مجوز» یافت نشد. دو resolver پویا موجود است: `run_command` (پس از رفع: هر عملیات تغییردهنده route → shell/L4) و `browser_click` (کف L2 + تحلیل پویا).

## 2. امنیت فرمان (§8.2) — `jarvis/tools/terminal.py` [سورس]

- `shell=False` همیشه؛ **تضعیف نشد** (خط 216 نسخه 22.4.2).
- `META` رجکس: `&&`, `||`, `|`, `;`, `&`, `>`, `<`, backtick, `$(`, CR, LF → **blocked** (pipe/redirection/chaining/newline injection بسته است).
- Allowlist مطلق اجرایی‌ها؛ خارج از allowlist → blocked. Git: مسدودسازی `-c`/`--exec-path`/`alias.`؛ Python: فقط `--version/-V/-VV` امن.
- Redaction: `CommandPolicy.SECRET` هر `token/password/secret/api_key/authorization=…` را در stdout/stderr/دستور بازتابی می‌پوشاند.
- Timeout 1..30s + cancel_event + kill/terminate با مدیریت خروجی؛ cwd اعتبارسنجی‌شده. env فقط کپی محیط (بدون تزریق).
- **جدید (BUG-001):** گرامر صریح route با fail-closed برای عملیات نامعلوم/غایب.
- نکته‌ی مستند: پس از رفع، `route nonsense` از dangerous→blocked تغییر کرد (سخت‌گیرانه‌تر؛ §5.4 مجاز می‌داند).

## 3. امنیت فایل‌سیستم (§8.3) — `jarvis/tools/files.py` [سورس]

- Path traversal: همه‌ی مسیرها `expanduser().resolve()`؛ عملیات روی مسیر واقعی.
- حذف: `_deletion_target` ریشه‌ی درایو (anchor)، Home، و در ویندوز `WINDIR/ProgramFiles(x86)/ProgramData` را محافظت می‌کند؛ آگاه از symlink (برای لینک از absolute استفاده می‌شود تا مقصد خارج از محدوده باز نشود).
- ZIP: فقط پیش‌نمایش/خواندن متن عضو؛ **بدون extract**؛ سقف `max_zip_entries / max_zip_member_bytes / max_zip_total_preview_bytes` (zip-bomb و عضو بزرگ محدود)؛ تست پوشش‌شده در `tests/test_files.py`.
- نوشتن/الحاق: سقف `max_file_bytes` و بازبینی سرریز الحاق.
- تخریب‌گرها (`rm/del/format/diskpart/…`) در `BLOCKED_EXECUTABLES` policy فرمان‌ها.
- آزمون تخریبی روی داده‌ی واقعی انجام نشد؛ تست‌ها tmp-only.

## 4. الگوهای اسکنر اسرار (§21، پس از BUG-002) [سورس + اندازه‌گیری]

| الگو | TP (فیک) | FP (منفی) | وضعیت تست |
|---|---|---|---|
| github_pat / classic / oauth / fine-grained(ghu,ghs,ghr) | ✓ | ✓ | در ماژول تست |
| api_key_generic (پیشونددار، JSON/.env) | 11 مثبت | 10 منفی | M9 kill |
| access/secret/bot token (نام‌محور) | ✓ | placeholder filtered | M10 kill |
| aws / private key / slack / google | شکل‌محور حفظ شد | — | رگرسیون |
| telegram bot shape + hardcoded_password (پیشونددار) | ✓ | ✓ | رگرسیون |

قانون صحرایی: **یافته فقط (file, secret_type) است؛ مقدار هرگز ثبت/چاپ نمی‌شود** — تست رگرسیون دارد.

## 5. سیاست حذف بسته (§22) [اندازه‌گیری روی ZIP نهایی]

فهرست حذف (`EXCLUDED_DIRS/SUFFIXES/NAMES`) روی محتوای واقعی ZIP v22.4.2 اعتبارسنجی شد: `forbidden_cache_files = 0`، `missing=0، extra=0، mismatch=0`، تک‌ریشه `Jarvis_v0.11.0/`، CRC پاک، مانیفست = محتوا minus خود مانیفست، بازسازی دوباره بایت‌یکسان (reproducible). جداکننده‌ی مسیر: پکیجدر POSIX rel-path مقایسه می‌کند و موارد تودرتوی cache (`__pycache__` در هر عمق) را پوشش می‌دهد.

## 6. اسکن اسرار — نتایج [اندازه‌گیری]

- درخت منبع v22.4.2: 1783 فایل، **0 secret** (الگوهای گسترش‌یافته — 16 الگو).
- ZIP نهایی: اسکن همه‌ی اعضای متنی — **0 secret** (`reports/v22_4_2/security_package.json`).
- فیکسچرهای تست با مونتاژ قطعه‌قطعه نوشته شدند تا منبعِ تست خودش trip نشود؛ این تکنیک در سند تست مستند است و اسکنر روی محتوای مونتاژشده مثبت می‌گیرد (تست‌شده).

## 7. موارد گزارش‌شده ولی خارج از محدوده‌ی این پچ

- طبقه‌بند فریمِ آموخته‌شده (learned) برای کلمات کاملاً خارج از واژگان شناخته‌شده — fail-closed به abstain (سالم)؛ گسترش نیازمند بازآموزی مدل: `V23_LANGUAGE_BRAIN_PLAN.md`.
- GUI/حافظه/نصب ویندوز: مطابق §19/§18/§20 بازرسی سطحی در همین اسکن (بدون نقص تکرارشونده‌ی جدید در محیط لینوکس؛ تست‌های GUI-headless اینجا اجرا نشدند — BLOCKED برای اجرای dynamic، پوشش از طریق مجموعه تست موجود).
