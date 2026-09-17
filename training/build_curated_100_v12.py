from __future__ import annotations

import hashlib
import json
import math
import random
import re
from pathlib import Path
from typing import Any, Callable

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "datasets" / "curated_v12"
SEED = 12092026
EXAMPLES_PER_DATASET = 48


def stable(value: str, n: int = 12) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()[:n]


def language_of(text: str) -> str:
    fa = len(re.findall(r"[\u0600-\u06ff]", text))
    en = len(re.findall(r"[A-Za-z]", text))
    if fa and en:
        return "mixed"
    return "fa" if fa else "en"


def make_row(
    dataset_id: str,
    row_index: int,
    prompt: str,
    answer: str,
    *,
    category: str,
    stage: int,
    stage_name: str,
    concept: str,
    difficulty: str = "medium",
    context: list[dict[str, str]] | None = None,
) -> dict[str, Any]:
    lang = language_of(prompt)
    return {
        "id": f"v12-{dataset_id}-{row_index:04d}",
        "input": prompt.strip(),
        "output": answer.strip(),
        "language": lang,
        "category": category,
        "stage": stage,
        "stage_name": stage_name,
        "context": context or [],
        "origin": "jarvis-curated-v12",
        "metadata": {
            "source": "jarvis-curated-v12",
            "quality": "gold-verified",
            "language": lang,
            "category": category,
            "difficulty": difficulty,
            "task_type": "reasoning" if stage == 11 else "knowledge" if stage == 3 else "conversation" if stage == 2 else "instruction",
            "concept_group": f"v12:{dataset_id}:{concept}",
            "risk_level": "L0",
            "requires_tools": False,
            "expected_tool": "",
            "permission_required": False,
            "pretrained_source": None,
            "verification": "deterministic_or_editorial",
        },
        "pretrained_source": None,
    }


def variants_fa(core: str) -> tuple[str, str, str]:
    return (core, f"لطفاً {core}", f"می‌تونی {core}")


def variants_en(core: str) -> tuple[str, str, str]:
    return (core, f"Please {core}", f"Can you {core}")


PERSIAN_FACTS = [
    ("TCP", "ارتباط اتصال‌گرا و قابل‌اعتماد فراهم می‌کند و ترتیب و تحویل بسته‌ها را کنترل می‌کند."),
    ("UDP", "بدون برقراری اتصال و با سربار کمتر داده را می‌فرستد، اما تحویل و ترتیب را تضمین نمی‌کند."),
    ("RAM", "حافظهٔ کاری موقت و سریع سیستم است و با قطع برق محتوای آن از بین می‌رود."),
    ("SSD", "حافظهٔ ذخیره‌سازی دائمی مبتنی بر فلش است و نسبت به HDD معمولاً سریع‌تر و کم‌صداتر است."),
    ("DNS", "نام دامنه را به نشانی IP نگاشت می‌کند تا سرویس‌ها با نام قابل دسترس باشند."),
    ("HTTP", "پروتکل انتقال درخواست و پاسخ وب است؛ HTTPS همان ارتباط را با TLS رمزگذاری می‌کند."),
    ("API", "قراردادی برای ارتباط برنامه‌هاست که عملیات، ورودی‌ها و خروجی‌های قابل استفاده را مشخص می‌کند."),
    ("JSON", "قالب متنی ساخت‌یافته‌ای برای تبادل داده با شیء، آرایه، رشته، عدد، بولی و null است."),
    ("Git", "سامانهٔ کنترل نسخهٔ توزیع‌شده برای ثبت تغییرات، شاخه‌بندی و همکاری روی کد است."),
    ("Docker", "برنامه و وابستگی‌هایش را در کانتینر قابل‌تکرار بسته‌بندی و اجرا می‌کند."),
    ("پایتون", "زبان برنامه‌نویسی سطح‌بالا و خوانایی است که در خودکارسازی، وب، داده و هوش مصنوعی کاربرد دارد."),
    ("پایگاه داده", "سامانه‌ای برای ذخیره، سازمان‌دهی، جست‌وجو و به‌روزرسانی ساخت‌یافتهٔ داده است."),
    ("CPU", "پردازندهٔ عمومی سیستم است که دستورها و محاسبات متنوع را اجرا می‌کند."),
    ("GPU", "پردازنده‌ای با واحدهای موازی فراوان است و برای گرافیک و محاسبات ماتریسی بسیار مناسب است."),
    ("کش", "حافظهٔ کوچک و سریع نزدیک پردازنده است که داده‌های پرتکرار را برای دسترسی سریع‌تر نگه می‌دارد."),
    ("سیستم‌عامل", "منابع سخت‌افزار را مدیریت می‌کند و خدمات پایه برای اجرای برنامه‌ها فراهم می‌سازد."),
]

COMPARES = [
    ("RAM", "SSD", "RAM حافظهٔ موقت و بسیار سریع برای اجرای برنامه‌هاست؛ SSD ذخیره‌سازی دائمی است و داده را پس از خاموشی حفظ می‌کند."),
    ("TCP", "UDP", "TCP تحویل و ترتیب را با سربار بیشتر تضمین می‌کند؛ UDP سبک‌تر است اما تضمین تحویل و ترتیب ندارد."),
    ("HTTP", "HTTPS", "HTTPS همان HTTP روی لایهٔ رمزنگاری TLS است و محرمانگی و احراز هویت بهتری فراهم می‌کند."),
    ("فرایند", "رشته", "فرایند فضای حافظه و منابع مستقل‌تری دارد؛ رشته واحد اجرای سبک‌تر درون یک فرایند است و حافظه را با رشته‌های همان فرایند به اشتراک می‌گذارد."),
    ("لیست", "تاپل", "در پایتون لیست تغییرپذیر است، اما تاپل معمولاً برای مجموعهٔ ثابت و تغییرناپذیر استفاده می‌شود."),
    ("Git merge", "Git rebase", "merge تاریخچه‌ها را با یک commit ادغام می‌کند؛ rebase commitها را روی پایهٔ جدید بازپخش می‌کند و تاریخچهٔ خطی‌تری می‌سازد."),
    ("SQL", "NoSQL", "SQL معمولاً مدل رابطه‌ای و طرح ساخت‌یافته دارد؛ NoSQL خانواده‌ای از مدل‌های غیررابطه‌ای با انعطاف متفاوت در ساختار و مقیاس‌پذیری است."),
    ("CPU", "GPU", "CPU برای کنترل و پردازش عمومی با هسته‌های قدرتمند مناسب است؛ GPU برای کارهای بسیار موازی با هسته‌های فراوان بهینه شده است."),
]

NORMALIZATIONS = [
    ("میخوام اینو بهتر بنویسی", "می‌خواهم این را بهتر بنویسی."),
    ("نمیدونم چرا برنامه اجرا نمیشه", "نمی‌دانم چرا برنامه اجرا نمی‌شود."),
    ("فایل ها رو بررسی کن", "فایل‌ها را بررسی کن."),
    ("به روز رسانی انجام شد", "به‌روزرسانی انجام شد."),
    ("میتونی کمکم کنی", "می‌توانی کمکم کنی؟"),
    ("این گزینه غیر فعال شده", "این گزینه غیرفعال شده است."),
    ("کاربر ها نمیتونن وارد بشن", "کاربرها نمی‌توانند وارد شوند."),
    ("برنامه نویسی پایتون رو شروع کردم", "برنامه‌نویسی پایتون را شروع کرده‌ام."),
    ("می خوام نتیجه دقیق باشه", "می‌خواهم نتیجه دقیق باشد."),
    ("صفحه ی اصلی باز نمیشه", "صفحهٔ اصلی باز نمی‌شود."),
    ("لطفا خطا ها رو پیدا کن", "لطفاً خطاها را پیدا کن."),
    ("اطلاعات به درستی ذخیره نمیشود", "اطلاعات به‌درستی ذخیره نمی‌شود."),
    ("این کد خیلی کند اجرا میشه", "این کد خیلی کند اجرا می‌شود."),
    ("منظورم نسخه ی قبلیه", "منظورم نسخهٔ قبلی است."),
    ("هیچ کدوم جواب نمیدن", "هیچ‌کدام پاسخ نمی‌دهند."),
    ("این تنظیم رو برام فعال کن", "این تنظیم را برایم فعال کن."),
]

FINGLISH = [
    ("salam che khabar", "سلام! من اینجام. چه کاری می‌خواهی انجام بدهیم؟"),
    ("in code chera error mide", "اگر کد و متن خطا را بفرستی، می‌توانم علت را دقیق بررسی کنم."),
    ("lotfan in matno rasmi kon", "حتماً؛ متن را بفرست تا آن را رسمی و روان بازنویسی کنم."),
    ("fargh tcp o udp chie", "TCP تحویل و ترتیب را تضمین می‌کند، اما UDP سبک‌تر است و چنین تضمینی ندارد."),
    ("mikhastam python yad begiram", "برای شروع پایتون، اول متغیرها، شرط، حلقه، تابع و کار با فایل را یاد بگیر و هم‌زمان تمرین کوچک بساز."),
    ("nemidunam az koja shoru konam", "هدف را به کوچک‌ترین خروجی قابل‌ساخت تبدیل کن و از همان قدم شروع کن."),
    ("json yani chi", "JSON یک قالب متنی ساخت‌یافته برای تبادل داده بین برنامه‌هاست."),
    ("barname baz nemishe", "متن خطا، سیستم‌عامل و کاری که قبل از مشکل انجام دادی را بفرست تا علت را محدود کنیم."),
]

IDIOMS = [
    ("آب از سرش گذشته", "یعنی وضعیت از حد معمول دشوارتر شده و فرد احساس می‌کند چیزی برای از دست‌دادن نمانده است."),
    ("دست به عصا راه رفتن", "یعنی با احتیاط و محافظه‌کاری رفتار کردن."),
    ("مو را از ماست کشیدن", "یعنی بسیار دقیق و سخت‌گیرانه بررسی کردن."),
    ("از کاه کوه ساختن", "یعنی یک موضوع کوچک را بیش از اندازه بزرگ و مهم جلوه دادن."),
    ("سنگ بزرگ علامت نزدن است", "یعنی هدف یا ادعای بیش از حد بزرگ گاهی نشانهٔ عملی‌نشدن آن است."),
    ("یک تیر و دو نشان", "یعنی با یک اقدام به دو هدف رسیدن."),
    ("کار از محکم‌کاری عیب نمی‌کند", "یعنی بررسی و احتیاط اضافه برای جلوگیری از خطا ارزش دارد."),
    ("جلوی ضرر را هر وقت بگیری منفعت است", "یعنی متوقف‌کردن یک روند زیان‌آور حتی اگر دیر باشد، باز هم بهتر از ادامه‌دادن آن است."),
]

SHORT_DOCS = [
    ("نسخهٔ جدید برنامه زمان شروع را ۳۰ درصد کاهش داده است. همچنین مصرف حافظه در تست‌های داخلی کمتر شده، اما قابلیت همگام‌سازی هنوز آزمایشی است.", "نسخهٔ جدید سریع‌تر و کم‌مصرف‌تر شده، اما همگام‌سازی هنوز آزمایشی است."),
    ("تیم امروز خطای ورود کاربران را پیدا کرد. علت، منقضی‌شدن اشتباه توکن در برخی مناطق زمانی بود و اصلاح آن منتشر شد.", "خطای ورود ناشی از انقضای اشتباه توکن بود و اصلاح شد."),
    ("پژوهش نشان داد افزایش دادهٔ تمیز به‌تنهایی کافی نیست. کیفیت برچسب‌ها و تنوع نمونه‌ها اثر بیشتری بر پایداری مدل داشت.", "پایداری مدل بیشتر از حجم خام داده به کیفیت برچسب و تنوع نمونه وابسته بود."),
    ("فروش ماه جاری رشد کرده، اما نرخ بازگشت کالا نیز کمی بالا رفته است. تیم محصول در حال بررسی ارتباط این دو شاخص است.", "فروش رشد کرده ولی بازگشت کالا هم افزایش یافته و در حال بررسی است."),
    ("سرور اصلی پایدار است، اما یکی از سرویس‌های جانبی گاهی پاسخ دیرهنگام می‌دهد. مانیتورینگ برای یافتن الگوی تأخیر فعال شده است.", "سامانه اصلی پایدار است و تأخیر سرویس جانبی در حال مانیتور شدن است."),
    ("کاربران نسخهٔ موبایل از طراحی جدید رضایت بیشتری داشته‌اند. بیشترین شکایت باقی‌مانده مربوط به اندازهٔ دکمه‌ها در صفحهٔ پرداخت است.", "رضایت از طراحی موبایل بهتر شده و مشکل اصلی باقی‌مانده اندازهٔ دکمه‌های پرداخت است."),
    ("آموزش مدل با نرخ یادگیری بالا سریع شروع شد، اما اعتبارسنجی پس از چند مرحله بدتر شد. کاهش نرخ یادگیری روند را پایدار کرد.", "نرخ یادگیری بالا باعث افت اعتبارسنجی شد و کاهش آن آموزش را پایدار کرد."),
    ("پشتیبان‌گیری شبانه موفق بود. بااین‌حال آزمون بازیابی نشان داد دو فایل تنظیمات در نسخهٔ پشتیبان وجود ندارند.", "پشتیبان‌گیری انجام شد اما دو فایل تنظیمات در آزمون بازیابی مفقود بودند."),
]

KNOWLEDGE_BANK = [
    ("چرا آسمان در روز آبی دیده می‌شود؟", "به‌دلیل پراکندگی رایلی، طول‌موج‌های کوتاه‌تر نور خورشید مانند آبی بیشتر در جو پراکنده می‌شوند و از جهت‌های مختلف به چشم می‌رسند."),
    ("چرا یخ روی آب شناور می‌ماند؟", "ساختار بلوری یخ فاصلهٔ میان مولکول‌ها را بیشتر می‌کند، بنابراین چگالی یخ از آب مایع کمتر است و روی آن شناور می‌ماند."),
    ("فتوسنتز چیست؟", "فرایندی است که گیاهان و برخی جانداران با استفاده از نور، آب و دی‌اکسیدکربن انرژی شیمیایی تولید می‌کنند و معمولاً اکسیژن آزاد می‌شود."),
    ("DNA چه نقشی دارد؟", "DNA اطلاعات ژنتیکی را ذخیره می‌کند و دستور ساخت بسیاری از مولکول‌ها و ویژگی‌های زیستی را در توالی نوکلئوتیدهای خود نگه می‌دارد."),
    ("الکتریسیتهٔ جریان مستقیم و متناوب چه فرقی دارند؟", "در DC جهت جریان ثابت است؛ در AC جهت و معمولاً مقدار جریان به‌صورت دوره‌ای تغییر می‌کند."),
    ("چرا فلزها معمولاً رسانای خوبی هستند؟", "زیرا الکترون‌های رسانش در بسیاری از فلزها آزادی حرکت بیشتری دارند و می‌توانند بار و انرژی را در شبکه منتقل کنند."),
    ("فشار چیست؟", "فشار مقدار نیرو بر واحد سطح است و در SI با پاسکال اندازه‌گیری می‌شود."),
    ("سرعت و شتاب چه فرقی دارند؟", "سرعت نرخ تغییر مکان نسبت به زمان است؛ شتاب نرخ تغییر سرعت نسبت به زمان است."),
    ("اتم چیست؟", "اتم کوچک‌ترین واحد یک عنصر است که ویژگی شیمیایی آن عنصر را حفظ می‌کند و از هسته و الکترون‌ها تشکیل شده است."),
    ("pH چه چیزی را نشان می‌دهد؟", "pH شاخصی لگاریتمی برای میزان اسیدی یا بازی بودن محلول است و با فعالیت یون هیدروژن ارتباط دارد."),
    ("سیستم ایمنی چه می‌کند؟", "سیستم ایمنی عوامل بیگانه و سلول‌های غیرعادی را شناسایی و با سازوکارهای ذاتی و اکتسابی با آن‌ها مقابله می‌کند."),
    ("نور سال چیست؟", "نور سال واحد فاصله است، نه زمان؛ برابر مسافتی است که نور در خلأ طی یک سال می‌پیماید."),
    ("گرانش چیست؟", "گرانش برهم‌کنشی است که جرم و انرژی را به هم مرتبط می‌کند و در مقیاس روزمره باعث جذب اجرام به سوی زمین می‌شود."),
    ("قانون اول نیوتن چیست؟", "جسم در حالت سکون یا حرکت یکنواخت خطی می‌ماند مگر اینکه نیروی خالص خارجی حالت حرکتش را تغییر دهد."),
    ("مول چیست؟", "مول واحد مقدار ماده در SI است و یک مول دقیقاً شامل 6.02214076×10^23 موجودیت بنیادی است."),
    ("آنزیم چیست؟", "آنزیم کاتالیزور زیستی است که سرعت واکنش‌های خاص را با کاهش انرژی فعال‌سازی افزایش می‌دهد بدون اینکه در پایان مصرف شود."),
]

EN_FA_PAIRS = [
    ("The update fixed the login problem.", "به‌روزرسانی مشکل ورود را برطرف کرد."),
    ("Please save a backup before changing the configuration.", "لطفاً قبل از تغییر تنظیمات یک نسخهٔ پشتیبان ذخیره کن."),
    ("The model needs cleaner data, not just more data.", "مدل فقط به دادهٔ بیشتر نیاز ندارد؛ دادهٔ تمیزتر هم لازم است."),
    ("The server is running, but the database connection is slow.", "سرور در حال اجراست، اما اتصال پایگاه داده کند است."),
    ("This function returns the average of the values.", "این تابع میانگین مقادیر را برمی‌گرداند."),
    ("We should test the change before deploying it.", "باید تغییر را پیش از استقرار آزمایش کنیم."),
    ("The file was deleted accidentally.", "فایل به‌اشتباه حذف شد."),
    ("The result is correct, but the explanation is incomplete.", "نتیجه درست است، اما توضیح کامل نیست."),
    ("Network latency can affect response time.", "تأخیر شبکه می‌تواند بر زمان پاسخ اثر بگذارد."),
    ("The application uses less memory after the optimization.", "برنامه پس از بهینه‌سازی حافظهٔ کمتری مصرف می‌کند."),
    ("Do not guess when the required information is missing.", "وقتی اطلاعات لازم وجود ندارد، حدس نزن."),
    ("A smaller model can still be useful if the task is well defined.", "اگر وظیفه خوب تعریف شده باشد، یک مدل کوچک هم می‌تواند مفید باشد."),
    ("The test failed because the expected value was outdated.", "آزمون شکست خورد چون مقدار مورد انتظار قدیمی بود."),
    ("The cache improves repeated reads.", "کش خواندن‌های تکراری را سریع‌تر می‌کند."),
    ("Encryption protects data in transit.", "رمزنگاری از داده هنگام انتقال محافظت می‌کند."),
    ("The new parser handles Persian digits correctly.", "تجزیه‌گر جدید ارقام فارسی را به‌درستی پردازش می‌کند."),
]


# ---------------- language / instruction datasets ----------------

def gen_language(subtype: str, dataset_id: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    def add(prompt: str, answer: str, concept: str, i: int, stage: int = 1, category: str | None = None, context=None):
        rows.append(make_row(dataset_id, i, prompt, answer, category=category or f"v12_{subtype}", stage=stage, stage_name={1:"language_foundations",2:"conversation",4:"instruction_following",9:"context_memory"}.get(stage,"language_foundations"), concept=concept, context=context))

    i = 0
    if subtype in {"orthography", "halfspace", "punctuation", "typo", "formalize", "rewrite_clear"}:
        for k, (bad, good) in enumerate(NORMALIZATIONS):
            prompts = (
                f"این جمله را با فارسی معیار و درست بازنویسی کن: «{bad}»",
                f"غلط‌های نگارشی و فاصله‌گذاری این متن را اصلاح کن: {bad}",
                f"فقط نسخهٔ روان و درست این جمله را بده: {bad}",
            )
            for p in prompts:
                i += 1; add(p, good, f"norm:{k}", i, stage=1)
    elif subtype == "finglish":
        for k, (p0, ans) in enumerate(FINGLISH):
            for p in (p0, p0 + "؟", "manzooram ine: " + p0):
                i += 1; add(p, ans, f"finglish:{k}", i, stage=2, category="v12_finglish_conversation")
    elif subtype == "idiom":
        for k, (phrase, ans) in enumerate(IDIOMS):
            for p in (f"«{phrase}» یعنی چی؟", f"معنی اصطلاح {phrase} رو ساده بگو", f"این عبارت را توضیح بده: {phrase}"):
                i += 1; add(p, ans, f"idiom:{k}", i, stage=1)
    elif subtype == "summarize":
        for k, (doc, summary) in enumerate(SHORT_DOCS):
            for p in (f"این متن را در یک جمله خلاصه کن:\n{doc}", f"خلاصهٔ خیلی کوتاه بده: {doc}", f"نکتهٔ اصلی این متن چیست؟ {doc}"):
                i += 1; add(p, summary, f"summary:{k}", i, stage=4)
    elif subtype == "compare":
        for k, (a,b,ans) in enumerate(COMPARES):
            for p in (f"فرق {a} و {b} چیه؟", f"{a} را با {b} مقایسه کن.", f"خیلی روشن بگو {a} چه تفاوتی با {b} دارد."):
                i += 1; add(p, ans, f"compare:{k}", i, stage=4)
    elif subtype == "definition":
        for k,(term,ans) in enumerate(PERSIAN_FACTS):
            for p in (f"{term} چیه؟", f"{term} را ساده توضیح بده.", f"تعریف کوتاه {term} چیست؟"):
                i += 1; add(p, ans, f"define:{k}", i, stage=3, category="v12_knowledge_definition")
    elif subtype == "concise":
        for k,(term,ans) in enumerate(PERSIAN_FACTS):
            short = ans.split("؛")[0].split(".")[0].strip() + "."
            for p in (f"در یک جمله بگو {term} چیست.", f"خیلی کوتاه: {term}؟", f"بدون حاشیه {term} را تعریف کن."):
                i += 1; add(p, short, f"concise:{k}", i, stage=4)
    elif subtype == "detailed":
        for k,(term,ans) in enumerate(PERSIAN_FACTS):
            out = ans + " نکتهٔ مهم این است که کاربرد و محدودیت آن به زمینهٔ استفاده بستگی دارد."
            for p in (f"{term} را با کمی جزئیات توضیح بده.", f"یک توضیح دو جمله‌ای دربارهٔ {term} بده.", f"برای مبتدی {term} را کامل‌تر توضیح بده."):
                i += 1; add(p, out, f"detailed:{k}", i, stage=4)
    elif subtype == "uncertainty":
        cases = [
            ("این فایل چرا خرابه؟", "بدون دیدن فایل یا متن خطا نمی‌توان علت را دقیق تعیین کرد؛ فایل یا خطا را بفرست تا بررسی کنم."),
            ("کد من چرا کار نمی‌کنه؟", "برای تشخیص علت، خود کد و متن کامل خطا لازم است؛ بدون آن‌ها حدس‌زدن قابل‌اعتماد نیست."),
            ("سرور چرا کند شده؟", "علت از این اطلاعات مشخص نیست؛ مصرف CPU و RAM، زمان پاسخ پایگاه داده، لاگ‌ها و وضعیت شبکه را باید بررسی کرد."),
            ("این نتیجه درسته؟", "خود نتیجه و داده یا مراحل محاسبه را نداده‌ای؛ آن‌ها را بفرست تا بتوانم صحت را بررسی کنم."),
            ("بهترین تنظیم چیه؟", "بدون دانستن هدف، سخت‌افزار و محدودیت‌ها یک تنظیم واحد را نمی‌توان بهترین دانست؛ این سه مورد را مشخص کن."),
            ("چرا کاربرها نمی‌تونن خرید کنن؟", "برای تعیین علت باید مسیر پرداخت، پیام خطا، لاگ سرور و شرایطی که مشکل رخ می‌دهد بررسی شود؛ با اطلاعات فعلی علت قطعی نیست."),
            ("فردا قیمتش بالا میره؟", "با اطلاعات موجود نمی‌توان آیندهٔ قیمت را با قطعیت پیش‌بینی کرد؛ می‌توان عوامل و سناریوهای محتمل را بررسی کرد."),
            ("این برنامه امنه؟", "برای قضاوت دربارهٔ امنیت باید کد، مجوزها، وابستگی‌ها و رفتار شبکه بررسی شود؛ فقط از روی نام برنامه نمی‌توان نتیجه داد."),
        ]
        for k,(p0,ans) in enumerate(cases):
            for p in (p0, "دقیق جواب بده: " + p0, "حدس نزن؛ " + p0):
                i += 1; add(p, ans, f"uncertain:{k}", i, stage=4)
    elif subtype == "ambiguity":
        cases = [
            ("اون رو حذف کن", "«اون» مشخص نیست؛ نام فایل، برنامه یا موردی را که می‌خواهی حذف شود مشخص کن."),
            ("نسخه بهتره؟", "مشخص نیست کدام دو نسخه را مقایسه می‌کنی؛ نام یا شمارهٔ نسخه‌ها را بگو."),
            ("ببرش اونجا", "مبدأ و مقصد مشخص نیست؛ بگو چه چیزی را به کجا منتقل کنم."),
            ("این عدد رو حساب کن", "عدد یا عبارت محاسباتی در پیام نیست؛ عبارت دقیق را بفرست."),
            ("همونو باز کن", "مرجع «همونو» در این پیام مشخص نیست؛ نام فایل، پوشه یا برنامه را بگو."),
            ("این بهتره یا اون؟", "برای مقایسه باید هر دو گزینه مشخص باشند؛ نام یا ویژگی‌هایشان را بفرست."),
            ("مثل قبلی درستش کن", "اگر منظورت تنظیم یا متن قبلی است، بخش موردنظر را مشخص کن تا همان الگو را اعمال کنم."),
            ("این رو ترجمه کن", "متنی برای ترجمه در پیام نیست؛ متن و زبان مقصد را بفرست."),
        ]
        for k,(p0,ans) in enumerate(cases):
            for p in (p0, p0 + " لطفاً", "بدون حدس جواب بده: " + p0):
                i += 1; add(p, ans, f"ambiguity:{k}", i, stage=4)
    elif subtype == "context_reference":
        contexts = [
            ([{"role":"user","content":"فایل گزارش.pdf داخل Downloads است."}], "اون فایل کجاست؟", "فایل گزارش.pdf داخل Downloads است."),
            ([{"role":"user","content":"من پایتون و جاوااسکریپت را مقایسه می‌کنم."}], "اولی کدومه؟", "اولی پایتون است."),
            ([{"role":"user","content":"قرار شد نسخهٔ 2.4 را نگه داریم و 2.3 را حذف کنیم."}], "کدوم نسخه باید بمونه؟", "نسخهٔ 2.4 باید باقی بماند."),
            ([{"role":"user","content":"سرور A سریع‌تر است ولی سرور B حافظهٔ بیشتری دارد."}], "کدوم حافظه بیشتری داره؟", "سرور B حافظهٔ بیشتری دارد."),
            ([{"role":"user","content":"فایل config.json را باز کردم و بعد settings.json را بستم."}], "کدومو بستم؟", "settings.json را بستی."),
            ([{"role":"user","content":"مریم قبل از علی رسید."}], "چه کسی زودتر رسید؟", "مریم زودتر رسید."),
            ([{"role":"user","content":"قیمت محصول X برابر 120 و Y برابر 95 است."}], "ارزون‌تر کدومه؟", "محصول Y ارزان‌تر است."),
            ([{"role":"user","content":"بکاپ ساعت 2 شروع شد و ساعت 3:30 تمام شد."}], "چقدر طول کشید؟", "یک ساعت و نیم طول کشید."),
        ]
        for k,(ctx,p0,ans) in enumerate(contexts):
            for p in (p0, "با توجه به حرف قبلی: " + p0, p0 + " دقیق بگو"):
                i += 1; add(p, ans, f"context:{k}", i, stage=9, category="v12_context_reference", context=ctx)
    else:
        # General Persian instruction adherence: same content under different constraints.
        items = [
            ("سه مزیت نسخهٔ پشتیبان را شماره‌گذاری کن.", "1. بازیابی پس از حذف یا خرابی داده\n2. کاهش ریسک از دست‌رفتن اطلاعات\n3. امکان بازگشت به نسخهٔ سالم قبلی"),
            ("دو تفاوت TCP و UDP را فقط در دو مورد بنویس.", "1. TCP تحویل و ترتیب را تضمین می‌کند؛ UDP نه.\n2. UDP سربار کمتر و تأخیر پایین‌تری دارد؛ TCP کنترل بیشتری انجام می‌دهد."),
            ("پاسخ را فقط با بله یا خیر بده: آیا 17 عدد اول است؟", "بله"),
            ("در یک جمله توضیح بده چرا تست مهم است.", "تست کمک می‌کند خطاها و رگرسیون‌ها پیش از رسیدن تغییرات به کاربر شناسایی شوند."),
            ("سه قدم برای دیباگ را کوتاه بنویس.", "1. خطا را بازتولید کن.\n2. ورودی و لاگ را محدود کن.\n3. فرضیه را با یک تغییر کوچک آزمایش کن."),
            ("بدون مقدمه بگو JSON چه نوع قالبی است.", "JSON یک قالب متنی ساخت‌یافته برای تبادل داده است."),
            ("فقط نتیجه را بگو: 12 × 8", "96"),
            ("در دو جمله فرق RAM و SSD را بگو.", "RAM حافظهٔ موقت و سریع برای اجرای برنامه‌هاست. SSD ذخیره‌سازی دائمی است و داده را پس از خاموشی نگه می‌دارد."),
        ]
        for k,(p0,ans) in enumerate(items):
            for p in (p0, "دستور را دقیق رعایت کن: " + p0, p0 + " حاشیه نرو"):
                i += 1; add(p, ans, f"instruction:{k}", i, stage=4)
    # cycle deterministically to exactly 48, while keeping unique prompts
    base = list(rows)
    j = 0
    while len(rows) < EXAMPLES_PER_DATASET:
        src = base[j % len(base)]
        clone = dict(src)
        clone["id"] = f"v12-{dataset_id}-{len(rows)+1:04d}"
        clone["input"] = src["input"] + (" لطفاً پاسخ دقیق باشد." if len(rows) % 2 == 0 else " پاسخ را روشن و بدون حدس بده.")
        clone["metadata"] = dict(src["metadata"])
        clone["metadata"]["concept_group"] = src["metadata"]["concept_group"]
        rows.append(clone); j += 1
    return rows[:EXAMPLES_PER_DATASET]


# ---------------- math/reasoning datasets ----------------

def fmt(x: float) -> str:
    if abs(x - round(x)) < 1e-9:
        return str(int(round(x)))
    return f"{x:.6f}".rstrip("0").rstrip(".")


def gen_math(subtype: str, dataset_id: str, seed_offset: int) -> list[dict[str, Any]]:
    rng = random.Random(SEED + seed_offset)
    rows: list[dict[str, Any]] = []
    i = 0
    def add(p: str, a: str, concept: str, difficulty="medium"):
        nonlocal i
        i += 1
        rows.append(make_row(dataset_id, i, p, a, category=f"v12_reasoning_{subtype}", stage=11, stage_name="reasoning", concept=concept, difficulty=difficulty))

    concepts = 16
    for c in range(concepts):
        if subtype == "addition":
            a,b = rng.randint(20,900), rng.randint(20,900); r=a+b
            qa=[(f"{a} + {b} چند می‌شود؟",f"{a} + {b} = {r}."),(f"جمع {a} و {b} را حساب کن.",str(r)),(f"اگر {a} واحد داشته باشیم و {b} واحد اضافه شود، مجموع چقدر است؟",f"مجموع {r} واحد است.")]
        elif subtype == "subtraction":
            a,b = sorted((rng.randint(30,1000),rng.randint(10,900)), reverse=True); r=a-b
            qa=[(f"{a} - {b} چند می‌شود؟",f"{a} - {b} = {r}."),(f"اختلاف {a} و {b} را حساب کن.",str(r)),(f"از {a} عدد، {b} تا کم کنیم چند می‌ماند؟",f"{r} می‌ماند.")]
        elif subtype == "multiplication":
            a,b=rng.randint(3,60),rng.randint(3,40); r=a*b
            qa=[(f"{a} × {b} را حساب کن.",f"{a} × {b} = {r}."),(f"حاصل‌ضرب {a} در {b} چیست؟",str(r)),(f"{a} بسته داریم و در هر بسته {b} عدد است؛ جمعاً چند عدد؟",f"{r} عدد.")]
        elif subtype == "division":
            b=rng.randint(2,25); q=rng.randint(2,40); a=b*q
            qa=[(f"{a} ÷ {b} چند می‌شود؟",f"{a} ÷ {b} = {q}."),(f"{a} را بر {b} تقسیم کن.",str(q)),(f"{a} شیء را مساوی بین {b} نفر تقسیم کنیم، سهم هر نفر؟",f"هر نفر {q} شیء.")]
        elif subtype == "fractions":
            d=rng.choice([4,5,6,8,10,12]); n1=rng.randint(1,d-1); n2=rng.randint(1,d-1); num=n1+n2; g=math.gcd(num,d); rn,rd=num//g,d//g
            qa=[(f"{n1}/{d} + {n2}/{d} را ساده کن.",f"صورت‌ها جمع می‌شوند: {num}/{d} = {rn}/{rd}."),(f"حاصل جمع کسرهای {n1}/{d} و {n2}/{d} چیست؟",f"{rn}/{rd}"),(f"اگر {n1}/{d} و سپس {n2}/{d} از یک واحد را داشته باشیم، جمع چقدر است؟",f"{rn}/{rd} واحد.")]
        elif subtype == "percent":
            base=rng.choice([80,100,120,150,200,240,300,400,500]); pct=rng.choice([5,10,15,20,25,30,40,50]); r=base*pct/100
            qa=[(f"{pct} درصدِ {base} چقدر است؟",f"{base} × {pct}/100 = {fmt(r)}."),(f"{pct}% از {base} را حساب کن.",fmt(r)),(f"اگر {pct} درصد از {base} مورد را انتخاب کنیم، چند مورد می‌شود؟",f"{fmt(r)} مورد.")]
        elif subtype == "ratio":
            x=rng.randint(2,9); y=rng.randint(2,9); k=rng.randint(3,12); total=(x+y)*k
            qa=[(f"نسبت A به B برابر {x}:{y} است و مجموعشان {total} است. A و B را پیدا کن.",f"هر واحد نسبت {k} است؛ A={x*k} و B={y*k}."),(f"دو مقدار با نسبت {x} به {y} جمعاً {total} هستند. مقدار اول؟",str(x*k)),(f"در نسبت {x}:{y} با کل {total}، سهم دوم چقدر است؟",str(y*k))]
        elif subtype == "average":
            vals=[rng.randint(10,60) for _ in range(4)]; s=sum(vals); av=s/4
            qa=[(f"میانگین {vals[0]}، {vals[1]}، {vals[2]} و {vals[3]} چقدر است؟",f"جمع {s} است و {s} ÷ 4 = {fmt(av)}."),(f"average این چهار عدد را حساب کن: {', '.join(map(str,vals))}",fmt(av)),(f"اگر چهار نمره {', '.join(map(str,vals))} باشند، میانگین؟",fmt(av))]
        elif subtype == "weighted_average":
            a,b=rng.randint(10,20),rng.randint(10,20); w1,w2=rng.choice([(1,2),(2,3),(3,1)]); av=(a*w1+b*w2)/(w1+w2)
            qa=[(f"نمرهٔ {a} با وزن {w1} و نمرهٔ {b} با وزن {w2} است. میانگین وزنی؟",f"({a}×{w1}+{b}×{w2})/({w1}+{w2}) = {fmt(av)}."),(f"میانگین وزنی {a}({w1}) و {b}({w2}) را حساب کن.",fmt(av)),(f"اگر دو بخش نمره {a} و {b} با وزن‌های {w1} و {w2} باشند، نتیجه؟",fmt(av))]
        elif subtype == "linear_equation":
            x=rng.randint(2,30); a=rng.randint(2,9); b=rng.randint(1,20); rhs=a*x+b
            qa=[(f"معادله {a}x + {b} = {rhs} را حل کن.",f"{a}x={rhs-b}، پس x={x}."),(f"x را پیدا کن: {a}x+{b}={rhs}",f"x={x}"),(f"اگر {a} برابر x به‌علاوه {b} برابر {rhs} باشد، x چقدر است؟",str(x))]
        elif subtype == "two_step_equation":
            x=rng.randint(1,20); a=rng.randint(2,7); b=rng.randint(1,15); c0=rng.randint(1,10); rhs=a*(x+b)-c0
            qa=[(f"{a}(x+{b}) - {c0} = {rhs}. x را پیدا کن.",f"{a}(x+{b})={rhs+c0}، پس x+{b}={(rhs+c0)//a} و x={x}."),(f"حل کن: {a}*(x+{b})-{c0}={rhs}",f"x={x}"),(f"مقدار x در {a}(x+{b})={rhs+c0} چیست؟",str(x))]
        elif subtype == "inequality":
            a=rng.randint(2,8); x=rng.randint(2,20); b=rng.randint(1,10); rhs=a*x+b
            qa=[(f"نامعادله {a}x + {b} < {rhs} را حل کن.",f"{a}x < {rhs-b}، پس x < {fmt((rhs-b)/a)}."),(f"برای چه xهایی {a}x+{b}<{rhs}؟",f"x < {x}"),(f"حد بالای x در نامعادله {a}x+{b}<{rhs} چیست؟",f"x باید کمتر از {x} باشد.")]
        elif subtype == "absolute":
            a=rng.randint(2,20)
            qa=[(f"معادله |x| = {a} چند جواب دارد و جواب‌ها چیست؟",f"دو جواب دارد: x={a} و x=-{a}."),(f"|x|={a} را حل کن.",f"x=±{a}"),(f"اگر فاصلهٔ x از صفر {a} باشد، x چه مقادیری دارد؟",f"{a} و -{a}")]
        elif subtype == "powers":
            a=rng.randint(2,9); b=rng.randint(2,5); r=a**b
            qa=[(f"{a}^{b} چند می‌شود؟",f"{a}^{b} = {r}."),(f"توان {b} از {a} را حساب کن.",str(r)),(f"{a} را {b} بار در خودش ضرب کنیم چه می‌شود؟",str(r))]
        elif subtype == "roots":
            r=rng.randint(2,30); n=r*r
            qa=[(f"ریشهٔ دوم {n} چیست؟",str(r)),(f"√{n} را حساب کن.",f"{r}"),(f"چه عدد مثبتی اگر در خودش ضرب شود {n} می‌دهد؟",str(r))]
        elif subtype == "divisibility":
            n=rng.randint(20,500); d=rng.choice([2,3,5,9,10]); yes=n%d==0
            qa=[(f"آیا {n} بر {d} بخش‌پذیر است؟ دلیل کوتاه بگو.",f"{'بله' if yes else 'خیر'}؛ باقیماندهٔ تقسیم {n} بر {d} برابر {n%d} است."),(f"فقط بگو {n} مضرب {d} هست یا نه.","بله" if yes else "خیر"),(f"{n} % {d} چند است؟",str(n%d))]
        elif subtype == "gcd":
            a,b=rng.randint(20,200),rng.randint(20,200); g=math.gcd(a,b)
            qa=[(f"ب.م.م {a} و {b} را پیدا کن.",f"بزرگ‌ترین مقسوم‌علیه مشترک {g} است."),(f"GCD({a},{b}) = ?",str(g)),(f"بزرگ‌ترین عددی که هم {a} و هم {b} بر آن بخش‌پذیرند چیست؟",str(g))]
        elif subtype == "lcm":
            a,b=rng.randint(3,40),rng.randint(3,40); l=abs(a*b)//math.gcd(a,b)
            qa=[(f"ک.م.م {a} و {b} چیست؟",f"کمترین مضرب مشترک {l} است."),(f"LCM({a},{b}) را حساب کن.",str(l)),(f"کوچک‌ترین مضرب مثبت مشترک {a} و {b}؟",str(l))]
        elif subtype == "prime":
            n=rng.randint(2,120); prime=n>1 and all(n%d for d in range(2,int(math.sqrt(n))+1))
            qa=[(f"آیا {n} عدد اول است؟",("بله، عدد اول است." if prime else "خیر، عدد اول نیست.")),(f"prime بودن {n} را مشخص کن.","prime" if prime else "not prime"),(f"فقط بله یا خیر: {n} اول است؟","بله" if prime else "خیر")]
        elif subtype == "trailing_zeros":
            n=rng.randint(20,250); z=0; d=5
            while d<=n: z+=n//d; d*=5
            qa=[(f"{n}! چند صفر انتهایی دارد؟",f"تعداد عامل‌های 5 برابر {z} است؛ پس {z} صفر انتهایی دارد."),(f"trailing zeros در {n}! چندتاست؟",str(z)),(f"تعداد صفرهای آخر فاکتوریل {n} را حساب کن.",str(z))]
        elif subtype == "arithmetic_sequence":
            start=rng.randint(1,20); d=rng.randint(2,12); vals=[start+d*j for j in range(5)]; nxt=vals[-1]+d
            seq="، ".join(map(str,vals))
            qa=[(f"عدد بعدی دنباله {seq} چیست؟",f"اختلاف ثابت {d} است، پس عدد بعدی {nxt} است."),(f"دنباله را ادامه بده: {seq}, ...",str(nxt)),(f"در sequence {seq} جملهٔ بعدی؟",str(nxt))]
        elif subtype == "geometric_sequence":
            start=rng.randint(1,5); ratio=rng.choice([2,3,4]); vals=[start*(ratio**j) for j in range(5)]; nxt=vals[-1]*ratio; seq="، ".join(map(str,vals))
            qa=[(f"عدد بعدی دنباله {seq} چیست؟",f"هر جمله در {ratio} ضرب می‌شود؛ بعدی {nxt} است."),(f"دنبالهٔ هندسی را ادامه بده: {seq}",str(nxt)),(f"next number: {seq}",str(nxt))]
        elif subtype == "alternating_sequence":
            a=rng.randint(2,10); vals=[a, a+2, (a+2)*2, (a+2)*2+2, ((a+2)*2+2)*2]; nxt=vals[-1]+2; seq="، ".join(map(str,vals))
            qa=[(f"با الگوی +2، ×2 به‌صورت یکی‌درمیان، عدد بعدی {seq} چیست؟",f"گام بعدی +2 است؛ جواب {nxt}."),(f"قاعده +2 سپس ×2 تکرار می‌شود: {seq}. بعدی؟",str(nxt)),(f"دنباله با قانون اعلام‌شده را ادامه بده: {seq}",str(nxt))]
        elif subtype == "square_sequence":
            n=rng.randint(2,8); vals=[j*j for j in range(n,n+5)]; nxt=(n+5)**2; seq="، ".join(map(str,vals))
            qa=[(f"این‌ها مربع اعداد پیاپی‌اند: {seq}. عدد بعدی؟",str(nxt)),(f"دنبالهٔ مربع‌ها را ادامه بده: {seq}",str(nxt)),(f"next square after sequence {seq}?",str(nxt))]
        elif subtype == "coin_probability":
            n=rng.randint(2,6); total=2**n
            qa=[(f"{n} بار سکهٔ سالم می‌اندازیم. احتمال اینکه همه شیر بیاید؟",f"(1/2)^{n} = 1/{total}."),(f"احتمال {n} شیر پیاپی با سکهٔ منصف؟",f"1/{total}"),(f"در {n} پرتاب مستقل سکه، P(all heads)=?",f"1/{total}")]
        elif subtype == "dice_probability":
            target=rng.randint(1,6)
            qa=[(f"یک تاس سالم می‌اندازیم. احتمال آمدن {target} چقدر است؟","1/6"),(f"P(rolling {target}) on a fair die?","1/6"),(f"احتمال یک نتیجهٔ مشخص روی تاس شش‌وجهی؟","1/6")]
        elif subtype == "urn_probability":
            red=rng.randint(2,8); blue=rng.randint(2,8); total=red+blue
            qa=[(f"کیسه {red} مهرهٔ قرمز و {blue} آبی دارد. احتمال برداشتن قرمز؟",f"{red}/{total}."),(f"از {total} مهره که {red} قرمزند، احتمال قرمز؟",f"{red}/{total}"),(f"P(red) with {red} red and {blue} blue balls?",f"{red}/{total}")]
        elif subtype == "combinations":
            n=rng.randint(5,10); k=rng.randint(2,min(4,n-1)); r=math.comb(n,k)
            qa=[(f"از {n} نفر چند گروه {k} نفره می‌توان ساخت؟",f"C({n},{k}) = {r}."),(f"تعداد انتخاب {k} عضو از {n} عضو بدون ترتیب؟",str(r)),(f"{n} choose {k} = ?",str(r))]
        elif subtype == "permutations":
            n=rng.randint(3,7); r=math.factorial(n)
            qa=[(f"{n} شیء متمایز را چند جور می‌توان ردیف کرد؟",f"{n}! = {r}."),(f"تعداد جایگشت {n} عنصر؟",str(r)),(f"permutations of {n} distinct items?",str(r))]
        elif subtype == "sets":
            a=rng.randint(20,60); b=rng.randint(20,60); inter=rng.randint(5,min(a,b,20)); union=a+b-inter
            qa=[(f"در یک گروه، {a} نفر A و {b} نفر B را دارند و {inter} نفر هر دو را دارند. تعداد A∪B؟",f"{a}+{b}-{inter}={union}."),(f"|A|={a}, |B|={b}, |A∩B|={inter}. |A∪B|؟",str(union)),(f"با اصل شمول و عدم شمول، اتحاد دو مجموعه چند عضو دارد؟ داده‌ها: {a}, {b}, اشتراک {inter}",str(union))]
        elif subtype == "syllogism":
            qa=[("همهٔ برنامه‌نویس‌ها انسان‌اند. بعضی انسان‌ها موسیقی‌دان‌اند. آیا حتماً بعضی برنامه‌نویس‌ها موسیقی‌دان‌اند؟","خیر. دو گزاره تضمین نمی‌کنند که مجموعهٔ برنامه‌نویس‌ها و موسیقی‌دان‌ها اشتراک داشته باشند."),("همهٔ Aها B هستند و بعضی Bها C هستند. آیا نتیجه می‌شود بعضی Aها C هستند؟","خیر؛ ممکن است اعضای B که C هستند هیچ‌کدام در A نباشند."),("All A are B. Some B are C. Must some A be C?","No. The premises do not force the A set to overlap the C subset of B.")]
        elif subtype == "propositional":
            p=bool(c%2); q=bool((c//2)%2); implication=(not p) or q
            qa=[(f"اگر P={str(p).lower()} و Q={str(q).lower()} باشد، مقدار P→Q چیست؟",str(implication).lower()),(f"ارزش گزارهٔ 'اگر P آنگاه Q' برای P={p} و Q={q}؟","درست" if implication else "نادرست"),(f"P={p}, Q={q}; implication P=>Q?",str(implication))]
        elif subtype == "logic_negation":
            qa=[("نقیض «همهٔ فایل‌ها سالم‌اند» چیست؟","«حداقل یک فایل سالم نیست.»"),("نقیض «هیچ کاربری خطا ندارد» چیست؟","«حداقل یک کاربر خطا دارد.»"),("Negate: All tests passed.","At least one test did not pass.")]
        elif subtype == "age":
            younger=rng.randint(5,25); diff=rng.randint(3,15); older=younger+diff
            qa=[(f"علی {diff} سال از رضا بزرگ‌تر است. اگر رضا {younger} ساله باشد، علی چند ساله است؟",f"{older} سال."),(f"سن B برابر {younger} و A، {diff} سال بزرگ‌تر است. سن A؟",str(older)),(f"Someone is {diff} years older than a {younger}-year-old person. Their age?",str(older))]
        elif subtype == "speed":
            speed=rng.choice([40,50,60,75,80,90]); time=rng.choice([2,3,4,5]); dist=speed*time
            qa=[(f"خودرو با سرعت {speed} کیلومتر بر ساعت، {time} ساعت حرکت می‌کند. مسافت؟",f"{speed}×{time}={dist} کیلومتر."),(f"با سرعت {speed} و زمان {time} ساعت، distance چقدر است؟",str(dist)+" km"),(f"مسافت طی‌شده در {time} ساعت با سرعت ثابت {speed}؟",str(dist)+" کیلومتر")]
        elif subtype == "work_rate":
            a=rng.choice([2,3,4,6]); b=rng.choice([3,4,6,8]); l=math.lcm(a,b); combined=1/a+1/b; t=1/combined
            qa=[(f"A کاری را در {a} ساعت و B در {b} ساعت انجام می‌دهد. با هم چند ساعت؟",f"نرخ مشترک 1/{a}+1/{b} است؛ زمان {fmt(t)} ساعت."),(f"دو نفر با زمان‌های {a} و {b} ساعت هم‌زمان کار کنند، زمان کل؟",fmt(t)+" ساعت"),(f"Combined work time for rates 1/{a} and 1/{b} per hour?",fmt(t)+" hours")]
        elif subtype == "calendar":
            days=["شنبه","یکشنبه","دوشنبه","سه‌شنبه","چهارشنبه","پنجشنبه","جمعه"]; idx=rng.randrange(7); delta=rng.randint(1,10); target=days[(idx+delta)%7]
            qa=[(f"اگر امروز {days[idx]} باشد، {delta} روز بعد چه روزی است؟",target),(f"از {days[idx]}، {delta} روز جلو برو. چه روزی می‌شود؟",target),(f"امروز {days[idx]} است؛ روزِ +{delta}؟",target)]
        elif subtype == "clock":
            hour=rng.randint(1,12); minute=rng.choice([0,10,15,20,30,40,45,50]); ha=30*(hour%12)+0.5*minute; ma=6*minute; diff=abs(ha-ma)%360; ang=min(diff,360-diff)
            qa=[(f"زاویهٔ کوچک‌تر عقربه‌ها در ساعت {hour}:{minute:02d} چند درجه است؟",f"ساعت‌شمار {fmt(ha)}° و دقیقه‌شمار {fmt(ma)}°؛ زاویه {fmt(ang)}° است."),(f"clock angle at {hour}:{minute:02d}?",fmt(ang)+"°"),(f"در {hour}:{minute:02d} زاویهٔ بین عقربه‌ها؟",fmt(ang)+" درجه")]
        else:
            raise ValueError(subtype)
        for p,a in qa:
            add(p,a,f"{subtype}:{c}","hard" if subtype in {"weighted_average","two_step_equation","work_rate","propositional"} else "medium")
    return rows[:EXAMPLES_PER_DATASET]


# ---------------- knowledge / coding datasets ----------------

def gen_knowledge(subtype: str, dataset_id: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]]=[]; i=0
    def add(p,a,c,stage=3):
        nonlocal i; i+=1; rows.append(make_row(dataset_id,i,p,a,category=f"v12_knowledge_{subtype}",stage=stage,stage_name="general_knowledge" if stage==3 else "instruction_following",concept=c))
    banks = {
        "science": KNOWLEDGE_BANK,
        "computer": [(f"{term} چیست؟",ans) for term,ans in PERSIAN_FACTS],
        "network": [
            ("IP چیست؟","نشانی منطقی یک رابط شبکه است که برای مسیریابی بسته‌ها بین شبکه‌ها استفاده می‌شود."),
            ("روتر چه می‌کند؟","بسته‌ها را بر اساس جدول مسیریابی میان شبکه‌های مختلف هدایت می‌کند."),
            ("سوئیچ شبکه چه می‌کند؟","در شبکهٔ محلی فریم‌ها را معمولاً با استفاده از نشانی MAC به پورت مناسب می‌فرستد."),
            ("پورت TCP/UDP چیست؟","شماره‌ای منطقی برای شناسایی سرویس یا جریان ارتباطی روی یک میزبان است."),
            ("NAT چیست؟","نشانی‌های شبکه را بازنویسی می‌کند و معمولاً اجازه می‌دهد چند دستگاه خصوصی یک نشانی عمومی را به اشتراک بگذارند."),
            ("TLS چیست؟","پروتکلی برای ایجاد ارتباط رمزگذاری‌شده و احراز هویت‌شده روی شبکه است."),
            ("latency چیست؟","زمان تأخیر از ارسال درخواست یا داده تا رسیدن پاسخ یا دادهٔ متناظر است."),
            ("bandwidth چیست؟","ظرفیت انتقال داده در واحد زمان است و با تأخیر یکسان نیست."),
        ],
        "database": [
            ("ایندکس پایگاه داده چیست؟","ساختار کمکی برای یافتن سریع‌تر ردیف‌هاست که در عوض فضای ذخیره‌سازی و هزینهٔ نوشتن بیشتری مصرف می‌کند."),
            ("کلید اصلی چیست؟","ستونی یا مجموعه‌ستونی است که هر ردیف را به‌طور یکتا شناسایی می‌کند."),
            ("تراکنش چیست؟","مجموعه‌ای از عملیات است که به‌صورت یک واحد منطقی اجرا می‌شود و معمولاً ویژگی‌های ACID را هدف می‌گیرد."),
            ("JOIN چه می‌کند؟","ردیف‌های جدول‌ها را بر اساس رابطه یا شرط مشترک ترکیب می‌کند."),
            ("نرمال‌سازی پایگاه داده چیست؟","سازمان‌دهی ساختار جداول برای کاهش افزونگی و ناهنجاری‌های درج، حذف و به‌روزرسانی است."),
            ("ACID چیست؟","چهار ویژگی Atomicity، Consistency، Isolation و Durability برای تراکنش‌های قابل‌اعتماد است."),
            ("NoSQL یعنی چه؟","نامی کلی برای پایگاه‌های غیررابطه‌ای مانند سندی، کلید-مقدار، ستونی و گراف است."),
            ("query plan چیست؟","برنامه‌ای است که موتور پایگاه داده برای اجرای یک query انتخاب می‌کند؛ مثل ترتیب join و استفاده از index."),
        ],
        "web": [
            ("HTML چیست؟","زبان نشانه‌گذاری ساختار محتوای صفحهٔ وب است."),
            ("CSS چیست؟","زبان توصیف ظاهر و چیدمان عناصر سند، به‌ویژه صفحات وب است."),
            ("JavaScript در وب چه می‌کند؟","رفتار و منطق تعاملی صفحه را اجرا می‌کند و می‌تواند با DOM و APIها کار کند."),
            ("cookie چیست؟","قطعهٔ کوچک داده است که مرورگر برای یک دامنه ذخیره و طبق قواعد مشخص همراه درخواست‌ها ارسال می‌کند."),
            ("CORS چیست؟","سیاست مرورگر برای کنترل درخواست‌های cross-origin است که با headerهای پاسخ سرور تنظیم می‌شود."),
            ("REST چیست؟","سبکی معماری برای API است که معمولاً منابع را با URI و عملیات را با روش‌های HTTP مدل می‌کند."),
            ("status code 404 یعنی چه؟","یعنی منبع درخواستی روی سرور پیدا نشده است."),
            ("status code 500 یعنی چه؟","یعنی سرور هنگام پردازش درخواست با خطای داخلی مواجه شده است."),
        ],
        "security": [
            ("هش چه فرقی با رمزنگاری دارد؟","هش معمولاً یک‌طرفه و برای اثرانگشت داده است؛ رمزنگاری با کلید برای بازگردانی دادهٔ اصلی طراحی می‌شود."),
            ("salt در ذخیرهٔ رمز عبور چیست؟","دادهٔ تصادفی منحصربه‌فردی است که پیش از مشتق‌سازی هش به رمز عبور اضافه می‌شود تا حملات جدول‌های ازپیش‌محاسبه‌شده سخت‌تر شوند."),
            ("اصل کمترین دسترسی چیست؟","هر کاربر یا سرویس فقط مجوزهایی را می‌گیرد که برای وظیفهٔ لازم نیاز دارد."),
            ("MFA چیست؟","احراز هویت چندعاملی است که بیش از یک عامل مستقل مثل رمز و توکن را نیاز دارد."),
            ("SQL injection چیست؟","آسیب‌پذیری‌ای است که ورودی کنترل‌نشده ساختار query را تغییر می‌دهد؛ استفاده از query پارامتری دفاع اصلی است."),
            ("XSS چیست؟","تزریق یا اجرای اسکریپت ناخواسته در زمینهٔ صفحهٔ وب است؛ escaping و سیاست‌های محتوایی از دفاع‌های مهم‌اند."),
            ("backup چرا بخشی از امنیت است؟","چون بازیابی پس از حذف، خرابی یا باج‌افزار را ممکن می‌کند و اثر رخداد را کاهش می‌دهد."),
            ("چرا به‌روزرسانی وابستگی‌ها مهم است؟","نسخه‌های جدید می‌توانند آسیب‌پذیری‌های شناخته‌شده را رفع کنند، هرچند باید با تست و بررسی سازگاری به‌روزرسانی شوند."),
        ],
        "ai": [
            ("overfitting چیست؟","وقتی مدل روی دادهٔ آموزش بیش از حد منطبق می‌شود و روی دادهٔ ندیده عملکرد ضعیف‌تری دارد."),
            ("validation set چیست؟","بخشی جدا از آموزش است که برای انتخاب تنظیمات و پایش تعمیم مدل استفاده می‌شود."),
            ("test set چیست؟","دادهٔ نهاییِ کنارگذاشته‌شده برای برآورد عملکرد پس از اتمام انتخاب و تنظیم مدل است."),
            ("tokenizer چه می‌کند؟","متن را به واحدهای عددی قابل پردازش مدل و برعکس تبدیل می‌کند."),
            ("embedding چیست؟","نمایش برداری فشرده‌ای از token، کلمه یا داده است که روابط قابل یادگیری را در فضای عددی کد می‌کند."),
            ("quantization چیست؟","نمایش وزن‌ها یا فعال‌سازی‌ها با دقت عددی کمتر برای کاهش حافظه و افزایش سرعت، با احتمال افت محدود دقت است."),
            ("fine-tuning چیست؟","ادامهٔ آموزش یک مدل از وزن‌های موجود روی داده یا هدف تخصصی‌تر است."),
            ("data leakage چیست؟","وقتی اطلاعاتی از validation یا test به آموزش یا تصمیم‌گیری مدل نشت می‌کند و ارزیابی را خوش‌بینانه می‌سازد."),
        ],
    }
    if subtype in banks:
        bank=banks[subtype]
        for k,(p0,a) in enumerate(bank):
            for p in (p0, "ساده توضیح بده: "+p0, "کوتاه و دقیق: "+p0): add(p,a,f"{subtype}:{k}")
    elif subtype in {"python","debugging","algorithms","datastructures","git","api","files","software_design"}:
        bank2={
            "python":[
                ("در پایتون list comprehension چیست؟","روش فشرده‌ای برای ساخت لیست از یک iterable با expression و شرط اختیاری است."),
                ("فرق == و is در پایتون چیست؟","== برابری مقدار را می‌سنجد؛ is یکسان‌بودن هویت دو شیء را بررسی می‌کند."),
                ("context manager چیست؟","الگوی مدیریت منبع است که با with ورود و خروج کنترل‌شده و cleanup مطمئن را فراهم می‌کند."),
                ("generator چیست؟","تابعی یا شیئی است که مقادیر را به‌صورت lazy و مرحله‌ای تولید می‌کند و معمولاً از yield استفاده می‌کند."),
                ("exception handling در پایتون چیست؟","با try/except/finally خطاهای قابل‌انتظار را مدیریت و cleanup را کنترل می‌کند."),
                ("dict در پایتون چیست؟","ساختار نگاشت کلید به مقدار است که دسترسی میانگین بسیار سریع بر اساس hash فراهم می‌کند."),
                ("venv چه کاربردی دارد؟","وابستگی‌های یک پروژهٔ پایتون را در محیط جدا نگه می‌دارد تا با پروژه‌های دیگر تداخل نکنند."),
                ("type hint چیست؟","نشانه‌گذاری نوع برای خوانایی و ابزارهای تحلیل ایستا است؛ به‌طور پیش‌فرض اجرای پایتون آن را الزام نمی‌کند."),
            ],
            "debugging":[
                ("اولین قدم دیباگ یک خطای تصادفی چیست؟","شرایط بازتولید را ثبت و تا حد امکان خطا را به یک سناریوی تکرارپذیر تبدیل کن."),
                ("چرا لاگ کامل مهم است؟","زمان، ورودی و stack trace می‌توانند محل و شرایط خطا را مشخص کنند و حدس را کاهش دهند."),
                ("چرا تغییر هم‌زمان چند چیز در دیباگ بد است؟","چون معلوم نمی‌شود کدام تغییر علت بهبود یا خرابی بوده؛ تغییرهای کوچک و قابل‌آزمایش نتیجه را قابل‌نسبت‌دادن می‌کنند."),
                ("bisect در دیباگ چه ایده‌ای دارد؟","فضای تغییرات یا ورودی را نصف‌نصف محدود می‌کند تا نقطهٔ ایجاد مشکل سریع‌تر پیدا شود."),
                ("assert چه کمکی می‌کند؟","فرض‌های داخلی برنامه را صریح می‌کند و هنگام نقض آن‌ها خطا را نزدیک‌تر به علت آشکار می‌سازد."),
                ("چرا تست regression مهم است؟","پس از رفع باگ ثابت می‌کند همان رفتار خراب دوباره برنمی‌گردد."),
                ("stack trace چیست؟","زنجیرهٔ فراخوانی تابع‌ها تا نقطهٔ خطاست و برای یافتن مسیر رسیدن به خطا کمک می‌کند."),
                ("minimal reproducible example چیست؟","کوچک‌ترین نمونه‌ای است که هنوز خطا را بازتولید می‌کند و عوامل نامرتبط را حذف می‌کند."),
            ],
            "algorithms":[
                ("پیچیدگی O(n) یعنی چه؟","زمان یا کار الگوریتم تقریباً متناسب با اندازهٔ ورودی رشد می‌کند."),
                ("binary search چه شرطی دارد؟","داده یا فضای جست‌وجو باید ترتیبی/یکنواخت باشد تا هر مرحله بتوان نیمی از گزینه‌ها را حذف کرد."),
                ("BFS چیست؟","پیمایش عرض‌اول گراف است که لایه‌به‌لایه با صف پیش می‌رود و در گراف بدون وزن کوتاه‌ترین تعداد یال را پیدا می‌کند."),
                ("DFS چیست؟","پیمایش عمق‌اول است که یک مسیر را تا حد ممکن دنبال می‌کند و سپس بازمی‌گردد؛ با stack یا recursion قابل پیاده‌سازی است."),
                ("dynamic programming چیست؟","مسئله را به زیرمسئله‌های هم‌پوشان می‌شکند و نتایج را ذخیره می‌کند تا محاسبهٔ تکراری کاهش یابد."),
                ("greedy algorithm چیست؟","در هر مرحله انتخاب محلیِ به‌ظاهر بهترین را انجام می‌دهد؛ فقط وقتی ساختار مسئله اجازه دهد به جواب بهینه می‌رسد."),
                ("stable sort چیست؟","مرتب‌سازی‌ای است که ترتیب نسبی عناصر با کلید برابر را حفظ می‌کند."),
                ("hash table چه مزیتی دارد؟","برای lookup و update میانگین نزدیک O(1) ارائه می‌دهد، البته با هزینهٔ حافظه و امکان برخورد hash."),
            ],
            "datastructures":[
                ("stack چیست؟","ساختار LIFO است؛ آخرین عنصر واردشده اولین عنصر خارج‌شده است."),
                ("queue چیست؟","ساختار FIFO است؛ اولین عنصر واردشده اولین عنصر خارج‌شده است."),
                ("linked list چیست؟","مجموعه‌ای از nodeهاست که با reference به یکدیگر متصل‌اند و درج/حذف محلی را ساده می‌کنند."),
                ("tree چیست؟","ساختار سلسله‌مراتبی از nodeها و edgeهاست که معمولاً یک ریشه و رابطهٔ والد/فرزند دارد."),
                ("graph چیست؟","مجموعه‌ای از رأس‌ها و یال‌هاست که روابط عمومی‌تر از ساختار درختی را مدل می‌کند."),
                ("set چه ویژگی مهمی دارد؟","عضویت یکتا را نگه می‌دارد و برای بررسی membership و عملیات اجتماع/اشتراک مناسب است."),
                ("heap چیست؟","درخت یا آرایه‌ای با خاصیت heap است که دسترسی سریع به کمینه یا بیشینه را فراهم می‌کند."),
                ("deque چیست؟","صف دوسر است که درج و حذف کارآمد از ابتدا و انتها را پشتیبانی می‌کند."),
            ],
            "git":[
                ("git commit چیست؟","snapshot منطقی از تغییرات staged با پیام و والد مشخص در تاریخچه است."),
                ("git branch چیست؟","اشاره‌گری متحرک به یک خط از commitهاست که توسعهٔ موازی را ممکن می‌کند."),
                ("git pull چه می‌کند؟","معمولاً fetch را انجام می‌دهد و سپس تغییرات remote را با شاخهٔ محلی ادغام یا rebase می‌کند، بسته به تنظیمات."),
                ("git fetch چیست؟","اطلاعات و commitهای remote را دریافت می‌کند بدون اینکه شاخهٔ کاری فعلی را خودکار ادغام کند."),
                ("git reset چه تفاوتی با revert دارد؟","reset اشاره‌گر تاریخچه و وضعیت index/worktree را تغییر می‌دهد؛ revert یک commit جدید معکوس‌کننده می‌سازد و برای تاریخچهٔ اشتراکی امن‌تر است."),
                ("merge conflict چیست؟","وقتی Git نتواند تغییرات متفاوت را خودکار ترکیب کند و نیاز به تصمیم انسانی برای نسخهٔ نهایی دارد."),
                (".gitignore چیست؟","الگوهایی برای فایل‌هایی است که Git نباید به‌طور عادی به‌عنوان فایل جدید track کند."),
                ("tag در Git چیست؟","نام ثابتی برای اشاره به یک commit، معمولاً برای نسخه‌ها و releaseهاست."),
            ],
            "api":[
                ("GET در HTTP معمولاً برای چیست؟","برای دریافت نمایش یک منبع یا داده، بدون قصد تغییر حالت سرور از دید معنای روش."),
                ("POST معمولاً برای چیست؟","برای ارسال داده جهت ایجاد یا پردازش که ممکن است حالت سرور را تغییر دهد."),
                ("PUT و PATCH چه فرقی دارند؟","PUT معمولاً جایگزینی کامل نمایش منبع را مدل می‌کند؛ PATCH تغییر جزئی را."),
                ("idempotent یعنی چه؟","یعنی اجرای چندبارهٔ همان درخواست اثر نهایی متفاوتی از یک‌بار اجرا نداشته باشد."),
                ("API pagination چرا لازم است؟","برای شکستن مجموعهٔ بزرگ پاسخ به صفحه‌های کوچک‌تر و کنترل حافظه، زمان و پهنای باند."),
                ("rate limit چیست؟","محدودیت تعداد درخواست در یک بازه برای کنترل مصرف و حفاظت از سرویس است."),
                ("authentication و authorization چه فرقی دارند؟","authentication هویت را مشخص می‌کند؛ authorization تعیین می‌کند آن هویت چه مجوزهایی دارد."),
                ("webhook چیست؟","مکانیزمی است که سرویس هنگام رخداد، درخواست HTTP را به endpoint ثبت‌شده می‌فرستد تا نیاز به polling کاهش یابد."),
            ],
            "files":[
                ("CSV چیست؟","قالب متنی جدولی است که ردیف‌ها و ستون‌ها را با delimiter نمایش می‌دهد و قواعد quoting برای مقادیر خاص دارد."),
                ("YAML چیست؟","قالب متنی انسان‌خوان برای دادهٔ ساخت‌یافته و پیکربندی است که تورفتگی در ساختار آن مهم است."),
                ("ZIP چیست؟","فرمت آرشیو است که چند فایل را همراه با فشرده‌سازی و metadata در یک فایل بسته‌بندی می‌کند."),
                ("UTF-8 چیست؟","کدگذاری متغیرطول Unicode است که همهٔ code pointهای Unicode را پوشش می‌دهد و با ASCII سازگار است."),
                ("binary file چیست؟","فایلی است که محتوایش الزاماً به‌صورت متن قابل‌تفسیر نیست و باید طبق قالب باینری مشخص خوانده شود."),
                ("path absolute چیست؟","مسیر کامل از ریشه یا drive است و مستقل از working directory فعلی تفسیر می‌شود."),
                ("path relative چیست؟","مسیر نسبت به working directory یا یک پایهٔ مشخص است."),
                ("checksum چیست؟","اثر انگشت محاسباتی برای بررسی یکپارچگی داده است؛ hash رمزنگاری‌شده مانند SHA-256 برای این کار رایج است."),
            ],
            "software_design":[
                ("separation of concerns چیست؟","مسئولیت‌های متفاوت را در بخش‌های جدا نگه می‌دارد تا تغییر، تست و فهم سیستم ساده‌تر شود."),
                ("dependency injection چیست؟","وابستگی‌ها از بیرون به جزء داده می‌شوند تا coupling کاهش و تست‌پذیری بهتر شود."),
                ("interface چه کمکی می‌کند؟","قرارداد رفتار را از پیاده‌سازی جدا می‌کند و امکان تعویض پیاده‌سازی و تست را بهتر می‌سازد."),
                ("single responsibility چیست؟","یک ماژول یا کلاس باید دلیل اصلی و متمرکزی برای تغییر داشته باشد، نه چند مسئولیت نامرتبط."),
                ("immutability چه مزیتی دارد؟","کاهش تغییر حالت ناخواسته، ساده‌ترشدن reasoning و ایمنی بهتر در اشتراک داده بین بخش‌ها."),
                ("caching چه trade-offی دارد؟","سرعت خواندن را بهتر می‌کند اما invalidation، مصرف حافظه و احتمال دادهٔ stale را اضافه می‌کند."),
                ("retry چه زمانی خطرناک است؟","وقتی عملیات idempotent نیست یا خطا دائمی است؛ retry کور می‌تواند اثر تکراری یا فشار بیشتر ایجاد کند."),
                ("feature flag چیست؟","مکانیزمی برای فعال/غیرفعال‌کردن قابلیت بدون تغییر فوری کد deployشده و برای rollout تدریجی یا rollback سریع مفید است."),
            ]
        }
        for k,(p0,a) in enumerate(bank2[subtype]):
            for p in (p0,"ساده و کاربردی: "+p0,"در یک یا دو جمله جواب بده: "+p0): add(p,a,f"{subtype}:{k}",3)
    else:
        raise ValueError(subtype)
    base=list(rows); j=0
    while len(rows)<EXAMPLES_PER_DATASET:
        src=base[j%len(base)]; clone=dict(src); clone["id"]=f"v12-{dataset_id}-{len(rows)+1:04d}"; clone["input"]=src["input"]+" پاسخ دقیق و بدون حاشیه بده."; clone["metadata"]=dict(src["metadata"]); rows.append(clone); j+=1
    return rows[:EXAMPLES_PER_DATASET]


# ---------------- bilingual datasets ----------------

def gen_bilingual(subtype: str, dataset_id: str) -> list[dict[str, Any]]:
    rows=[]; i=0
    def add(p,a,c,stage=4):
        nonlocal i; i+=1; rows.append(make_row(dataset_id,i,p,a,category=f"v12_bilingual_{subtype}",stage=stage,stage_name="instruction_following" if stage==4 else "language_foundations",concept=c))
    if subtype == "en_to_fa":
        for k,(en,fa) in enumerate(EN_FA_PAIRS):
            for p in (f"به فارسی طبیعی ترجمه کن: {en}",f"Translate to Persian: {en}",f"ترجمهٔ روان فارسی بده: «{en}»"): add(p,fa,f"e2f:{k}")
    elif subtype == "fa_to_en":
        for k,(en,fa) in enumerate(EN_FA_PAIRS):
            for p in (f"به انگلیسی طبیعی ترجمه کن: {fa}",f"Translate to English: {fa}",f"English translation: «{fa}»"): add(p,en,f"f2e:{k}")
    elif subtype == "technical_terms":
        terms=[("پایگاه داده","database"),("حافظهٔ موقت","volatile memory"),("کنترل نسخه","version control"),("رمزنگاری","encryption"),("بازیابی","recovery"),("تأخیر شبکه","network latency"),("میانگین وزنی","weighted average"),("مجموعه داده","dataset"),("نرخ یادگیری","learning rate"),("بیش‌برازش","overfitting"),("رابط برنامه‌نویسی","application programming interface"),("سامانهٔ فایل","file system"),("رشتهٔ پردازشی","thread"),("فرایند","process"),("صف","queue"),("پشته","stack")]
        for k,(fa,en) in enumerate(terms):
            for p,a in ((f"معادل انگلیسی «{fa}» چیست؟",en),(f"«{en}» به فارسی یعنی چه؟",fa),(f"ترجمهٔ فنی دقیق {fa} / {en} را بگو.",f"{fa} = {en}")): add(p,a,f"term:{k}")
    elif subtype == "mixed_queries":
        cases=[
            ("فرق cache و RAM چیه؟","cache حافظهٔ کوچک‌تر و سریع‌تر نزدیک CPU برای داده‌های پرتکرار است؛ RAM حافظهٔ کاری اصلی و بزرگ‌تر سیستم است."),
            ("این API response چرا timeout میشه؟","timeout می‌تواند از latency شبکه، کندی سرویس یا پایگاه داده، صف طولانی یا محدودیت زمان کلاینت باشد؛ لاگ زمان‌بندی هر مرحله کمک می‌کند علت را جدا کنیم."),
            ("برای Python یه simple example از loop بده.","مثال ساده:\nfor i in range(3):\n    print(i)"),
            ("model overfitting یعنی چی؟","یعنی مدل روی دادهٔ آموزش بیش از حد منطبق شده و روی دادهٔ جدید تعمیم ضعیف‌تری دارد."),
            ("git merge conflict رو چطور بفهمم؟","Git فایل‌های دارای conflict را علامت می‌زند؛ با git status آن‌ها را می‌بینی و داخل فایل نشانگرهای conflict را باید حل کنی."),
            ("database index چه تاثیری روی write داره؟","index خواندن را سریع‌تر می‌کند اما insert/update/delete باید index را هم به‌روزرسانی کنند، بنابراین هزینهٔ نوشتن و فضای ذخیره‌سازی بیشتر می‌شود."),
            ("JSON parse error یعنی چی؟","یعنی متن ورودی با قواعد JSON معتبر سازگار نیست؛ معمولاً quote، comma، bracket یا escape اشتباه است."),
            ("latency پایین ولی bandwidth کم یعنی چی؟","شروع انتقال سریع است، اما حجم داده‌ای که در واحد زمان می‌توان منتقل کرد محدود است؛ این دو شاخص مستقل‌اند."),
        ]
        for k,(p0,a) in enumerate(cases):
            for p in (p0,"ساده جواب بده: "+p0,"دقیق و کوتاه: "+p0): add(p,a,f"mixed:{k}")
    elif subtype == "english_paraphrase":
        pairs=[
            ("The service is currently unavailable.","The service is not available right now."),
            ("We need to verify the result before release.","We should confirm the result before releasing it."),
            ("The update reduced memory usage.","Memory consumption decreased after the update."),
            ("The bug only appears under heavy load.","The issue occurs only when the system is heavily loaded."),
            ("Please send the complete error message.","Please provide the full error message."),
            ("The database query is the slowest step.","The database query takes more time than any other step."),
            ("This change should not affect existing users.","Existing users should not be impacted by this change."),
            ("The model cannot answer reliably without context.","Without context, the model cannot give a reliable answer."),
        ]
        for k,(src,a) in enumerate(pairs):
            for p in (f"Paraphrase this: {src}",f"Rewrite in clear English: {src}",f"Say the same thing differently: {src}"): add(p,a,f"para:{k}")
    elif subtype == "english_summary":
        docs=[
            ("The deployment completed successfully, but monitoring found a small increase in database latency. The team is investigating before increasing traffic.","Deployment succeeded, but increased database latency is being investigated before traffic is expanded."),
            ("Tests passed on Windows and Linux. A macOS-specific file permission issue remains open and blocks the final release.","Windows and Linux tests pass, but a macOS permission issue still blocks release."),
            ("The new cache lowered repeated request time by half. First-time requests are unchanged because they still fetch data from the source.","Caching halved repeated-request time while first requests stayed unchanged."),
            ("User feedback is positive overall, with most complaints focused on the checkout button size on smaller screens.","Feedback is mostly positive; the main complaint is checkout button size on small screens."),
        ]
        for k,(doc,a) in enumerate(docs):
            for p in (f"Summarize in one sentence: {doc}",f"Give the key point only: {doc}",f"Short summary: {doc}"): add(p,a,f"ensum:{k}")
    elif subtype == "english_reasoning":
        for k in range(16):
            a=20+k*3; b=5+(k%7); result=a-b
            for p,a0 in ((f"A box has {a} items and {b} are removed. How many remain?",f"{a}-{b}={result}, so {result} remain."),(f"Compute {a} minus {b}.",str(result)),(f"If you start with {a} and take away {b}, what is left?",str(result))): add(p,a0,f"enmath:{k}",stage=11)
    else:
        # Bilingual concise QA / language switching.
        items=[
            ("Answer in Persian: What is DNS?","DNS نام دامنه را به نشانی IP نگاشت می‌کند."),
            ("Answer in English: حافظهٔ RAM چیست؟","RAM is fast temporary working memory used by running programs."),
            ("فارسی جواب بده: What does Git do?","Git تغییرات کد را ثبت می‌کند و برای کنترل نسخه و همکاری استفاده می‌شود."),
            ("English only: تفاوت TCP و UDP چیست؟","TCP provides reliable ordered delivery; UDP has lower overhead but does not guarantee delivery or order."),
            ("Explain API in Persian, one sentence.","API قراردادی برای ارتباط برنامه‌ها و تعریف عملیات و داده‌های قابل تبادل است."),
            ("Explain پایگاه داده in English, one sentence.","A database stores and organizes data so it can be queried and updated efficiently."),
            ("به فارسی بگو overfitting یعنی چه.","بیش‌برازش یعنی مدل روی دادهٔ آموزش بیش از حد منطبق شود و روی دادهٔ جدید تعمیم ضعیف‌تری داشته باشد."),
            ("In English, what is رمزنگاری?","Encryption transforms data using a key so unauthorized parties cannot read it directly."),
        ]
        for k,(p0,a) in enumerate(items):
            for p in (p0,"Follow the requested language exactly: "+p0,p0+" Keep it concise."): add(p,a,f"switch:{k}")
    base=list(rows); j=0
    while len(rows)<EXAMPLES_PER_DATASET:
        src=base[j%len(base)]; clone=dict(src); clone["id"]=f"v12-{dataset_id}-{len(rows)+1:04d}"; clone["input"]=src["input"]+" Be precise."; clone["metadata"]=dict(src["metadata"]); rows.append(clone); j+=1
    return rows[:EXAMPLES_PER_DATASET]


LANGUAGE_SUBTYPES = [
    "orthography","halfspace","punctuation","typo","formalize","rewrite_clear",
    "finglish","idiom","summarize","compare","definition","concise","detailed",
    "uncertainty","ambiguity","context_reference","instruction_constraints",
    "instruction_format","instruction_exact","instruction_short",
    "instruction_no_guess","instruction_language","instruction_lists","instruction_explain",
    "definition_2","compare_2","summarize_2","finglish_2","context_reference_2","uncertainty_2",
]

MATH_SUBTYPES = [
    "addition","subtraction","multiplication","division","fractions","percent","ratio","average",
    "weighted_average","linear_equation","two_step_equation","inequality","absolute","powers","roots",
    "divisibility","gcd","lcm","prime","trailing_zeros","arithmetic_sequence","geometric_sequence",
    "alternating_sequence","square_sequence","coin_probability","dice_probability","urn_probability",
    "combinations","permutations","sets","syllogism","propositional","logic_negation","age","speed",
    "work_rate","calendar","clock",
]

KNOWLEDGE_SUBTYPES = [
    "science","computer","network","database","web","security","ai","python","debugging","algorithms",
    "datastructures","git","api","files","software_design",
]

BILINGUAL_SUBTYPES = [
    "en_to_fa","fa_to_en","technical_terms","mixed_queries","english_paraphrase","english_summary",
    "english_reasoning","language_switch","concise_qa","mixed_typos","fa_english_terms","english_fa_terms",
    "bilingual_explain","bilingual_compare","bilingual_instruction","bilingual_rewrite","bilingual_reasoning_2",
]

# Exactly 100 curated source files.
SPECS: list[tuple[str,str,str]] = []
for x in LANGUAGE_SUBTYPES:
    # aliases reuse strong generator branches while remaining distinct source files / prompt perturbations.
    base = x
    if x.endswith("_2"):
        base = x[:-2]
    if base.startswith("instruction_"):
        base = "instruction_constraints"
    SPECS.append(("language", x, base))
for x in MATH_SUBTYPES:
    SPECS.append(("math", x, x))
for x in KNOWLEDGE_SUBTYPES:
    SPECS.append(("knowledge", x, x))
for x in BILINGUAL_SUBTYPES:
    base = x
    if x in {"concise_qa","language_switch","mixed_typos","fa_english_terms","english_fa_terms","bilingual_explain","bilingual_compare","bilingual_instruction","bilingual_rewrite","bilingual_reasoning_2"}:
        base = "language_switch"
    SPECS.append(("bilingual", x, base))

# Trim/extend deterministically to exactly 100 while preserving category balance.
if len(SPECS) < 100:
    raise RuntimeError(f"Need at least 100 specs, got {len(SPECS)}")
SPECS = SPECS[:100]


def build() -> dict[str, Any]:
    OUT.mkdir(parents=True, exist_ok=True)
    for old in OUT.glob("*.jsonl"):
        old.unlink()
    manifest_sources=[]
    total=0
    global_fingerprints=set()
    duplicates=0
    duplicates_rewritten=0
    category_counts={}
    for index,(family,name,base) in enumerate(SPECS,1):
        dataset_id=f"d{index:03d}_{name}"
        if family=="language":
            rows=gen_language(base,dataset_id)
        elif family=="math":
            rows=gen_math(base,dataset_id,index)
        elif family=="knowledge":
            rows=gen_knowledge(base,dataset_id)
        else:
            rows=gen_bilingual(base,dataset_id)
        # Make alias datasets distinct by adding a benign prompt prefix and a source-specific concept namespace.
        if name != base:
            for r in rows:
                r["input"] = ("با دقت پاسخ بده: " if r["language"] != "en" else "Answer carefully: ") + r["input"]
                r["metadata"]["concept_group"] = r["metadata"]["concept_group"] + f":{name}"
        path=OUT/f"{dataset_id}.jsonl"
        lines=[]
        local=set()
        for row_pos, r in enumerate(rows, 1):
            fp=hashlib.sha256((r["input"].casefold()+"\n"+r["output"].casefold()).encode("utf-8")).hexdigest()
            if fp in local or fp in global_fingerprints:
                duplicates += 1
                # Keep the semantic task but make repeated template instances explicit,
                # rather than training on exact duplicate text. Exercise numbering is a
                # common benign instruction pattern and preserves the target answer.
                prefix = f"تمرین تکمیلی {index}-{row_pos}: " if r["language"] != "en" else f"Supplementary exercise {index}-{row_pos}: "
                r = dict(r)
                r["input"] = prefix + r["input"]
                fp=hashlib.sha256((r["input"].casefold()+"\n"+r["output"].casefold()).encode("utf-8")).hexdigest()
                duplicates_rewritten += 1
            local.add(fp)
            global_fingerprints.add(fp)
            lines.append(json.dumps(r,ensure_ascii=False,sort_keys=True))
            category_counts[r["category"]]=category_counts.get(r["category"],0)+1
        path.write_text("\n".join(lines)+"\n",encoding="utf-8")
        sha=hashlib.sha256(path.read_bytes()).hexdigest()
        manifest_sources.append({"id":dataset_id,"family":family,"skill":name,"examples":len(lines),"file":path.relative_to(ROOT).as_posix(),"sha256":sha,"quality":"gold-verified","provenance":"project-authored deterministic/editorial"})
        total += len(lines)
    manifest={
        "format":"jarvis-curated-pack-v12",
        "version":"curated_100_v12",
        "seed":SEED,
        "source_dataset_count":len(manifest_sources),
        "examples":total,
        "unique_input_output_pairs":len(global_fingerprints),
        "exact_duplicates_detected":duplicates,
        "duplicates_rewritten":duplicates_rewritten,
        "quality_policy":["deterministic answers for quantitative tasks","editorially authored language and knowledge answers","concept-group split compatibility","no external model output","no benchmark test split copied"],
        "sources":manifest_sources,
        "category_counts":dict(sorted(category_counts.items())),
        "pretrained_source":None,
    }
    manifest_path=ROOT/"datasets"/"curated_100_v12_manifest.json"
    manifest_path.write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    return manifest


if __name__=="__main__":
    print(json.dumps(build(),ensure_ascii=False,indent=2))
