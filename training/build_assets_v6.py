from __future__ import annotations

import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]


def _knowledge_entry(
    identifier: str,
    title: str,
    keywords: list[str],
    fa: str,
    en: str,
) -> dict[str, Any]:
    return {
        "id": identifier,
        "title": title,
        "keywords": keywords,
        "answers": {"fa": fa, "en": en},
    }


def build_knowledge() -> dict[str, Any]:
    base_path = ROOT / "data" / "knowledge_v2.json"
    base = json.loads(base_path.read_text(encoding="utf-8"))
    entries = {
        str(item.get("id")): item
        for item in base.get("entries", [])
        if isinstance(item, dict) and item.get("id")
    }
    entries["jarvis-v6-identity"] = _knowledge_entry(
        "jarvis-v6-identity",
        "JARVIS v0.6 identity developer Younes Gohari یونس گوهری",
        ["جارویس", "سازنده", "توسعه دهنده", "یونس گوهری", "version", "developer"],
        "من JARVIS v0.6 هستم؛ Intelligence & Desktop Mastery، توسعه‌یافته توسط یونس گوهری. مدل‌های من در همین پروژه طراحی و از مقداردهی تصادفی روی داده‌های پروژه آموزش داده شده‌اند.",
        "I'm JARVIS v0.6, Intelligence & Desktop Mastery, developed by Younes Gohari. My models were designed here and trained from random initialization on project-authored data.",
    )
    facts = (
        ("human-testicle", "testicle anatomy بیضه چیست دستگاه تولیدمثل", ["بیضه", "testicle", "اسپرم", "تستوسترون", "تولید مثل"], "بیضه یکی از دو غدهٔ تولیدمثلی مردانه است که اسپرم و هورمون‌هایی مانند تستوسترون تولید می‌کند و معمولاً در کیسهٔ بیضه قرار دارد. درد یا تورم ناگهانی و شدید نیازمند ارزیابی فوری پزشکی است.", "A testicle is one of the male reproductive glands that produces sperm and hormones such as testosterone. Sudden severe pain or swelling needs urgent medical assessment."),
        ("sky-blue", "why sky is blue چرا آسمان آبی است Rayleigh scattering", ["آسمان", "آبی", "sky", "blue", "پراکندگی", "Rayleigh"], "مولکول‌های جو طول موج‌های کوتاه‌تر نور مرئی، به‌ویژه آبی، را بیشتر از طول موج‌های بلند پراکنده می‌کنند؛ بنابراین نور آبی از جهت‌های مختلف به چشم می‌رسد.", "Air molecules scatter shorter visible wavelengths, especially blue, more strongly than longer wavelengths, so blue light reaches our eyes from many directions."),
        ("ram", "RAM memory حافظه رم چیست volatile", ["رم", "RAM", "حافظه", "memory", "موقت"], "RAM حافظهٔ سریع و موقتی برای برنامه‌ها و داده‌های در حال استفاده است. ظرفیت بیشتر آن اجرای هم‌زمان برنامه‌های بیشتری را ممکن می‌کند، اما جایگزین SSD نیست.", "RAM is fast, volatile working memory for active programs and data. More capacity helps multitasking, but RAM does not replace persistent storage."),
        ("cpu", "CPU processor پردازنده چیست cores", ["CPU", "پردازنده", "هسته", "processor", "core"], "CPU دستورهای برنامه را اجرا و کار عمومی سیستم را هماهنگ می‌کند. تعداد هسته، کارایی هر هسته، حافظهٔ نهان و محدودیت توان همگی بر عملکرد اثر دارند.", "A CPU executes program instructions and coordinates general computation. Core count, per-core performance, cache, and power limits all affect speed."),
        ("gpu", "GPU graphics پردازنده گرافیکی چیست", ["GPU", "گرافیک", "پردازنده گرافیکی", "parallel"], "GPU تعداد زیادی واحد محاسباتی موازی دارد و برای گرافیک و برخی محاسبات ماتریسی مناسب است. JARVIS v0.6 برای اجرای اصلی به GPU نیاز ندارد.", "A GPU has many parallel compute units suited to graphics and some matrix workloads. JARVIS v0.6 does not require a GPU for its core runtime."),
        ("storage", "SSD HDD storage ذخیره سازی تفاوت", ["SSD", "HDD", "ذخیره", "storage", "NVMe"], "SSD بدون قطعات متحرک است و معمولاً تأخیر کمتر و سرعت بیشتری دارد؛ HDD اغلب برای ظرفیت زیاد ارزان‌تر است. NVMe نوعی رابط سریع SSD روی PCIe است.", "SSDs have no moving parts and usually provide lower latency and higher speed; HDDs are often cheaper for large capacity. NVMe is a fast PCIe interface for SSDs."),
        ("api", "API application programming interface رابط برنامه نویسی", ["API", "رابط", "برنامه نویسی", "interface"], "API یک قرارداد مشخص است که نرم‌افزار از طریق آن داده یا رفتار جزء دیگری را درخواست می‌کند، بدون اینکه به جزئیات داخلی آن وابسته باشد.", "An API is a defined contract through which software requests data or behavior from another component without depending on its internals."),
        ("rest", "REST API HTTP resources", ["REST", "HTTP", "resource", "منبع"], "REST سبکی برای طراحی API بر پایهٔ منبع، درخواست‌های stateless و معناهای استاندارد HTTP است؛ هر API مبتنی بر HTTP الزاماً RESTful نیست.", "REST is an API design style based on resources, stateless requests, and standard HTTP semantics; not every HTTP API is RESTful."),
        ("dns", "DNS domain name system دامنه آی پی", ["DNS", "دامنه", "domain", "IP", "نام"], "DNS نام دامنه را به رکوردهای شبکه مانند IP نگاشت می‌کند. پاسخ‌ها سلسله‌مراتبی‌اند و برای کاهش تأخیر cache می‌شوند.", "DNS maps domain names to network records such as IP addresses. Its hierarchy and caching reduce repeated lookup cost."),
        ("http-https", "HTTP HTTPS TLS security", ["HTTP", "HTTPS", "TLS", "امنیت", "وب"], "HTTPS همان HTTP روی TLS است و برای ارتباط، رمزنگاری مسیر، یکپارچگی و احراز هویت سرور را فراهم می‌کند؛ امنیت خود برنامه هنوز نیازمند طراحی درست است.", "HTTPS carries HTTP over TLS to provide transport encryption, integrity, and server authentication; application security still requires correct design."),
        ("ip-address", "IP address آدرس آی پی IPv4 IPv6", ["IP", "IPv4", "IPv6", "آدرس", "شبکه"], "آدرس IP شناسهٔ مسیریابی یک رابط شبکه است. IPv4 آدرس ۳۲ بیتی و IPv6 آدرس ۱۲۸ بیتی دارد؛ آدرس عمومی و خصوصی کاربرد متفاوت دارند.", "An IP address identifies a network interface for routing. IPv4 uses 32 bits and IPv6 uses 128; public and private addresses serve different scopes."),
        ("sqlite", "SQLite database پایگاه داده", ["SQLite", "دیتابیس", "database", "SQL", "تراکنش"], "SQLite یک پایگاه‌دادهٔ رابطه‌ای تعبیه‌شده و بدون سرور است که معمولاً در یک فایل ذخیره می‌شود و از تراکنش پشتیبانی می‌کند.", "SQLite is an embedded, serverless relational database usually stored in one file, with transactional guarantees."),
        ("json", "JSON data format ساختار داده", ["JSON", "داده", "فرمت", "object", "array"], "JSON قالب متنی برای object، array، رشته، عدد، boolean و null است. برای تنظیمات کوچک مناسب است اما جای تراکنش و query پایگاه‌داده را نمی‌گیرد.", "JSON is a text format for objects, arrays, strings, numbers, booleans, and null. It suits small configuration data but does not provide database transactions or queries."),
        ("python", "Python programming language پایتون چیست", ["پایتون", "Python", "زبان", "برنامه نویسی"], "پایتون زبان برنامه‌نویسی سطح‌بالا با خوانایی زیاد و کتابخانهٔ استاندارد گسترده است که برای اتوماسیون، وب، علم و ابزارهای دسکتاپ استفاده می‌شود.", "Python is a readable high-level programming language with a broad standard library, used for automation, web, science, and desktop tooling."),
        ("oop", "object oriented programming OOP شی گرایی", ["OOP", "شی گرایی", "class", "object", "کلاس"], "برنامه‌نویسی شی‌گرا داده و رفتار مرتبط را در objectها سازمان می‌دهد. encapsulation، composition و polymorphism ابزارند، نه الزام برای هر مسئله.", "Object-oriented programming organizes related state and behavior in objects. Encapsulation, composition, and polymorphism are tools rather than requirements for every problem."),
        ("algorithm", "algorithm الگوریتم complexity پیچیدگی", ["الگوریتم", "algorithm", "پیچیدگی", "Big O"], "الگوریتم دنباله‌ای مشخص از گام‌ها برای حل مسئله است. پیچیدگی زمانی و فضایی چگونگی رشد هزینه با اندازهٔ ورودی را توصیف می‌کند.", "An algorithm is a defined sequence of steps for solving a problem. Time and space complexity describe how resource cost grows with input size."),
        ("process-thread", "process thread تفاوت پردازش رشته", ["process", "thread", "پردازش", "رشته", "race condition"], "Process فضای آدرس جدا دارد؛ threadهای یک process حافظه را به اشتراک می‌گذارند. اشتراک حافظه سبک‌تر است اما synchronization برای جلوگیری از race condition لازم می‌شود.", "Processes have separate address spaces; threads in one process share memory. Sharing is lighter but requires synchronization to prevent race conditions."),
        ("transformer", "Transformer attention مدل عصبی", ["Transformer", "attention", "ترنسفورمر", "توکن", "neural"], "Transformer معماری عصبی مبتنی بر attention است که روابط میان توکن‌ها را مدل می‌کند. اندازهٔ مدل، داده و روش آموزش همگی کیفیت را تعیین می‌کنند.", "A Transformer is a neural architecture based on attention for modeling relationships among tokens. Model size, data, and training procedure all affect quality."),
        ("neural-network", "neural network شبکه عصبی training inference", ["شبکه عصبی", "neural network", "training", "inference", "وزن"], "شبکهٔ عصبی تابعی پارامتری است که وزن‌هایش هنگام آموزش برای کاهش خطا تنظیم می‌شوند؛ inference از وزن‌های آموخته‌شده برای خروجی جدید استفاده می‌کند.", "A neural network is a parameterized function whose weights are adjusted during training to reduce error; inference uses those learned weights on new input."),
        ("machine-learning", "machine learning یادگیری ماشین", ["یادگیری ماشین", "machine learning", "داده", "مدل"], "یادگیری ماشین روش ساخت مدل‌هایی است که از داده الگو می‌آموزند. کیفیت داده، split درست و ارزیابی روی نمونهٔ دیده‌نشده برای ادعای عملکرد ضروری است.", "Machine learning builds models that learn patterns from data. Data quality, proper splits, and evaluation on unseen examples are essential for performance claims."),
        ("black-hole", "black hole سیاهچاله event horizon", ["سیاهچاله", "black hole", "افق رویداد", "گرانش"], "سیاهچاله ناحیه‌ای از فضا-زمان با گرانش بسیار شدید و افق رویداد است؛ پس از افق رویداد مسیر خروجی برای نور وجود ندارد.", "A black hole is a region of spacetime with extreme gravity and an event horizon beyond which light has no outward path."),
        ("solar-system", "solar system منظومه شمسی planets", ["منظومه شمسی", "خورشید", "سیاره", "solar system"], "منظومهٔ شمسی شامل خورشید، هشت سیاره، سیاره‌های کوتوله، قمرها و اجرام کوچک‌تر است. ترتیب سیاره‌ها از خورشید با عطارد آغاز می‌شود و به نپتون می‌رسد.", "The Solar System contains the Sun, eight planets, dwarf planets, moons, and smaller bodies. The planets run from Mercury outward to Neptune."),
        ("photosynthesis", "photosynthesis فتوسنتز گیاه", ["فتوسنتز", "photosynthesis", "گیاه", "نور", "کلروفیل"], "در فتوسنتز، گیاهان و برخی جانداران انرژی نور را برای ساخت مواد آلی از دی‌اکسیدکربن و آب به کار می‌گیرند و معمولاً اکسیژن آزاد می‌کنند.", "In photosynthesis, plants and some organisms use light energy to build organic molecules from carbon dioxide and water, commonly releasing oxygen."),
        ("dna", "DNA ژن ماده وراثتی", ["DNA", "ژن", "وراثت", "genetic"], "DNA مولکول اصلی ذخیرهٔ اطلاعات وراثتی در بسیاری از جانداران است. ژن بخشی از DNA است که می‌تواند در ساخت محصولی عملکردی نقش داشته باشد.", "DNA is the primary hereditary information molecule in many organisms. A gene is a DNA region that can contribute to a functional product."),
        ("cell", "cell biology سلول زیست شناسی", ["سلول", "cell", "زیست", "غشا", "هسته"], "سلول واحد بنیادی ساختار و عملکرد جانداران است. سلول‌های یوکاریوتی هسته و اندامک‌های غشادار دارند؛ پروکاریوت‌ها هستهٔ غشادار ندارند.", "The cell is the fundamental structural and functional unit of life. Eukaryotic cells have a nucleus and membrane-bound organelles; prokaryotes do not have a membrane-bound nucleus."),
        ("water-cycle", "water cycle چرخه آب", ["چرخه آب", "تبخیر", "بارش", "water cycle"], "چرخهٔ آب شامل تبخیر و تعرق، میعان، بارش، نفوذ و رواناب است و آب را میان جو، سطح و زیرزمین جابه‌جا می‌کند.", "The water cycle includes evaporation and transpiration, condensation, precipitation, infiltration, and runoff, moving water among atmosphere, surface, and ground."),
        ("electricity", "electricity voltage current برق ولتاژ جریان", ["برق", "ولتاژ", "جریان", "مقاومت", "electricity"], "ولتاژ اختلاف پتانسیل الکتریکی، جریان نرخ عبور بار و مقاومت میزان مخالفت با جریان است. قانون اهم در شرایط مناسب رابطهٔ V=IR را بیان می‌کند.", "Voltage is electric potential difference, current is charge flow rate, and resistance opposes current. Under suitable conditions, Ohm's law is V=IR."),
        ("encryption-hash", "encryption hashing رمزنگاری هش", ["رمزنگاری", "هش", "encryption", "hash", "password"], "رمزنگاری با کلید برای بازگردانی مجاز داده طراحی می‌شود؛ هش یک‌طرفه است. گذرواژه باید با الگوریتم مشتق‌سازی کند و salt تصادفی ذخیره شود، نه هش سریع عمومی.", "Encryption uses a key for authorized recovery; hashing is one-way. Passwords should use a slow password KDF with a random salt, not a fast general-purpose hash."),
        ("git", "Git version control گیت کنترل نسخه", ["Git", "گیت", "commit", "branch", "کنترل نسخه"], "Git سامانهٔ کنترل نسخهٔ توزیع‌شده است. commit یک snapshot منطقی ثبت می‌کند و branch اشاره‌گری متحرک به تاریخچه است.", "Git is a distributed version-control system. A commit records a logical snapshot, and a branch is a movable reference into history."),
        ("zip", "ZIP archive فشرده سازی zip bomb", ["ZIP", "فشرده", "آرشیو", "zip bomb"], "ZIP ظرفی برای چند فایل فشرده یا بدون فشرده‌سازی است. پیش از extract باید مسیر member، حجم بازشده و نسبت فشرده‌سازی محدود شود تا zip-slip و zip-bomb رخ ندهد.", "ZIP is a container for compressed or stored files. Before extraction, member paths, expanded size, and compression ratio should be bounded to prevent zip-slip and zip-bomb attacks."),
        ("sqlite-json", "SQLite versus JSON comparison", ["SQLite", "JSON", "مقایسه", "settings", "history"], "برای تنظیمات کوچک و خواندنی JSON ساده است؛ برای تاریخچه، query، چند جدول و ثبت اتمیک SQLite مناسب‌تر است. انتخاب به الگوی دسترسی بستگی دارد.", "JSON is simple for tiny human-readable settings; SQLite is better for history, queries, multiple tables, and atomic updates. The access pattern should drive the choice."),
        ("internet-current", "current latest information internet search freshness", ["آخرین", "امروز", "قیمت", "هوا", "latest", "current", "today"], "اطلاعات متغیر مانند نسخهٔ جدید، قیمت، هوا و خبر باید هنگام درخواست از منبع زنده بررسی و با تاریخ و منبع گزارش شود؛ دانش آفلاین برای آن کافی نیست.", "Changing facts such as releases, prices, weather, and news should be checked against live sources at request time and reported with dates and sources; offline knowledge is insufficient."),
    )
    for identifier, title, keywords, fa, en in facts:
        entries[identifier] = _knowledge_entry(identifier, title, keywords, fa, en)
    payload = {"version": 6, "entries": list(entries.values())}
    output = ROOT / "data" / "knowledge_v3.json"
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return {"entries": len(payload["entries"]), "path": str(output)}


def build_responses() -> dict[str, Any]:
    payload = json.loads((ROOT / "data" / "responses_v2.json").read_text(encoding="utf-8"))
    payload["version"] = 6
    intents = payload.setdefault("intents", {})
    intents["ask_name"] = {
        "fa": ["من JARVIS v0.6 هستم؛ Intelligence & Desktop Mastery، توسعه‌یافته توسط یونس گوهری."],
        "en": ["I'm JARVIS v0.6—Intelligence & Desktop Mastery, developed by Younes Gohari."],
    }
    intents["self_intro"] = {
        "fa": ["من JARVIS v0.6 هستم؛ یک Agent دسکتاپ محلی با Brain عصبی کوچک، حافظه، دانش، برنامه‌ریز و ابزارهای کنترل Windows. توسط یونس گوهری توسعه داده شده‌ام."],
        "en": ["I'm JARVIS v0.6, a local desktop agent with a small neural brain, memory, knowledge, planning, and Windows tools, developed by Younes Gohari."],
    }
    intents["capabilities"] = {
        "fa": ["می‌تونم گفتگو و Context را حفظ کنم، فایل و ZIP را بررسی کنم، دانش محلی را پاسخ بدهم، در وب تحقیق کنم و با تأییدهای امنیتی برنامه‌ها، پنجره‌ها، مرورگر و بخش‌های Windows را کنترل کنم."],
        "en": ["I can chat with context, inspect files and ZIPs, answer local knowledge, research the web, and—with safety confirmations—control apps, windows, browsers, and selected Windows functions."],
    }
    intents["tell_joke"] = {
        "fa": ["برنامه‌نویس به باگ گفت: «تو چرا همیشه برمی‌گردی؟» باگ گفت: «چون من feature نسل بعدم!»", "چرا کامپیوتر رفت دکتر؟ چون حافظه‌اش هی کم می‌شد!"],
        "en": ["Why did the computer visit the doctor? It kept losing its memory.", "A programmer's favorite place is the cache—everything feels familiar there."],
    }
    intents["short_story"] = {
        "fa": ["نیمه‌شب، چراغ کوچک کیس روشن ماند. جارویس صدای فن را شنید و فهمید یک فایل گمشده هنوز میان پوشه‌ها منتظر است. مسیر را پیدا کرد، اما پیش از بازکردنش پرسید: «همین فایل را می‌خواستی؟» و آن شب، احتیاط قهرمان داستان شد."],
        "en": ["At midnight, one small case light stayed awake. Jarvis traced a missing file through quiet folders, but before opening it asked, “Is this the one?” That night, caution became the hero."],
    }
    intents.setdefault("tool_confirm_dangerous", {
        "fa": ["این عمل می‌تواند حالت سیستم یا داده را تغییر دهد. اگر دقیقاً همین کار را می‌خواهی، صریحاً تأیید کن."],
        "en": ["This action can change system state or data. Explicitly confirm if this exact action is intended."],
    })
    output = ROOT / "data" / "responses_v3.json"
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return {"intents": len(intents), "path": str(output)}


def build_hybrid_intents() -> dict[str, Any]:
    intents: dict[str, list[str]] = {
        "minimize_app": ["کروم رو مینیمایز کن", "فایرفاکس رو جمع کن", "تلگرام رو بفرست تسک بار", "پنجره نوت پد رو کوچک کن", "minimize Chrome", "minimize Firefox", "send Notepad to taskbar", "minimize that app", "برنامه رو مینیمایز کن", "دیسکورد رو جمعش کن", "minimize VS Code", "اسپاتیفای رو مینیمایز کن"],
        "maximize_app": ["کروم رو تمام صفحه کن", "فایرفاکس رو بزرگ کن", "پنجره تلگرام رو maximize کن", "نوت پد رو فول اسکرین کن", "maximize Chrome", "make Firefox full screen", "maximize that app", "maximize VS Code", "برنامه رو بزرگ کن", "دیسکورد رو تمام صفحه کن", "maximize Notepad", "اسپاتیفای رو بزرگ کن"],
        "restore_app": ["کروم رو از مینیمایز دربیار", "پنجره فایرفاکس رو برگردون", "تلگرام رو restore کن", "نوت پد رو برگردون", "restore Chrome", "unminimize Firefox", "restore that app", "restore VS Code window", "برنامه رو به حالت عادی برگردون", "دیسکورد رو restore کن", "restore Notepad", "اسپاتیفای رو برگردون"],
        "create_folder": ["روی دسکتاپ پوشه بساز", "یک فولدر جدید ایجاد کن", "پوشه Jarvis Test رو بساز", "داخل Documents پوشه بساز", "create a folder on Desktop", "make a new directory", "create folder Jarvis Test", "make a folder in Documents", "پوشه جدید درست کن", "یک directory بساز", "create new folder", "build a folder here"],
        "create_file": ["فایل notes.txt رو بساز", "یک فایل جدید ایجاد کن", "روی دسکتاپ فایل text بساز", "فایل پایتون بساز", "create notes.txt", "make a new file", "create a text file", "create Python file", "فایل جدید درست کن", "یک document بساز", "create file here", "make README.md"],
        "delete_file": ["این فایل رو حذف کن", "فایل notes.txt رو پاک کن", "این سند رو delete کن", "فایل انتخاب شده را حذف کن", "delete this file", "remove notes.txt", "delete the selected document", "erase this file", "فایل رو برای حذف آماده کن", "پاک کردن فایل", "remove this file", "delete report.csv"],
        "enumerate_windows": ["لیست پنجره ها رو نشون بده", "همه پنجره های باز", "فهرست window ها", "چه پنجره هایی بازه", "list visible windows", "show open windows", "enumerate windows", "which windows are open", "لیست پنجره های ویندوز", "پنجره ها رو بشمار", "show all windows", "list desktop windows"],
        "battery_info": ["وضعیت باتری رو بگو", "باتری چند درصده", "اطلاعات باتری", "آیا لپتاپ شارژ میشه", "show battery information", "battery status", "what is the battery percentage", "is the battery charging", "مشخصات باتری", "باتری وصله به برق", "check battery", "read battery status"],
        "network_info": ["اطلاعات شبکه رو بگو", "وضعیت network", "آدرس IP سیستم", "مشخصات اتصال شبکه", "show network information", "network status", "show local IP", "inspect the network", "شبکه رو بررسی کن", "اطلاعات اتصال", "check network info", "list network addresses"],
        "run_command": ["دستور python --version رو اجرا کن", "command git status", "فرمان hostname را اجرا کن", "دستور whoami", "run command python --version", "execute git status", "run hostname", "execute whoami", "ترمینال دستور رو اجرا کن", "این command را بزن", "run a terminal command", "execute this command"],
        "web_research": ["درباره آخرین پایتون تحقیق کن", "از چند منبع اینترنتی بررسی کن", "تحقیق اینترنتی انجام بده", "این موضوع را cross check کن", "research the latest Python release", "cross-check this on the web", "do web research", "research from multiple sources", "در اینترنت تحقیق کن", "منابع مختلف را بررسی کن", "research this topic", "verify current information online"],
    }
    payload = {
        "version": 6,
        "intents": [
            {"tag": tag, "route_type": "tool", "examples": examples}
            for tag, examples in intents.items()
        ],
    }
    output = ROOT / "data" / "training" / "intelligence_v6.json"
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return {"intents": len(intents), "examples": sum(len(value) for value in intents.values())}


def build() -> dict[str, Any]:
    return {
        "knowledge": build_knowledge(),
        "responses": build_responses(),
        "hybrid_intents": build_hybrid_intents(),
    }


if __name__ == "__main__":
    print(json.dumps(build(), ensure_ascii=False, indent=2))
