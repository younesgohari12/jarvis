from __future__ import annotations

import hashlib
import json
import random
import sys
from collections import Counter
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from jarvis.neural.tokenizer import normalize_language_text  # noqa: E402


VERSION = "dataset_v001"
SEED = 5052026


def _json_line(value: dict[str, Any]) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(_json_line(row) + "\n" for row in rows), encoding="utf-8")


def build() -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    seen: set[tuple[str, str, str]] = set()

    def add(
        category: str,
        language: str,
        stage: int,
        input_text: str,
        output_text: str,
        **extra: Any,
    ) -> None:
        source = normalize_language_text(input_text)
        target = normalize_language_text(output_text)
        key = (category, source.casefold(), target.casefold())
        if not source or not target or key in seen:
            return
        seen.add(key)
        row: dict[str, Any] = {
            "id": f"jv5-{len(rows) + 1:06d}",
            "dataset_version": VERSION,
            "category": category,
            "language": language,
            "stage": stage,
            "input": source,
            "output": target,
            "origin": "jarvis-authored",
            "pretrained_source": None,
        }
        row.update(extra)
        rows.append(row)

    greetings_fa = [
        "سلام", "سلام جارویس", "درود", "صبحت بخیر", "عصرت بخیر", "شب بخیر",
        "سلام خوبی", "چه خبر جارویس", "هی جارویس", "سلام رفیق", "salam", "salam jarvis",
    ]
    greetings_en = [
        "hello", "hi jarvis", "good morning", "good evening", "hey", "hello there",
        "how are you", "are you ready", "what's up jarvis", "nice to meet you",
    ]
    answers_fa = [
        "سلام، من اینجام. چه کاری انجام بدم؟", "درود، آماده‌ام. از کجا شروع کنیم؟",
        "سلام یونس، گوش می‌دم.", "خوبم و آماده‌ام؛ تو چطوری؟",
    ]
    answers_en = [
        "Hello. I'm here and ready to help.", "Hi, Younes. What should we work on?",
        "I'm ready. Tell me what you need.", "I'm doing well. How are you?",
    ]
    for index, prompt in enumerate(greetings_fa):
        for answer in answers_fa:
            add("persian_conversation", "fa", 1 if index < 6 else 2, prompt, answer)
    for index, prompt in enumerate(greetings_en):
        for answer in answers_en:
            add("english_conversation", "en", 1 if index < 5 else 2, prompt, answer)

    identity = [
        ("اسمت چیه؟", "من JARVIS هستم؛ هوش دسکتاپی که توسط یونس گوهری توسعه داده شده‌ام.", "fa"),
        ("کی ساختت؟", "توسط یونس گوهری توسعه داده شده‌ام.", "fa"),
        ("سازنده ات کیه", "توسعه‌دهندهٔ من یونس گوهری است.", "fa"),
        ("خودتو معرفی کن", "من JARVIS، یک Neural Desktop Intelligence مستقل و قابل‌آموزش هستم.", "fa"),
        ("what is your name?", "I'm JARVIS, a trainable neural desktop intelligence.", "en"),
        ("who developed you?", "I was developed by Younes Gohari.", "en"),
        ("tell me about yourself", "I'm JARVIS. My own neural models work with memory, tools, local knowledge, and research.", "en"),
    ]
    for prompt, answer, language in identity:
        for prefix in ("", "لطفاً " if language == "fa" else "please ", "میشه " if language == "fa" else "can you "):
            add("model_identity", language, 8, prefix + prompt, answer)

    knowledge = [
        ("پایتون چیست؟", "پایتون یک زبان برنامه‌نویسی سطح‌بالا و خواناست که برای اتوماسیون، وب، علم داده و هوش مصنوعی استفاده می‌شود.", "fa"),
        ("چرا آسمان آبی است؟", "نور آبی در جو بیشتر از طول موج‌های بلند پراکنده می‌شود؛ به همین دلیل آسمان روز آبی دیده می‌شود.", "fa"),
        ("رم چه کاری انجام میده؟", "RAM حافظهٔ سریع و موقت سیستم برای نگهداری برنامه‌ها و داده‌های در حال استفاده است.", "fa"),
        ("فرق SSD و HDD چیه؟", "SSD بدون قطعهٔ مکانیکی و سریع‌تر است؛ HDD معمولاً ارزان‌تر و برای ظرفیت بالا مناسب‌تر است.", "fa"),
        ("REST API چیست؟", "REST API رابطی مبتنی بر منابع و عملیات استاندارد HTTP برای ارتباط بین نرم‌افزارها است.", "fa"),
        ("سیاهچاله چیه؟", "سیاهچاله ناحیه‌ای از فضا با گرانش بسیار شدید است که پس از افق رویداد حتی نور هم از آن فرار نمی‌کند.", "fa"),
        ("what is Python?", "Python is a readable high-level programming language used for automation, web development, data, and AI.", "en"),
        ("why is the sky blue?", "Air molecules scatter shorter blue wavelengths more strongly than longer visible wavelengths.", "en"),
        ("what does RAM do?", "RAM is fast temporary memory that holds active programs and working data.", "en"),
        ("what is a REST API?", "A REST API exposes resources through standard HTTP methods and stateless requests.", "en"),
    ]
    question_frames_fa = ("{}", "لطفاً توضیح بده {}", "ساده بگو {}", "میشه بگی {}", "کوتاه بگو {}")
    question_frames_en = ("{}", "please explain {}", "briefly explain {}", "can you tell me {}", "in simple terms, {}")
    for question, answer, language in knowledge:
        for frame in question_frames_fa if language == "fa" else question_frames_en:
            add("general_knowledge", language, 2, frame.format(question), answer)

    programming = [
        ("یک تابع پایتون برای جمع دو عدد بنویس", "```python\ndef add(a: float, b: float) -> float:\n    return a + b\n```", "fa"),
        ("چرا type hint خوبه؟", "Type hint قرارداد ورودی و خروجی را روشن می‌کند و تحلیل ایستا، تکمیل کد و نگهداری را بهتر می‌کند.", "fa"),
        ("SQLite یا JSON برای تاریخچه چت؟", "برای تاریخچهٔ قابل‌جستجو و نوشتن اتمیک SQLite مناسب‌تر است؛ JSON برای تنظیمات کوچک ساده‌تر است.", "fa"),
        ("write a Python function that adds two numbers", "```python\ndef add(a: float, b: float) -> float:\n    return a + b\n```", "en"),
        ("when should I use SQLite?", "Use SQLite for structured local data that needs queries, indexes, transactions, or concurrent-safe writes.", "en"),
        ("what is a race condition?", "A race condition occurs when behavior depends on the timing of unsynchronized concurrent operations.", "en"),
    ]
    for prompt, answer, language in programming:
        for suffix in ("", " با مثال" if language == "fa" else " with an example", " کوتاه" if language == "fa" else " briefly"):
            add("programming", language, 2, prompt + suffix, answer)

    apps = {
        "chrome": ("کروم", "Chrome"), "edge": ("مایکروسافت اج", "Microsoft Edge"),
        "firefox": ("فایرفاکس", "Firefox"), "steam": ("استیم", "Steam"),
        "discord": ("دیسکورد", "Discord"), "telegram": ("تلگرام", "Telegram"),
        "spotify": ("اسپاتیفای", "Spotify"), "vscode": ("وی اس کد", "VS Code"),
        "notepad": ("نوت پد", "Notepad"), "calculator": ("ماشین حساب", "Calculator"),
        "explorer": ("فایل اکسپلورر", "File Explorer"), "powershell": ("پاورشل", "PowerShell"),
        "cmd": ("سی ام دی", "Command Prompt"), "paint": ("پینت", "Paint"),
    }
    open_fa = ("{name} رو باز کن", "{name}و بیار", "بزن {name}", "برنامه {name} رو اجرا کن", "یه {name} باز کن")
    close_fa = ("{name} رو ببند", "{name}و ببند", "برنامه {name} رو ببند", "{name} رو terminate کن", "از {name} خارج شو")
    open_en = ("open {name}", "launch {name}", "start {name}", "run the {name} app")
    close_en = ("close {name}", "quit {name}", "terminate {name}", "close the {name} app")
    for app, (fa_name, en_name) in apps.items():
        for template in open_fa:
            prompt = template.format(name=fa_name)
            add("tool_calling", "fa", 4, prompt, f"<tool>open_app app={app}", action="open_app", arguments={"app": app})
            add("instruction_following", "fa", 3, "لطفاً " + prompt, f"<tool>open_app app={app}", action="open_app", arguments={"app": app})
        for template in close_fa:
            prompt = template.format(name=fa_name)
            add("tool_calling", "fa", 4, prompt, f"<tool>close_app app={app}", action="close_app", arguments={"app": app})
        for template in open_en:
            prompt = template.format(name=en_name)
            add("tool_calling", "en", 4, prompt, f"<tool>open_app app={app}", action="open_app", arguments={"app": app})
        for template in close_en:
            prompt = template.format(name=en_name)
            add("tool_calling", "en", 4, prompt, f"<tool>close_app app={app}", action="close_app", arguments={"app": app})

    drive_words = {
        "C": ("سی", "c"), "D": ("دی", "d"), "E": ("ای", "e"),
        "F": ("اف", "f"), "G": ("جی", "g"),
    }
    for drive, (fa_word, en_word) in drive_words.items():
        path = f"{drive}:\\"
        prompts = (
            f"درایو {drive} رو باز کن", f"درایو {fa_word} رو بیار", f"برو تو درایو {drive}",
            f"{drive} رو باز کن", f"open drive {en_word}", f"go to {drive} drive",
        )
        for prompt in prompts:
            language = "en" if prompt.startswith(("open", "go")) else "fa"
            add("tool_calling", language, 4, prompt, f"<tool>open_folder path={path}", action="open_folder", arguments={"path": path})

    folders = {
        "desktop": "دسکتاپ", "downloads": "دانلودها", "documents": "اسناد",
        "pictures": "تصاویر", "videos": "ویدیوها", "music": "موزیک",
        "home": "خانه", "appdata": "اپ دیتا", "temp": "تمپ",
    }
    for folder, fa_name in folders.items():
        for prompt in (f"{fa_name} رو باز کن", f"پوشه {fa_name} رو بیار", f"open {folder}", f"show my {folder} folder"):
            language = "en" if prompt.startswith(("open", "show")) else "fa"
            add("tool_calling", language, 4, prompt, f"<tool>open_folder known={folder}", action="open_folder", arguments={"known_folder": folder})

    queries = [
        "یونس گوهری", "چت جی پی تی", "آموزش پایتون", "هوش مصنوعی", "قیمت بیت کوین",
        "OpenAI", "Python tutorial", "JARVIS", "Windows 11", "latest AI news",
    ]
    browsers = (("chrome", "کروم"), ("edge", "اج"), ("firefox", "فایرفاکس"))
    for browser, label in browsers:
        for query in queries:
            for connector in ("و", "بعد", "بعدش", "سپس"):
                prompt = f"{label} رو باز کن {connector} {query} رو سرچ کن داخلش"
                plan = [
                    {"tool": "open_app", "arguments": {"app": browser}},
                    {"tool": "web_search", "arguments": {"query": query, "target_browser": browser}},
                ]
                add("multi_step_tasks", "mixed", 6, prompt, f"<tool_plan>{json.dumps(plan, ensure_ascii=False)}", plan=plan)
            prompt_en = f"open {browser} and search for {query} in it"
            plan_en = [
                {"tool": "open_app", "arguments": {"app": browser}},
                {"tool": "web_search", "arguments": {"query": query, "target_browser": browser}},
            ]
            add("multi_step_tasks", "en", 6, prompt_en, f"<tool_plan>{json.dumps(plan_en)}", plan=plan_en)

    context_pairs = []
    for app, (fa_name, en_name) in apps.items():
        context_pairs.extend(
            [
                (f"{fa_name} رو باز کن", "ببندش", "fa", app),
                (f"open {en_name}", "close it", "en", app),
                (f"{fa_name} رو باز کن", "همونو ببند", "fa", app),
                (f"open {en_name}", "restart it", "en", app),
            ]
        )
    for first, followup, language, app in context_pairs:
        action = "restart_app" if "restart" in followup else "close_app"
        conversation = [
            {"role": "user", "content": first},
            {"role": "assistant", "content": f"<tool>open_app app={app}"},
            {"role": "user", "content": followup},
        ]
        add(
            "context_followup", language, 5, followup, f"<tool>{action} app={app}",
            context=conversation, action=action, arguments={"app": app},
        )

    for drive in drive_words:
        for folder in ("Projects", "Users", "Games", "Younes Gohari", "Jarvis"):
            prompt = f"فولدر {folder} رو داخل درایو {drive} پیدا کن"
            add(
                "entity_arguments", "mixed", 4, prompt,
                f"<tool>find_folder root={drive}:\\ name={folder}",
                action="find_folder", arguments={"root": f"{drive}:\\", "name": folder},
            )
            add(
                "multi_step_tasks", "mixed", 6,
                f"درایو {drive} رو باز کن و فولدر {folder} رو داخلش پیدا کن",
                f"<tool_plan>open_folder {drive}:\\; find_folder {folder}",
                plan=[
                    {"tool": "open_folder", "arguments": {"path": f"{drive}:\\"}},
                    {"tool": "find_folder", "arguments": {"root": f"{drive}:\\", "name": folder}},
                ],
            )

    personality_prompts = {
        "Normal": ("مود رو عادی کن", "Personality روی Normal تنظیم شد."),
        "Kind": ("مود رو مهربون کن", "با کمال میل؛ Personality روی Kind تنظیم شد. ❤️"),
        "Angry": ("مود رو عصبانی کن", "باشه، مود Angry فعال شد."),
        "Loti": ("مود رو لوتی کن", "روی چشم داش؛ مود Loti فعال شد."),
        "Gang": ("مود گنگ رو فعال کن", "Gang mode is active."),
        "Funny": ("مود خنده دار رو بزن", "مود Funny فعال شد؛ بریم که داشته باشیم!"),
        "Professional": ("حرفه ای صحبت کن", "Professional mode is now active."),
    }
    for personality, (prompt, answer) in personality_prompts.items():
        for variant in (prompt, "لطفاً " + prompt, prompt + " جارویس", prompt.replace("رو", "و")):
            add("personality", "fa", 8, variant, answer, action="set_personality", arguments={"personality": personality})

    reasoning_items = [
        ("برای تنظیمات کوچک JSON بهتره یا SQLite؟", "اگر داده کم و یکپارچه است JSON ساده‌تر است؛ اگر query، transaction یا رشد داده مهم است SQLite انتخاب مطمئن‌تری است."),
        ("سیستم 4 گیگ رم برای مدل 10 میلیون پارامتری کافیه؟", "برای inference یک مدل 10M پارامتری معمولاً کافی است، اما training به optimizer و activation memory بیشتری نیاز دارد؛ batch و sequence را کوچک نگه دار."),
        ("اول بکاپ بگیرم یا migration اجرا کنم؟", "اول بکاپ قابل‌بازیابی بگیر، سپس migration را روی clone تست کن و بعد با rollback plan روی production اجرا کن."),
        ("آیا نتیجه فردا را قطعی میدانی؟", "نه. درباره آیندهٔ قطعی نباید ادعا ساخت؛ فقط می‌توان بر اساس شواهد فعلی احتمال یا سناریو ارائه کرد."),
        ("Which is safer for chat history, JSON or SQLite?", "SQLite is generally safer for growing chat history because it supports transactions, indexes, and atomic writes."),
        ("Should I optimize before measuring?", "Measure first, identify the actual bottleneck, then optimize and benchmark again."),
    ]
    for prompt, answer in reasoning_items:
        language = "en" if prompt.isascii() else "fa"
        for prefix in ("", "تحلیل کن: " if language == "fa" else "analyze: ", "با دلیل بگو: " if language == "fa" else "explain with reasons: "):
            add("reasoning", language, 7, prefix + prompt, answer, reasoning_mode="think")

    research_topics = [
        "آخرین نسخه پایتون", "قیمت بیت کوین", "هوای امروز تهران", "خبرهای امروز هوش مصنوعی",
        "نتیجه بازی امشب", "قیمت RTX", "latest Python release", "current Bitcoin price",
        "today's AI news", "current weather in Skopje",
    ]
    for topic in research_topics:
        language = "en" if topic.isascii() else "fa"
        for frame in (("درباره {} تحقیق کن", "از اینترنت {} رو بررسی کن", "{} رو پیدا کن") if language == "fa" else ("research {}", "check the web for {}", "find current information about {}")):
            prompt = frame.format(topic)
            add("search_decision", language, 7, prompt, f"<research>query={topic}", action="research", arguments={"query": topic})

    unknowns = [
        "zxqv plmokn 773", "فلان چیز عجیب رو انجام بده", "راز نگفته من چیه", "نتیجه قطعی فردا رو بگو",
        "run an undefined magical tool", "tell me a secret I never shared", "guarantee tomorrow's exact price",
    ]
    for prompt in unknowns:
        language = "en" if prompt.isascii() else "fa"
        answer = "اطلاعات کافی ندارم؛ لطفاً دقیق‌تر توضیح بده." if language == "fa" else "I don't have enough information; please clarify."
        add("unknown_low_confidence", language, 3, prompt, answer, low_confidence=True)

    # Natural, deterministic surface variations. They expand language coverage without
    # changing semantic labels or manufacturing external knowledge.
    base_rows = list(rows)
    for row in base_rows:
        prompt = row["input"]
        language = row["language"]
        variants: list[str] = []
        if language in {"fa", "mixed"}:
            variants.extend((f"لطفا {prompt}", f"{prompt} لطفا", f"خب {prompt}"))
            variants.append(prompt.replace(" رو ", "و "))
        else:
            variants.extend((f"please {prompt}", f"{prompt}, please", f"hey jarvis, {prompt}"))
        for variant in variants:
            add(
                row["category"], language, int(row["stage"]), variant, row["output"],
                **{key: value for key, value in row.items() if key not in {
                    "id", "dataset_version", "category", "language", "stage", "input",
                    "output", "origin", "pretrained_source",
                }},
            )

    random.Random(SEED).shuffle(rows)
    for index, row in enumerate(rows, 1):
        row["id"] = f"jv5-{index:06d}"

    split_rows: dict[str, list[dict[str, Any]]] = {"train": [], "validation": [], "test": []}
    for row in rows:
        digest = hashlib.sha256(row["input"].casefold().encode("utf-8")).digest()[0]
        split = "validation" if digest < 26 else "test" if digest < 52 else "train"
        row["split"] = split
        split_rows[split].append(row)

    normalized = ROOT / "datasets" / "normalized" / f"{VERSION}.jsonl"
    _write_jsonl(normalized, rows)
    _write_jsonl(ROOT / "datasets" / "raw" / "seed_v001.jsonl", rows)
    _write_jsonl(ROOT / "datasets" / "cleaned" / f"{VERSION}.jsonl", rows)
    for split, values in split_rows.items():
        _write_jsonl(ROOT / "datasets" / "splits" / f"{VERSION}_{split}.jsonl", values)
    routes = {
        "tools": {"tool_calling", "entity_arguments", "multi_step_tasks"},
        "conversations": {"persian_conversation", "english_conversation", "context_followup"},
        "reasoning": {"reasoning", "search_decision"},
        "knowledge": {"general_knowledge", "programming", "model_identity"},
    }
    for directory, categories in routes.items():
        _write_jsonl(
            ROOT / "datasets" / directory / f"{VERSION}.jsonl",
            [row for row in rows if row["category"] in categories],
        )
    (ROOT / "datasets" / "corrections" / "learning_queue.jsonl").touch(exist_ok=True)

    raw_bytes = normalized.read_bytes()
    manifest = {
        "dataset_version": VERSION,
        "created_by": "JARVIS training pipeline",
        "pretrained_sources": [],
        "pretrained_source": None,
        "seed": SEED,
        "examples": len(rows),
        "splits": {key: len(value) for key, value in split_rows.items()},
        "categories": dict(sorted(Counter(row["category"] for row in rows).items())),
        "languages": dict(sorted(Counter(row["language"] for row in rows).items())),
        "stages": {str(key): value for key, value in sorted(Counter(row["stage"] for row in rows).items())},
        "sha256": hashlib.sha256(raw_bytes).hexdigest(),
    }
    (ROOT / "datasets" / "manifest_v001.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return manifest


if __name__ == "__main__":
    print(json.dumps(build(), ensure_ascii=False, indent=2))
