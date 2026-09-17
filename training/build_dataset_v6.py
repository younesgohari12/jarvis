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

from jarvis.learning.failures import FailureCollector  # noqa: E402
from jarvis.neural.tokenizer import normalize_language_text  # noqa: E402


VERSION = "dataset_v002"
SEED = 6062026
STAGES = {
    1: "language_foundations",
    2: "conversation",
    3: "general_knowledge",
    4: "instruction_following",
    5: "action_recognition",
    6: "entity_extraction",
    7: "tool_calling",
    8: "argument_extraction",
    9: "context_memory",
    10: "planning",
    11: "reasoning",
    12: "recovery_verification",
    13: "search_decision",
    14: "jarvis_personality",
}


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def _legacy_stage(row: dict[str, Any]) -> int:
    category = str(row.get("category", ""))
    action = str(row.get("action", ""))
    if int(row.get("stage", 1)) == 1:
        return 1
    if category in {"persian_conversation", "english_conversation", "model_identity"}:
        return 2
    if category in {"general_knowledge", "programming"}:
        return 3
    if category in {"instruction_following", "unknown_low_confidence"}:
        return 4
    if category == "tool_calling" and action in {"open_app", "close_app"}:
        return 5
    if category == "tool_calling":
        return 7
    if category == "entity_arguments":
        return 8
    if category == "context_followup":
        return 9
    if category == "multi_step_tasks":
        return 10
    if category == "reasoning":
        return 11
    if category == "search_decision":
        return 13
    if category == "personality":
        return 14
    return 4


def build() -> dict[str, Any]:
    from training.build_dataset_v5 import build as build_v5

    build_v5()
    legacy_path = ROOT / "datasets" / "normalized" / "dataset_v001.jsonl"
    legacy = [json.loads(line) for line in legacy_path.read_text(encoding="utf-8").splitlines() if line]

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
        if stage not in STAGES:
            raise ValueError(f"Invalid curriculum stage: {stage}")
        seen.add(key)
        row: dict[str, Any] = {
            "id": "",
            "dataset_version": VERSION,
            "category": category,
            "language": language,
            "stage": stage,
            "stage_name": STAGES[stage],
            "input": source,
            "output": target,
            "origin": "jarvis-authored",
            "pretrained_source": None,
        }
        row.update(extra)
        rows.append(row)

    for row in legacy:
        extras = {
            key: value for key, value in row.items()
            if key not in {
                "id", "dataset_version", "category", "language", "stage", "input",
                "output", "origin", "pretrained_source", "split",
            }
        }
        add(
            str(row["category"]), str(row["language"]), _legacy_stage(row),
            str(row["input"]), str(row["output"]), inherited_from="dataset_v001", **extras,
        )

    # Stage 3: compact, grounded bilingual knowledge authored for this project.
    knowledge = (
        ("بیضه چیست؟", "بیضه یکی از دو غدهٔ تولیدمثلی مردانه است که اسپرم و هورمون‌هایی مانند تستوسترون تولید می‌کند و معمولاً در کیسهٔ بیضه قرار دارد.", "fa"),
        ("testicle چیست؟", "بیضه اندام تولیدمثلی مردانه برای تولید اسپرم و تستوسترون است؛ درد یا تورم ناگهانی آن نیازمند ارزیابی فوری پزشکی است.", "fa"),
        ("what is a testicle?", "A testicle is a male reproductive gland that produces sperm and hormones such as testosterone. Sudden severe pain or swelling needs urgent medical assessment.", "en"),
        ("چرا آسمان آبی است؟", "مولکول‌های جو طول موج‌های کوتاه‌تر نور مرئی، به‌ویژه آبی، را بیشتر از طول موج‌های بلند پراکنده می‌کنند؛ بنابراین آسمان روز آبی دیده می‌شود.", "fa"),
        ("why is the sky blue?", "Air molecules scatter shorter visible wavelengths more strongly than longer wavelengths, so blue light reaches our eyes from many directions.", "en"),
        ("RAM چیست؟", "RAM حافظهٔ سریع و موقتی است که برنامه‌ها و داده‌های در حال استفاده را نگه می‌دارد؛ با خاموش شدن سیستم محتوای معمول آن از بین می‌رود.", "fa"),
        ("what is RAM?", "RAM is fast, volatile working memory used by active programs and data. Its ordinary contents disappear when power is removed.", "en"),
        ("CPU چیست؟", "CPU دستورهای برنامه را اجرا و محاسبات و هماهنگی عمومی سیستم را مدیریت می‌کند.", "fa"),
        ("GPU چیست؟", "GPU پردازنده‌ای با واحدهای موازی فراوان برای گرافیک و برخی محاسبات داده‌محور است؛ برای اجرای Jarvis الزامی نیست.", "fa"),
        ("API چیست؟", "API یک قرارداد مشخص برای درخواست قابلیت یا داده از نرم‌افزار دیگر است و جزئیات داخلی آن را پنهان می‌کند.", "fa"),
        ("what is an API?", "An API is a defined interface through which software requests data or behavior from another component.", "en"),
        ("DNS چیست؟", "DNS نام دامنه را به آدرس‌های شبکه مانند IP نگاشت می‌کند و به‌صورت سلسله‌مراتبی و cache‌شونده کار می‌کند.", "fa"),
        ("what does DNS do?", "DNS maps domain names to network records such as IP addresses through a hierarchical, cacheable system.", "en"),
        ("SQLite چیست؟", "SQLite یک پایگاه‌دادهٔ رابطه‌ای تراکنشی و بدون سرور است که داده را معمولاً در یک فایل محلی ذخیره می‌کند.", "fa"),
        ("what is SQLite?", "SQLite is an embedded, transactional relational database that usually stores its data in one local file.", "en"),
        ("ترنسفورمر چیست؟", "Transformer معماری عصبی مبتنی بر attention است که وابستگی میان توکن‌ها را بدون تکرار زمانی اجباری مدل می‌کند.", "fa"),
        ("what is a transformer model?", "A Transformer is a neural architecture that uses attention to model relationships among tokens without mandatory recurrent steps.", "en"),
        ("سیاهچاله چیست؟", "سیاهچاله ناحیه‌ای از فضا-زمان با گرانش بسیار شدید است؛ پس از افق رویداد مسیر خروجی برای نور وجود ندارد.", "fa"),
        ("what is a black hole?", "A black hole is a region of spacetime whose gravity creates an event horizon beyond which light cannot escape.", "en"),
        ("HTTPS چیست؟", "HTTPS همان HTTP روی TLS است و محرمانگی، یکپارچگی و احراز هویت سرور را برای ارتباط فراهم می‌کند.", "fa"),
        ("what is HTTPS?", "HTTPS carries HTTP over TLS to provide transport encryption, integrity, and server authentication.", "en"),
        ("فرق process و thread چیست؟", "Process فضای آدرس جدا دارد؛ threadها درون یک process حافظه را به اشتراک می‌گذارند و هماهنگی دقیق‌تری می‌خواهند.", "fa"),
        ("difference between a process and a thread", "Processes have separate address spaces; threads share memory inside a process and therefore require careful synchronization.", "en"),
        ("هش چیست؟", "تابع هش داده با طول دلخواه را به خروجی با طول ثابت نگاشت می‌کند؛ هش رمزنگاری باید در برابر برخورد و پیش‌تصویر مقاوم باشد.", "fa"),
        ("what is a cryptographic hash?", "A cryptographic hash maps arbitrary input to fixed-size output and is designed to resist preimage and collision attacks.", "en"),
        ("یادگیری ماشین چیست؟", "یادگیری ماشین روش ساخت سامانه‌هایی است که الگوها را از داده می‌آموزند و روی نمونه‌های جدید پیش‌بینی یا تصمیم می‌گیرند.", "fa"),
        ("what is machine learning?", "Machine learning builds systems that learn patterns from data to make predictions or decisions on new examples.", "en"),
    )
    for question, answer, language in knowledge:
        frames = (
            ("{}", "ساده توضیح بده: {}", "مختصر بگو {}", "جارویس، {}")
            if language == "fa" else
            ("{}", "explain simply: {}", "briefly answer: {}", "Jarvis, {}")
        )
        for frame in frames:
            add("general_knowledge_v6", language, 3, frame.format(question), answer, grounded=True)

    apps = {
        "chrome": ("کروم", "Chrome"), "edge": ("مایکروسافت اج", "Edge"),
        "firefox": ("فایرفاکس", "Firefox"), "notepad": ("نوت پد", "Notepad"),
        "vscode": ("وی اس کد", "VS Code"), "calculator": ("ماشین حساب", "Calculator"),
        "paint": ("پینت", "Paint"), "telegram": ("تلگرام", "Telegram"),
        "discord": ("دیسکورد", "Discord"), "spotify": ("اسپاتیفای", "Spotify"),
    }
    actions = {
        "open_app": ("{fa} رو باز کن", "open {en}"),
        "close_app": ("{fa} رو ببند", "close {en}"),
        "focus_app": ("{fa} رو بیار جلو", "focus {en}"),
        "restart_app": ("{fa} رو دوباره اجرا کن", "restart {en}"),
        "minimize_app": ("{fa} رو مینیمایز کن", "minimize {en}"),
        "maximize_app": ("{fa} رو تمام صفحه کن", "maximize {en}"),
        "restore_app": ("پنجره {fa} رو برگردون", "restore {en} window"),
    }
    for app, names in apps.items():
        for action, templates in actions.items():
            for language, template, name in (("fa", templates[0], names[0]), ("en", templates[1], names[1])):
                base = template.format(fa=name, en=name)
                output = f"<tool>{action} app={app}"
                for variant in (base, f"لطفاً {base}" if language == "fa" else f"please {base}", f"جارویس {base}" if language == "fa" else f"Jarvis, {base}"):
                    add("desktop_action", language, 5, variant, output, action=action, arguments={"app": app})

    entities = (
        ("پنجره Chrome رو پیدا کن", "<tool>find_window window=Chrome", "fa", "window", "Chrome"),
        ("find the Notepad window", "<tool>find_window window=Notepad", "en", "window", "Notepad"),
        ("برنامه Blender رو پیدا کن", "<tool>find_installed_app query=Blender", "fa", "app", "Blender"),
        ("find installed app VLC", "<tool>find_installed_app query=VLC", "en", "app", "VLC"),
        ("پوشه Projects رو داخل درایو D پیدا کن", "<tool>find_folder root=D:\\ name=Projects", "fa", "folder", "Projects"),
        ("find folder Games in drive E", "<tool>find_folder root=E:\\ name=Games", "en", "folder", "Games"),
        ("فایل report.csv رو داخل Downloads پیدا کن", "<tool>find_file folder=Downloads query=report.csv", "fa", "file", "report.csv"),
        ("find notes.txt in Documents", "<tool>find_file folder=Documents query=notes.txt", "en", "file", "notes.txt"),
    )
    for prompt, output, language, entity_type, entity in entities:
        for prefix in ("", "لطفاً " if language == "fa" else "please ", "دقیقاً " if language == "fa" else "exactly "):
            add("entity_extraction", language, 6, prefix + prompt, output, entity_type=entity_type, entity=entity)

    tools = (
        ("لیست پنجره‌ها رو نشون بده", "<tool>enumerate_windows", "fa"),
        ("list visible windows", "<tool>enumerate_windows", "en"),
        ("متن صفحه وب رو بخون", "<tool>browser_read_page", "fa"),
        ("list links on this page", "<tool>browser_links", "en"),
        ("تب جدید باز کن", "<tool>new_tab", "fa"),
        ("go back in the browser", "<tool>navigate_back", "en"),
        ("وضعیت باتری رو بگو", "<tool>battery_info", "fa"),
        ("show network information", "<tool>network_info", "en"),
        ("کلیپ بورد رو بخون", "<tool>clipboard_read", "fa"),
        ("clear the clipboard", "<tool>clipboard_clear", "en"),
        ("لیست برنامه‌های نصب شده", "<tool>list_installed_apps", "fa"),
        ("show running processes", "<tool>process_info", "en"),
    )
    for prompt, output, language in tools:
        for suffix in ("", " لطفاً" if language == "fa" else ", please", " جارویس" if language == "fa" else ", Jarvis"):
            add("tool_selection", language, 7, prompt + suffix, output, action=output.split(">", 1)[1])

    arguments = (
        ("صدا رو روی 35 درصد بذار", "<tool>set_volume percent=35", "fa"),
        ("set volume to 72 percent", "<tool>set_volume percent=72", "en"),
        ("روشنایی رو 60 درصد کن", "<tool>set_brightness percent=60", "fa"),
        ("move Chrome window to 100 200", "<tool>move_window window=Chrome x=100 y=200", "en"),
        ("پنجره Notepad رو 900 در 600 کن", "<tool>resize_window window=Notepad width=900 height=600", "fa"),
        ("یک پوشه به نام Jarvis Test روی دسکتاپ بساز", "<tool>create_folder path=Desktop/Jarvis Test", "fa"),
        ("create file notes.txt in Documents", "<tool>create_file path=Documents/notes.txt", "en"),
        ("داخل notes.txt بنویس سلام دنیا", "<tool>write_file path=notes.txt content=سلام دنیا", "fa"),
        ("append done to notes.txt", "<tool>append_file path=notes.txt content=done", "en"),
        ("run command python --version", "<tool>run_command command=python --version", "en"),
    )
    for prompt, output, language in arguments:
        for prefix in ("", "لطفاً " if language == "fa" else "please ", "جارویس " if language == "fa" else "Jarvis, "):
            add("argument_extraction", language, 8, prefix + prompt, output, structured_arguments=True)

    context_pairs = (
        ("کروم رو باز کن", "حالا مینیمایزش کن", "<tool>minimize_app app=chrome", "fa"),
        ("مایکروسافت اج رو باز کن", "همونو تمام صفحه کن", "<tool>maximize_app app=edge", "fa"),
        ("open Firefox", "close it", "<tool>close_app app=firefox", "en"),
        ("open Notepad", "type hello there", "<tool>type_text text=hello there", "en"),
        ("پوشه Projects رو پیدا کن", "بازش کن", "<tool>open_folder reference=Projects", "fa"),
        ("search Python 3.13", "open the first result", "<tool>browser_open_link index=0", "en"),
    )
    for first, follow, output, language in context_pairs:
        for pronoun in (follow, f"بعد {follow}" if language == "fa" else f"then {follow}", f"{follow} لطفاً" if language == "fa" else f"{follow}, please"):
            add(
                "context_followup_v6", language, 9, pronoun, output,
                context=[{"role": "user", "content": first}, {"role": "assistant", "content": "<completed>"}],
            )

    plans = (
        ("کروم رو باز کن و Python 3.13 رو داخلش سرچ کن", ["open_app", "web_search"], "fa"),
        ("open Edge then search for SQLite WAL", ["open_app", "web_search"], "en"),
        ("پوشه Test رو روی دسکتاپ بساز و فایل readme.md رو داخلش ایجاد کن", ["create_folder", "create_file"], "fa"),
        ("open Chrome, search for Python docs, then open the first result", ["open_app", "web_search", "browser_open_link"], "en"),
        ("فایرفاکس رو بیار جلو بعد مینیمایزش کن", ["focus_app", "minimize_app"], "fa"),
        ("find VLC and open it", ["find_installed_app", "open_app"], "en"),
    )
    for prompt, steps, language in plans:
        encoded = json.dumps([{"tool": tool} for tool in steps], ensure_ascii=False)
        for prefix in ("", "لطفاً " if language == "fa" else "please ", "قدم به قدم: " if language == "fa" else "step by step: "):
            add("goal_planning", language, 10, prefix + prompt, f"<tool_plan>{encoded}", plan=steps)

    reasoning = (
        ("اگر RAM کم باشد اول مدل را بزرگ کنم یا context را کاهش بدهم؟", "اول اندازه‌گیری کن؛ برای کاهش حافظه معمولاً context و batch را کم کن، سپس فقط اگر کیفیت نیاز داشت معماری را تغییر بده.", "fa"),
        ("Should I delete a failing file before making a backup?", "No. Preserve a recoverable copy first, diagnose the failure, then perform the smallest confirmed change.", "en"),
        ("چرا نتیجه یک ابزار باید verify شود؟", "چون نبودن exception فقط آغاز عملیات را نشان می‌دهد؛ verification بررسی می‌کند حالت واقعی سیستم به هدف رسیده باشد.", "fa"),
        ("When should an agent ask a clarification?", "It should ask when a required argument is missing or multiple plausible targets would materially change the action.", "en"),
    )
    for prompt, output, language in reasoning:
        for prefix in ("", "تحلیل کن: " if language == "fa" else "analyze: ", "با دلیل: " if language == "fa" else "explain why: "):
            add("reasoning_v6", language, 11, prefix + prompt, output, reasoning_mode="private_bounded")

    recovery = (
        ("برنامه پیدا نشد؛ چه کار کن؟", "فهرست برنامه‌ها را refresh کن، دوباره با نام نزدیک جستجو کن و فقط در صورت یافتن launcher معتبر دوباره اجرا کن.", "fa"),
        ("the window did not close; recover", "Re-enumerate matching windows, report whether it is still open, and ask before escalating to process termination.", "en"),
        ("چند پوشه هم‌نام پیدا شد", "مسیرهای محدود و شماره‌گذاری‌شده را نشان بده و قبل از بازکردن از کاربر انتخاب بخواه.", "fa"),
        ("a command was blocked by policy", "Explain the policy decision and request a safer, explicit alternative; never bypass the command policy.", "en"),
        ("عملیات موفق گزارش شد ولی verify نشد", "موفقیت را قطعی اعلام نکن؛ وضعیت را دوباره بخوان و نتیجهٔ تأییدشده یا محدودیت را شفاف گزارش بده.", "fa"),
    )
    for prompt, output, language in recovery:
        for frame in ("{}", "سناریو: {}", "راه بازیابی برای {}") if language == "fa" else ("{}", "scenario: {}", "recovery for: {}"):
            add("recovery_verification", language, 12, frame.format(prompt), output, hard_example_weight=1.5)

    current_topics = (
        "آخرین نسخه پایدار پایتون", "قیمت امروز بیت کوین", "آب و هوای امروز تهران",
        "آخرین اخبار هوش مصنوعی", "current Python release", "Bitcoin price today",
        "weather in London today", "latest Windows update",
    )
    for topic in current_topics:
        language = "en" if topic.isascii() else "fa"
        frames = ("درباره {} تحقیق اینترنتی کن", "{} رو از چند منبع بررسی کن", "اطلاعات به روز {} رو پیدا کن") if language == "fa" else ("research {}", "cross-check {} on the web", "find current information about {}")
        for frame in frames:
            add("search_decision_v6", language, 13, frame.format(topic), f"<tool>web_research query={topic}", current_information=True)

    personalities = {
        "Normal": ("جارویس", "بله، اینجام."),
        "Kind": ("جارویس", "جانم؟ ❤️"),
        "Angry": ("جارویس", "چیه باز؟"),
        "Loti": ("جارویس", "جان داش، بگو."),
        "Gang": ("Jarvis", "Yeah. What's up?"),
        "Funny": ("جارویس", "بله قربان؛ پردازنده روشن، شوخی هم آماده!"),
        "Professional": ("Jarvis", "At your service. How may I assist?"),
    }
    for personality, (prompt, output) in personalities.items():
        language = "en" if prompt.isascii() else "fa"
        add("personality_behavior_v6", language, 14, prompt, output, personality=personality)
        add("personality_behavior_v6", language, 14, f"[{personality}] {prompt}", output, personality=personality)
        command = f"مود رو {personality} کن" if language == "fa" else f"switch personality to {personality}"
        add("personality_switch_v6", language, 14, command, f"<personality>{personality}", personality=personality)

    # Reviewed failures are opt-in hard examples. Pending rows never enter training.
    failure_path = ROOT / "runtime_data" / "failure_review_queue.jsonl"
    if failure_path.is_file():
        for item in FailureCollector(failure_path).approved_training_rows():
            add(
                "approved_failure_correction", "mixed", 12,
                str(item["input"]), str(item["output"]),
                hard_example_weight=float(item.get("hard_example_weight", 2.0)),
                source_failure_id=str(item.get("source_failure_id", "")),
            )

    random.Random(SEED).shuffle(rows)
    for index, row in enumerate(rows, 1):
        row["id"] = f"jv6-{index:06d}"

    splits: dict[str, list[dict[str, Any]]] = {"train": [], "validation": [], "test": []}
    for row in rows:
        digest = hashlib.sha256(f"{row['category']}|{row['input'].casefold()}".encode("utf-8")).digest()[0]
        split = "validation" if digest < 26 else "test" if digest < 52 else "train"
        row["split"] = split
        splits[split].append(row)

    normalized = ROOT / "datasets" / "normalized" / f"{VERSION}.jsonl"
    _write_jsonl(normalized, rows)
    _write_jsonl(ROOT / "datasets" / "raw" / "seed_v002.jsonl", rows)
    _write_jsonl(ROOT / "datasets" / "cleaned" / f"{VERSION}.jsonl", rows)
    for split, values in splits.items():
        _write_jsonl(ROOT / "datasets" / "splits" / f"{VERSION}_{split}.jsonl", values)
    routes = {
        "tools": {5, 6, 7, 8, 10},
        "conversations": {1, 2, 9, 14},
        "reasoning": {11, 12, 13},
        "knowledge": {3},
    }
    for directory, stage_ids in routes.items():
        _write_jsonl(
            ROOT / "datasets" / directory / f"{VERSION}.jsonl",
            [row for row in rows if int(row["stage"]) in stage_ids],
        )
    (ROOT / "datasets" / "corrections" / "learning_queue.jsonl").touch(exist_ok=True)

    counts = Counter(int(row["stage"]) for row in rows)
    missing = [stage for stage in STAGES if counts[stage] == 0]
    if missing:
        raise RuntimeError(f"Curriculum stages have no examples: {missing}")
    manifest = {
        "dataset_version": VERSION,
        "created_by": "JARVIS v0.6 project-owned training pipeline",
        "provenance": "random initialization; project-authored data only",
        "pretrained_sources": [],
        "pretrained_source": None,
        "seed": SEED,
        "examples": len(rows),
        "splits": {key: len(value) for key, value in splits.items()},
        "categories": dict(sorted(Counter(str(row["category"]) for row in rows).items())),
        "languages": dict(sorted(Counter(str(row["language"]) for row in rows).items())),
        "curriculum": [
            {"stage": stage, "name": name, "examples": counts[stage]}
            for stage, name in STAGES.items()
        ],
        "sha256": hashlib.sha256(normalized.read_bytes()).hexdigest(),
    }
    (ROOT / "datasets" / "manifest_v002.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return manifest


if __name__ == "__main__":
    print(json.dumps(build(), ensure_ascii=False, indent=2))
