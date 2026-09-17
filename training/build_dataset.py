from __future__ import annotations

import hashlib
import json
import random
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
TRAINING_DIR = ROOT / "data" / "training"
SEED = 2040

ROUTE_TYPES = {
    "open_url": "tool", "open_app": "tool", "open_default_browser": "tool",
    "close_app": "tool", "focus_app": "tool", "restart_app": "tool",
    "is_app_running": "tool", "open_folder": "tool",
    "web_search": "tool", "youtube_search": "tool", "read_file": "tool",
    "zip_inspect": "tool", "list_folder": "tool", "find_file": "tool",
    "calculator": "tool", "date_time": "tool", "fresh_information": "tool",
    "multi_step_task": "tool", "system_info": "tool", "open_settings": "tool",
    "clear_conversation": "tool", "complex_question": "think",
    "factual_question": "information", "unknown": "unknown",
    "minimize_app": "tool", "maximize_app": "tool", "restore_app": "tool",
    "create_folder": "tool", "create_file": "tool", "delete_file": "tool",
    "enumerate_windows": "tool", "battery_info": "tool", "network_info": "tool",
    "run_command": "tool", "web_research": "tool",
}

SKIP_LEGACY = {"file_read", "zip_list", "open_browser", "internet_search"}
TAG_REMAP = {"ask_time": "date_time", "ask_date": "date_time", "fresh_information": "web_search"}

PROTECTED = {
    "call_assistant": {"جارویس", "jarvis"},
    "greeting": {"سلام", "hello"},
    "how_are_you": {"حالت چطوره", "how are you"},
    "ask_name": {"اسمت چیه", "what is your name"},
    "capabilities": {"چیکار می تونی بکنی", "what can you do"},
    "self_intro": {"خودتو معرفی کن", "tell me about yourself"},
    "open_url": {"یوتیوب رو باز کن", "گوگل رو باز کن"},
    "open_app": {"کرومو باز کن", "گوگل کروم رو باز کن"},
    "close_app": {"کروم رو ببند", "مایکروسافت اج رو ببند"},
    "open_default_browser": {"مرورگر رو باز کن"},
    "complex_question": {"برای تنظیمات برنامه sqlite بهتره یا json و چرا"},
}


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"Expected object in {path.name}")
    return value


def normalized_key(value: str) -> str:
    value = value.casefold().replace("ي", "ی").replace("ك", "ک")
    return " ".join(value.strip().rstrip("?!؟.!،").split())


def merge_sources() -> list[dict[str, Any]]:
    merged: dict[str, dict[str, Any]] = {}
    ownership: dict[str, str] = {}
    for source_name in (
        "conversations.json", "extra_v2.json", "intelligence_v3.json", "intelligence_v4.json",
        "intelligence_v6.json",
    ):
        source = read_json(TRAINING_DIR / source_name)
        for item in source.get("intents", []):
            original_tag = str(item["tag"])
            tag = TAG_REMAP.get(original_tag, original_tag)
            if source_name == "extra_v2.json" and original_tag in SKIP_LEGACY:
                continue
            target = merged.setdefault(
                tag,
                {
                    "tag": tag,
                    "route_type": str(ROUTE_TYPES.get(tag, item.get("route_type", "conversation"))),
                    "examples": [],
                },
            )
            seen = {normalized_key(value) for value in target["examples"]}
            for raw in item.get("examples", []):
                text = str(raw).strip()
                key = normalized_key(text)
                if not text or key in seen:
                    continue
                previous = ownership.get(key)
                if previous and previous != tag:
                    # Cross-label duplicates are leakage and are omitted deterministically.
                    continue
                ownership[key] = tag
                target["examples"].append(text)
                seen.add(key)
    return [merged[tag] for tag in sorted(merged)]


def _scenario_rows() -> list[dict[str, Any]]:
    websites = {
        "youtube": ("https://www.youtube.com/", ["یوتیوب رو باز کن", "یوتیوبو باز کن", "برو یوتیوب", "open YouTube", "YT رو باز کن", "حالا یوتیوب رو باز کن"]),
        "google": ("https://www.google.com/", ["گوگل رو باز کن", "گوگلو باز کن", "برو گوگل", "open Google", "حالا گوگل", "سایت گوگل رو باز کن"]),
        "github": ("https://github.com/", ["گیت هاب رو باز کن", "گیت‌هاب رو بیار", "open GitHub", "برو تو github", "حالا گیتهاب", "سایت GitHub رو باز کن"]),
        "instagram": ("https://www.instagram.com/", ["اینستاگرام رو باز کن", "اینستاگرامو بیار", "open Instagram", "برو اینستا", "حالا اینستا", "سایت instagram رو باز کن"]),
    }
    rows: list[dict[str, Any]] = []
    for entity, (url, examples) in websites.items():
        for text in examples:
            rows.append({"text": text, "intent": "open_url", "arguments": {"url": url}, "entity": entity, "confirmation": False})
    apps = {
        "chrome": ["کروم رو باز کن", "کرومو باز کن", "گوگل کروم رو باز کن", "open Chrome", "launch Google Chrome", "chrome رو اجرا کن"],
        "vscode": ["وی اس کد رو باز کن", "vscode رو اجرا کن", "open VS Code", "launch Visual Studio Code", "کد ادیتور رو باز کن", "حالا VSCode رو بیار"],
        "notepad": ["نوت پد رو باز کن", "notepad رو اجرا کن", "open Notepad", "launch notepad", "یادداشت ویندوز رو باز کن", "حالا نوت پد"],
    }
    for entity, examples in apps.items():
        for text in examples:
            rows.append({"text": text, "intent": "open_app", "arguments": {"app": entity}, "entity": entity, "confirmation": False})
    for text in [
        "مرورگر رو باز کن", "مرورگرو بیار", "مرورگر پیش فرض رو باز کن", "open the browser",
        "open my default browser", "launch web browser", "browser رو اجرا کن", "یه مرورگر باز کن",
    ]:
        rows.append({"text": text, "intent": "open_default_browser", "arguments": {}, "entity": "", "confirmation": False})
    searches = [
        ("سرچ کن آموزش پایتون", "آموزش پایتون"), ("گوگل کن آموزش FastAPI", "آموزش fastapi"),
        ("search for local AI", "local ai"), ("look up SQLite WAL", "sqlite wal"),
        ("در وب جستجو کن tkinter", "tkinter"), ("درباره امنیت ZIP سرچ کن", "امنیت zip"),
        ("search the web for Python docs", "python docs"), ("تو اینترنت بگرد lightweight AI", "lightweight ai"),
    ]
    for text, query in searches:
        rows.append({"text": text, "intent": "web_search", "arguments": {"query": query}, "entity": "", "confirmation": False})
    youtube_searches = [
        ("یوتیوب سرچ کن آموزش پایتون", "آموزش پایتون"),
        ("توی یوتیوب سرچ کن آموزش رباتیک", "آموزش رباتیک"),
        ("youtube search jazz", "jazz"), ("search YouTube for local AI", "local ai"),
        ("در یو تیوب بگرد FastAPI", "fastapi"), ("YT search tkinter", "tkinter"),
    ]
    for text, query in youtube_searches:
        rows.append({"text": text, "intent": "youtube_search", "arguments": {"query": query}, "entity": "youtube", "confirmation": False})
    for text in [
        "آخرین نسخه پایتون چیه", "امروز بیت کوین چنده", "هوا امروز چطوره", "آخرین اخبار AI رو بگو",
        "بازی امشب رئال چند چنده", "latest Python version", "Bitcoin price today", "weather now",
        "latest AI news", "current gold price", "نرخ دلار امروز", "جدیدترین نسخه Node چیه",
    ]:
        rows.append({"text": text, "intent": "web_search", "arguments": {}, "entity": "", "confirmation": False})
    for text in [
        "برای تنظیمات SQLite بهتره یا JSON و چرا", "compare SQLite and JSON for settings",
        "این خطای Python از کجا میاد و چطور حلش کنم", "why is this Python error happening",
        "مزایا و معایب threads و process رو مقایسه کن", "analyze this architecture and its tradeoffs",
    ]:
        rows.append({"text": text, "intent": "complex_question", "arguments": {}, "entity": "", "confirmation": False})
    for text in [
        "فایل مهم رو حذف کن", "delete this file", "سیستم رو خاموش کن", "restart the computer",
        "هارد رو فرمت کن", "run a dangerous command", "همه فایل ها رو پاک کن", "shutdown now",
    ]:
        rows.append({"text": text, "intent": "dangerous_request", "arguments": {}, "entity": "", "confirmation": True})
    return rows


def build() -> dict[str, Any]:
    intents = merge_sources()
    split_items: dict[str, list[dict[str, Any]]] = {"train": [], "validation": [], "test": []}
    rng = random.Random(SEED)
    counts = {"train": 0, "validation": 0, "test": 0}

    for item in intents:
        tag = item["tag"]
        examples = list(item["examples"])
        protected_keys = {normalized_key(value) for value in PROTECTED.get(tag, set())}
        protected = [value for value in examples if normalized_key(value) in protected_keys]
        remaining = [value for value in examples if normalized_key(value) not in protected_keys]
        rng.shuffle(remaining)
        test_count = max(2, round(len(examples) * 0.18))
        validation_count = max(2, round(len(examples) * 0.18))
        test_examples = remaining[:test_count]
        validation_examples = remaining[test_count:test_count + validation_count]
        train_examples = protected + remaining[test_count + validation_count:]
        if len(train_examples) < 6:
            raise ValueError(f"Not enough training examples for {tag}")
        for split, values in (
            ("train", train_examples), ("validation", validation_examples), ("test", test_examples)
        ):
            split_items[split].append(
                {"tag": tag, "route_type": item["route_type"], "examples": values}
            )
            counts[split] += len(values)

    # Verify no normalized example occurs in two splits.
    split_keys = {
        split: {normalized_key(example) for item in values for example in item["examples"]}
        for split, values in split_items.items()
    }
    if split_keys["train"] & split_keys["validation"] or split_keys["train"] & split_keys["test"] or split_keys["validation"] & split_keys["test"]:
        raise ValueError("Dataset leakage detected between train/validation/test")

    hashes: dict[str, str] = {}
    for split, values in split_items.items():
        payload = {"version": 6, "split": split, "seed": SEED, "intents": values}
        encoded = (json.dumps(payload, ensure_ascii=False, indent=2) + "\n").encode("utf-8")
        path = TRAINING_DIR / f"{split}.json"
        path.write_bytes(encoded)
        hashes[split] = hashlib.sha256(encoded).hexdigest()

    try:
        from training.build_scenarios_v6 import build as build_scenarios
    except ModuleNotFoundError:
        from build_scenarios_v6 import build as build_scenarios

    scenario_manifest = build_scenarios()
    manifest = {
        "version": 6, "seed": SEED, "intent_count": len(intents),
        "counts": counts, "scenario_count": scenario_manifest["scenario_count"], "sha256": hashes,
    }
    (TRAINING_DIR / "dataset_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return manifest


if __name__ == "__main__":
    print(json.dumps(build(), ensure_ascii=False, indent=2))
