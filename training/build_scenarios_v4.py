from __future__ import annotations

import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "data" / "training" / "scenarios_v4.json"

APPS: dict[str, tuple[str, str]] = {
    "chrome": ("کروم", "Chrome"),
    "edge": ("مایکروسافت اج", "Microsoft Edge"),
    "firefox": ("فایرفاکس", "Firefox"),
    "steam": ("استیم", "Steam"),
    "discord": ("دیسکورد", "Discord"),
    "telegram": ("تلگرام", "Telegram"),
    "spotify": ("اسپاتیفای", "Spotify"),
    "vscode": ("وی اس کد", "VS Code"),
    "notepad": ("نوت پد", "Notepad"),
    "calculator": ("ماشین حساب", "Calculator"),
    "explorer": ("اکسپلورر", "Explorer"),
    "cmd": ("سی ام دی", "CMD"),
    "powershell": ("پاورشل", "PowerShell"),
    "paint": ("پینت", "Paint"),
}

WEBSITES: dict[str, tuple[str, str, str]] = {
    "youtube": ("یوتیوب", "YouTube", "https://www.youtube.com/"),
    "google": ("گوگل", "Google", "https://www.google.com/"),
    "github": ("گیت هاب", "GitHub", "https://github.com/"),
    "stackoverflow": ("استک اورفلو", "Stack Overflow", "https://stackoverflow.com/"),
    "reddit": ("ردیت", "Reddit", "https://www.reddit.com/"),
    "instagram": ("اینستاگرام", "Instagram", "https://www.instagram.com/"),
    "facebook": ("فیسبوک", "Facebook", "https://www.facebook.com/"),
    "gmail": ("جیمیل", "Gmail", "https://mail.google.com/"),
    "wikipedia": ("ویکی پدیا", "Wikipedia", "https://www.wikipedia.org/"),
    "spotify": ("اسپاتیفای", "Spotify", "https://open.spotify.com/"),
}


def turn(text: str, calls: list[dict[str, Any]] | None = None, **expected: Any) -> dict[str, Any]:
    result: dict[str, Any] = {"text": text, "expected": expected}
    result["expected"]["calls"] = calls or []
    return result


def scenario(identifier: str, category: str, turns: list[dict[str, Any]]) -> dict[str, Any]:
    return {"id": identifier, "category": category, "turns": turns}


def build_rows() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []

    for app_id, (fa, en) in APPS.items():
        opens = (
            f"{fa} رو باز کن", f"{fa}و باز کن", f"برنامه {fa} رو اجرا کن",
            f"open {en}", f"launch {en}", f"{fa} رو بیار",
        )
        closes = (
            f"{fa} رو ببند", f"{fa}و ببند", f"مرورگر {fa} رو ببند" if app_id in {"chrome", "edge", "firefox"} else f"برنامه {fa} رو ببند",
            f"close {en}", f"terminate {en}", f"از {fa} خارج شو",
        )
        for index, text in enumerate(opens, 1):
            rows.append(scenario(
                f"open-{app_id}-{index}", "open_app",
                [turn(text, [{"tool": "open_app", "arguments": {"app": app_id}}], action="open", entity=app_id)],
            ))
        for index, text in enumerate(closes, 1):
            rows.append(scenario(
                f"close-{app_id}-{index}", "close_app",
                [turn(text, [{"tool": "close_app", "arguments": {"app": app_id}}], action="close", entity=app_id)],
            ))

    drive_names = {"C": "سی", "D": "دی", "E": "ای", "F": "اف"}
    for drive, fa_name in drive_names.items():
        examples = (
            f"درایو {drive} رو باز کن", f"درایو {fa_name} رو باز کن",
            f"برو تو درایو {drive}", f"open drive {drive}", f"{drive} drive رو بیار",
        )
        for index, text in enumerate(examples, 1):
            rows.append(scenario(
                f"drive-{drive.lower()}-{index}", "drive",
                [turn(text, [{"tool": "open_folder", "arguments": {"path": f"{drive}:\\"}}], action="open", entity=drive)],
            ))

    folders = {
        "desktop": ("دسکتاپ", "Desktop"), "downloads": ("دانلودها", "Downloads"),
        "documents": ("اسناد", "Documents"), "pictures": ("تصاویر", "Pictures"),
        "videos": ("ویدیوها", "Videos"), "music": ("موزیک", "Music"),
        "home": ("پوشه کاربر", "Home"), "appdata": ("اپ دیتا", "AppData"),
        "temp": ("پوشه موقت", "Temp"),
    }
    for folder_id, (fa, en) in folders.items():
        for index, text in enumerate((f"{fa} رو باز کن", f"{fa} رو بیار", f"open {en}", f"go to {en}"), 1):
            rows.append(scenario(
                f"folder-{folder_id}-{index}", "known_folder",
                [turn(text, [{"tool": "open_folder", "arguments": {"path_nonempty": True}}], action="open", entity=folder_id)],
            ))

    for website_id, (fa, en, url) in WEBSITES.items():
        examples = (
            (f"سایت {fa} رو باز کن", f"برو سایت {fa}", f"open {en} website")
            if website_id == "spotify"
            else (f"{fa} رو باز کن", f"برو {fa}", f"open {en}")
        )
        for index, text in enumerate(examples, 1):
            rows.append(scenario(
                f"website-{website_id}-{index}", "website",
                [turn(text, [{"tool": "open_url", "arguments": {"url": url}}], action="open", entity=website_id)],
            ))

    search_queries = ("یونس گوهری", "OpenAI", "Python 3.13", "REST API")
    for browser_id, (_, en) in {key: APPS[key] for key in ("chrome", "edge", "firefox")}.items():
        for index, query in enumerate(search_queries, 1):
            text = f"داخل {en} {query} رو سرچ کن"
            rows.append(scenario(
                f"browser-search-{browser_id}-{index}", "browser_search",
                [turn(text, [{"tool": "web_search", "arguments": {"query": query.casefold(), "target_browser": browser_id}}], action="search", entity=browser_id)],
            ))

    contexts = (
        ("کروم رو باز کن", "ببندش", "chrome"),
        ("مایکروسافت اج رو باز کن", "همون مرورگر رو ببند", "edge"),
        ("فایرفاکس رو باز کن", "اون برنامه رو ببند", "firefox"),
        ("استیم رو باز کن", "همونو ببند", "steam"),
        ("تلگرام رو باز کن", "ببندش", "telegram"),
        ("دیسکورد رو باز کن", "همون قبلیه رو ببند", "discord"),
        ("اسپاتیفای رو باز کن", "close it", "spotify"),
        ("وی اس کد رو باز کن", "quit that app", "vscode"),
    )
    for index, (first, second, app_id) in enumerate(contexts, 1):
        rows.append(scenario(
            f"context-close-{index}", "reference",
            [
                turn(first, [{"tool": "open_app", "arguments": {"app": app_id}}], action="open", entity=app_id),
                turn(second, [{"tool": "close_app", "arguments": {"app": app_id}}], action="close", entity=app_id, reference=True),
            ],
        ))

    browser_contexts = (
        ("کروم رو باز کن", "حالا داخلش OpenAI رو سرچ کن", "chrome", "openai"),
        ("مایکروسافت اج رو باز کن", "توش Python رو سرچ کن", "edge", "python"),
        ("فایرفاکس رو باز کن", "search REST API in it", "firefox", "rest api"),
    )
    for index, (first, second, app_id, query) in enumerate(browser_contexts, 1):
        rows.append(scenario(
            f"context-search-{index}", "reference_search",
            [
                turn(first, [{"tool": "open_app", "arguments": {"app": app_id}}], action="open", entity=app_id),
                turn(second, [{"tool": "web_search", "arguments": {"query": query, "target_browser": app_id}}], action="search", entity=app_id, reference=True),
            ],
        ))

    ambiguous = (
        ("یه فولدر باز کن", "open_folder", "path"),
        ("یک پوشه باز کن", "open_folder", "path"),
        ("open a folder", "open_folder", "path"),
        ("یه برنامه باز کن", "open_app", "app"),
        ("برنامه رو اجرا کن", "open_app", "app"),
        ("یک اپلیکیشن باز کن", "open_app", "app"),
        ("سرچ کن", "web_search", "query"),
        ("search for", "web_search", "query"),
    )
    for index, (text, tool, missing) in enumerate(ambiguous, 1):
        rows.append(scenario(
            f"clarify-{index}", "clarification",
            [turn(text, [], clarification=True, missing=missing, blocked_tool=tool)],
        ))

    multisteps = (
        ("کروم رو باز کن و یونس گوهری رو سرچ کن داخلش", "chrome", "یونس گوهری"),
        ("مایکروسافت اج رو باز کن و OpenAI رو سرچ کن داخلش", "edge", "openai"),
        ("فایرفاکس رو باز کن و Python رو سرچ کن توش", "firefox", "python"),
        ("open Chrome and search for local AI", "chrome", "local ai"),
        ("launch Edge then look up REST API", "edge", "rest api"),
        ("open Firefox and search SQLite WAL in it", "firefox", "sqlite wal"),
    )
    for index, (text, app_id, query) in enumerate(multisteps, 1):
        rows.append(scenario(
            f"multi-browser-search-{index}", "multi_step",
            [turn(text, [
                {"tool": "open_app", "arguments": {"app": app_id}},
                {"tool": "web_search", "arguments": {"query": query, "target_browser": app_id}},
            ], multistep=True, action="multi", entity=app_id)],
        ))

    for index, app_id in enumerate(("chrome", "edge", "steam", "telegram", "vscode"), 1):
        fa, en = APPS[app_id]
        rows.extend((
            scenario(
                f"focus-{index}", "process",
                [turn(f"{fa} رو بیار جلو", [{"tool": "focus_app", "arguments": {"app": app_id}}], action="focus", entity=app_id)],
            ),
            scenario(
                f"restart-{index}", "process",
                [turn(f"restart {en}", [{"tool": "restart_app", "arguments": {"app": app_id}}], action="restart", entity=app_id)],
            ),
            scenario(
                f"running-{index}", "process",
                [turn(f"is {en} running", [{"tool": "is_app_running", "arguments": {"app": app_id}}], action="is_running", entity=app_id)],
            ),
        ))

    return rows


def build() -> dict[str, Any]:
    rows = build_rows()
    identifiers = [str(row["id"]) for row in rows]
    if len(rows) < 300 or len(identifiers) != len(set(identifiers)):
        raise ValueError("Scenario suite must contain at least 300 unique scenarios")
    payload = {"version": 4, "scenario_count": len(rows), "scenarios": rows}
    OUTPUT.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return {"scenario_count": len(rows), "path": str(OUTPUT)}


if __name__ == "__main__":
    print(json.dumps(build(), ensure_ascii=False, indent=2))
