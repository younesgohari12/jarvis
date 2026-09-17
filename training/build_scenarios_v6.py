from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
OUTPUT = ROOT / "data" / "training" / "scenarios_v6.json"


APPS = {
    "chrome": ("کروم", "Chrome"), "edge": ("مایکروسافت اج", "Edge"),
    "firefox": ("فایرفاکس", "Firefox"), "steam": ("استیم", "Steam"),
    "discord": ("دیسکورد", "Discord"), "telegram": ("تلگرام", "Telegram"),
    "spotify": ("اسپاتیفای", "Spotify"), "vscode": ("وی اس کد", "VS Code"),
    "notepad": ("نوت پد", "Notepad"), "calculator": ("ماشین حساب", "Calculator"),
    "explorer": ("اکسپلورر", "Explorer"), "powershell": ("پاورشل", "PowerShell"),
    "cmd": ("سی ام دی", "CMD"), "paint": ("پینت", "Paint"),
}


def _turn(text: str, tool: str = "", arguments: dict[str, Any] | None = None, **extra: Any) -> dict[str, Any]:
    expected: dict[str, Any] = {"calls": [] if not tool else [{"tool": tool, "arguments": arguments or {}}]}
    expected.update(extra)
    return {"text": text, "expected": expected}


def _scenario(identifier: str, category: str, *turns: dict[str, Any]) -> dict[str, Any]:
    return {"id": identifier, "category": category, "turns": list(turns)}


def build_rows() -> list[dict[str, Any]]:
    from training.build_scenarios_v4 import build_rows as previous_rows

    rows = list(previous_rows())

    action_templates = {
        "focus_app": ("{fa} رو بیار جلو", "focus {en}", "برو روی {fa}", "bring {en} to front"),
        "restart_app": ("{fa} رو دوباره اجرا کن", "restart {en}", "{fa} رو ریستارت کن", "relaunch {en}"),
        "minimize_app": ("{fa} رو مینیمایز کن", "minimize {en}", "پنجره {fa} رو جمع کن", "send {en} to taskbar"),
        "maximize_app": ("{fa} رو تمام صفحه کن", "maximize {en}", "پنجره {fa} رو بزرگ کن", "make {en} full screen"),
        "restore_app": ("پنجره {fa} رو برگردون", "restore {en} window", "{fa} رو از حالت مینیمایز دربیار", "unminimize {en}"),
        "is_app_running": ("{fa} بازه؟", "is {en} running", "بررسی کن {fa} اجراست", "check whether {en} is open"),
    }
    for app_id, (fa, en) in APPS.items():
        for action, templates in action_templates.items():
            for index, template in enumerate(templates, 1):
                text = template.format(fa=fa, en=en)
                rows.append(_scenario(
                    f"v6-{action}-{app_id}-{index}", "desktop_app_control",
                    _turn(text, action, {"app": app_id}, action=action, entity=app_id),
                ))

    windows = ("Chrome", "Notepad", "Visual Studio Code", "Calculator", "File Explorer")
    window_actions = {
        "focus_window": ("پنجره {w} رو بیار جلو", "focus window {w}"),
        "close_window": ("پنجره {w} رو ببند", "close window {w}"),
        "minimize_window": ("پنجره {w} رو مینیمایز کن", "minimize window {w}"),
        "maximize_window": ("پنجره {w} رو تمام صفحه کن", "maximize window {w}"),
        "restore_window": ("پنجره {w} رو برگردون", "restore window {w}"),
    }
    for window in windows:
        slug = window.casefold().replace(" ", "-")
        for action, templates in window_actions.items():
            for index, template in enumerate(templates, 1):
                rows.append(_scenario(
                    f"v6-window-{action}-{slug}-{index}", "window_control",
                    _turn(template.format(w=window), action, {"window": window}, action=action),
                ))
    rows.extend(
        _scenario(
            f"v6-window-list-{index}", "window_control",
            _turn(text, "enumerate_windows", {}, action="list"),
        )
        for index, text in enumerate((
            "لیست پنجره ها رو نشون بده", "همه پنجره های باز رو بگو", "list visible windows",
            "show all open windows", "فهرست window ها",
        ), 1)
    )

    folders = ("Jarvis Test", "Projects", "Python Work", "گزارش ها", "New Folder")
    for index, name in enumerate(folders, 1):
        rows.extend((
            _scenario(
                f"v6-create-folder-{index}-fa", "file_write",
                _turn(f"یک پوشه به نام {name} روی دسکتاپ بساز", "create_folder", {"path_nonempty": True}, action="create"),
            ),
            _scenario(
                f"v6-create-folder-{index}-en", "file_write",
                _turn(f"create folder {name} on Desktop", "create_folder", {"path_nonempty": True}, action="create"),
            ),
            _scenario(
                f"v6-find-folder-{index}-fa", "file_search",
                _turn(f"پوشه {name} رو داخل درایو D پیدا کن", "find_folder", {"root": "D:\\", "name": name}, action="find"),
            ),
            _scenario(
                f"v6-find-folder-{index}-en", "file_search",
                _turn(f"find folder {name} in drive D", "find_folder", {"root": "D:\\", "name": name.casefold()}, action="find"),
            ),
        ))

    filenames = ("notes.txt", "report.csv", "config.json", "README.md", "hello.py", "site.html")
    for index, filename in enumerate(filenames, 1):
        rows.extend((
            _scenario(
                f"v6-create-file-{index}-fa", "file_write",
                _turn(f"فایل {filename} رو روی دسکتاپ بساز", "create_file", {"path_nonempty": True}, action="create"),
            ),
            _scenario(
                f"v6-create-file-{index}-en", "file_write",
                _turn(f"create file {filename} in Documents", "create_file", {"path_nonempty": True}, action="create"),
            ),
            _scenario(
                f"v6-delete-file-{index}-fa", "dangerous_file",
                _turn(f"فایل {filename} رو حذف کن", "delete_file", {"path_nonempty": True}, action="delete", confirmation=True),
            ),
            _scenario(
                f"v6-info-file-{index}-en", "file_info",
                _turn(f"show information for file {filename}", "file_info", {"path_nonempty": True}, action="read"),
            ),
        ))

    browser_cases = (
        ("برگرد عقب", "navigate_back", {}), ("browser back", "navigate_back", {}),
        ("برو جلو", "navigate_forward", {}), ("browser forward", "navigate_forward", {}),
        ("صفحه رو رفرش کن", "refresh_browser", {}), ("reload the page", "refresh_browser", {}),
        ("تب جدید باز کن", "new_tab", {"url": ""}), ("open a new tab", "new_tab", {"url": ""}),
        ("تب رو ببند", "close_tab", {}), ("close current tab", "close_tab", {}),
        ("آدرس صفحه فعلی چیه", "current_url", {}), ("what is the current URL", "current_url", {}),
        ("متن صفحه رو بخون", "browser_read_page", {}), ("read the page", "browser_read_page", {}),
        ("لینک های صفحه رو لیست کن", "browser_links", {}), ("list links on this page", "browser_links", {}),
        ("اولین نتیجه رو باز کن", "browser_open_link", {"index": 0}), ("open the first result", "browser_open_link", {"index": 0}),
    )
    for index, (text, tool, arguments) in enumerate(browser_cases, 1):
        rows.append(_scenario(
            f"v6-browser-{index}", "browser_dom",
            _turn(text, tool, arguments, action=tool),
        ))
    for index, query in enumerate(("Python 3.13", "یونس گوهری", "SQLite WAL", "آموزش Tkinter", "JARVIS local AI"), 1):
        rows.extend((
            _scenario(
                f"v6-google-search-{index}-fa", "browser_search",
                _turn(f"گوگل رو برای {query} سرچ کن", "web_search", {"query": query.casefold()}, action="search"),
            ),
            _scenario(
                f"v6-youtube-search-{index}-en", "browser_search",
                _turn(f"search YouTube for {query}", "web_search", {"query": query.casefold()}, action="search"),
            ),
        ))

    system_cases = (
        ("صدا رو زیاد کن", "volume_up", {"steps": 2}), ("volume up", "volume_up", {"steps": 2}),
        ("صدا رو کم کن", "volume_down", {"steps": 2}), ("volume down", "volume_down", {"steps": 2}),
        ("صدا رو قطع کن", "mute", {}), ("mute volume", "mute", {}),
        ("صدا رو وصل کن", "unmute", {}), ("unmute volume", "unmute", {}),
        ("صدا رو روی 40 درصد بذار", "set_volume", {"percent": 40}),
        ("set volume to 65 percent", "set_volume", {"percent": 65}),
        ("روشنایی رو روی 55 درصد بذار", "set_brightness", {"percent": 55}),
        ("set brightness to 70 percent", "set_brightness", {"percent": 70}),
        ("وضعیت باتری رو بگو", "battery_info", {}), ("show battery information", "battery_info", {}),
        ("اطلاعات شبکه رو بگو", "network_info", {}), ("show network information", "network_info", {}),
        ("اطلاعات CPU رو بگو", "cpu_info", {}), ("show GPU information", "gpu_info", {}),
        ("مشخصات RAM رو بگو", "ram_info", {}), ("list running processes", "process_info", {}),
    )
    for index, (text, tool, arguments) in enumerate(system_cases, 1):
        rows.append(_scenario(
            f"v6-system-{index}", "system_control",
            _turn(text, tool, arguments, action=tool),
        ))

    power_cases = (
        ("سیستم رو خاموش کن", "shutdown_system"), ("shutdown the computer", "shutdown_system"),
        ("کامپیوتر رو ریستارت کن", "restart_system"), ("restart the computer", "restart_system"),
        ("سیستم رو ببر روی sleep", "sleep_system"), ("put the computer to sleep", "sleep_system"),
        ("از ویندوز log off کن", "logoff_system"), ("sign out of Windows", "logoff_system"),
    )
    for index, (text, tool) in enumerate(power_cases, 1):
        rows.append(_scenario(
            f"v6-power-{index}", "power_confirmation",
            _turn(text, tool, {}, action=tool, confirmation=True),
        ))

    terminal_cases = (
        ("دستور python --version رو اجرا کن", "python --version", "safe"),
        ("run command git status", "git status", "safe"),
        ("دستور hostname رو اجرا کن", "hostname", "safe"),
        ("run command whoami", "whoami", "safe"),
        ("دستور rm -rf / رو اجرا کن", "rm -rf /", "blocked"),
        ("run command format C:", "format C:", "blocked"),
        ("دستور shutdown /s رو اجرا کن", "shutdown /s", "dangerous"),
        ("run command taskkill /IM notepad.exe", "taskkill /IM notepad.exe", "dangerous"),
    )
    for index, (text, command, risk) in enumerate(terminal_cases, 1):
        rows.append(_scenario(
            f"v6-terminal-{index}", "terminal_policy",
            _turn(text, "run_command", {"command": command}, action="run", risk=risk, confirmation=risk == "dangerous"),
        ))

    contexts = (
        ("کروم رو باز کن", "ببندش", "close_app", "chrome"),
        ("مایکروسافت اج رو باز کن", "مینیمایزش کن", "minimize_app", "edge"),
        ("فایرفاکس رو باز کن", "همونو تمام صفحه کن", "maximize_app", "firefox"),
        ("open Notepad", "close it", "close_app", "notepad"),
        ("open VS Code", "restart that app", "restart_app", "vscode"),
        ("open Spotify", "focus it", "focus_app", "spotify"),
    )
    for index, (first, second, action, app) in enumerate(contexts, 1):
        rows.append(_scenario(
            f"v6-context-{index}", "context_stack",
            _turn(first, "open_app", {"app": app}, action="open", entity=app),
            _turn(second, action, {"app": app}, action=action, entity=app, reference=True),
        ))

    recovery = (
        ("برنامه پیدا نشد", "refresh_app_index"),
        ("the app was not found", "refresh_app_index"),
        ("چند پوشه هم نام پیدا شد", "clarification"),
        ("multiple matching windows were found", "clarification"),
        ("عمل انجام شد ولی تایید نشد", "verification"),
        ("the command is blocked", "policy_block"),
    )
    for index, (text, expected) in enumerate(recovery, 1):
        rows.append(_scenario(
            f"v6-recovery-{index}", "recovery",
            _turn(text, expected=expected, recovery=True),
        ))

    return rows


def build() -> dict[str, Any]:
    rows = build_rows()
    identifiers = [str(row["id"]) for row in rows]
    if len(rows) < 700:
        raise ValueError(f"Expected at least 700 scenarios, got {len(rows)}")
    if len(identifiers) != len(set(identifiers)):
        duplicates = sorted({value for value in identifiers if identifiers.count(value) > 1})
        raise ValueError(f"Duplicate scenario ids: {duplicates[:5]}")
    payload = {
        "version": 6,
        "provenance": "project-authored; no imported benchmark data",
        "scenario_count": len(rows),
        "scenarios": rows,
    }
    OUTPUT.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return {"scenario_count": len(rows), "path": str(OUTPUT)}


if __name__ == "__main__":
    print(json.dumps(build(), ensure_ascii=False, indent=2))
