from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
import unittest
import zipfile
from pathlib import Path

from jarvis import __version__
from jarvis.runtime.bootstrap import build_runtime


ROOT = Path(__file__).resolve().parent


def _core_smoke_test(debug: bool = False) -> dict[str, object]:
    previous_data_dir = os.environ.get("JARVIS_DATA_DIR")
    with tempfile.TemporaryDirectory(prefix="jarvis-v09-self-test-") as temporary:
        temp_path = Path(temporary)
        os.environ["JARVIS_DATA_DIR"] = str(temp_path / "runtime")
        runtime = None
        try:
            runtime = build_runtime(ROOT, debug=debug)
            prompts = ["سلام", "hello", "جارویس", "حالت چطوره؟", "what is your name?"]
            replies = [runtime.agent.respond(prompt) for prompt in prompts]
            runtime.agent.respond("اسم من یونس هست")
            memory_reply = runtime.agent.respond("اسمم چی بود؟")
            text_reasoning = runtime.agent.respond(
                "متن: علی فردا نمی‌آید چون بیمار است. چرا علی نمی‌آید؟"
            )
            formal_reasoning = runtime.agent.respond(
                "همه پرنده‌ها بال دارند و گنجشک پرنده است. چه نتیجه‌ای می‌گیری؟"
            )
            knowledge_reply = runtime.agent.respond("NAT چیکار می‌کند؟")
            runtime.agent.set_personality("Kind")
            kind_reply = runtime.agent.respond("جارویس")

            text_path = temp_path / "sample.txt"
            text_path.write_text("Hello Jarvis\nسلام جارویس", encoding="utf-8")
            text_result = runtime.agent.attach_file(text_path)
            zip_path = temp_path / "sample.zip"
            with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
                archive.writestr("docs/readme.md", "# Jarvis\nOffline test")
                archive.writestr("src/demo.py", "print('ok')\n")
            zip_result = runtime.agent.attach_file(zip_path)

            close_edge = runtime.agent.router.route("مرورگر مایکروسافت edge رو ببند")
            close_default_browser = runtime.agent.router.route("مرورگر پیش فرض سیستم رو ببند")
            drive_c = runtime.agent.router.route("درایو c رو باز کن")
            ambiguous_folder = runtime.agent.router.route("یه فولدر باز کن")
            ambiguous_plan = runtime.agent.planner.plan(
                "یه فولدر باز کن", ambiguous_folder
            )

            first_integrity = runtime.memory.integrity_check()
            runtime.close()
            runtime = build_runtime(ROOT, debug=debug)
            reopened_name = runtime.agent.respond("اسمم چی بود؟")
            parameter_count = runtime.neural.parameter_count if runtime.neural else 0
            checks = {
                "brain": runtime.neural is not None and parameter_count == 23_077_376,
                "database": first_integrity and runtime.memory.integrity_check(),
                "database_reopen": "یونس" in reopened_name.text,
                "basic_intents": [reply.intent for reply in replies]
                == ["greeting", "greeting", "call_assistant", "how_are_you", "ask_name"],
                "memory": "یونس" in memory_reply.text,
                "grounded_text_reasoning": text_reasoning.intent == "grounded_text_answer"
                and "بیمار" in text_reasoning.text,
                "formal_reasoning": formal_reasoning.intent == "grounded_text_answer"
                and "گنجشک" in formal_reasoning.text and "بال" in formal_reasoning.text,
                "expanded_knowledge": knowledge_reply.intent == "knowledge_answer"
                and "IP" in knowledge_reply.text,
                "personality": "جانم" in kind_reply.text,
                "files": text_result.intent == zip_result.intent == "file_attached",
                "tool_registry": {
                    "open_url", "open_app", "close_app", "close_default_browser",
                    "focus_app", "restart_app",
                    "open_folder", "web_search", "calculator", "enumerate_windows",
                    "create_folder", "browser_read_page", "run_command", "battery_info",
                }.issubset(runtime.tools.names()),
                "close_polarity": close_edge.intent == "close_app"
                and close_edge.arguments == {"app": "edge"},
                "default_browser_close": close_default_browser.intent
                == "close_default_browser" and close_default_browser.arguments == {},
                "drive_resolution": drive_c.intent == "open_folder"
                and drive_c.arguments == {"path": "C:\\"},
                "argument_guard": ambiguous_plan is not None
                and not ambiguous_plan.actionable
                and ambiguous_plan.needs_input == "path",
            }
            return {
                "version": __version__,
                "parameters": parameter_count,
                "checks": checks,
                "passed": all(checks.values()),
            }
        finally:
            if runtime is not None:
                runtime.close()
            if previous_data_dir is None:
                os.environ.pop("JARVIS_DATA_DIR", None)
            else:
                os.environ["JARVIS_DATA_DIR"] = previous_data_dir


def _self_test(debug: bool = False) -> int:
    try:
        smoke = _core_smoke_test(debug)
    except Exception as exc:
        print(f"Jarvis core self-test failed: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(smoke, ensure_ascii=False, indent=2))
    if not smoke["passed"]:
        return 1
    suite = unittest.defaultTestLoader.discover(str(ROOT / "tests"), pattern="test_*.py")
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    print(
        json.dumps(
            {
                "tests_run": result.testsRun,
                "failures": len(result.failures),
                "errors": len(result.errors),
                "successful": result.wasSuccessful(),
            },
            indent=2,
        )
    )
    return 0 if result.wasSuccessful() else 1


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="JARVIS v0.11.0 — Bilingual Trained Local Desktop Agent")
    parser.add_argument("--version", action="store_true", help="show version and exit")
    parser.add_argument("--debug", action="store_true", help="enable decision and tool diagnostics")
    parser.add_argument("--self-test", action="store_true", help="run core checks and the complete test suite")
    parser.add_argument(
        "--gui-smoke", action="store_true",
        help="open the GUI and close it automatically (display-enabled environments)",
    )
    args = parser.parse_args(argv)
    if args.version:
        print(f"Jarvis {__version__}")
        return 0
    if args.self_test:
        return _self_test(args.debug)

    runtime = None
    try:
        runtime = build_runtime(ROOT, debug=args.debug)
        from jarvis.gui.app import JarvisGUI

        gui = JarvisGUI(runtime, auto_close_ms=900 if args.gui_smoke else None)
        gui.run()
        return 0
    except Exception as exc:
        if runtime is not None:
            runtime.close()
        print(f"Jarvis could not start: {exc}", file=sys.stderr)
        if "display" in str(exc).casefold():
            print("A graphical desktop session is required for the Tkinter window.", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
