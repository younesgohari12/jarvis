from __future__ import annotations

import unittest
from unittest import mock

from jarvis.agent.task_context import TaskContext
from jarvis.nlu.action_parser import Action, ActionParser
from jarvis.nlu.task_parser import TaskSegmenter
from jarvis.tools.browser import BrowserActionResult
from jarvis.tools.processes import ProcessResult
from jarvis.tools.registry import ArgumentValidationError
from tests.helpers import TemporaryRuntime


def _process_result(name: str, arguments: dict[str, object], *, confirmed: bool = False):  # type: ignore[no-untyped-def]
    del confirmed
    app = str(arguments.get("app", ""))
    labels = {"chrome": "Google Chrome", "edge": "Microsoft Edge", "steam": "Steam"}
    if name in {"open_app", "close_app", "focus_app", "restart_app"}:
        return ProcessResult(
            True, app, labels.get(app, app), name.removesuffix("_app"),
            name != "open_app", True, "closed" if name == "close_app" else "opened",
        )
    if name == "web_search":
        return BrowserActionResult(
            True, "search", "https://www.google.com/search", str(arguments.get("target_browser", "default")),
            True, "opened",
        )
    if name == "open_folder":
        return str(arguments["path"])
    if name == "open_url":
        return BrowserActionResult(True, "open_url", str(arguments["url"]), "default", True, "opened")
    return True


class StatefulAgentV4Tests(unittest.TestCase):
    def test_close_edge_polarity_regression(self) -> None:
        with TemporaryRuntime() as runtime:
            route = runtime.agent.router.route("مرورگر مایکروسافت edge رو ببند")
            self.assertEqual(route.intent, "close_app")
            self.assertEqual(route.arguments, {"app": "edge"})
            self.assertEqual(route.action, Action.CLOSE.value)

    def test_close_pronoun_uses_last_app(self) -> None:
        with TemporaryRuntime() as runtime, mock.patch.object(
            runtime.tools, "invoke", side_effect=_process_result
        ) as invoked:
            runtime.agent.respond("کروم رو باز کن")
            confirmation = runtime.agent.respond("ببندش")
            self.assertEqual(confirmation.intent, "tool_confirm_dangerous")
            reply = runtime.agent.respond("بله")
            self.assertEqual(reply.intent, "tool_close_app_result")
            self.assertIn("بستم", reply.text)
            self.assertEqual(invoked.call_args_list[-1].args[:2], ("close_app", {"app": "chrome"}))

    def test_explicit_browser_reference_uses_last_browser(self) -> None:
        with TemporaryRuntime() as runtime, mock.patch.object(
            runtime.tools, "invoke", side_effect=_process_result
        ):
            runtime.agent.respond("Edge رو باز کن")
            confirmation = runtime.agent.respond("همون مرورگری که باز کردی رو ببند")
            self.assertEqual(confirmation.intent, "tool_confirm_dangerous")
            reply = runtime.agent.respond("بله")
            call = next(
                step["tool_call"] for step in reply.data["plan"]["steps"] if "tool_call" in step
            )
            self.assertEqual((call["tool"], call["arguments"]), ("close_app", {"app": "edge"}))

    def test_open_chrome_and_search_executes_both_steps(self) -> None:
        with TemporaryRuntime() as runtime, mock.patch.object(
            runtime.tools, "invoke", side_effect=_process_result
        ) as invoked:
            reply = runtime.agent.respond("کروم رو باز کن و یونس گوهری رو سرچ کن داخلش")
            self.assertEqual([call.args[0] for call in invoked.call_args_list], ["open_app", "web_search"])
            self.assertEqual(invoked.call_args_list[1].args[1]["target_browser"], "chrome")
            self.assertIn("یونس گوهری", reply.text)
            self.assertEqual(reply.data["step_count"], 2)

    def test_open_chrome_then_youtube_targets_chrome(self) -> None:
        with TemporaryRuntime() as runtime, mock.patch.object(
            runtime.tools, "invoke", side_effect=_process_result
        ) as invoked:
            runtime.agent.respond("کروم رو باز کن بعد برو یوتیوب")
            self.assertEqual([call.args[0] for call in invoked.call_args_list], ["open_app", "open_url"])
            self.assertEqual(invoked.call_args_list[1].args[1]["browser"], "chrome")

    def test_drive_then_child_folder_is_windows_path(self) -> None:
        with TemporaryRuntime() as runtime:
            route = runtime.agent.router.route("درایو C رو باز کن و پوشه Users رو باز کن")
            self.assertEqual(route.intent, "multi_step_task")
            context: dict[str, object] = {"_planning_chain": True}
            first = runtime.agent.router.route(route.arguments["segments"][0], context, allow_multi=False)
            context = TaskContext.update(context, "open_folder", first.arguments)
            context["_planning_chain"] = True
            second = runtime.agent.router.route(route.arguments["segments"][1], context, allow_multi=False)
            self.assertEqual(second.arguments["path"], "C:\\Users")

    def test_downloads_then_find_zip_uses_folder_context(self) -> None:
        with TemporaryRuntime() as runtime:
            route = runtime.agent.router.route("پوشه Downloads رو باز کن و فایل zip رو پیدا کن")
            context: dict[str, object] = {"_planning_chain": True}
            first = runtime.agent.router.route(route.arguments["segments"][0], context, allow_multi=False)
            context = TaskContext.update(context, "open_folder", first.arguments)
            context["_planning_chain"] = True
            second = runtime.agent.router.route(route.arguments["segments"][1], context, allow_multi=False)
            self.assertEqual(second.intent, "find_file")
            self.assertTrue(second.arguments["folder"])
            self.assertEqual(second.arguments["query"], "zip")

    def test_drive_c_resolves_to_windows_root(self) -> None:
        with TemporaryRuntime() as runtime:
            route = runtime.agent.router.route("درایو c رو باز کن")
            self.assertEqual(route.arguments, {"path": "C:\\"})

    def test_ambiguous_folder_clarifies_without_invocation(self) -> None:
        with TemporaryRuntime() as runtime, mock.patch.object(runtime.tools, "invoke") as invoked:
            reply = runtime.agent.respond("یک فولدر باز کن")
            invoked.assert_not_called()
            self.assertEqual(reply.intent, "tool_clarification")
            self.assertEqual(reply.text, "کدوم فولدر رو باز کنم؟")

    def test_ambiguous_app_clarifies_without_invocation(self) -> None:
        with TemporaryRuntime() as runtime, mock.patch.object(runtime.tools, "invoke") as invoked:
            reply = runtime.agent.respond("یه برنامه باز کن")
            invoked.assert_not_called()
            self.assertEqual(reply.intent, "tool_clarification")
            self.assertIn("کدوم برنامه", reply.text)

    def test_folder_clarification_followup_executes_resolved_drive(self) -> None:
        with TemporaryRuntime() as runtime, mock.patch.object(
            runtime.tools, "invoke", side_effect=_process_result
        ) as invoked:
            first = runtime.agent.respond("یه فولدر باز کن")
            second = runtime.agent.respond("درایو سی")
            self.assertEqual(first.intent, "tool_clarification")
            self.assertEqual(second.intent, "tool_folder_opened")
            invoked.assert_called_once_with("open_folder", {"path": "C:\\"}, confirmed=False)

    def test_app_clarification_followup_resolves_static_app(self) -> None:
        with TemporaryRuntime() as runtime, mock.patch.object(
            runtime.tools, "invoke", side_effect=_process_result
        ) as invoked:
            runtime.agent.respond("یه برنامه باز کن")
            reply = runtime.agent.respond("استیم")
            self.assertEqual(reply.intent, "tool_open_app_result")
            invoked.assert_called_once_with("open_app", {"app": "steam"}, confirmed=False)

    def test_registry_rejects_empty_required_argument(self) -> None:
        with TemporaryRuntime() as runtime:
            with self.assertRaises(ArgumentValidationError):
                runtime.tools.invoke("open_app", {"app": ""})

    def test_steam_not_found_is_natural_and_hides_internal_error(self) -> None:
        with TemporaryRuntime() as runtime, mock.patch.object(
            runtime.processes, "open_app",
            return_value=ProcessResult(False, "steam", "Steam", "open", False, True, "not_found"),
        ):
            reply = runtime.agent.respond("برنامه steam رو باز کن")
            self.assertIn("پیدا نکردم", reply.text)
            self.assertNotIn("Tool open_app failed", reply.text)
            self.assertNotIn("Missing tool argument", reply.text)

    def test_failed_verification_does_not_claim_success(self) -> None:
        with TemporaryRuntime() as runtime, mock.patch.object(
            runtime.tools, "invoke",
            return_value=ProcessResult(False, "edge", "Microsoft Edge", "close", True, True, "still_running"),
        ):
            confirmation = runtime.agent.respond("Edge رو ببند")
            self.assertEqual(confirmation.intent, "tool_confirm_dangerous")
            reply = runtime.agent.respond("بله")
            self.assertEqual(reply.intent, "tool_failure")
            self.assertIn("هنوز در حال اجراست", reply.text)
            self.assertNotIn("رو بستم", reply.text)

    def test_context_stack_is_bounded_and_entity_based(self) -> None:
        state: dict[str, object] = {}
        for app in ("chrome", "edge", "firefox", "steam", "telegram", "discord", "spotify"):
            state = TaskContext.update(state, "open_app", {"app": app})
        self.assertEqual(len(state["recent_entities"]), 5)
        self.assertEqual(state["last_entity"], {"type": "app", "id": "spotify", "label": "spotify"})

    def test_task_segmenter_does_not_split_ordinary_conjunction(self) -> None:
        self.assertEqual(
            TaskSegmenter.split("مزایا و معایب SQLite و JSON رو مقایسه کن"),
            ("مزایا و معایب SQLite و JSON رو مقایسه کن",),
        )

    def test_task_segmenter_splits_two_actions(self) -> None:
        self.assertEqual(
            TaskSegmenter.split("کروم رو باز کن و OpenAI رو سرچ کن داخلش"),
            ("کروم رو باز کن", "OpenAI رو سرچ کن داخلش"),
        )

    def test_action_parser_close_wins_over_open_entity_words(self) -> None:
        self.assertEqual(ActionParser.parse("همون مرورگری که باز کردی رو ببند").action, Action.CLOSE)

    def test_capability_registry_exposes_required_arguments_and_risk(self) -> None:
        with TemporaryRuntime() as runtime:
            schemas = {item["name"]: item for item in runtime.tools.capabilities()}
            self.assertEqual(schemas["close_app"]["required"], ["app"])
            self.assertEqual(schemas["close_app"]["permission_level"], "L3")

    def test_tool_aware_system_question_collects_system_info(self) -> None:
        with TemporaryRuntime() as runtime, mock.patch.object(
            runtime.tools, "invoke",
            return_value={
                "cpu": "Test CPU", "ram_total_mb": 8192,
                "physical_cores": 4, "logical_cores": 8,
            },
        ) as invoked:
            reply = runtime.agent.respond("CPU من برای اجرای این پروژه خوبه؟")
            invoked.assert_called_once_with("system_info", {}, confirmed=False)
            self.assertEqual(reply.intent, "system_assessment")
            self.assertIn("Test CPU", reply.text)


if __name__ == "__main__":
    unittest.main()
