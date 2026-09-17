from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest import mock

from jarvis.agent.decision import DecisionEngine
from jarvis.agent.intent_router import IntentRoute
from jarvis.agent.planner import ToolPlanner
from jarvis.search.engine import SearchEvidence, SearchEngine, SearchReport
from jarvis.tools.internet import InternetResult, SearchResult
from tests.helpers import TemporaryRuntime


class IntelligenceV3Tests(unittest.TestCase):
    def test_youtube_and_chrome_are_distinct_entities(self) -> None:
        with TemporaryRuntime() as runtime:
            youtube = runtime.agent.router.route("یوتیوب رو باز کن")
            chrome = runtime.agent.router.route("کروم رو باز کن")
            self.assertEqual((youtube.intent, youtube.arguments["website"]), ("open_url", "youtube"))
            self.assertEqual((chrome.intent, chrome.arguments["app"]), ("open_app", "chrome"))

    def test_google_and_google_chrome_hard_negative(self) -> None:
        with TemporaryRuntime() as runtime:
            google = runtime.agent.router.route("گوگل رو باز کن")
            chrome = runtime.agent.router.route("گوگل کروم رو باز کن")
            self.assertEqual(google.arguments["url"], "https://www.google.com/")
            self.assertEqual(chrome.arguments, {"app": "chrome"})

    def test_default_browser_has_no_fake_url_argument(self) -> None:
        with TemporaryRuntime() as runtime:
            route = runtime.agent.router.route("مرورگر پیش فرض رو باز کن")
            self.assertEqual(route.intent, "open_default_browser")
            self.assertEqual(route.arguments, {})

    def test_safe_open_url_runs_without_confirmation(self) -> None:
        with TemporaryRuntime() as runtime, mock.patch.object(
            runtime.tools, "invoke", return_value=True
        ) as invoked:
            reply = runtime.agent.respond("یوتیوب رو باز کن")
            self.assertEqual(reply.intent, "tool_open_url_result")
            self.assertIn("YouTube", reply.text)
            invoked.assert_called_once_with(
                "open_url",
                {"url": "https://www.youtube.com/", "website": "youtube"},
                confirmed=False,
            )

    def test_safe_open_app_runs_without_confirmation(self) -> None:
        with TemporaryRuntime() as runtime, mock.patch.object(
            runtime.tools, "invoke", return_value="Google Chrome"
        ):
            reply = runtime.agent.respond("کروم رو باز کن")
            call = reply.data["plan"]["steps"][1]["tool_call"]
            self.assertEqual(call["tool"], "open_app")
            self.assertFalse(call["requires_confirmation"])

    def test_youtube_search_extracts_query_and_url(self) -> None:
        with TemporaryRuntime() as runtime:
            route = runtime.agent.router.route("توی یوتیوب سرچ کن آموزش رباتیک")
            self.assertEqual(route.intent, "youtube_search")
            self.assertEqual(route.arguments["query"], "آموزش رباتیک")
            self.assertIn("search_query=", route.arguments["url"])

    def test_web_search_extracts_persian_query(self) -> None:
        with TemporaryRuntime() as runtime:
            route = runtime.agent.router.route("گوگل کن آموزش FastAPI")
            self.assertEqual(route.intent, "web_search")
            self.assertEqual(route.arguments["query"], "آموزش fastapi")

    def test_fresh_information_automatically_routes_to_search(self) -> None:
        for prompt in (
            "آخرین نسخه پایتون چیه؟", "امروز بیت کوین چنده؟",
            "هوا امروز چطوره؟", "latest AI news",
        ):
            with self.subTest(prompt=prompt), TemporaryRuntime() as runtime:
                route = runtime.agent.router.route(prompt)
                self.assertEqual(route.intent, "web_search")
                self.assertEqual(route.decision_mode, "search")

    def test_stable_local_question_does_not_require_search(self) -> None:
        self.assertFalse(DecisionEngine.needs_fresh_information("پایتون چیست؟"))
        with TemporaryRuntime() as runtime:
            self.assertNotEqual(runtime.agent.router.route("پایتون چیست؟").intent, "web_search")

    def test_search_result_is_summarized_with_sources(self) -> None:
        report = SearchReport(
            "latest Python", "Python 3.x is the current stable family.",
            (SearchEvidence("Python", "https://python.org/", "Official", 1.0),), 1,
        )
        with TemporaryRuntime() as runtime, mock.patch.object(
            runtime.tools, "invoke", return_value=report
        ):
            reply = runtime.agent.respond("latest Python version")
            self.assertEqual(reply.intent, "tool_search_result")
            self.assertIn("Sources", reply.text)
            self.assertIn("https://python.org/", reply.text)

    def test_search_status_is_emitted(self) -> None:
        statuses: list[str] = []
        report = SearchReport("news", "summary", (), 0)
        with TemporaryRuntime() as runtime, mock.patch.object(
            runtime.tools, "invoke", return_value=report
        ):
            runtime.agent.set_status_callback(statuses.append)
            runtime.agent.respond("latest AI news")
        self.assertIn("SEARCHING", statuses)
        self.assertEqual(statuses[-1], "READY")

    def test_complex_sqlite_json_question_uses_think_mode(self) -> None:
        statuses: list[str] = []
        with TemporaryRuntime() as runtime:
            runtime.agent.set_status_callback(statuses.append)
            reply = runtime.agent.respond("برای تنظیمات برنامه SQLite بهتره یا JSON و چرا؟")
            self.assertEqual(reply.intent, "reasoned_answer")
            self.assertIn("SQLite", reply.text)
            self.assertIn("JSON", reply.text)
            self.assertIn("THINKING", statuses)
            self.assertIn("self_check_passed", reply.data["self_checks"])

    def test_reasoning_does_not_expose_private_chain_of_thought(self) -> None:
        with TemporaryRuntime() as runtime:
            reply = runtime.agent.respond("compare SQLite and JSON for desktop settings")
            self.assertNotIn("Pass 1", reply.text)
            self.assertNotIn("Chain-of-Thought", reply.text)
            self.assertGreater(reply.data["confidence"], 0.8)

    def test_calculator_bypasses_neural_brain(self) -> None:
        with TemporaryRuntime() as runtime, mock.patch.object(
            runtime.brain, "classify", side_effect=AssertionError("brain should not run")
        ):
            reply = runtime.agent.respond("125 * 43")
            self.assertEqual(reply.intent, "tool_calculation_result")
            self.assertIn("5375", reply.text)

    def test_datetime_tool_returns_date_and_weekday(self) -> None:
        with TemporaryRuntime() as runtime:
            reply = runtime.agent.respond("امروز چه روزیه؟")
            self.assertEqual(reply.intent, "tool_datetime_result")
            self.assertRegex(reply.text, r"\d{4}-\d{2}-\d{2}")

    def test_correction_is_learned_without_retraining(self) -> None:
        with TemporaryRuntime() as runtime:
            learned = runtime.agent.respond(
                "نه، وقتی میگم یوتیوب رو باز کن باید سایت یوتیوب رو باز کنی."
            )
            route = runtime.agent.router.route("یوتیوب رو باز کن")
            self.assertEqual(learned.intent, "correction_learned")
            self.assertEqual(runtime.memory.correction_count(), 1)
            self.assertEqual(route.source, "correction_memory")
            self.assertEqual(route.arguments["website"], "youtube")

    def test_teach_command_can_define_spotify_shortcut(self) -> None:
        with TemporaryRuntime() as runtime:
            runtime.agent.respond("یاد بگیر وقتی گفتم موزیک بزن Spotify رو باز کنی")
            route = runtime.agent.router.route("موزیک بزن")
            self.assertEqual(route.intent, "open_app")
            self.assertEqual(route.arguments["app"], "spotify")

    def test_bare_when_command_stores_project_shortcut(self) -> None:
        with TemporaryRuntime() as runtime:
            learned = runtime.agent.respond(
                "وقتی گفتم بزن بریم پروژه Robot رو باز کن"
            )
            route = runtime.agent.router.route("بزن بریم")
            self.assertEqual(learned.intent, "correction_learned")
            self.assertEqual(route.intent, "open_folder")
            self.assertEqual(route.arguments["path"], "robot")

    def test_feedback_is_stored_for_future_training(self) -> None:
        with TemporaryRuntime() as runtime:
            runtime.agent.respond("سلام")
            row_id = runtime.agent.add_feedback(True, "good answer")
            self.assertGreater(row_id, 0)
            self.assertEqual(runtime.memory.feedback_count(), 1)

    def test_power_action_requires_explicit_confirmation_before_execution(self) -> None:
        with TemporaryRuntime() as runtime:
            first = runtime.agent.respond("سیستم رو خاموش کن")
            self.assertEqual(first.intent, "tool_confirm_dangerous")
            self.assertIn("plan", first.data)
            calls = [
                step.get("tool_call", {})
                for step in first.data["plan"]["steps"]
                if "tool_call" in step
            ]
            self.assertEqual(calls[0]["tool"], "shutdown_system")
            self.assertTrue(calls[0]["requires_confirmation"])

    def test_rejecting_dangerous_action_cancels_it(self) -> None:
        with TemporaryRuntime() as runtime:
            runtime.agent.respond("همه فایل ها رو پاک کن")
            reply = runtime.agent.respond("نه")
            self.assertEqual(reply.intent, "tool_cancelled")

    def test_multi_step_plan_executes_two_safe_calls(self) -> None:
        calls: list[str] = []
        with TemporaryRuntime() as runtime:
            original = runtime.tools.invoke

            def invoke(name: str, arguments: dict[str, object], *, confirmed: bool = False):  # type: ignore[no-untyped-def]
                if name in {"open_app", "open_url"}:
                    calls.append(name)
                    return True
                return original(name, arguments, confirmed=confirmed)

            with mock.patch.object(runtime.tools, "invoke", side_effect=invoke):
                reply = runtime.agent.respond("کروم رو باز کن بعد یوتیوب رو بیار")
            self.assertEqual(calls, ["open_app", "open_url"])
            self.assertEqual(reply.intent, "tool_multi_step_result")
            self.assertEqual(reply.data["step_count"], 2)

    def test_tool_result_updates_task_context(self) -> None:
        with TemporaryRuntime() as runtime, mock.patch.object(
            runtime.tools, "invoke", return_value=True
        ):
            runtime.agent.respond("گوگل رو باز کن")
            context = runtime.memory.get_context(runtime.agent.session_id)
            self.assertEqual(context["last_opened_url"], "https://www.google.com/")
            runtime.agent.respond("حالا یوتیوب رو باز کن")
            self.assertEqual(runtime.memory.get_fact("last_opened_url"), "https://www.youtube.com/")

    def test_attached_file_context_resolves_relative_followup(self) -> None:
        with TemporaryRuntime() as runtime, tempfile.TemporaryDirectory(prefix="jarvis-context-") as temporary:
            path = Path(temporary) / "main.py"
            path.write_text("print('context works')", encoding="utf-8")
            runtime.agent.attach_file(path)
            reply = runtime.agent.respond("حالا main.py رو بخون")
            self.assertEqual(reply.intent, "tool_file_result")
            self.assertIn("context works", reply.text)

    def test_entity_registries_are_config_driven_and_complete(self) -> None:
        with TemporaryRuntime() as runtime:
            self.assertGreaterEqual(len(runtime.entities.websites), 10)
            self.assertGreaterEqual(len(runtime.entities.apps), 7)
            self.assertIsNotNone(runtime.entities.website("youtube"))
            self.assertIsNotNone(runtime.entities.app("powershell"))

    def test_personality_does_not_change_tool_selection(self) -> None:
        routes: set[tuple[str, str]] = set()
        with TemporaryRuntime() as runtime:
            for personality in runtime.personalities.available_names():
                runtime.agent.set_personality(personality)
                route = runtime.agent.router.route("یوتیوب رو باز کن")
                routes.add((route.intent, route.arguments["website"]))
        self.assertEqual(routes, {("open_url", "youtube")})

    def test_search_engine_ranks_and_extracts_bounded_evidence(self) -> None:
        class FakeInternet:
            def search(self, query: str) -> list[SearchResult]:
                return [SearchResult("Python release", "https://example.com/python", "Python release information and stable version details.")]

            def fetch_text(self, url: str) -> InternetResult:
                return InternetResult(
                    url, 200, "text/html",
                    "<html><body><main>Python stable release provides security fixes and language improvements.</main></body></html>",
                    False,
                )

        report = SearchEngine(FakeInternet(), 1).search("Python stable release")  # type: ignore[arg-type]
        self.assertEqual(report.fetched_pages, 1)
        self.assertTrue(report.summary)
        self.assertEqual(len(report.evidence), 1)

    def test_planner_preserves_structured_arguments(self) -> None:
        route = IntentRoute(
            "open_url", "tool", 0.99, 0.8,
            arguments={"url": "https://www.youtube.com/", "website": "youtube"},
        )
        plan = ToolPlanner().plan("open youtube", route)
        self.assertIsNotNone(plan)
        assert plan is not None
        self.assertEqual(plan.steps[1].tool_call.arguments["website"], "youtube")
        self.assertFalse(plan.needs_confirmation)


if __name__ == "__main__":
    unittest.main()
