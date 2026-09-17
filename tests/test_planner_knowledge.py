from __future__ import annotations

import unittest
from pathlib import Path

from jarvis.agent.intent_router import FactExtractor, IntentRoute
from jarvis.agent.planner import ToolPlanner
from jarvis.tools.internet import InternetTool, InternetToolError
from tests.helpers import TemporaryRuntime


class KnowledgeAndPlannerTests(unittest.TestCase):
    def test_builtin_knowledge_is_seeded_and_searchable(self) -> None:
        with TemporaryRuntime() as runtime:
            self.assertGreaterEqual(runtime.knowledge.count(), 8)
            hits = runtime.knowledge.search("Python programming language")
            self.assertTrue(hits)
            self.assertIn("python", hits[0].title.casefold())

    def test_unknown_knowledge_query_has_no_false_hit(self) -> None:
        with TemporaryRuntime() as runtime:
            self.assertEqual(runtime.knowledge.search("zxqv plmokn 773"), [])

    def test_user_knowledge_document_can_be_added(self) -> None:
        with TemporaryRuntime() as runtime:
            runtime.knowledge.add_document(
                "Project Aurora",
                "[en] Aurora uses a local SQLite queue.",
                ["aurora", "sqlite", "queue"],
                external_id="test:aurora",
            )
            hits = runtime.knowledge.search("Aurora SQLite")
            self.assertTrue(hits)
            self.assertEqual(hits[0].source, "user")

    def test_file_plan_has_multiple_structured_steps(self) -> None:
        route = IntentRoute("file_read", "tool", 0.9, 0.5)
        plan = ToolPlanner().plan("read the file", route, Path("C:/tmp/demo.txt"))
        self.assertIsNotNone(plan)
        assert plan is not None
        self.assertEqual(len(plan.steps), 3)
        self.assertTrue(plan.actionable)
        payload = plan.to_dict()
        self.assertEqual(payload["type"], "plan")
        self.assertEqual(payload["steps"][1]["tool_call"]["type"], "tool")

    def test_file_plan_requests_attachment_when_missing(self) -> None:
        route = IntentRoute("zip_list", "tool", 0.9, 0.5)
        plan = ToolPlanner().plan("show zip contents", route)
        self.assertIsNotNone(plan)
        assert plan is not None
        self.assertFalse(plan.actionable)
        self.assertEqual(plan.needs_input, "attachment")

    def test_browser_and_search_plans_are_safe_without_confirmation(self) -> None:
        planner = ToolPlanner()
        browser = planner.plan(
            "open https://example.com", IntentRoute("open_browser", "tool", 0.9, 0.5)
        )
        search = planner.plan(
            "search the internet local AI", IntentRoute("internet_search", "tool", 0.9, 0.5)
        )
        self.assertTrue(browser and not browser.needs_confirmation)
        self.assertTrue(search and not search.needs_confirmation)
        assert browser is not None
        self.assertEqual(browser.steps[1].tool_call.tool, "open_default_browser")

    def test_system_information_plan_is_read_only(self) -> None:
        plan = ToolPlanner().plan(
            "system info", IntentRoute("system_info", "tool", 0.9, 0.5)
        )
        self.assertIsNotNone(plan)
        assert plan is not None
        self.assertFalse(plan.needs_confirmation)
        self.assertEqual(plan.steps[1].tool_call.tool, "system_info")

    def test_fact_extractor_supports_persian_and_english(self) -> None:
        self.assertEqual(FactExtractor.user_name("اسم من یونس هست"), "یونس")
        self.assertEqual(FactExtractor.user_name("my name is Alice"), "Alice")
        self.assertIsNone(FactExtractor.user_name("what is my name"))

    def test_internet_tool_rejects_non_http_urls(self) -> None:
        tool = InternetTool()
        for url in ("file:///etc/passwd", "javascript:alert(1)", "not a url"):
            with self.subTest(url=url), self.assertRaises(InternetToolError):
                tool.open_in_browser(url)

    def test_internet_fetch_blocks_private_addresses(self) -> None:
        with self.assertRaises(InternetToolError):
            InternetTool().fetch_text("http://127.0.0.1/private")

    def test_clear_conversation_confirmation_can_be_cancelled(self) -> None:
        with TemporaryRuntime() as runtime:
            first = runtime.agent.respond("گفتگو رو پاک کن")
            second = runtime.agent.respond("نه")
            self.assertEqual(first.intent, "tool_confirm_clear")
            self.assertEqual(second.intent, "tool_cancelled")


if __name__ == "__main__":
    unittest.main()
