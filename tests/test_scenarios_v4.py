from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path
from typing import Any
from unittest import mock

from jarvis.runtime.bootstrap import build_runtime
from jarvis.search.engine import SearchEvidence, SearchReport
from jarvis.tools.browser import BrowserActionResult
from jarvis.tools.processes import ProcessResult


ROOT = Path(__file__).resolve().parents[1]


def _load_scenarios() -> list[dict[str, Any]]:
    payload = json.loads(
        (ROOT / "data" / "training" / "scenarios_v4.json").read_text(encoding="utf-8")
    )
    scenarios = [value for value in payload.get("scenarios", []) if isinstance(value, dict)]
    if len(scenarios) < 300:
        raise AssertionError("v0.4 requires at least 300 real scenario tests")
    return scenarios


class StatefulScenarioTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls._temporary = tempfile.TemporaryDirectory(prefix="jarvis-v04-scenarios-")
        cls._previous = os.environ.get("JARVIS_DATA_DIR")
        os.environ["JARVIS_DATA_DIR"] = cls._temporary.name
        cls.runtime = build_runtime(ROOT)

    @classmethod
    def tearDownClass(cls) -> None:
        cls.runtime.close()
        if cls._previous is None:
            os.environ.pop("JARVIS_DATA_DIR", None)
        else:
            os.environ["JARVIS_DATA_DIR"] = cls._previous
        cls._temporary.cleanup()

    @staticmethod
    def _result(name: str, arguments: dict[str, Any], *, confirmed: bool = False) -> Any:
        del confirmed
        if name in {"open_app", "close_app", "focus_app", "restart_app"}:
            app = str(arguments["app"])
            action = name.removesuffix("_app")
            code = "closed" if name == "close_app" else "opened"
            return ProcessResult(True, app, app, action, name != "open_app", True, code)
        if name == "is_app_running":
            return True
        if name == "open_url":
            return BrowserActionResult(
                True, "open_url", str(arguments["url"]),
                str(arguments.get("browser", "default")), True, "opened",
            )
        if name == "open_default_browser":
            return BrowserActionResult(True, "open_browser", "about:blank", "default", False, "opened")
        if name == "web_search":
            if arguments.get("target_browser") or arguments.get("url"):
                return BrowserActionResult(
                    True, "search", str(arguments.get("url", "https://www.google.com/search")),
                    str(arguments.get("target_browser", "default")), True, "opened",
                )
            query = str(arguments["query"])
            return SearchReport(
                query, f"Grounded summary for {query}",
                (SearchEvidence("Source", "https://example.com/", "Evidence", 1.0),), 1,
            )
        if name == "open_folder":
            return str(arguments["path"])
        return True


def _assert_call(
    test: StatefulScenarioTests,
    actual: tuple[str, dict[str, Any]],
    expected: dict[str, Any],
    message: str,
) -> None:
    test.assertEqual(actual[0], expected["tool"], message)
    expected_arguments = expected.get("arguments", {})
    for key, value in expected_arguments.items():
        if key == "path_nonempty":
            test.assertTrue(str(actual[1].get("path", "")).strip(), message)
            continue
        actual_value = actual[1].get(key)
        if key == "query":
            test.assertEqual(str(actual_value).casefold(), str(value).casefold(), message)
        else:
            test.assertEqual(actual_value, value, message)


def _scenario_test(scenario: dict[str, Any]):  # type: ignore[no-untyped-def]
    def test(self: StatefulScenarioTests) -> None:
        self.runtime.agent.new_session()
        captured: list[tuple[str, dict[str, Any]]] = []

        def invoke(name: str, arguments: dict[str, Any], *, confirmed: bool = False) -> Any:
            captured.append((name, dict(arguments)))
            return self._result(name, arguments, confirmed=confirmed)

        with mock.patch.object(self.runtime.tools, "invoke", side_effect=invoke):
            for turn_index, item in enumerate(scenario["turns"]):
                text = str(item["text"])
                expected = dict(item["expected"])
                context = self.runtime.memory.get_context(self.runtime.agent.session_id)
                route = self.runtime.agent.router.route(text, context)
                before = len(captured)
                reply = self.runtime.agent.respond(text)
                actual_calls = captured[before:]
                expected_calls = list(expected.get("calls", []))
                if (
                    expected_calls
                    and not actual_calls
                    and reply.intent.startswith("tool_confirm_")
                ):
                    reply = self.runtime.agent.respond("بله")
                    actual_calls = captured[before:]
                message = f"{scenario['id']} turn {turn_index + 1}: {text}"
                self.assertEqual(len(actual_calls), len(expected_calls), message)
                for actual, wanted in zip(actual_calls, expected_calls, strict=True):
                    _assert_call(self, actual, wanted, message)
                if expected.get("clarification"):
                    self.assertEqual(reply.intent, "tool_clarification", message)
                    self.assertNotIn("Missing tool argument", reply.text, message)
                    self.assertEqual(reply.data.get("missing"), expected.get("missing"), message)
                expected_action = str(expected.get("action", ""))
                if expected_action and expected_action != "multi":
                    self.assertEqual(route.action, expected_action, message)
                entity = str(expected.get("entity", ""))
                if entity and not expected.get("multistep"):
                    self.assertIn(entity, route.entities, message)
                if expected.get("reference"):
                    self.assertTrue(route.referenced, message)
                if expected.get("multistep"):
                    self.assertEqual(route.intent, "multi_step_task", message)
                    self.assertEqual(reply.intent, "tool_multi_step_result", message)
    return test


for _index, _scenario in enumerate(_load_scenarios(), 1):
    setattr(
        StatefulScenarioTests,
        f"test_scenario_{_index:03d}_{str(_scenario['id']).replace('-', '_')}",
        _scenario_test(_scenario),
    )


if __name__ == "__main__":
    unittest.main()
