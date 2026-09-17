from __future__ import annotations

import json
import platform
import tempfile
import unittest
from pathlib import Path

from jarvis.agent.task_context import TaskContext
from jarvis.learning.failures import FailureCollector
from jarvis.tools.files import FileManager, FileToolError
from jarvis.tools.input_control import InputController
from jarvis.tools.system_control import SystemController
from jarvis.tools.terminal import CommandPolicy, CommandRunner
from jarvis.tools.windows import WindowManager
from tests.helpers import TemporaryRuntime


ROOT = Path(__file__).resolve().parents[1]


class DesktopAgentV06Tests(unittest.TestCase):
    def test_v6_dataset_has_fourteen_populated_curriculum_stages(self) -> None:
        manifest = json.loads(
            (ROOT / "datasets" / "manifest_v002.json").read_text(encoding="utf-8")
        )
        stages = manifest["curriculum"]
        self.assertEqual([row["stage"] for row in stages], list(range(1, 15)))
        self.assertTrue(all(row["examples"] > 0 for row in stages))
        self.assertEqual(manifest["examples"], sum(manifest["splits"].values()))
        self.assertEqual(manifest["pretrained_sources"], [])

    def test_v6_scenario_bank_is_large_unique_and_project_owned(self) -> None:
        payload = json.loads(
            (ROOT / "data" / "training" / "scenarios_v6.json").read_text(encoding="utf-8")
        )
        scenarios = payload["scenarios"]
        identifiers = [row["id"] for row in scenarios]
        self.assertGreaterEqual(len(scenarios), 800)
        self.assertEqual(payload["scenario_count"], len(scenarios))
        self.assertEqual(len(identifiers), len(set(identifiers)))
        self.assertIn("project", payload["provenance"].casefold())
        self.assertTrue(all(row.get("turns") for row in scenarios))

    def test_required_local_knowledge_and_creative_prompts(self) -> None:
        with TemporaryRuntime() as runtime:
            cases = {
                "بیضه چیست؟": "تستوسترون",
                "Why is the sky blue?": "Rayleigh",
                "RAM چیست؟": "حافظه",
                "یک جوک بگو": "باگ",
                "یک داستان کوتاه بگو": "جارویس",
            }
            for prompt, fragment in cases.items():
                with self.subTest(prompt=prompt):
                    reply = runtime.agent.respond(prompt)
                    self.assertIn(fragment.casefold(), reply.text.casefold())
                    self.assertNotEqual(reply.intent, "tool_failure")

    def test_desktop_routes_extract_actions_and_arguments(self) -> None:
        with TemporaryRuntime() as runtime:
            cases = (
                ("صدا را روی ۳۰ درصد بگذار", "set_volume", {"percent": 30}),
                ("دستور python --version را اجرا کن", "run_command", {"command": "python --version"}),
                ("همه مرورگرها رو ببند", "close_all_browsers", {}),
                ("سیستم رو خاموش کن", "shutdown_system", {}),
                ("یک پوشه به اسم Test روی Desktop بساز", "create_folder", {}),
            )
            for text, intent, arguments in cases:
                with self.subTest(text=text):
                    route = runtime.agent.router.route(text)
                    self.assertEqual(route.intent, intent)
                    for key, value in arguments.items():
                        self.assertEqual(route.arguments.get(key), value)
            self.assertTrue(runtime.agent.router.route("سیستم رو خاموش کن").requires_confirmation)

    def test_pronoun_resolution_uses_entity_context_stack(self) -> None:
        with TemporaryRuntime() as runtime:
            context = TaskContext.update({}, "open_app", {"app": "chrome"})
            route = runtime.agent.router.route("ببندش", context)
            self.assertEqual(route.intent, "close_app")
            self.assertEqual(route.arguments.get("app"), "chrome")
            self.assertTrue(route.referenced)

    def test_command_policy_blocks_composition_and_bulk_deletion(self) -> None:
        self.assertEqual(CommandPolicy.assess("python --version").level, "safe")
        self.assertEqual(CommandPolicy.assess("git status").level, "dangerous")
        self.assertEqual(CommandPolicy.assess("git --version").level, "safe")
        self.assertEqual(CommandPolicy.assess("git -c alias.pwn=!calc pwn").level, "blocked")
        self.assertEqual(CommandPolicy.assess("python script.py").level, "dangerous")
        for command in ("del /s C:\\*", "rm -rf /tmp/example", "whoami && shutdown /s"):
            with self.subTest(command=command):
                self.assertEqual(CommandPolicy.assess(command).level, "blocked")
        result = CommandRunner().run("python --version")
        self.assertTrue(result.success)
        self.assertTrue(result.verified)
        self.assertIn("Python", result.stdout or result.stderr)

    def test_failure_queue_redacts_secrets_and_requires_review(self) -> None:
        with tempfile.TemporaryDirectory(prefix="jarvis-v06-failures-") as temporary:
            collector = FailureCollector(Path(temporary) / "failures.jsonl")
            row = collector.add(
                user_input="run token=private-value",
                predicted_intent="run_command",
                arguments={"password": "hidden", "command": "python app.py"},
                failure_code="policy_block",
            )
            stored = collector.rows("pending")[0]
            self.assertNotIn("private-value", json.dumps(stored))
            self.assertEqual(stored["prediction"]["arguments"]["password"], "[REDACTED]")
            self.assertEqual(collector.approved_training_rows(), [])
            self.assertTrue(
                collector.review(
                    row["id"], "approved", correct_action="run_command",
                    correct_arguments={"command": "python --version"},
                )
            )
            self.assertEqual(len(collector.approved_training_rows()), 1)

    def test_bounded_file_lifecycle_and_empty_folder_only_deletion(self) -> None:
        with TemporaryRuntime() as runtime, tempfile.TemporaryDirectory(
            prefix="jarvis-v06-files-"
        ) as temporary:
            manager = FileManager(runtime.config.files)
            root = Path(temporary)
            folder = root / "work"
            created_folder = manager.create_folder(folder)
            self.assertTrue(created_folder.success and created_folder.verified)
            file_path = folder / "note.txt"
            manager.create_file(file_path, "سلام")
            manager.append_text(file_path, " JARVIS")
            self.assertEqual(manager.inspect(file_path).preview, "سلام JARVIS")
            with self.assertRaises(FileToolError):
                manager.delete_folder(folder)
            self.assertTrue(manager.delete_file(file_path).verified)
            self.assertTrue(manager.delete_folder(folder).verified)

    def test_platform_specific_controls_fail_honestly_off_windows(self) -> None:
        if platform.system() == "Windows":
            self.skipTest("Non-Windows honesty check")
        window_result = WindowManager().close("Jarvis nonexistent window")
        input_result = InputController().type_text("سلام")
        system_result = SystemController().set_volume(20)
        for result in (window_result, input_result, system_result):
            self.assertFalse(result.success)
            self.assertTrue(result.verified)
            self.assertEqual(result.code, "unsupported_platform")

    def test_registry_exposes_v6_capabilities_and_sensitive_click_risk(self) -> None:
        with TemporaryRuntime() as runtime:
            self.assertGreaterEqual(len(runtime.tools.names()), 100)
            sensitive = runtime.tools.spec("browser_click")
            self.assertIsNotNone(sensitive)
            assert sensitive is not None and sensitive.risk_resolver is not None
            self.assertEqual(
                sensitive.risk_resolver({"label": "پرداخت نهایی"}),
                "privileged",
            )
            shutdown = runtime.tools.spec("shutdown_system")
            self.assertIsNotNone(shutdown)
            self.assertEqual(shutdown.category, "power_control")  # type: ignore[union-attr]


if __name__ == "__main__":
    unittest.main()
