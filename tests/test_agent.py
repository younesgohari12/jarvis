from __future__ import annotations

import tempfile
import unittest
import zipfile
from pathlib import Path
from unittest import mock

from jarvis.search.engine import SearchEvidence, SearchReport
from tests.helpers import TemporaryRuntime


class AgentIntegrationTests(unittest.TestCase):
    def test_required_bilingual_conversation(self) -> None:
        with TemporaryRuntime() as runtime:
            expected = {
                "سلام": "greeting",
                "hello": "greeting",
                "جارویس": "call_assistant",
                "حالت چطوره؟": "how_are_you",
                "what is your name?": "ask_name",
            }
            for prompt, intent in expected.items():
                with self.subTest(prompt=prompt):
                    self.assertEqual(runtime.agent.respond(prompt).intent, intent)

    def test_name_fact_survives_follow_up(self) -> None:
        with TemporaryRuntime() as runtime:
            saved = runtime.agent.respond("اسم من یونس هست")
            answer = runtime.agent.respond("اسمم چی بود؟")
            self.assertEqual(saved.intent, "set_user_name")
            self.assertEqual(answer.intent, "ask_user_name_known")
            self.assertIn("یونس", answer.text)

    def test_low_mood_context_is_supportive(self) -> None:
        with TemporaryRuntime() as runtime:
            answer = runtime.agent.respond("حالم خیلی بده")
            self.assertEqual(answer.intent, "context_low_mood")
            self.assertIn("امروز", answer.text)
            self.assertEqual(runtime.memory.get_context(runtime.agent.session_id)["topic"], "wellbeing")

    def test_project_context_handles_short_follow_up(self) -> None:
        with TemporaryRuntime() as runtime:
            first = runtime.agent.respond("دارم روی یه پروژه کار می‌کنم")
            second = runtime.agent.respond("یک برنامه پایتون")
            self.assertEqual(first.intent, "context_project_start")
            self.assertEqual(second.intent, "context_project_kind")
            self.assertEqual(runtime.memory.get_fact("last_project"), "python")

    def test_general_question_is_grounded_in_local_knowledge(self) -> None:
        with TemporaryRuntime() as runtime:
            answer = runtime.agent.respond("چرا آسمون آبیه؟")
            self.assertEqual(answer.intent, "knowledge_answer")
            self.assertIn("پراکندگی", answer.text)
            self.assertEqual(answer.data["reasoning_source"], "local_knowledge")

    def test_local_knowledge_answers_supported_question(self) -> None:
        with TemporaryRuntime() as runtime:
            answer = runtime.agent.respond("what is Python?")
            self.assertEqual(answer.intent, "knowledge_answer")
            self.assertIn("programming language", answer.text)
            self.assertEqual(answer.data["reasoning_source"], "local_knowledge")

    def test_word_inside_language_question_is_not_treated_as_greeting(self) -> None:
        with TemporaryRuntime() as runtime:
            answer = runtime.agent.respond("مخفف انگلیسی کلمه سلام چیست؟")
            self.assertEqual(answer.intent, "knowledge_answer")
            self.assertIn("Hello", answer.text)
            self.assertIn("Hi", answer.text)
            self.assertIn("مخفف", answer.text)

    def test_all_personality_call_overrides_are_distinct(self) -> None:
        expected_fragments = {
            "Normal": "اینجام",
            "Kind": "جانم",
            "Angry": "چیه باز",
            "Loti": "داش",
            "Gang": "Yeah",
            "Professional": "در خدمتم",
            "Funny": "آفلاین",
        }
        answers: set[str] = set()
        with TemporaryRuntime() as runtime:
            for name, fragment in expected_fragments.items():
                with self.subTest(personality=name):
                    runtime.agent.set_personality(name)
                    answer = runtime.agent.respond("جارویس").text
                    self.assertIn(fragment, answer)
                    answers.add(answer)
        self.assertEqual(len(answers), len(expected_fragments))

    def test_personality_changes_safe_tool_tone_not_semantics(self) -> None:
        with TemporaryRuntime() as runtime:
            with mock.patch.object(runtime.tools, "invoke", return_value="Google Chrome"):
                runtime.agent.set_personality("Angry")
                angry = runtime.agent.respond("کروم رو باز کن")
                runtime.agent.set_personality("Kind")
                kind = runtime.agent.respond("کروم رو باز کن")
            self.assertEqual(angry.intent, "tool_open_app_result")
            self.assertEqual(kind.intent, "tool_open_app_result")
            self.assertIn("کروم", angry.text)
            self.assertIn("کروم", kind.text)
            self.assertNotEqual(angry.text, kind.text)
            self.assertFalse(angry.data["plan"]["steps"][1]["tool_call"]["requires_confirmation"])

    def test_attach_text_and_zip_with_follow_up(self) -> None:
        with TemporaryRuntime() as runtime, tempfile.TemporaryDirectory(prefix="jarvis-agent-") as temporary:
            root = Path(temporary)
            text_path = root / "note.txt"
            text_path.write_text("سلام از فایل", encoding="utf-8")
            text_reply = runtime.agent.attach_file(text_path, "fa")
            self.assertEqual(text_reply.event, "attachment")
            self.assertIn("سلام از فایل", text_reply.text)

            zip_path = root / "code.zip"
            with zipfile.ZipFile(zip_path, "w") as archive:
                archive.writestr("src/main.py", "print('Jarvis')")
            runtime.agent.attach_file(zip_path, "fa")
            follow_up = runtime.agent.respond("ساختار فایل زیپ رو بگو")
            self.assertEqual(follow_up.intent, "tool_file_result")
            self.assertIn("src/main.py", follow_up.text)

    def test_explicit_search_runs_without_confirmation_and_can_be_disabled(self) -> None:
        with TemporaryRuntime() as runtime:
            self.assertTrue(runtime.agent.internet_enabled)
            report = SearchReport(
                "پایتون", "Python is a programming language.",
                (SearchEvidence("Python", "https://python.org/", "Official site", 1.0),), 1,
            )
            with mock.patch.object(runtime.tools, "invoke", return_value=report) as invoked:
                answer = runtime.agent.respond("در اینترنت درباره پایتون جستجو کن")
            self.assertEqual(answer.intent, "tool_search_result")
            self.assertIn("https://python.org/", answer.text)
            invoked.assert_called_once()
            runtime.agent.update_setting("internet_enabled", False)
            disabled = runtime.agent.respond("آخرین نسخه پایتون چیه؟")
            self.assertEqual(disabled.intent, "internet_disabled")

    def test_user_settings_are_persisted(self) -> None:
        with TemporaryRuntime() as runtime:
            runtime.agent.update_setting("performance", "LOW")
            runtime.agent.update_setting("timestamps", True)
            runtime.agent.set_personality("Funny")
            self.assertEqual(runtime.memory.get_setting("performance"), "LOW")
            self.assertTrue(runtime.memory.get_bool_setting("timestamps"))
            self.assertEqual(runtime.memory.get_setting("personality"), "Funny")

    def test_new_session_keeps_long_term_fact(self) -> None:
        with TemporaryRuntime() as runtime:
            runtime.agent.respond("اسم من یونسه")
            old_session = runtime.agent.session_id
            new_session = runtime.agent.new_session()
            answer = runtime.agent.respond("اسمم چی بود؟")
            self.assertNotEqual(old_session, new_session)
            self.assertIn("یونس", answer.text)

    def test_brain_databases_and_tools_are_available(self) -> None:
        with TemporaryRuntime() as runtime:
            self.assertIsNotNone(runtime.brain)
            self.assertTrue(runtime.memory.integrity_check())
            self.assertTrue(runtime.knowledge.integrity_check())
            self.assertTrue({"file_inspect", "system_info", "internet_search"}.issubset(runtime.tools.names()))


if __name__ == "__main__":
    unittest.main()
