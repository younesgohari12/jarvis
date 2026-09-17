from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest import mock

from jarvis.agent.dialogue_subject import DialogueSubjectResolver
from jarvis.agent.task_context import TaskContext
from jarvis.gui.chat_surface import ModernChatSurface
from jarvis.gui.app import JarvisGUI
from jarvis.research.engine import ResearchEngine, ResearchReport
from jarvis.search.engine import EvidencePassage, SearchEngine, SearchEvidence, SearchReport
from jarvis.tools.internet import SearchDiagnostics, SearchResult
from jarvis.tools.processes import ProcessManager, ProcessResult, RunningProcess
from tests.helpers import TemporaryRuntime


class _InternetFixture:
    def search_detailed(self, _query: str) -> SearchDiagnostics:
        return SearchDiagnostics(
            (
                SearchResult(
                    "تعداد گل‌های کریستیانو رونالدو",
                    "https://bad.example/ronaldo",
                    "کریستیانو رونالدو بیش از ۹۰۰ گل زده است.",
                    "fixture",
                    "کریستیانو رونالدو بیش از ۹۰۰ گل زده است.",
                ),
                SearchResult(
                    "آمار دوران حرفه‌ای لیونل مسی",
                    "https://good.example/messi",
                    "لیونل مسی صدها گل رسمی در دوران حرفه‌ای خود ثبت کرده است.",
                    "fixture",
                    "لیونل مسی صدها گل رسمی در دوران حرفه‌ای خود ثبت کرده است.",
                ),
            ),
            ("fixture",),
            (),
        )


class _AppIndexFixture:
    class App:
        app_id = "chrome"
        name = "Google Chrome"
        process_names = ("chrome.exe",)
        launchers = ("chrome.exe",)

    def resolve(self, _query: str, refresh_if_missing: bool = True) -> object:
        return self.App()


class ContextResearchUIV072Tests(unittest.TestCase):
    def test_subject_extraction_normalizes_arabic_persian_letters(self) -> None:
        subject = DialogueSubjectResolver.extract_subject(
            "درباره ليونل مسي تحقيق کن"
        )
        self.assertEqual(subject, "لیونل مسی")

    def test_short_information_followup_inherits_subject(self) -> None:
        resolved = DialogueSubjectResolver.resolve(
            "چقدر گل زده تا الان؟",
            {"last_information_subject": "لیونل مسی"},
        )
        self.assertTrue(resolved.used_context)
        self.assertEqual(resolved.subject, "لیونل مسی")
        self.assertIn("لیونل مسی", resolved.query)

    def test_explicit_new_research_subject_replaces_old_subject(self) -> None:
        resolved = DialogueSubjectResolver.resolve(
            "درباره کریستیانو رونالدو تحقیق کن",
            {"last_information_subject": "لیونل مسی"},
        )
        self.assertFalse(resolved.used_context)
        self.assertEqual(resolved.subject, "کریستیانو رونالدو")

    def test_research_context_saves_information_subject(self) -> None:
        result = ResearchReport("q", ("q",), "answer", (), 1.0, 1, subject="Ada Lovelace")
        state = TaskContext.update(
            {}, "web_research", {"query": "research Ada Lovelace", "subject": "Ada Lovelace"}, result
        )
        self.assertEqual(state["last_information_subject"], "Ada Lovelace")
        self.assertEqual(state["last_research_subject"], "Ada Lovelace")

    def test_search_subject_lock_rejects_wrong_person(self) -> None:
        report = SearchEngine(_InternetFixture(), fetch_top_pages=0).search(  # type: ignore[arg-type]
            "لیونل مسی چقدر گل زده تا الان؟", subject="لیونل مسی"
        )
        self.assertTrue(report.evidence)
        self.assertTrue(all("messi" in item.url for item in report.evidence))
        self.assertNotIn("رونالدو", report.summary)

    def test_research_synthesis_deduplicates_urls_and_drops_noise(self) -> None:
        encoded = "https://fa.wikipedia.org/wiki/%D9%84%DB%8C%D9%88%D9%86%D9%84_%D9%85%D8%B3%DB%8C"
        unicode_url = "https://fa.wikipedia.org/wiki/لیونل_مسی"
        wrong_url = "https://bad.example/ronaldo"
        sources = (
            SearchEvidence("لیونل مسی", encoded, "", 0.9, source_quality=0.76),
            SearchEvidence("لیونل مسی - ویکی‌پدیا", unicode_url, "", 0.88, source_quality=0.76),
            SearchEvidence("رونالدو", wrong_url, "", 0.8, source_quality=0.4),
        )
        passages = (
            EvidencePassage(
                "لیونل مسی بازیکن فوتبال اهل آرژانتین است و در سطح اول فوتبال جهان بازی کرده است.",
                encoded, "لیونل مسی", 0.9, 1,
            ),
            EvidencePassage(
                "عکس کودکی لیونل مسی و ثروت و همسر او را در این مطلب ببینید.",
                unicode_url, "لیونل مسی", 0.7, 2,
            ),
            EvidencePassage(
                "کریستیانو رونالدو بیش از ۹۰۰ گل زده است.",
                wrong_url, "رونالدو", 0.8, 3,
            ),
        )
        raw = SearchReport("مسی", "", sources, 0, 0.8, "verified", passages)
        summary, selected, confidence, status = ResearchEngine._synthesize(
            "درباره لیونل مسی تحقیق کن", "لیونل مسی", raw
        )
        self.assertIn("جمع‌بندی تحقیق", summary)
        self.assertNotIn("رونالدو", summary)
        self.assertNotIn("عکس کودکی", summary)
        self.assertEqual(len(selected), 1)
        self.assertEqual(summary.count("[1]"), 1)
        self.assertLessEqual(confidence, 0.72)
        self.assertEqual(status, "supported")

    def test_runtime_carries_messi_into_goal_followup(self) -> None:
        source = SearchEvidence(
            "Lionel Messi career goals", "https://example.org/messi",
            "Lionel Messi career statistics", 0.91, "fixture", "example.org", 0.85,
        )
        research = ResearchReport(
            "درباره لیونل مسی", ("درباره لیونل مسی",),
            "جمع‌بندی تحقیق:\nلیونل مسی فوتبالیست آرژانتینی است. [1]",
            (source,), 1.0, 1, 0.8, "supported", ("fixture",), (), "لیونل مسی",
        )
        followup = SearchReport(
            "لیونل مسی چقدر گل زده", "آمار رسمی لیونل مسی در منبع ثبت شده است. [1]",
            (source,), 0, 0.79, "supported",
            (EvidencePassage("آمار رسمی لیونل مسی در منبع ثبت شده است.", source.url, source.title, 0.9, 1),),
            ("fixture",),
        )
        with TemporaryRuntime() as runtime:
            original = runtime.tools.invoke
            calls: list[tuple[str, dict[str, object]]] = []

            def invoke(name: str, args: dict[str, object] | None = None, **kwargs: object) -> object:
                if name in {"web_research", "web_search"}:
                    calls.append((name, dict(args or {})))
                    return research if name == "web_research" else followup
                return original(name, args, **kwargs)

            with mock.patch.object(runtime.tools, "invoke", side_effect=invoke):
                runtime.agent.respond("درباره ليونل مسي تحقيق کن")
                answer = runtime.agent.respond("چقدر گل زده تا الان؟")
        self.assertEqual(calls[1][0], "web_search")
        self.assertEqual(calls[1][1]["subject"], "لیونل مسی")
        self.assertIn("لیونل مسی", str(calls[1][1]["query"]))
        self.assertNotIn("رونالدو", answer.text)

    def test_greeting_prefix_does_not_hide_a_real_question(self) -> None:
        with TemporaryRuntime() as runtime, mock.patch.object(
            runtime.tools, "invoke", side_effect=AssertionError("web/tool call was not expected")
        ):
            answer = runtime.agent.respond("سلام بيضه چيست")
        self.assertEqual(answer.intent, "knowledge_answer")
        self.assertIn("تستوسترون", answer.text)

    def test_language_followups_keep_the_referenced_word(self) -> None:
        with TemporaryRuntime() as runtime:
            runtime.agent.update_setting("internet_enabled", False)
            first = runtime.agent.respond("مخفف انگليسي کلمه سلام چيست")
            second = runtime.agent.respond("و کلمه کير چطور؟")
            third = runtime.agent.respond("خب مخفف انگليسيش چيميشه؟")
        self.assertIn("Hello", first.text)
        self.assertIn("penis", second.text)
        self.assertIn("penis", third.text)
        self.assertEqual(second.intent, "knowledge_answer")
        self.assertTrue(third.data["conversation_subject_used"])
        self.assertEqual(third.data["information_subject"], "کیر")

    def test_common_anatomy_definitions_stay_local_and_neutral(self) -> None:
        with TemporaryRuntime() as runtime, mock.patch.object(
            runtime.tools, "invoke", side_effect=AssertionError("web/tool call was not expected")
        ):
            penis = runtime.agent.respond("کير چيست")
            vagina = runtime.agent.respond("واژن چيست")
        self.assertEqual(penis.data["reasoning_source"], "local_knowledge")
        self.assertEqual(vagina.data["reasoning_source"], "local_knowledge")
        self.assertIn("آلت مردانه", penis.text)
        self.assertIn("فرج", vagina.text)

    def test_medical_search_drops_sensational_results_and_encoded_duplicates(self) -> None:
        class MedicalFixture:
            def search_detailed(self, _query: str) -> SearchDiagnostics:
                return SearchDiagnostics(
                    (
                        SearchResult(
                            "واژن چیست؟ | تشخیص اندازه + عکس | 5 مشخصات یک واژن خوب",
                            "https://parmaclinic.com/medical-articles/vagina/",
                            "تشخیص اندازه از چهره و عکس؛ مشخصات یک واژن خوب.",
                            "fixture",
                            "",
                        ),
                        SearchResult(
                            "واژن",
                            "https://fa.wikipedia.org/wiki/%D9%88%D8%A7%DA%98%D9%86",
                            "واژن مجرایی عضلانی در دستگاه تولیدمثلی است.",
                            "fixture",
                            "واژن مجرایی عضلانی در دستگاه تولیدمثلی است.",
                        ),
                        SearchResult(
                            "واژن - ویکی‌پدیا",
                            "https://fa.wikipedia.org/wiki/واژن",
                            "واژن از دهلیز تا دهانه رحم ادامه دارد.",
                            "fixture",
                            "",
                        ),
                        SearchResult(
                            "Vaginal health",
                            "https://medlineplus.gov/vaginaldiseases.html",
                            "واژن بخشی از دستگاه تولیدمثلی است و برخی علائم نیازمند ارزیابی پزشکی‌اند.",
                            "fixture",
                            "واژن بخشی از دستگاه تولیدمثلی است و برخی علائم نیازمند ارزیابی پزشکی‌اند.",
                        ),
                    ),
                    ("fixture",),
                    (),
                )

        report = SearchEngine(MedicalFixture(), fetch_top_pages=0).search("واژن چیست؟")  # type: ignore[arg-type]
        urls = [item.url for item in report.evidence]
        self.assertFalse(any("parmaclinic" in url for url in urls))
        self.assertEqual(sum("wikipedia.org" in url for url in urls), 1)
        self.assertNotIn("تشخیص اندازه", report.summary)

    def test_default_browser_close_has_dedicated_route(self) -> None:
        with TemporaryRuntime() as runtime:
            route = runtime.agent.router.route("مرورگر پیش فرض سیستم رو ببند")
        self.assertEqual(route.intent, "close_default_browser")
        self.assertEqual(route.arguments, {})

    def test_visible_browser_window_close_accepts_background_process(self) -> None:
        manager = ProcessManager(_AppIndexFixture())  # type: ignore[arg-type]
        manager.system = "windows"
        running = (RunningProcess(41, "chrome.exe"), RunningProcess(42, "chrome.exe"))
        with mock.patch.object(manager, "find_running_app", return_value=running), mock.patch.object(
            manager, "_visible_window_count", side_effect=(1, 0)
        ), mock.patch.object(manager, "_close_visible_windows", return_value=1), mock.patch.object(
            manager, "is_app_running", return_value=True
        ), mock.patch("jarvis.tools.processes.time.sleep", return_value=None):
            result = manager.close_app("chrome")
        self.assertTrue(result.success)
        self.assertTrue(result.verified)
        self.assertEqual(result.code, "windows_closed")

    def test_close_default_browser_uses_detected_browser(self) -> None:
        manager = ProcessManager(_AppIndexFixture())  # type: ignore[arg-type]
        expected = ProcessResult(True, "chrome", "Google Chrome", "close", False, True, "closed")
        with mock.patch.object(manager, "default_browser_id", return_value="chrome"), mock.patch.object(
            manager, "close_app", return_value=expected
        ) as close:
            self.assertEqual(manager.close_default_browser(), expected)
        close.assert_called_once_with("chrome")

    def test_chat_copy_removes_bidi_control_characters(self) -> None:
        self.assertEqual(
            ModernChatSurface._clean_copy("\u2067سلام\u2069 \u2066C:\\Work\u2069"),
            "سلام C:\\Work",
        )

    def test_chat_display_never_exposes_bidi_control_boxes(self) -> None:
        unsafe = "\u2067سلام\u2069 \u202ehello\u202c \u2066https://example.com\u2069"
        clean = ModernChatSurface._clean_display(unsafe)
        self.assertEqual(clean, "سلام hello https://example.com")
        self.assertFalse(any(mark in clean for mark in "\u061c\u200e\u200f\u202a\u202b\u202c\u202d\u202e\u2066\u2067\u2068\u2069"))

    def test_conversation_copy_uses_selected_last_message_count(self) -> None:
        messages = [
            ("user", "اول"),
            ("assistant", "پاسخ اول"),
            ("user", "\u2067دوم\u2069"),
            ("assistant", "پاسخ دوم"),
        ]
        copied = JarvisGUI._format_conversation_copy(messages, "۲ پیام")
        self.assertNotIn("اول", copied)
        self.assertEqual(copied, "شما:\nدوم\n\nJARVIS:\nپاسخ دوم")
        self.assertNotIn("\u2067", copied)

    def test_conversation_copy_can_include_all_messages(self) -> None:
        messages = [("user", "سلام"), ("assistant", "درود")]
        copied = JarvisGUI._format_conversation_copy(messages, "همه پیام‌ها")
        self.assertEqual(copied, "شما:\nسلام\n\nJARVIS:\nدرود")

    def test_input_registers_explicit_copy_and_paste_shortcuts(self) -> None:
        class FakeText:
            def __init__(self) -> None:
                self.bindings: dict[str, object] = {}

            def bind(self, sequence: str, callback: object) -> None:
                self.bindings[sequence] = callback

        gui = object.__new__(JarvisGUI)
        gui.message_input = FakeText()  # type: ignore[assignment]
        gui._bind_input_clipboard_shortcuts()
        self.assertTrue(
            {"<Control-c>", "<Control-C>", "<Control-v>", "<Control-V>"}
            .issubset(gui.message_input.bindings)
        )

    def test_modern_chat_dependency_is_runtime_only_and_lightweight(self) -> None:
        root = Path(__file__).resolve().parents[1]
        runtime_requirements = (root / "requirements-runtime.txt").read_text(encoding="utf-8")
        self.assertIn("customtkinter==6.0.0", runtime_requirements)
        self.assertNotIn("electron", runtime_requirements.casefold())
        self.assertNotIn("chromium", runtime_requirements.casefold())


if __name__ == "__main__":
    unittest.main()
