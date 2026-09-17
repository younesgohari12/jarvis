from __future__ import annotations

import inspect
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from jarvis.agent.deliberation import (
    DeliberationOutcome,
    EvidenceVerifier,
    QueryAnalyzer,
    TextReasoner,
)
from jarvis.agent.reasoning import ReasoningEngine
from jarvis.neural.conversation import NeuralConversationEngine
from jarvis.research.engine import ResearchEngine
from jarvis.search.engine import EvidencePassage, SearchEngine, SearchEvidence, SearchReport
from jarvis.tools.internet import (
    InternetResult,
    InternetTool,
    InternetToolError,
    SearchDiagnostics,
    SearchResult,
    _DuckDuckGoParser,
)
from tests.helpers import TemporaryRuntime


class _EvidenceInternet:
    def __init__(self) -> None:
        self.calls = 0

    def search_detailed(self, query: str) -> SearchDiagnostics:
        self.calls += 1
        return SearchDiagnostics(
            (
                SearchResult(
                    "Python release information",
                    "https://www.python.org/downloads/",
                    "Python stable release information and security maintenance.",
                    "duckduckgo",
                    "Python stable release information includes the current supported release and its security maintenance status.",
                ),
                SearchResult(
                    "Python",
                    "https://en.wikipedia.org/wiki/Python_(programming_language)",
                    "Python is a programming language with stable release information.",
                    "wikipedia_en",
                    "Python is a programming language. Its stable release information documents the current supported version.",
                ),
            ),
            ("duckduckgo", "wikipedia_en"),
            (),
        )

    def fetch_text(self, url: str) -> InternetResult:
        raise AssertionError("embedded provider content should avoid another fetch")


class _UnavailableInternet:
    def search_detailed(self, query: str) -> SearchDiagnostics:
        return SearchDiagnostics((), (), ("duckduckgo: blocked", "wikipedia: timeout"))

    def fetch_text(self, url: str) -> InternetResult:
        raise AssertionError("no page should be fetched")


class GroundedIntelligenceV071Tests(unittest.TestCase):
    def test_query_analysis_separates_passage_from_question(self) -> None:
        analysis = QueryAnalyzer().analyze(
            "این متن را دقیق تحلیل کن: علی فردا نمی‌آید چون بیمار است. چرا علی نمی‌آید؟"
        )
        self.assertEqual(analysis.kind, "text_causality")
        self.assertIn("بیمار", analysis.supplied_text)
        self.assertEqual(analysis.questions, ("چرا علی نمی‌آید؟",))

    def test_causal_answer_is_taken_from_supplied_text(self) -> None:
        outcome = TextReasoner().answer_question(
            "متن: علی فردا نمی‌آید چون بیمار است. چرا علی نمی‌آید؟"
        )
        self.assertIsNotNone(outcome)
        assert outcome is not None
        self.assertIn("بیمار", outcome.text)
        self.assertEqual(outcome.source, "supplied_text")
        self.assertIn("answer", " ".join(outcome.checks))

    def test_irrelevant_passage_does_not_create_an_answer(self) -> None:
        outcome = TextReasoner().answer_question(
            "متن: باران به دلیل تراکم بخار آب شکل گرفت. قیمت طلا چقدر است؟"
        )
        self.assertIsNone(outcome)

    def test_persian_formal_deduction_is_generic(self) -> None:
        outcome = TextReasoner().logical_inference(
            "اگر همه پرنده‌ها بال دارند و گنجشک پرنده است، چه نتیجه‌ای می‌گیری؟", "fa"
        )
        self.assertIsNotNone(outcome)
        assert outcome is not None
        self.assertIn("گنجشک", outcome.text)
        self.assertIn("بال", outcome.text)
        self.assertEqual(outcome.source, "formal_deduction")

    def test_english_multi_hop_deduction(self) -> None:
        outcome = TextReasoner().logical_inference(
            "All birds are animals. All animals are living things. A sparrow is a bird. What follows?",
            "en",
        )
        self.assertIsNotNone(outcome)
        assert outcome is not None
        self.assertIn("sparrow", outcome.text.casefold())
        self.assertIn("living", outcome.text.casefold())

    def test_document_analysis_is_extractive_and_explicitly_bounded(self) -> None:
        text = (
            "پروژه آلفا یک ابزار دسکتاپ آفلاین است. حافظه آن با SQLite کار می‌کند. "
            "رابط برنامه با Tkinter ساخته شده و عملیات طولانی در worker اجرا می‌شود."
        )
        outcome = TextReasoner().analyze_document("این متن را تحلیل کن", text)
        self.assertIsNotNone(outcome)
        assert outcome is not None
        self.assertIn("جمع‌بندی", outcome.text)
        self.assertIn("خارج", outcome.text)
        self.assertTrue(outcome.evidence)

    def test_multiple_questions_are_answered_separately(self) -> None:
        outcome = TextReasoner().answer_question(
            "متن: سارا در تبریز زندگی می‌کند. سارا با قطار سفر می‌کند. "
            "سارا کجا زندگی می‌کند؟ سارا با چه چیزی سفر می‌کند؟"
        )
        self.assertIsNotNone(outcome)
        assert outcome is not None
        self.assertIn("1.", outcome.text)
        self.assertIn("2.", outcome.text)
        self.assertIn("تبریز", outcome.text)
        self.assertIn("قطار", outcome.text)

    def test_evidence_verifier_rejects_unrelated_claim(self) -> None:
        outcome = DeliberationOutcome(
            "قیمت امروز طلا افزایش یافت.", 0.9, "supplied_text",
            ("آسمان به دلیل پراکندگی رایلی آبی دیده می‌شود.",), ("candidate",),
        )
        self.assertIsNone(EvidenceVerifier().verify(outcome, "fa"))

    def test_runtime_answers_text_question_before_any_web_tool(self) -> None:
        with TemporaryRuntime() as runtime, mock.patch.object(
            runtime.tools, "invoke", wraps=runtime.tools.invoke
        ) as invoked:
            reply = runtime.agent.respond(
                "این متن را بخوان: مینا جلسه را لغو کرد چون بیمار بود. چرا جلسه لغو شد؟"
            )
            self.assertEqual(reply.intent, "grounded_text_answer")
            self.assertIn("بیمار", reply.text)
            self.assertFalse(any(call.args[0] == "web_search" for call in invoked.call_args_list))

    def test_runtime_uses_formal_reasoning_without_internet(self) -> None:
        with TemporaryRuntime() as runtime:
            runtime.agent.update_setting("internet_enabled", False)
            reply = runtime.agent.respond(
                "همه فلزها رسانا هستند و مس فلز است، چه نتیجه‌ای می‌گیری؟"
            )
            self.assertEqual(reply.intent, "grounded_text_answer")
            self.assertIn("مس", reply.text)
            self.assertIn("رسانا", reply.text)

    def test_attached_text_can_be_analyzed_directly(self) -> None:
        with TemporaryRuntime() as runtime, tempfile.TemporaryDirectory(prefix="jarvis-doc-") as temporary:
            path = Path(temporary) / "notes.txt"
            path.write_text(
                "سامانه بتا آفلاین اجرا می‌شود. داده‌ها در SQLite ذخیره می‌شوند. "
                "رابط کاربری در زمان پردازش مسدود نمی‌شود.",
                encoding="utf-8",
            )
            runtime.agent.attach_file(path)
            reply = runtime.agent.respond("این فایل را دقیق تحلیل کن")
            self.assertEqual(reply.intent, "document_analysis")
            self.assertTrue(reply.data["attachment_used"])
            self.assertIn("SQLite", reply.text)

    def test_previous_user_passage_can_be_referenced(self) -> None:
        with TemporaryRuntime() as runtime:
            runtime.agent.update_setting("internet_enabled", False)
            runtime.agent.respond(
                "گزارش امروز می‌گوید پروژه آلفا تکمیل شد و سه تست نهایی آن پاس شدند."
            )
            reply = runtime.agent.respond("متن بالا را خلاصه کن")
            self.assertEqual(reply.intent, "document_analysis")
            self.assertTrue(reply.data["conversation_context_used"])
            self.assertIn("پروژه", reply.text)

    def test_expanded_local_knowledge_is_seeded(self) -> None:
        with TemporaryRuntime() as runtime:
            self.assertGreaterEqual(runtime.knowledge.count(), 97)
            reply = runtime.agent.respond("NAT چیکار می‌کنه؟")
            self.assertEqual(reply.intent, "knowledge_answer")
            self.assertIn("IP", reply.text)

    def test_hyphenated_backup_rule_is_not_misread_as_subtraction(self) -> None:
        with TemporaryRuntime() as runtime:
            reply = runtime.agent.respond("قانون بکاپ 3-2-1 چیه؟")
            self.assertEqual(reply.intent, "knowledge_answer")
            self.assertIn("سه نسخه", reply.text)

    def test_question_word_inside_affirmation_vocabulary_keeps_question_polarity(self) -> None:
        with TemporaryRuntime() as runtime:
            reply = runtime.agent.respond("همبستگی یعنی حتما علت؟")
            self.assertEqual(reply.intent, "knowledge_answer")
            self.assertIn("علیت", reply.text)

    def test_persian_grammar_with_english_terms_gets_persian_answer(self) -> None:
        with TemporaryRuntime() as runtime:
            reply = runtime.agent.respond("فرق authentication و authorization چیست؟")
            self.assertEqual(reply.language, "fa")
            self.assertIn("احراز هویت", reply.text)

    def test_number_in_unknown_fact_question_does_not_trigger_calculator(self) -> None:
        with TemporaryRuntime() as runtime:
            runtime.agent.update_setting("internet_enabled", False)
            reply = runtime.agent.respond(
                "این سؤال ساختگی بی پاسخ را جواب بده: رنگ دقیق سیاره خیالی زتا-۹ چیست؟"
            )
            self.assertNotEqual(reply.intent, "tool_calculation_result")
            self.assertIn("اطلاعات کافی", reply.text)

    def test_ddg_html_and_lite_layouts_are_supported(self) -> None:
        samples = (
            '<a class="result__a" href="//duckduckgo.com/l/?uddg=https%3A%2F%2Fexample.com%2Fa">A</a>'
            '<div class="result__snippet">first snippet</div>',
            '<a class="result-link" href="https://example.org/b">B</a>'
            '<td class="result-snippet">second snippet</td>',
        )
        for sample in samples:
            with self.subTest(sample=sample):
                parser = _DuckDuckGoParser()
                parser.feed(sample)
                parser.close()
                self.assertEqual(len(parser.results), 1)
                self.assertTrue(parser.results[0].snippet.endswith("snippet"))

    def test_wikipedia_provider_parses_official_action_api_shape(self) -> None:
        payload = {
            "query": {"pages": [{
                "title": "Python",
                "fullurl": "https://en.wikipedia.org/wiki/Python_(programming_language)",
                "extract": "Python is a high-level programming language.",
            }]}
        }
        tool = InternetTool()
        response = InternetResult("https://en.wikipedia.org/w/api.php", 200, "application/json", json.dumps(payload), False)
        with mock.patch.object(tool, "_fetch_with_retry", return_value=response):
            results = tool._search_wikipedia("Python", "en")
        self.assertEqual(results[0].provider, "wikipedia_en")
        self.assertIn("programming", results[0].content)

    def test_bing_rss_provider_parses_bounded_results(self) -> None:
        rss = """<?xml version="1.0"?><rss><channel><item>
        <title>Python releases</title><link>https://www.python.org/downloads/</link>
        <description>Official &lt;b&gt;Python&lt;/b&gt; release downloads.</description>
        <pubDate>Thu, 27 Aug 2026 00:00:00 GMT</pubDate>
        </item></channel></rss>"""
        tool = InternetTool()
        response = InternetResult("https://www.bing.com/search", 200, "application/rss+xml", rss, False)
        with mock.patch.object(tool, "_fetch_with_retry", return_value=response):
            results = tool._search_bing_rss("Python", "en")
        self.assertEqual(results[0].provider, "bing_rss")
        self.assertEqual(results[0].url, "https://www.python.org/downloads/")
        self.assertNotIn("<b>", results[0].snippet)

    def test_search_detailed_merges_independent_providers(self) -> None:
        tool = InternetTool(max_search_results=4)
        ddg = [SearchResult("Official", "https://python.org/", "release", "duckduckgo")]
        wiki = [SearchResult("Python", "https://en.wikipedia.org/wiki/Python", "language", "wikipedia_en")]
        with mock.patch.object(tool, "_search_duckduckgo", return_value=ddg), mock.patch.object(
            tool, "_search_wikipedia", return_value=wiki
        ), mock.patch.object(tool, "_search_bing_rss", return_value=[]):
            result = tool.search_detailed("Python")
        self.assertEqual(set(result.providers), {"duckduckgo", "wikipedia_en"})
        self.assertEqual(len(result.results), 2)

    def test_search_engine_returns_unavailable_instead_of_throwing(self) -> None:
        report = SearchEngine(_UnavailableInternet(), 2).search("latest Python")  # type: ignore[arg-type]
        self.assertEqual(report.status, "unavailable")
        self.assertFalse(report.summary)
        self.assertEqual(len(report.errors), 2)

    def test_search_engine_produces_cited_multi_source_evidence(self) -> None:
        report = SearchEngine(_EvidenceInternet(), 2).search("latest stable Python release")  # type: ignore[arg-type]
        self.assertIn(report.status, {"verified", "supported"})
        self.assertGreater(report.confidence, 0.5)
        self.assertGreaterEqual(len(report.evidence), 2)
        self.assertRegex(report.summary, r"\[[12]\]")
        self.assertTrue(all(passage.url for passage in report.passages))

    def test_search_cache_avoids_repeating_same_stable_query(self) -> None:
        internet = _EvidenceInternet()
        engine = SearchEngine(internet, 1)  # type: ignore[arg-type]
        first = engine.search("Python stable release")
        second = engine.search("Python stable release")
        self.assertIs(first, second)
        self.assertEqual(internet.calls, 1)

    def test_irrelevant_search_text_is_not_presented_as_an_answer(self) -> None:
        class Irrelevant:
            def search(self, query: str) -> list[SearchResult]:
                return [SearchResult("Cooking", "https://example.com/food", "A recipe uses onions and oil.")]

            def fetch_text(self, url: str) -> InternetResult:
                return InternetResult(url, 200, "text/html", "<p>Cooking rice needs water.</p>", False)

        report = SearchEngine(Irrelevant(), 1).search("quantum entanglement")  # type: ignore[arg-type]
        self.assertFalse(report.summary)
        self.assertEqual(report.status, "insufficient")

    def test_html_extraction_drops_script_and_navigation(self) -> None:
        raw = (
            "<html><nav>menu privacy</nav><main><p>Python stable release provides security fixes.</p>"
            "<script>malicious irrelevant words</script></main></html>"
        )
        text = SearchEngine._extract_text(raw, "text/html")
        self.assertIn("security fixes", text)
        self.assertNotIn("malicious", text)
        self.assertNotIn("menu privacy", text)

    def test_runtime_explains_provider_failure_and_does_not_guess(self) -> None:
        unavailable = SearchReport(
            "latest Python", "", (), 0, 0.0, "unavailable", (), (), ("timeout",), True
        )
        with TemporaryRuntime() as runtime, mock.patch.object(
            runtime.tools, "invoke", return_value=unavailable
        ):
            reply = runtime.agent.respond("آخرین نسخه پایتون چیست؟")
        self.assertEqual(reply.intent, "tool_search_unavailable")
        self.assertIn("حدسی", reply.text)

    def test_runtime_search_output_includes_confidence_and_sources(self) -> None:
        report = SearchReport(
            "Python", "Python is a programming language. [1]",
            (SearchEvidence("Python", "https://python.org/", "official", 0.9),),
            1, 0.82, "supported",
            (EvidencePassage("Python is a programming language.", "https://python.org/", "Python", 0.8, 1),),
        )
        with TemporaryRuntime() as runtime, mock.patch.object(
            runtime.tools, "invoke", return_value=report
        ):
            reply = runtime.agent.respond("latest Python version")
        self.assertIn("82", reply.text)
        self.assertIn("https://python.org/", reply.text)

    def test_research_rewrites_are_generic_not_topic_table_translations(self) -> None:
        class Stub:
            def search(self, query: str) -> SearchReport:
                return SearchReport(query, "", (), 0, status="no_results")

        with tempfile.TemporaryDirectory(prefix="jarvis-rewrite-") as temporary:
            engine = ResearchEngine(Stub(), Path(temporary) / "search.jsonl")  # type: ignore[arg-type]
            rewrites = engine.rewrite_queries("آخرین نسخه ابزار ناشناخته زتا")
        self.assertIn("زتا", rewrites[1])
        self.assertNotIn("Python", " ".join(rewrites))

    def test_neural_guard_rejects_structured_leak_echo_and_repetition(self) -> None:
        guard = NeuralConversationEngine._quality
        self.assertFalse(guard("<tool>{}</tool>", "hello")[0])
        self.assertFalse(guard("hello there", "hello there")[0])
        self.assertFalse(guard("test test test test test test", "test")[0])

    def test_neural_guard_rejects_irrelevant_language_flip(self) -> None:
        accepted, reason, _confidence = NeuralConversationEngine._quality(
            "این خروجی فارسی هیچ ارتباطی ندارد", "Please suggest a short coding activity"
        )
        self.assertFalse(accepted)
        self.assertEqual(reason, "language_consistency_guard")

    def test_complex_reasoning_no_longer_contains_topic_specific_sqlite_branch(self) -> None:
        source = inspect.getsource(ReasoningEngine.answer_complex)
        self.assertNotIn('"sqlite" in normalized', source)
        self.assertNotIn("_sqlite_json_answer", source)

    def test_url_credentials_and_local_targets_remain_blocked(self) -> None:
        with self.assertRaises(InternetToolError):
            InternetTool._parsed_url("https://user:password@example.com/")
        with self.assertRaises(InternetToolError):
            InternetTool().fetch_text("http://127.0.0.1/private")


if __name__ == "__main__":
    unittest.main()
