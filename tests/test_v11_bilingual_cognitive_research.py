from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from jarvis.agent.bilingual_fluency import BilingualFluencyEngine
from jarvis.agent.cognitive_model import CognitiveSkillModel
from jarvis.knowledge.store import KnowledgeStore
from jarvis.research.engine import ResearchEngine
from jarvis.search.engine import EvidencePassage, SearchEngine, SearchEvidence, SearchReport
from jarvis.tools.internet import SearchDiagnostics, SearchResult
from tests.helpers import TemporaryRuntime


ROOT = Path(__file__).resolve().parents[1]


class _SearchInternet:
    def __init__(self, results: tuple[SearchResult, ...]) -> None:
        self.results = results

    def search_detailed(self, _query: str) -> SearchDiagnostics:
        return SearchDiagnostics(self.results, ("test",), ())


class _MultiReportSearch:
    def __init__(self) -> None:
        self.calls = 0

    def search(self, query: str, subject: str = "") -> SearchReport:
        del subject
        self.calls += 1
        if self.calls % 2:
            evidence = SearchEvidence(
                "Python security guide", "https://docs.python.org/3/security.html",
                "Python package security requires trusted sources and dependency review.",
                0.84, "docs", "docs.python.org", 0.9,
            )
            passage = EvidencePassage(
                "Python package security starts with trusted package sources and careful dependency review.",
                evidence.url, evidence.title, 0.78, 1,
            )
        else:
            evidence = SearchEvidence(
                "Supply-chain guidance", "https://www.cisa.gov/software-supply-chain",
                "Software supply-chain guidance recommends pinned dependencies and integrity checks.",
                0.86, "official", "cisa.gov", 0.95,
            )
            passage = EvidencePassage(
                "Software supply-chain guidance recommends pinned dependencies and package integrity checks.",
                evidence.url, evidence.title, 0.8, 1,
            )
        return SearchReport(
            query, passage.text + " [1]", (evidence,), 0, 0.72,
            "supported", (passage,), (evidence.provider,), (), False,
        )


class BilingualCognitiveResearchV11Tests(unittest.TestCase):
    def test_bilingual_model_passed_real_holdout_gate(self) -> None:
        model = BilingualFluencyEngine(ROOT / "models" / "bilingual_fluency_v11.json")
        metrics = json.loads(
            (ROOT / "models" / "bilingual_fluency_metrics_v11.json").read_text(encoding="utf-8")
        )
        self.assertTrue(model.ready)
        self.assertEqual(metrics["dataset_examples"], 105)
        self.assertEqual(metrics["held_out_examples"], 14)
        self.assertTrue(metrics["accepted_for_release"])
        self.assertGreaterEqual(metrics["classification_accuracy"], 0.85)
        self.assertEqual(metrics["generation_acceptance"], 1.0)
        self.assertIsNone(metrics["pretrained_source"])

    def test_cognitive_model_passed_real_holdout_gate(self) -> None:
        model = CognitiveSkillModel(ROOT / "models" / "cognitive_skills_v11.json")
        metrics = json.loads(
            (ROOT / "models" / "cognitive_skills_metrics_v11.json").read_text(encoding="utf-8")
        )
        self.assertTrue(model.ready)
        self.assertEqual(model.training_examples, 96)
        self.assertEqual(metrics["held_out_examples"], 16)
        self.assertTrue(metrics["accepted_for_release"])
        self.assertGreaterEqual(metrics["classification_accuracy"], 0.875)
        self.assertIsNone(metrics["pretrained_source"])

    def test_english_fluency_generates_grounded_requested_formats_without_web(self) -> None:
        prompts = (
            ("Write a thoughtful paragraph about persistence", "Persistence"),
            ("Write a formal message telling a customer their order ships tomorrow", "shipped tomorrow"),
            ("Tell me a short story about courage", "courage"),
            ("I feel overwhelmed and need someone to talk to", "we can slow it down"),
        )
        with TemporaryRuntime() as runtime, mock.patch.object(
            runtime.tools, "invoke", side_effect=AssertionError("web/tool call was not expected")
        ):
            for prompt, expected in prompts:
                with self.subTest(prompt=prompt):
                    answer = runtime.agent.respond(prompt)
                    self.assertEqual(answer.intent, "fluent_response")
                    self.assertEqual(answer.language, "en")
                    self.assertIn(expected.casefold(), answer.text.casefold())

    def test_reported_reasoning_failures_are_now_solved_locally(self) -> None:
        cases = (
            ("همهٔ وارها تِپ هستند. بعضی تِپ‌ها سردند. آیا حتماً بعضی وارها سردند؟", "مثال نقض"),
            ("اگر پریروز، فردای دوشنبه بود، امروز چه روزی است؟", "پنجشنبه"),
            ("اگر احتمال موفقیت هر تلاش مستقل ۳۵٪ باشد، احتمال حداقل دو موفقیت در چهار تلاش چقدر است؟", "43.701875٪"),
            ("سه جعبه با برچسب‌های قرمز، آبی و مخلوط داریم و هر سه برچسب اشتباه‌اند. فقط با بیرون‌آوردن یک مهره، چطور اصلاح می‌شوند؟", "برچسب «مخلوط»"),
        )
        with TemporaryRuntime() as runtime, mock.patch.object(
            runtime.tools, "invoke", side_effect=AssertionError("web/tool call was not expected")
        ):
            for prompt, expected in cases:
                with self.subTest(prompt=prompt):
                    route = runtime.agent.router.route(prompt)
                    answer = runtime.agent.respond(prompt)
                    self.assertEqual(answer.intent, "reasoned_answer")
                    self.assertIn(expected, answer.text)
                    self.assertNotEqual(route.intent, "web_search")

    def test_bilingual_reasoning_generalizes_to_english(self) -> None:
        cases = (
            ("All zargs are mips and no mip is blue. Can a zarg be blue?", "cannot overlap"),
            ("If the day before yesterday was the day after Monday, what day is it today?", "Thursday"),
            ("With a 30 percent independent success rate, what is the chance of at least two successes in four trials?", "34.83%"),
        )
        with TemporaryRuntime() as runtime, mock.patch.object(
            runtime.tools, "invoke", side_effect=AssertionError("web/tool call was not expected")
        ):
            for prompt, expected in cases:
                with self.subTest(prompt=prompt):
                    answer = runtime.agent.respond(prompt)
                    self.assertEqual(answer.intent, "reasoned_answer")
                    self.assertIn(expected, answer.text)

    def test_tcp_summary_and_language_formal_rewrite_keep_context(self) -> None:
        with TemporaryRuntime() as runtime, mock.patch.object(
            runtime.tools, "invoke", side_effect=AssertionError("web/tool call was not expected")
        ):
            runtime.agent.respond("تفاوت TCP و UDP چیست؟")
            summary = runtime.agent.respond("پاسخ قبلی را در یک جمله خلاصه کن")
            runtime.agent.respond("معادل انگلیسی زن چیست؟")
            runtime.agent.respond("جمع آن در انگلیسی چیست؟")
            sentence = runtime.agent.respond("با همان کلمه یک جمله بساز")
            formal = runtime.agent.respond("همان جمله را رسمی‌تر کن، بدون اینکه معنایش عوض شود")
        self.assertIn("TCP", summary.text)
        self.assertIn("UDP", summary.text)
        self.assertEqual(sentence.text, "The woman presented the report.")
        self.assertEqual(formal.text, "The woman formally presented the report.")

    def test_general_knowledge_is_larger_bilingual_and_english_retrieval_works(self) -> None:
        payload = json.loads((ROOT / "data" / "knowledge_v4.json").read_text(encoding="utf-8"))
        self.assertGreaterEqual(len(payload["entries"]), 125)
        self.assertEqual(len(payload["entries"]), len({row["id"] for row in payload["entries"]}))
        with tempfile.TemporaryDirectory(prefix="jarvis-v11-knowledge-") as temporary:
            store = KnowledgeStore(
                Path(temporary) / "knowledge.db", ROOT / "data" / "knowledge_v4.json", 15, 750
            )
            try:
                hit = store.search("What is the difference between precision and recall?", 1)[0]
                self.assertGreaterEqual(hit.score, 0.29)
                self.assertIn("Precision", hit.content)
            finally:
                store.close()

    def test_search_rejects_unrequested_social_and_low_value_results(self) -> None:
        results = (
            SearchResult(
                "Python security tweet", "https://x.com/example/status/1",
                "Python dependency security advice and package integrity tips.", "social",
            ),
            SearchResult(
                "Sign in", "https://accounts.example.com/login",
                "Sign in to see Python dependency security search results.", "weak",
            ),
            SearchResult(
                "Python Packaging Security", "https://docs.python.org/3/security.html",
                "Official Python dependency security guidance covers trusted package sources.",
                "official", "Official Python dependency security guidance covers trusted package sources and integrity checks.",
            ),
        )
        report = SearchEngine(_SearchInternet(results), fetch_top_pages=0).search(
            "Python dependency security guidance"
        )
        urls = [item.url for item in report.evidence]
        self.assertEqual(urls, ["https://docs.python.org/3/security.html"])
        self.assertTrue(report.summary)

    def test_search_allows_social_when_user_explicitly_requests_it(self) -> None:
        result = SearchResult(
            "Python release tweet", "https://x.com/python/status/1",
            "Official Python release tweet announces package security guidance.", "social",
            "Official Python release tweet announces package security guidance for developers.",
        )
        report = SearchEngine(_SearchInternet((result,)), fetch_top_pages=0).search(
            "Python package security tweet"
        )
        self.assertEqual(len(report.evidence), 1)
        self.assertLessEqual(report.evidence[0].source_quality, 0.16)

    def test_research_combines_independent_sources_across_rewrites(self) -> None:
        with tempfile.TemporaryDirectory(prefix="jarvis-v11-research-") as temporary:
            search = _MultiReportSearch()
            report = ResearchEngine(search, Path(temporary) / "research.jsonl").research(
                "Python package security best practices"
            )
        self.assertGreaterEqual(report.attempts, 2)
        self.assertEqual(len(report.sources), 2)
        self.assertIn("docs.python.org", {source.domain or source.url.split('/')[2] for source in report.sources})
        self.assertIn("[1]", report.summary)
        self.assertIn("[2]", report.summary)

    def test_research_rewrites_are_domain_aware(self) -> None:
        with tempfile.TemporaryDirectory(prefix="jarvis-v11-rewrites-") as temporary:
            engine = ResearchEngine(_MultiReportSearch(), Path(temporary) / "research.jsonl")
            technical = " ".join(engine.rewrite_queries("Python API authentication"))
            medical = " ".join(engine.rewrite_queries("medical treatment evidence"))
        self.assertIn("official documentation", technical)
        self.assertIn("authoritative medical guidance", medical)


if __name__ == "__main__":
    unittest.main()
