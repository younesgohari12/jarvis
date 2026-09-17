from __future__ import annotations

import json
import re
import unittest
from pathlib import Path
from unittest import mock

from jarvis.agent.constraints_v15 import ConstraintExtractor, ResponseVerifier
from jarvis.agent.semantic_router_v15 import SemanticIntentRouterV15
from jarvis.utils.text import normalize_text
from tests.helpers import ROOT, TemporaryRuntime


E2E_HOLDOUT_CASES = (
    ("101 به جز 1 و خودش مقسومعلیه دیگری دارد یا نه؟".replace("\x1f", "‌"), ("عدد اول است",)),
    ("97 غیر از یک و خودش مقسوم‌علیه دیگری دارد؟", ("عدد اول است",)),
    ("بین 90 تا 100 کدام عدد اول است؟", ("97",)),
    ("احتمال دقیقاً 2 بار شیر در 4 پرتاب سکه سالم چقدر است؟", ("3/8", "37.5")),
    ("What is the probability of exactly 3 heads in 5 coin tosses?", ("5/16", "31.25")),
    ("از 11 عضو چند انتخاب 4 عضوی می‌توان داشت؟", ("330",)),
    ("P(9,3) را حساب کن", ("504",)),
    ("دو کارگر کاری را جداگانه در 6 و 3 ساعت تمام می‌کنند؛ با هم چند ساعت؟", ("2",)),
    ("کتابی 120 صفحه دارد؛ روز اول 35 و روز دوم 28 صفحه خواندم. چند صفحه باقی مانده؟", ("57",)),
    ("علی 12 ساله است. 5 سال دیگر چند ساله است؟", ("17",)),
    ("اگر 3 کتاب 120 هزار تومان باشد، 5 کتاب چقدر می‌شود؟", ("200",)),
    ("سرعت خودرو 72 کیلومتر بر ساعت است و 2.5 ساعت حرکت می‌کند؛ مسافت؟", ("180",)),
    ("حل کن: 4x - 7 = 21", ("x = 7",)),
    ("دنباله حسابی از 5 با اختلاف 4 داریم. جمله 9 چیست؟", ("37",)),
    ("اگر A قبل از B و B قبل از C باشد، کدام‌یک اول است؟", ("A",)),
    ("به انگلیسی بگو: لطفاً فایل نهایی را ارسال کنید", ("Please send the final file.",)),
    ("این جمله رو انگلیسی کن: جلسه به هفته بعد منتقل شد", ("meeting", "next week")),
    ("Translate this into Farsi: The result is ready", ("نتیجه آماده است",)),
    ("خروجی این کد چیست؟ x=3; x += 2; print(x)", ("5",)),
    ("خروجی این کد چیست؟ x=2; for i in range(3): x += i; print(x)", ("2", "3", "5")),
    ("این متن را رسمی‌تر بنویس: فایل رو زود بفرست", ("فایل را", "ارسال کنید")),
    ("حداکثر 18 کلمه درباره امنیت داده بنویس", ("امنیت", "داده")),
)


ROUTER_HARD_CASES = (
    ("خروجی این کد چیست؟ x=3; print(x+4)", "code_trace"),
    ("یک تابع پایتون برای حذف مقادیر تکراری بنویس", "coding"),
    ("این جمله را به انگلیسی بگو: نتیجه آماده است", "translation"),
    ("این پیام را رسمی‌تر و روان‌تر کن", "rewrite"),
    ("احتمال دقیقاً دو شیر در چهار پرتاب سکه چقدر است؟", "probability"),
    ("اگر A قبل از B و B قبل از C باشد چه نتیجه‌ای می‌گیریم؟", "logic"),
    ("کتاب 180 صفحه است و روزی 30 صفحه می‌خوانم؛ چند روز؟", "word_problem"),
    ("آیا 113 عدد اول است؟", "math"),
    ("حداکثر 40 کلمه درباره امنیت داده بنویس", "constraint_writing"),
    ("آخرین خبرهای OpenAI امروز چیست؟", "fresh_information"),
    ("صدای سیستم را روی 30 درصد بگذار", "desktop_command"),
    ("TCP و UDP چه تفاوتی دارند؟", "general_question"),
)


class IntelligenceV15Tests(unittest.TestCase):
    def test_v15_end_to_end_holdout_without_tools(self) -> None:
        with TemporaryRuntime() as runtime:
            with mock.patch.object(runtime.tools, "invoke", side_effect=AssertionError("unexpected tool invocation")):
                for prompt, fragments in E2E_HOLDOUT_CASES:
                    with self.subTest(prompt=prompt):
                        answer = runtime.agent.respond(prompt)
                        lowered = answer.text.casefold()
                        for fragment in fragments:
                            self.assertIn(fragment.casefold(), lowered)
                        self.assertNotIn(answer.intent, {"unknown_information", "context_clarification", "tool_search_result"})

    def test_v15_holdout_prompts_are_not_exact_training_examples(self) -> None:
        seen: set[str] = set()
        for split in ("train", "validation", "test"):
            path = ROOT / "datasets" / "splits" / f"dataset_v007_{split}.jsonl"
            for line in path.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    row = json.loads(line)
                    seen.add(normalize_text(str(row.get("input", ""))).casefold())
        for prompt, _ in E2E_HOLDOUT_CASES:
            with self.subTest(prompt=prompt):
                self.assertNotIn(normalize_text(prompt).casefold(), seen)

    def test_v15_semantic_router_hard_cases(self) -> None:
        router = SemanticIntentRouterV15(ROOT / "models" / "semantic_router_v15.npz")
        self.assertTrue(router.ready)
        self.assertEqual(router.parameter_count, 3_145_740)
        for prompt, expected in ROUTER_HARD_CASES:
            with self.subTest(prompt=prompt):
                result = router.predict(prompt)
                self.assertIsNotNone(result)
                assert result is not None
                self.assertEqual(result.intent, expected)

    def test_v15_probability_roles_are_semantic(self) -> None:
        cases = (
            ("احتمال دقیقاً 2 بار شیر در 4 پرتاب سکه سالم چقدر است؟", "C(4,2)", "37.5"),
            ("احتمال دقیقاً 3 بار شیر در 5 پرتاب سکه سالم چقدر است؟", "C(5,3)", "31.25"),
            ("What is the probability of exactly 2 heads in 4 coin tosses?", "C(4,2)", "37.5"),
        )
        with TemporaryRuntime() as runtime:
            with mock.patch.object(runtime.tools, "invoke", side_effect=AssertionError("unexpected tool invocation")):
                for prompt, comb, pct in cases:
                    with self.subTest(prompt=prompt):
                        answer = runtime.agent.respond(prompt)
                        self.assertIn(comb, answer.text)
                        self.assertIn(pct, answer.text)

    def test_v15_constraints_extract_verify_and_repair(self) -> None:
        requests = (
            "دقیقاً 3 جمله درباره امنیت داده بنویس و فقط فارسی باشد.",
            "حداکثر 18 کلمه درباره امنیت داده بنویس.",
            "دقیقاً دو جمله درباره یادگیری بنویس؛ شامل کلمه «تمرین» و بدون استفاده از کلمه «سریع».",
        )
        with TemporaryRuntime() as runtime:
            with mock.patch.object(runtime.tools, "invoke", side_effect=AssertionError("unexpected tool invocation")):
                for prompt in requests:
                    with self.subTest(prompt=prompt):
                        constraints = ConstraintExtractor.extract(prompt)
                        self.assertTrue(constraints.active)
                        answer = runtime.agent.respond(prompt)
                        verified = ResponseVerifier.verify(answer.text, constraints)
                        self.assertTrue(verified.valid, verified.violations)

    def test_v15_dataset_and_model_quality_gates(self) -> None:
        merged = json.loads((ROOT / "datasets" / "manifest_v007.json").read_text(encoding="utf-8"))
        report = json.loads((ROOT / "reports" / "semantic_router_v15_training_report.json").read_text(encoding="utf-8"))
        tok = json.loads((ROOT / "models" / "tokenizer_metrics_v007.json").read_text(encoding="utf-8"))

        self.assertEqual(merged["dataset_version"], "dataset_v007")
        self.assertEqual(merged["total_examples"], 36_530)
        self.assertEqual(merged["new_examples"], 21_800)
        self.assertEqual(merged["source_dataset_count"], 500)
        self.assertEqual(merged["cross_split_concept_leakage"], 0)

        quality = merged["pack_quality"]
        self.assertEqual(quality["generalization"]["examples"], 5000)
        self.assertEqual(quality["generalization"]["unique_inputs"], 5000)
        self.assertGreaterEqual(quality["generalization"]["unique_output_ratio"], 0.999)
        self.assertEqual(quality["reasoning"]["unique_outputs"], 5000)
        self.assertEqual(quality["persian"]["unique_outputs"], 4000)
        self.assertEqual(quality["persian"]["fa_language_contamination"], 0)
        self.assertEqual(quality["routing"]["unique_inputs"], 4800)
        self.assertEqual(quality["routing"]["conflicting_input_labels"], 0)
        self.assertEqual(set(quality["routing"]["labels"].values()), {400})
        self.assertEqual(quality["constraints"]["unique_outputs"], 3000)
        self.assertEqual(quality["constraints"]["fa_language_contamination"], 0)

        self.assertEqual(report["parameter_count"], 3_145_740)
        self.assertEqual(report["test"]["accuracy"], 1.0)
        self.assertEqual(tok["dataset_version"], "dataset_v007")
        self.assertEqual(tok["normalized_round_trip_accuracy"], 1.0)
        self.assertEqual(tok["unknown_token_rate"], 0.0)


if __name__ == "__main__":
    unittest.main()
