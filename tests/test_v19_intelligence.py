from __future__ import annotations

import json
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

from jarvis.agent.local_intelligence_v19 import LocalIntelligenceV19
from jarvis.agent.semantic_ir_v19 import SemanticIRParserV19
from jarvis.agent.semantic_models_v19 import (
    CodeIntentClassifierV19,
    ExecutionPatternClassifierV19,
    NumericRoleTaggerV19,
    SemanticFrameClassifierV19,
)
from jarvis.nlu.semantic_slots_v19 import SemanticSlotBinderV19
from jarvis.utils.text import normalize_text
from tests.helpers import ROOT, TemporaryRuntime
from tests.v19_holdout_cases import V19_HOLDOUT_CASES


class IntelligenceV19Tests(unittest.TestCase):
    def test_v19_fresh_holdout_semantic_path_50_of_50(self) -> None:
        local = LocalIntelligenceV19()
        self.assertEqual(len(V19_HOLDOUT_CASES), 50)
        for prompt, fragments in V19_HOLDOUT_CASES:
            with self.subTest(prompt=prompt):
                language = "fa" if any("\u0600" <= ch <= "\u06ff" for ch in prompt) else "en"
                answer = local.solve(prompt, language)
                self.assertIsNotNone(answer)
                assert answer is not None
                low = answer.text.casefold()
                for fragment in fragments:
                    self.assertIn(fragment.casefold(), low)
                self.assertNotIn(answer.intent, {"tool_clarification", "unknown_information", "fact_not_found"})

    def test_v19_holdout_has_zero_exact_overlap_with_v19_training_data(self) -> None:
        seen: set[str] = set()
        paths = (
            ROOT / "datasets" / "numeric_roles_v19" / "numeric_role_24000_real.jsonl",
            ROOT / "datasets" / "multistep_v19" / "execution_graph_12000_real.jsonl",
            ROOT / "datasets" / "code_v19" / "code_intelligence_5000_real.jsonl",
            ROOT / "datasets" / "semantic_ir_v19" / "semantic_frame_18000_real.jsonl",
        )
        for path in paths:
            for line in path.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                row = json.loads(line)
                text = row.get("text") or row.get("input")
                if text:
                    seen.add(normalize_text(str(text)).casefold())
        for prompt, _ in V19_HOLDOUT_CASES:
            with self.subTest(prompt=prompt):
                self.assertNotIn(normalize_text(prompt).casefold(), seen)

    def test_v19_models_are_real_trained_and_runtime_loadable(self) -> None:
        models = (
            ("numeric_role_v19.npz", NumericRoleTaggerV19),
            ("semantic_frame_v19.npz", SemanticFrameClassifierV19),
            ("execution_pattern_v19.npz", ExecutionPatternClassifierV19),
            ("code_intent_v19.npz", CodeIntentClassifierV19),
        )
        total = 0
        for filename, model_cls in models:
            with self.subTest(filename=filename):
                path = ROOT / "models" / filename
                self.assertTrue(path.is_file())
                model = model_cls(path)
                self.assertTrue(model.ready)
                self.assertGreater(model.parameter_count, 100_000)
                self.assertEqual(model.metadata.get("dataset_version"), "dataset_v019_real")
                self.assertEqual(model.metadata.get("holdout_strategy"), "template_id_disjoint")
                total += model.parameter_count
        self.assertGreater(total, 2_000_000)

    def test_v19_training_report_passes_real_quality_gates(self) -> None:
        report = json.loads((ROOT / "reports" / "v19_real_training_report.json").read_text(encoding="utf-8"))
        self.assertEqual(report["dataset_version"], "dataset_v019_real")
        self.assertEqual(report["holdout_strategy"], "template_id_disjoint")
        self.assertTrue(report["passed"])
        observed = report["quality_gates"]["observed"]
        minimum = report["quality_gates"]["minimum"]
        for model_id, threshold in minimum.items():
            self.assertGreaterEqual(observed[model_id], threshold)
        self.assertGreaterEqual(observed["numeric_role_v19"], .95)
        self.assertGreaterEqual(observed["semantic_frame_v19"], .96)
        self.assertGreaterEqual(observed["execution_pattern_v19"], .99)
        self.assertGreaterEqual(observed["code_intent_v19"], .95)

    def test_v19_dataset_diversity_is_template_level_not_number_substitution(self) -> None:
        report = json.loads((ROOT / "reports" / "v19_real_training_report.json").read_text(encoding="utf-8"))
        ds = report["datasets"]
        self.assertEqual(ds["numeric_roles"]["rows"], 24_000)
        self.assertEqual(ds["multistep"]["rows"], 12_000)
        self.assertEqual(ds["code"]["rows"], 5_000)
        self.assertEqual(ds["semantic_frame"]["rows"], 18_000)
        for family, count in ds["numeric_roles"]["templates"].items():
            with self.subTest(dataset="numeric", family=family):
                self.assertGreaterEqual(count, 30)
        for graph, count in ds["multistep"]["templates"].items():
            with self.subTest(dataset="multistep", family=graph):
                self.assertGreaterEqual(count, 30)
        for intent, count in ds["code"]["templates"].items():
            with self.subTest(dataset="code", family=intent):
                self.assertGreaterEqual(count, 30)
        for frame, count in ds["semantic_frame"]["templates"].items():
            with self.subTest(dataset="frame", family=frame):
                self.assertGreaterEqual(count, 30)

    def test_v19_numeric_binding_is_order_independent(self) -> None:
        binder = SemanticSlotBinderV19(ROOT)
        frame = binder.bind("در 6 ساعت 4 کارگر 120 قطعه تولید می‌کنند")
        self.assertEqual(frame.task, "work_rate_observation")
        self.assertEqual(frame.slots["hours"], 6.0)
        self.assertEqual(frame.slots["workers"], 4.0)
        self.assertEqual(frame.slots["output"], 120.0)

    def test_v19_ratio_525_ir_and_solver_use_same_semantics(self) -> None:
        prompt = "مبلغ ۵۲۵ را با نسبت ۴ به ۳ بین دو نفر تقسیم کن"
        ir = SemanticIRParserV19.parse(prompt)
        self.assertIsNotNone(ir)
        assert ir is not None
        self.assertEqual(ir.task, "ratio_split")
        self.assertEqual(ir.slots, {"total": 525.0, "a": 4.0, "b": 3.0})
        answer = LocalIntelligenceV19().solve(prompt, "fa")
        self.assertIsNotNone(answer)
        assert answer is not None
        self.assertIn("300", answer.text)
        self.assertIn("225", answer.text)

    def test_v19_execution_graph_is_parsed_and_executed(self) -> None:
        prompt = "500 را 20 درصد کم کن و بعد 30 اضافه کن"
        ir = SemanticIRParserV19.parse(prompt)
        self.assertIsNotNone(ir)
        assert ir is not None
        self.assertEqual(ir.task, "execution_graph")
        self.assertEqual(ir.slots["initial"], 500.0)
        self.assertEqual(ir.slots["steps"], [
            {"op": "percentage_remove", "percent": 20.0},
            {"op": "add", "value": 30.0},
        ])
        answer = LocalIntelligenceV19().solve(prompt, "fa")
        self.assertIsNotNone(answer)
        assert answer is not None
        self.assertIn("430", answer.text)

    def test_v19_router_priority_prevents_percentage_collisions(self) -> None:
        with TemporaryRuntime() as runtime:
            volume = runtime.agent.router.route("صدا را روی ۳۰ درصد بگذار")
            brightness = runtime.agent.router.route("روشنایی را روی ۴۰ درصد تنظیم کن")
            probability = runtime.agent.router.route("احتمال موفقیت 30 درصد است؛ در 4 بار دقیقاً 2 موفقیت؟")
            sequence = runtime.agent.router.route("دنباله هندسی از 4 با نسبت 3 داریم. جمله 6 چیست؟")
            self.assertEqual((volume.intent, volume.arguments.get("percent")), ("set_volume", 30))
            self.assertEqual((brightness.intent, brightness.arguments.get("percent")), ("set_brightness", 40))
            self.assertNotEqual(probability.intent, "percentage")
            self.assertNotEqual(sequence.intent, "ratio_split")

    def test_v19_core_integration_hits_new_ir_before_old_wrong_ratio_solver(self) -> None:
        with TemporaryRuntime() as runtime:
            with mock.patch.object(runtime.tools, "invoke", side_effect=AssertionError("unexpected tool invocation")):
                cases = (
                    ("مبلغ ۵۲۵ را با نسبت ۴ به ۳ بین دو نفر تقسیم کن", ("300", "225")),
                    ("احتمال موفقیت 30 درصد است؛ در 4 بار دقیقاً 2 موفقیت؟", ("26.46",)),
                    ("دنباله هندسی از 4 با نسبت 3 داریم. جمله 6 چیست؟", ("972",)),
                    ("500 را 20 درصد کم کن و بعد 30 اضافه کن", ("430",)),
                )
                for prompt, fragments in cases:
                    with self.subTest(prompt=prompt):
                        answer = runtime.agent.respond(prompt)
                        for fragment in fragments:
                            self.assertIn(fragment, answer.text)
                        self.assertNotEqual(answer.intent, "tool_failure")


if __name__ == "__main__":
    unittest.main()
