from __future__ import annotations

import json
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

from jarvis.agent.semantic_ir_v18 import SemanticIRParserV18
from jarvis.agent.semantic_router_v18 import SemanticIntentRouterV18
from jarvis.neural.transformer import JarvisTransformer
from jarvis.utils.text import normalize_text
from tests.helpers import ROOT, TemporaryRuntime
from tests.v18_holdout_cases import V18_HOLDOUT_CASES


class IntelligenceV18Tests(unittest.TestCase):
    def test_v18_holdout_end_to_end_without_web_or_tools(self) -> None:
        with TemporaryRuntime() as runtime:
            with mock.patch.object(runtime.tools, "invoke", side_effect=AssertionError("unexpected tool invocation")):
                for prompt, fragments in V18_HOLDOUT_CASES:
                    with self.subTest(prompt=prompt):
                        answer = runtime.agent.respond(prompt)
                        low = answer.text.casefold()
                        for fragment in fragments:
                            self.assertIn(fragment.casefold(), low)
                        self.assertNotIn(answer.intent, {"tool_clarification", "unknown_information", "fact_not_found"})

    def test_v18_holdout_has_zero_exact_overlap_v009(self) -> None:
        seen: set[str] = set()
        for split in ("train", "validation", "test"):
            path = ROOT / "datasets" / "splits" / f"dataset_v009_{split}.jsonl"
            for line in path.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    seen.add(normalize_text(json.loads(line)["input"]).casefold())
        self.assertEqual(len(V18_HOLDOUT_CASES), 50)
        for prompt, _ in V18_HOLDOUT_CASES:
            with self.subTest(prompt=prompt):
                self.assertNotIn(normalize_text(prompt).casefold(), seen)

    def test_v18_transformer_is_really_trained_on_v009(self) -> None:
        metadata = JarvisTransformer.peek_metadata(ROOT / "models" / "jarvis_nano_v18.npz")
        self.assertEqual(metadata.get("model_id"), "jarvis_nano_v18")
        self.assertEqual(metadata.get("dataset_version"), "dataset_v009")
        self.assertEqual(int(metadata.get("parameter_count", 0)), 23_077_376)
        gate = json.loads((ROOT / "models" / "release_gate_v18.json").read_text(encoding="utf-8"))
        self.assertTrue(gate["passed"])

    def test_v18_semantic_models_are_v009_models(self) -> None:
        router = SemanticIntentRouterV18(ROOT / "models" / "semantic_router_v18.npz")
        self.assertTrue(router.ready)
        self.assertEqual(router.metadata.get("dataset_version"), "dataset_v009")
        for name in ("semantic_frame_v18.npz", "numeric_slot_tagger_v18.npz"):
            with np.load(ROOT / "models" / name, allow_pickle=False) as payload:
                self.assertEqual(str(payload["dataset_version"].item()), "dataset_v009")

    def test_v18_registry_activates_v18_models(self) -> None:
        registry = json.loads((ROOT / "models" / "model_registry_v2.json").read_text(encoding="utf-8"))
        active = next(row for row in registry["profiles"] if row.get("profile") == registry["active_profile"])
        self.assertEqual(active["id"], "jarvis_nano_v18")
        self.assertEqual(active["dataset_version"], "dataset_v009")
        aux = {row["id"]: row for row in registry["auxiliary_models"]}
        for model_id in ("semantic_router_v18", "semantic_frame_v18", "numeric_slot_tagger_v18"):
            self.assertEqual(aux[model_id]["dataset_version"], "dataset_v009")
        self.assertEqual(registry["total_model_oriented_parameters"], 28_713_512)

    def test_v18_ir_question_polarity_and_order_independence(self) -> None:
        cases = (
            ("آیا 103 غیر از 1 و خودش عامل دیگری دارد؟", "has_other_divisor"),
            ("آیا 121 فقط بر 1 و خودش بخش‌پذیر است؟", "only_one_self"),
        )
        for prompt, predicate in cases:
            ir = SemanticIRParserV18.parse(prompt)
            self.assertIsNotNone(ir)
            assert ir is not None
            self.assertEqual(ir.predicate, predicate)
            self.assertEqual(ir.polarity, "yes_no")
        speed = SemanticIRParserV18.parse("در 2 ساعت، 150 کیلومتر طی شد؛ سرعت متوسط؟")
        self.assertIsNotNone(speed)
        assert speed is not None
        self.assertEqual(speed.slots["distance_km"], 150.0)
        self.assertEqual(speed.slots["time_hours"], 2.0)


if __name__ == "__main__":
    unittest.main()
