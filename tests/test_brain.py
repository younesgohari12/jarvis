from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path

from jarvis.brain.model import HybridNeuralBrain
from jarvis.config import load_config
from jarvis.utils.text import normalize_text


ROOT = Path(__file__).resolve().parents[1]


class HybridBrainTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls._temporary = tempfile.TemporaryDirectory(prefix="jarvis-brain-")
        cls._previous = os.environ.get("JARVIS_DATA_DIR")
        os.environ["JARVIS_DATA_DIR"] = cls._temporary.name
        cls.config = load_config(ROOT)
        cls.brain = HybridNeuralBrain.load(cls.config.paths.model, cls.config.brain)

    @classmethod
    def tearDownClass(cls) -> None:
        if cls._previous is None:
            os.environ.pop("JARVIS_DATA_DIR", None)
        else:
            os.environ["JARVIS_DATA_DIR"] = cls._previous
        cls._temporary.cleanup()

    def test_model_is_quantized_and_under_ten_megabytes(self) -> None:
        self.assertLess(self.config.paths.model.stat().st_size, 10 * 1024 * 1024)
        self.assertEqual(self.brain.metadata.get("quantization"), "symmetric_int8")

    def test_model_parameter_budget_is_realistic(self) -> None:
        self.assertGreaterEqual(self.brain.parameter_count, 2_000_000)
        self.assertLessEqual(self.brain.parameter_count, 8_000_000)
        self.assertEqual(self.brain.embedding_size, 64)
        self.assertEqual(self.brain.hidden_size, 80)

    def test_required_persian_intents(self) -> None:
        expected = {
            "سلام": "greeting",
            "جارویس": "call_assistant",
            "حالت چطوره؟": "how_are_you",
            "اسمت چیه؟": "ask_name",
            "چیکار می تونی بکنی؟": "capabilities",
        }
        for text, intent in expected.items():
            with self.subTest(text=text):
                self.assertEqual(self.brain.classify(text).intent, intent)

    def test_required_english_intents(self) -> None:
        expected = {
            "hello": "greeting",
            "how are you?": "how_are_you",
            "what is your name?": "ask_name",
            "tell me about yourself": "self_intro",
            "what can you do?": "capabilities",
        }
        for text, intent in expected.items():
            with self.subTest(text=text):
                self.assertEqual(self.brain.classify(text).intent, intent)

    def test_out_of_distribution_input_becomes_unknown(self) -> None:
        prediction = self.brain.classify("zxqv plmokn 773")
        self.assertTrue(prediction.is_unknown)
        self.assertEqual(prediction.intent, "unknown")
        self.assertTrue(prediction.unknown_reason)

    def test_classifier_is_deterministic(self) -> None:
        first = self.brain.classify("tell me about yourself")
        second = self.brain.classify("tell me about yourself")
        self.assertEqual(first, second)

    def test_full_router_metrics_exceed_quality_floor(self) -> None:
        metrics = json.loads(
            (ROOT / "models" / "training_metrics_fast_v7.json").read_text(encoding="utf-8")
        )
        # These are measured held-out neural/hybrid scores, not hard-coded
        # scenario answers.  Router/tool metrics are reported separately by
        # training/evaluate_brain.py.
        self.assertGreaterEqual(metrics["final_float_neural_test_accuracy"], 0.65)
        self.assertGreaterEqual(metrics["test_hybrid_accuracy"], 0.60)
        self.assertGreaterEqual(metrics["test_unknown_recall"], 0.80)
        self.assertIsNone(metrics.get("pretrained_source"))

    def test_explicit_dataset_splits_do_not_overlap(self) -> None:
        split_values: dict[str, set[str]] = {}
        for split in ("train", "validation", "test"):
            payload = json.loads(
                (ROOT / "data" / "training" / f"{split}.json").read_text(encoding="utf-8")
            )
            split_values[split] = {
                normalize_text(str(example))
                for item in payload["intents"]
                for example in item["examples"]
            }
        self.assertFalse(split_values["train"] & split_values["validation"])
        self.assertFalse(split_values["train"] & split_values["test"])
        self.assertFalse(split_values["validation"] & split_values["test"])


if __name__ == "__main__":
    unittest.main()
