from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from jarvis.neural.config import TransformerConfig
from jarvis.neural.tokenizer import JarvisTokenizer, normalize_language_text
from jarvis.neural.transformer import JarvisTransformer, NeuralModelError
from training.numpy_trainer import NumpyTransformerTrainer


ROOT = Path(__file__).resolve().parents[1]


class NeuralV07Tests(unittest.TestCase):
    def test_release_model_is_project_owned_and_exact_size(self) -> None:
        model = JarvisTransformer.load(ROOT / "models" / "jarvis_nano_v07.npz")
        self.assertEqual(model.parameter_count, 23_077_376)
        self.assertEqual(model.config.architecture, "decoder_only_transformer_gqa_rope_swiglu")
        self.assertIsNone(model.metadata.get("pretrained_source"))
        self.assertTrue(model.metadata.get("training_complete"))
        self.assertEqual(model.metadata.get("dataset_version"), "dataset_v003")
        self.assertEqual(model.metadata.get("model_format"), "jarvis-numpy-int8-v2")
        self.assertIn("weights_sha256", model.metadata.get("integrity", {}))

    def test_tokenizer_round_trip_and_provenance(self) -> None:
        tokenizer = JarvisTokenizer.load(ROOT / "models" / "jarvis_tokenizer_v003.json")
        self.assertEqual(tokenizer.vocab_size, 4096)
        for text in ("سلام JARVIS", "Open D:\\Projects", "print('یونس')", "فارسی + English"):
            with self.subTest(text=text):
                self.assertEqual(tokenizer.decode(tokenizer.encode(text)), normalize_language_text(text))
        payload = json.loads((ROOT / "models" / "jarvis_tokenizer_v003.json").read_text(encoding="utf-8"))
        self.assertIsNone(payload["pretrained_source"])

    def test_kv_cache_matches_full_forward(self) -> None:
        model = JarvisTransformer.load(ROOT / "models" / "jarvis_nano_v07.npz")
        tokenizer = JarvisTokenizer.load(ROOT / "models" / "jarvis_tokenizer_v003.json")
        tokens = [tokenizer.bos_id, tokenizer.special_to_id["<user>"], *tokenizer.encode("hello")]
        expected = model.forward(tokens)[0, -1]
        actual, cache = model.prefill_kv_cache(tokens)
        self.assertTrue(np.allclose(expected, actual, rtol=2e-4, atol=2e-4))
        self.assertEqual(cache[0]["key"].shape[1], len(tokens))

    def test_external_pretrained_metadata_is_rejected(self) -> None:
        config = self._tiny_config()
        model = JarvisTransformer.initialize(config)
        with tempfile.TemporaryDirectory(prefix="jarvis-provenance-") as temporary:
            source = Path(temporary) / "source.npz"
            target = Path(temporary) / "tampered.npz"
            model.save_quantized(source, {"training_complete": True})
            with np.load(source, allow_pickle=False) as archive:
                arrays = {name: archive[name] for name in archive.files}
            metadata = json.loads(bytes(arrays["__metadata__"]).decode("utf-8"))
            metadata["pretrained_source"] = "external-model"
            arrays["__metadata__"] = np.frombuffer(json.dumps(metadata).encode("utf-8"), dtype=np.uint8)
            np.savez_compressed(target, **arrays)
            with self.assertRaises(NeuralModelError):
                JarvisTransformer.load(target)

    def test_full_gradient_training_reduces_repeated_batch_loss(self) -> None:
        model = JarvisTransformer.initialize(self._tiny_config())
        trainer = NumpyTransformerTrainer(
            model, learning_rate=0.01, minimum_learning_rate=0.002,
            weight_decay=0.0, gradient_clip=5.0, warmup_steps=1, total_steps=20,
        )
        inputs = np.asarray([[1, 5, 10, 11, 6, 12, 13, 2]], dtype=np.int64)
        targets = np.asarray([[-100, -100, -100, 6, 12, 13, 2, 2]], dtype=np.int64)
        before = trainer.evaluate_batch(inputs, targets)
        for _ in range(20):
            trainer.train_batch(inputs, targets)
        after = trainer.evaluate_batch(inputs, targets)
        self.assertLess(after, before * 0.7)

    @staticmethod
    def _tiny_config() -> TransformerConfig:
        return TransformerConfig.from_dict(
            {
                "name": "Test Nano", "profile": "test",
                "format": "jarvis-transformer-v1", "architecture": "decoder_only_transformer",
                "vocab_size": 64, "d_model": 32, "n_layers": 1,
                "n_heads": 4, "n_kv_heads": 4, "ffn_hidden": 64,
                "max_seq_len": 16, "tie_embeddings": True, "seed": 42,
                "training": {"sequence_length": 8, "micro_batch_size": 1,
                    "gradient_accumulation": 1, "learning_rate": 0.01,
                    "minimum_learning_rate": 0.002, "weight_decay": 0.0,
                    "gradient_clip": 5.0, "warmup_steps": 1, "steps_per_stage": 1},
            }
        )

    def test_dataset_has_thousands_and_disjoint_splits(self) -> None:
        manifest = json.loads((ROOT / "datasets" / "manifest_v003.json").read_text(encoding="utf-8"))
        self.assertGreaterEqual(manifest["total_examples"], 5000)
        self.assertEqual(manifest["cross_split_concept_leakage"], 0)
        self.assertIsNone(manifest["provenance"]["pretrained_source"])
        split_ids: list[set[str]] = []
        stages: set[int] = set()
        for split in ("train", "validation", "test"):
            rows = [json.loads(line) for line in (ROOT / "datasets" / "splits" / f"dataset_v003_{split}.jsonl").read_text(encoding="utf-8").splitlines()]
            split_ids.append({str(row["id"]) for row in rows})
            self.assertTrue(all(row["pretrained_source"] is None for row in rows))
            stages.update(int(row["stage"]) for row in rows)
        self.assertEqual(stages, set(range(1, 15)))
        self.assertFalse(split_ids[0] & split_ids[1])
        self.assertFalse(split_ids[0] & split_ids[2])
        self.assertFalse(split_ids[1] & split_ids[2])


if __name__ == "__main__":
    unittest.main()
