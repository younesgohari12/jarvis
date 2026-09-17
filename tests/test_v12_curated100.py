from __future__ import annotations

import json
import unittest
from pathlib import Path

from jarvis.config import load_config
from jarvis.neural.transformer import JarvisTransformer

ROOT = Path(__file__).resolve().parents[1]


class Curated100V12Tests(unittest.TestCase):
    def test_curated_pack_has_exactly_100_sources_and_4800_unique_examples(self) -> None:
        manifest = json.loads((ROOT / "datasets" / "curated_100_v12_manifest.json").read_text(encoding="utf-8"))
        self.assertEqual(manifest["source_dataset_count"], 100)
        self.assertEqual(manifest["examples"], 4800)
        self.assertEqual(manifest["unique_input_output_pairs"], 4800)
        self.assertEqual(len(manifest["sources"]), 100)
        self.assertTrue(all(int(source["examples"]) == 48 for source in manifest["sources"]))
        self.assertTrue(all((ROOT / source["file"]).is_file() for source in manifest["sources"]))

    def test_dataset_v004_is_leakage_free_and_contains_curated_pack(self) -> None:
        manifest = json.loads((ROOT / "datasets" / "manifest_v004.json").read_text(encoding="utf-8"))
        self.assertEqual(manifest["total_examples"], 10329)
        self.assertEqual(manifest["inherited_v003_examples"], 5529)
        self.assertEqual(manifest["curated_source_dataset_count"], 100)
        self.assertEqual(manifest["curated_examples_kept"], 4800)
        self.assertEqual(manifest["cross_split_concept_leakage"], 0)
        self.assertEqual(sum(manifest["split_counts"].values()), manifest["total_examples"])
        self.assertIn("lab_imports", manifest)

    def test_release_model_passed_gate_and_is_default(self) -> None:
        gate = json.loads((ROOT / "models" / "release_gate_v12.json").read_text(encoding="utf-8"))
        self.assertTrue(gate["passed"])
        self.assertGreater(
            gate["curated_v12_holdout_gate"]["candidate_metrics"]["by_language_token_accuracy"]["fa"],
            gate["curated_v12_holdout_gate"]["baseline_metrics"]["by_language_token_accuracy"]["fa"],
        )
        model = JarvisTransformer.load(ROOT / "models" / "jarvis_nano_v12.npz")
        self.assertEqual(model.metadata.get("dataset_version"), "dataset_v004")
        self.assertEqual(model.metadata.get("pretrained_source"), None)
        config = load_config(ROOT)
        self.assertEqual(config.paths.neural_model.name, "jarvis_nano_v18.npz")


if __name__ == "__main__":
    unittest.main()
