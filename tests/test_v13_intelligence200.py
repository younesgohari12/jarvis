from __future__ import annotations
import json, unittest
from pathlib import Path
from jarvis.config import load_config
from jarvis.neural.transformer import JarvisTransformer
ROOT=Path(__file__).resolve().parents[1]

class Intelligence200V13Tests(unittest.TestCase):
    def test_exactly_100_generalization_and_100_reasoning_sources(self):
        g=json.loads((ROOT/'datasets/generalization_100_v13_manifest.json').read_text(encoding='utf-8'))
        r=json.loads((ROOT/'datasets/reasoning_100_v13_manifest.json').read_text(encoding='utf-8'))
        self.assertEqual(g['source_dataset_count'],100); self.assertEqual(r['source_dataset_count'],100)
        self.assertEqual(g['examples'],3600); self.assertEqual(r['examples'],3600)
        self.assertEqual(g['unique_input_output_pairs'],3600); self.assertEqual(r['unique_input_output_pairs'],3600)
        self.assertTrue(all((ROOT/x['file']).is_file() for x in g['sources']))
        self.assertTrue(all((ROOT/x['file']).is_file() for x in r['sources']))
    def test_dataset_v005_has_no_concept_leakage(self):
        m=json.loads((ROOT/'datasets/manifest_v005.json').read_text(encoding='utf-8'))
        self.assertEqual(m['total_examples'],17529); self.assertEqual(m['inherited_v004_examples'],10329)
        self.assertEqual(m['generalization_examples_added'],3600); self.assertEqual(m['reasoning_examples_added'],3600)
        self.assertEqual(m['cross_split_concept_leakage'],0)
    def test_v13_release_gate_passes_both_axes_and_regression(self):
        g=json.loads((ROOT/'models/release_gate_v13.json').read_text(encoding='utf-8'))
        self.assertTrue(g['passed']); q=g['quality_gate']['groups']
        self.assertGreater(q['generalization']['cross_entropy_improvement_pct'],0)
        self.assertGreater(q['reasoning']['cross_entropy_improvement_pct'],0)
        self.assertGreater(q['reasoning']['token_accuracy_delta_points'],0)
        self.assertGreaterEqual(q['v004_regression']['token_accuracy_delta_points'],0)
    def test_v13_is_default_release(self):
        model=JarvisTransformer.load(ROOT/'models/jarvis_nano_v13.npz')
        self.assertEqual(model.metadata.get('dataset_version'),'dataset_v005')
        self.assertEqual(model.metadata.get('pretrained_source'),None)
        self.assertEqual(load_config(ROOT).paths.neural_model.name,'jarvis_nano_v18.npz')

if __name__=='__main__': unittest.main()
