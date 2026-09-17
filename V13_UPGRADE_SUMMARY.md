# JARVIS v13 Upgrade Summary

## Added intelligence data

- Generalization: 100 curated source datasets, 3,600 unique examples.
- Reasoning: 100 curated source datasets, 3,600 unique examples.
- Previous dataset retained: 10,329 examples from dataset_v004.
- New combined dataset_v005: 17,529 examples.
- Cross-split concept leakage: 0.
- Tokenizer normalized round-trip accuracy: 100%.
- Unknown-token rate: 0% with the retained release tokenizer.

## Release model

- `models/jarvis_nano_v13.npz`
- 23,077,376 parameters.
- INT8 project-native model format.
- Based on v12 continual training plus v13 Generalization/Reasoning training and 40% weight interpolation to control catastrophic drift.

## Independent release gate vs v12

120 held-out cases per new axis plus 120 historical regression cases:

- Generalization cross-entropy: improved by 0.407%; token accuracy changed by -0.085 percentage points (effectively flat within the strict non-regression gate).
- Reasoning cross-entropy: improved by 1.077%; token accuracy improved by +0.214 percentage points.
- Historical v004 regression: cross-entropy improved by 0.090%; token accuracy improved by +0.311 percentage points.

Full gate: `models/release_gate_v13.json`.

## Verification

- 663 unit/integration tests passed.
- 0 test failures.
- v13 is the default neural release; v12 and v08 remain as fallbacks.
