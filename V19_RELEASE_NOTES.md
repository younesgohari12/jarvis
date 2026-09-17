# JARVIS v19 — Real Trained Semantic IR Final

This release replaces the prototype v19 binder with a trained, gated semantic pipeline.

## Runtime architecture

Text → high-confidence system-action gate → Semantic IR v19 → trained slot/frame models → typed solver / ExecutionGraph executor → legacy fallback only when v19 abstains.

Key regressions fixed:
- ratio split `525 : 4 : 3` now returns `300` and `225`;
- volume and brightness are no longer stolen by generic percentage handling;
- probability outranks generic percentage;
- geometric sequence mentions of ratio no longer route to ratio-split;
- numeric roles are context-bound instead of assigned by number position;
- multistep IR is executed by the v19 ExecutionGraph path.

## Real trained v19 models

All four files are runtime-loadable `.npz` trained artifacts using `dataset_v019_real` with `template_id_disjoint` holdouts:

- `numeric_role_v19.npz` — 1,179,666 parameters — test accuracy 95.40%
- `semantic_frame_v19.npz` — 589,833 parameters — test accuracy 96.68%
- `execution_pattern_v19.npz` — 196,614 parameters — test accuracy 100.00%
- `code_intent_v19.npz` — 196,614 parameters — test accuracy 95.10%

Total new trained semantic parameters: 2,162,727. The main v18 transformer weights were intentionally not enlarged or falsely relabeled as newly trained.

## Real training datasets

- Numeric Role: 24,000 examples
- Multistep Execution Graph: 12,000 examples
- Code Intelligence: 5,000 examples
- Semantic Frame: 18,000 examples

Each principal family has at least 30 independent template IDs; holdouts are split by template ID rather than random rows from the same wording template.

## Verification

Fresh v19 intelligence holdout: 50/50, with zero exact overlap against the four v19 training datasets.

Complete regression suite was executed in four shards:
- 696 standard tests passed
- 585 subtests passed
- 0 failures

The release packager also validates the four v19 trained models, quality gates, regression report, ZIP CRC, extracted self-test, and extracted targeted v06/v18/v19 regression suite.
