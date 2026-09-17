# JARVIS Intelligence v14 — Runtime + Generalization + Reasoning

This release fixes the main bottleneck found in v13: the Agent often routed a task to `unknown`/web before the improved neural model could answer it.

## What changed

- Added `LocalIntelligenceV14` before generic unknown/web routing for verifiable local tasks.
- Deterministic math/reasoning questions bypass conversational follow-up inference, preventing phrases such as «جملهٔ دنباله» from being misread as language follow-ups.
- Unknown routes no longer fall into unrelated creative/story generation before reasoning.
- Smart Brain remains a bounded fallback for genuinely uncertain routes; fresh/current questions can still use web research.
- LAB and tokenization now use `dataset_v006`; enabled user-authored LAB supervised data is merged into the active tokenized training corpus.

## High-precision local reasoning families

Factorial, prime test, GCD/LCM, percentage-of, linear equations, fair-coin probability, weekday offsets, combinations, permutations, averages and weighted averages, ratio reduction/splitting, unit conversion, Celsius/Fahrenheit conversion, arithmetic/geometric sequences, speed-distance problems, quotient/remainder, parity, selected Python coding requests, and bounded Persian↔English translation patterns.

## Dataset v006

- Total active examples: **15,329**
- Stable inherited v004 examples: **10,329**
- Generalization v14: **100 shards / 2,000 examples / 2,000 unique outputs (100.0%)**
- Reasoning v14: **100 shards / 3,000 examples / 3,000 unique outputs (100.0%)**
- Cross-split concept leakage: **0**
- Old v13 paraphrase/template-heavy Generalization/Reasoning packs are **not inherited into the active v006 corpus**.
- Tokenizer normalized round-trip: **100.0%**; unknown-token rate: **0.0%**.

The v14 examples are project-authored deterministic tasks with computed/verified labels. Output uniqueness is achieved by preserving meaningful problem-specific computation/evidence, not by adding fake exercise numbers.

## End-to-End holdout gate

A 30-case local/no-web holdout was created across the target failure families. Exact prompt overlap with every `dataset_v006` split is checked and must be zero.

- Original v13 ZIP: **0/30**
- v14 runtime: **30/30**

This benchmark is intentionally targeted at the routing/reasoning failures reported for v13; it is not a general-IQ score.

## Regression gate

**666/666 pytest tests passed** plus **169 subtests passed**. The v14 test file contributes 30 end-to-end cases, 30 dataset holdout checks, and dataset-quality checks.

## Neural model policy

The active transformer remains **`jarvis_nano_v13.npz` (23,077,376 parameters)** as the neural fallback. A new checkpoint is not labelled v14 unless actual continual training finishes and a separate neural quality gate passes. This release therefore does **not** disguise a copied/renamed v13 checkpoint as a new model.
