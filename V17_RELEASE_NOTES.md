# JARVIS Intelligence v17

## Goal
v17 targets sentence-form sensitivity and generalization failures that remained after v16. The central change is a Semantic Intermediate Representation (SIR) layer rather than another large collection of regex-only solvers.

## Runtime architecture
The local reasoning path is now:

`input -> semantic frame parse -> role binding -> bounded plan -> deterministic solve -> verification -> answer`

Examples of frames:

- `task=binomial_probability, p=0.70, n=4, k=3`
- `task=permutation, n=11, k=4`
- `task=speed, distance_km=180, time_hours=3`
- `task=work_scaling, workers1=4, hours1=6, output1=120, workers2=2, hours2=3`

The solver consumes bound roles instead of rescanning arbitrary numbers from the raw sentence. This reduces number/regex collisions and makes multiple surface forms map to the same operation.

## Major fixes
- Biased coin probability recognizes Persian/English variants including «احتمال»، «شانس», chance, probability and attached forms such as «احتمال شیرش».
- Permutation recognizes ordered selections, no-replacement wording and ranked first/second/third selections.
- Two-dice probability recognizes `add up to`, sum/total and Persian equivalents.
- Direct speed problems bind distance and time roles before division.
- Worker productivity problems use worker-hours and preserve the requested output variable.
- Transitive symbolic chains are solved before tool/action routing.
- Prime yes/no answers now align the polarity with the explanation.
- Python even-number generation stays local and no longer tries Tool/Web.
- Persian rewrite v17 preserves the payload and normalizes colloquial clitics such as `قراردادو`, `برام`, `مقاله رو`, `کد رو`.
- Self-contained rewrite requests bypass stale context, while follow-ups such as «همان جمله را رسمی‌تر کن» still use conversation context.

## Multi-step reasoning
v17 uses bounded, internal multi-stage reasoning rather than free-form chain generation:
1. parse semantic task,
2. bind roles,
3. build a solver plan,
4. compute,
5. verify the result,
6. render a concise answer.

This is intended to improve reliability without exposing or depending on long free-form hidden reasoning text.

## Dataset v008
- Parent: dataset_v007
- New curated examples: 1,570
- Total examples: 38,100
- 40 auditable v17 shards
- New inputs are 100% unique inside the v17 addition
- Exact overlap with parent: 0
- Cross-split concept leakage: 0
- Tokenizer normalized round-trip: 100%
- Unknown-token rate: 0%

The v17 additions emphasize semantic role binding and checkable multi-step problems rather than paraphrase volume.

## Quality gates
- 40/40 targeted v17 holdout prompts pass without Web/Tool.
- Exact overlap of those 40 prompts with dataset_v008: 0/40.
- Full regression suite: 680/680 tests passed in three independent batches, 0 failures.

## Model parameters
The neural/learned model parameter count is intentionally unchanged in this release. The existing Transformer and learned semantic router remain in place; v17 improves runtime cognition through SIR and verified solvers rather than reporting a fake parameter increase without a quality-gated training run.
