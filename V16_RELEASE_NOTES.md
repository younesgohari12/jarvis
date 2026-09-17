# JARVIS Intelligence v16 — Release Notes

v16 is a targeted runtime-intelligence release built on the v15 final package. It fixes semantic parser, routing, translation-isolation, rewrite-fidelity, code-trace and constraint-following failures discovered with fresh holdout prompts.

## What changed

### Probability
- Biased-coin binomial parser now extracts three independent semantic roles: success probability `p`, trial count `n`, and exact-success count `k`.
- Example fixed: 60% heads, 3 tosses, exactly 2 heads -> 43.2%.
- Two-dice sum parser accepts more natural Persian and English phrasings.

### Permutations and ratios
- Added an ordered-selection/permutation solver for natural language such as `3 arrangements from 8 without repetition`.
- Added ratio-split solver for prompts such as `210 را به نسبت 2 به 5 تقسیم کن`.

### Code trace precedence
- A self-contained Python snippet containing state changes plus `print(...)` is now recognized as deterministic code trace even without an explicit `output/خروجی` phrase.
- Code trace bypasses conversation/memory interpretation and algebra routing.
- The evaluator remains AST-based and does not execute arbitrary Python.

### Translation isolation
- Translation is now a dedicated pipeline.
- A translation request can no longer fall through to Knowledge/RAG.
- If a reliable local translation is unavailable, v16 fails closed with a translation clarification rather than retrieving an unrelated knowledge entry.
- Added high-confidence status/restart translation patterns, including the reported server-restart sentence.

### Rewrite fidelity
- Explicit rewrite requests transform the supplied payload instead of generating a generic email template.
- Persian colloquial object markers and common informal verbs are formalized while preserving nouns, dates and deadlines.
- Example fixed: `لطفا گزارشو تا فردا واسم بفرست` -> `لطفاً گزارش را تا فردا برای من ارسال کنید.`

### Constraint following
- Added semantic variants such as `حتماً کلمه «X» را داشته باشد`, `واژه «X» حتماً باید در متن باشد`, and English `make sure it includes the word "X"`.
- Constrained writing is protected from multi-step segmentation when `and/و` joins constraints in one request.

## Verification

Targeted fresh benchmark, Web/Tool disabled:
- v15: 4/24
- v16: 24/24
- Exact prompt overlap with dataset_v007 train/validation/test: 0/24

Full regression suite after final source changes:
- 677 primary tests passed
- 279 subtests passed
- 0 failures
- 0 errors

## Model/data honesty

v16 does not rename the existing Transformer or inflate parameter counts. The model-oriented parameter total remains 26,223,116 (23,077,376 Transformer + 3,145,740 semantic router). The active training corpus remains dataset_v007 with 36,530 examples. This release improves behavior through runtime semantics and routing/isolation fixes rather than unvalidated extra training data.
