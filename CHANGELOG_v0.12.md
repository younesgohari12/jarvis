# CHANGELOG v0.12.0 — Intelligence v23.0.0

Base: v0.11.0 — Intelligence v22.4.2. Scope: V23_LANGUAGE_BRAIN_PLAN.md Phase 1.

## Added
- **Unit conversion service** (`units_service_v23.py`, کارتابل واحد): standalone FA/EN conversions over mass/distance/time/volume + compound speed (km/h↔m/s↔mph), word-numbers (نیم/ربع/سه/…), fail-closed on unknown units/temperature/currency mixes, identity-conversion and single-number guards, inverse-restoration witness.
- **Work-rate algebra service** (`work_rate_v23.py`): forward scale (hours/minutes), inverted hours, harmonic combined inversion, per-worker rate sums, same-crew extrapolation, mid-task half-leave phases, per-plural rates, extended rate nouns (pages/bottles/…), currency rate application («dollars an hour», «ساعتی ۱۵ دلار», «10 dollars for every 2 hours»).
- **Narrative arithmetic service** (`narrative_math_v23.py`): sum/difference, price drop, net weight, fraction inverse (نصف/یک‌سوم/یک‌چهارم — Half/One third/One quarter), pack multiply, rectangle perimeter, recipe scaling, passenger get-off/get-on chains, original amount, rope pieces, temperature reverse, page-rate application, «ثمن و مابقی» with «هزار تومان» scale preservation.
- **Engine wiring** (`local_intelligence_v23.py`, engine 6.0.0-v23.0): zero-regression hook — v22.4.2 pipeline verbatim first; v23 services only on abstain/refuse; structured dimension refusals never overridden; every v23 answer witness-verified.
- **Router guard** (`local_intelligence_router_v23`): fires only when a v23 service fully solves the text now; packaging math («بسته‌بندی») no longer stolen by close-app; translation/code guards keep precedence.
- **Datasets** (`datasets/v23/`): train_core 12,535 + dev_paraphrase 1,706, seeded, deduped, contamination-audited (zero frozen-eval rows, 3-gram threshold 0.34), SHA-256 recorded (`datasets_manifest.json`).
- **Frozen blind set** (`benchmarks/v23_blind_set.jsonl`): 1,245 rows, frozen + read-only + SHA recorded BEFORE any v23 engine code existed; honest authoring labels (literal/authored-template/synthetic); hard negatives included.
- **Tests**: 86 service tests, 7 mutation-guard tests (G1–G7, valid-kill), 6 BUG-004 regression tests → suite 1292→1391 (+585 subtests unchanged).
- `benchmarks/run_v23_blind.py` (one-blind-run runner with freeze-SHA gate), `benchmarks/package_v23.py` (deterministic packager).

## Fixed
- **BUG-004 (MEDIUM, root cause, two layers)**: mixed-dimension chains verified when the parsed initial was untyped/dimensionless («۱۵۰ دلار + ۲۰ کالا = 170»). The chain audit now adopts the first typed operand's dimension, keeps auditing every later typed operand, and runs even for untyped initials; bare dimensionless operands never poison typed chains (v21 semantics kept); «جمعش/مجموعش» recognized as chain language. Reproduction evidence in `tests/test_v23_bugfixes.py`.

## Measured results
- Diagnostic v22.4.2 (frozen set `8e4aac45df…`): 32.71% → **73.83%** numeric; answerable-only 25.77% → **71.13%**; hard_negatives 10/10.
- Blind v23 (frozen set `a65ed2fd27…`, 1245): **79.60%** numeric; answerable-only **78.88%**; hard_negatives 46/47.
- Performance: diagnostic wall 156.3 s → 74.1 s.
