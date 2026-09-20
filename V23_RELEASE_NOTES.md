# JARVIS v23.0.0 — Language Brain Phase 1 (RELEASE NOTES)

**Release:** v0.12.0-intelligence-v23.0.0
**Base:** v0.11.0 — Intelligence v22.4.2 (frozen deterministic core, release-integrity lock)
**Scope:** V23_LANGUAGE_BRAIN_PLAN.md «مرحله 1» — deterministic rule-based generalization + unit service + verified datasets. No model retraining (Phase 2), no neural generator (Phase 3) — both remain future work by design.

---

## What v23.0.0 adds

Three new **verified numeric services** run ONLY when the complete v22.4.2
pipeline produces no verified answer (zero-regression hook), and each result
must pass an independent witness re-derivation plus `numeric_guard` before
the engine may speak:

| Service | Module | Families covered (v22.4.2 diagnostic baseline) |
|---|---|---|
| Unit conversion service (کارتابل واحد) | `jarvis/agent/units_service_v23.py` | units **0/10** → mass/distance/time/volume FA+EN, compound speed km/h↔m/s↔mph |
| Work-rate algebra | `jarvis/agent/work_rate_v23.py` | work_rate **7/20** → forward scale (h/min), inverted hours, harmonic combined inversion, per-worker rate sums, same-crew extrapolation, mid-task crew leave, per-plural rates («هر ۲ ساعت»), extended rate nouns (pages/bottles/…), currency rates («dollars an hour», «ساعتی ۱۵ دلار», «10 dollars for every 2 hours») |
| Narrative arithmetic | `jarvis/agent/narrative_math_v23.py` | persian_math **0/8** + english_math **0/8** → sum/difference, price drop, net weight, fraction inverse (نصف/یک‌سوم/یک‌چهارم), pack multiply, perimeter, recipe scale, passenger chains, original amount, rope pieces, temperature reverse, page rates, «ثمن و مابقی» with «هزار تومان» scale preservation |

**Routing:** the v23 numeric route (`local_intelligence_router_v23`) fires
only when a service fully solves the text right now — it outranks the
close-app collision («بسته‌بندی» is packaging, not close-app) and learned
word-problem labels that had no verified engine behind them. Translation and
other deterministic guards keep precedence.

## Root-cause bug fix carried in this release

**BUG-004 (MEDIUM — fail-open dimension hole).** «موجودی علی ۱۵۰ دلار و ۲۰
کالاست؛ جمعش چقدر است؟» computed 150+20=170 and passed verification.
Root causes: (1) `validate_operation_chain` disabled the whole chain audit
whenever the chain initial was dimensionless; (2) the typed audit skipped the
chain entirely when the parsed initial had no typed quantity (a parsed 0).
Fix: the chain now ADOPTS the first typed operand's dimension and keeps
auditing every later typed operand; bare dimensionless operands never poison
a typed chain (v21 semantics preserved); the audit also runs for untyped
initials; «جمعش/مجموعش» added to the chain language. Reproduction evidence:
170 before, structured refusal after (`tests/test_v23_bugfixes.py`).

## Evaluation (all measured, never hand-typed)

- Regression: **1391 tests + 585 subtests, 0 failures, 1 environment-dependent skip** (v22.4.2 baseline 1292+585 held; 86 new v23 service tests + 7 mutation-guard tests + 6 BUG-004 regression tests added; the engine-version wiring test was updated to the v23 contract while still asserting the v22 label is intact)
- Diagnostic v22.4.2 (same frozen set, SHA `8e4aac45df…`): **32.71% → 73.83%** numeric, answerable-only **25.77% → 71.13%**, hard_negatives **10/10** preserved; by-domain: units 0/10→**10/10**, work_rate 7/20→**18/20**, persian_math 0/8→**7/8**, english_math 0/8→**7/8**
- Blind v23 (frozen 1245-case set, SHA `a65ed2fd27…`, frozen + read-only BEFORE any v23 engine code existed): **79.60%** numeric (answerable-only **78.88%**), hard_negatives **46/47**; by-domain: units_conversion 181/182, narrative_math 259/260, work_rate 359/421, rates 83/102, finance 40/89, age 31/63, ownership 24/114
- Contamination: zero rows from any frozen evaluation set (3-gram guard, threshold 0.34, 4279 frozen source rows checked; 172 blind + 86 train candidates dropped by the guard)
- Mutation: **7 new guards (G1–G7) with valid-kill semantics** (`tests/test_v23_mutation_guards.py`) — identity guard, single-number guard, unknown-unit refusal, work-rate witness, positivity guard, narrative witness, inverse-restoration witness
- Performance: diagnostic wall time **74.1 s** (v22.4.2 baseline 156.3 s — faster because the v23 services answer before deeper abstention paths)

## Datasets (spec §3 — honest authoring labels from day one)

| Set | Rows | SHA-256 |
|---|---|---|
| `datasets/v23/train_core.jsonl` | 12,535 | `43a738c2…` |
| `datasets/v23/dev_paraphrase.jsonl` | 1,706 | `6c770056…` |
| `benchmarks/v23_blind_set.jsonl` | 1,245 (frozen, chmod 444) | `a65ed2fd…` |

The bounded template space saturates at 12,535 unique train rows (nominal
target ~15k; expanding further would duplicate rows). Phase 2 classifier
retraining consumes these sets; it is intentionally NOT part of this release.

## Honest remaining gaps (v23.1+ targets)

ownership paraphrase breadth (24/114 blind — multi-step quotes, pronoun
chains), finance margin/inverse forms (40/89), age relative frames (31/63),
speed×time application with mixed units, and the Phase 2 classifier
retraining + Phase 3 neural generator per the plan.

## Assets

- `Jarvis_v0.12.0_Intelligence_v23.0.0.zip` — the release package
- `package_integrity.json` — external finalization artifact measured from the final ZIP

**In-tree reports: reports/v23/REPORT_FA.md · reports/v23/blind_freeze.json · reports/v23/blind_eval.json · reports/v23/datasets_manifest.json · reports/v23/diagnostic_v22_4_2_with_v23.json · CHANGELOG_v0.12.md · TEST_RESULTS.json · RELEASE_V23.json**
