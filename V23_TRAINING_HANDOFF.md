# V23 TRAINING HANDOFF — Generalized Weakness Targets

**From:** JARVIS v22.4.1 (frozen deterministic core, release-integrity lock)
**To:** v23 — GPU training / real Language Brain work
**Status:** informational. No v23 work begins in v22.4.1.

---

## Purpose and rules

This file contains **only generalized weaknesses** observed across the v22.4
evaluation evidence (broad 500, legacy fresh regression, the frozen 2075-case
blind benchmark, domain reports). It deliberately contains **no copied blind
benchmark questions** — the blind benchmark is frozen evaluation evidence and
its exact rows must never be fed into v23 training (contamination). Design
*vnew* training data from the abstractions below instead.

## Priority-ordered generalization targets

### 1. Work-rate paraphrase generalization (blind 52/90 — weakest family)
The engine solves canonical work-rate formulas when the surface form matches
its trained patterns. It fails on varied paraphrases: combined-rate inversion
("how long together?"), partial contributions, worker joins/leaves
mid-task, per-unit rates ("each worker …"), and negative constraints.
**v23 dataset direction:** generate *new* work-rate corpora with heavy
paraphrase diversity (inverted queries, collective nouns, mixed time units,
implicit "remaining work" framings). Teach the model the underlying algebra
(rate × time = output; additive rates), not surface cues.

### 2. Ownership linguistic diversity (blind 194/235)
The event model is correct (legacy 60/60) but blind paraphrases expose
brittle verb/argument recognition: "pays", "sends", indirect object
constructions, split transfers ("gives 5 to A and 5 to B"), and pronoun
chains across >2 sentences. **v23 direction:** paraphrase-rich transfer
corpora in FA/EN with varied verbs (می‌فرستد / واریز می‌کند / حواله می‌کند /
hands / wires), multi-leg chains, and conversational context. Keep the
audited event-model semantics as the verifier.

### 3. Rate expressions (blind 176/210)
Compound-dimension arithmetic works when units are explicit; misses occur on
implicit units ("dollars an hour"), inverted FA forms, per-plural nouns
("per 2 hours"), and speed/time/distance blends. **v23 direction:** corpora
that force dimension analysis rather than keyword matching; include unknown
units that must trigger abstention.

### 4. Finance paraphrases (blind 124/150)
Discount/tax/markup with reordering, "off" vs "reduced by" vs "discounted
to", multi-step (discount then tax), and percentage-of-remainder cases.
**v23 direction:** multi-step finance chains with explicit provenance
expectations; separate price_delta / discount / discount_percentage roles in
every example.

### 5. Age relation variants (blind 139/163)
The entity→attribute→value engine handles canonical forms; variants that
remain hard: relative frames ("when Ali was Reza's age…"), future/past
projection ("in 5 years"), ratio-of-ages, and named-multiple comparisons.
**v23 direction:** temporal-frame-aware age corpora; keep the abs()
difference and query-binding invariants as verifier constraints.

### 6. Persian/English natural generation (broad: FA 6/50, EN 0/50)
Free-form generation is the standing gap — the deterministic core answers
reasoning tasks but cannot produce natural language paragraphs. **v23
direction:** this is the Language Brain / neural scope: sequence-level
training with fact-locked rendering (numbers/units must be post-verified by
the v22 verifier — never let the generator emit unverified arithmetic).
Use failure-family abstractions to build training sets; validate every
generated number through the deterministic verifier before accepting it.

## Guardrails carried into v23

1. **Never train on the frozen 2075-case blind benchmark rows** (SHA
   `063aef8f…f7be1`) — evaluation only.
2. Keep the v22.4.1 verifier invariants as hard constraints: operation-aware
   dimension algebra, exact SourceSpan provenance, fail-closed verification,
   typed units with no silent substitution.
3. New benchmarks for v23 must be **frozen + SHA-256 recorded before first
   execution**, with honest authoring labels (literal / authored-template /
   synthetic) from day one.
4. Maintain the mutation-integrity discipline: every guard added in v23 gets
   a mutation test with valid-kill semantics.
5. Regression baseline to protect: 1227 tests + 585 subtests, 0 failures.
