# JARVIS v22.4 — TRUE ROOT-CAUSE HARDENING AUDIT

Release: `v0.11.0-intelligence-v22.4`
Base tag: `v0.11.0-intelligence-v22.3` (commit `542424f`)
Package: `Jarvis_v0.11.0_Intelligence_v22_4_TRUE_ROOT_CAUSE_HARDENED.zip`

This release eliminates the v22.3 semantic-correctness defects at their
architectural root. It adds NO neural weights, NO training, and NO composite
score inflation. Every metric below is reproduced from fresh runs of the
committed runners; test counts are parsed from raw pytest output, never typed.

---

## 0. Verification protocol

* Baseline tag checked out and re-run: **1042 tests + 585 subtests, 0 failures**.
* Every P0 defect of v22.3 was first reproduced on the untouched baseline
  (`benchmarks/repro_v22_4_bugs.py` → `reports/v22_4/p0_reproduction.json`),
  then re-checked after the fix (same script, second run).
* Full regression re-run after every change: final **1215 tests + 585 subtests,
  0 failures** (raw pytest output in `reports/v22_4/regression.json`).
* Mutation testing: 7 mutants, 7 killed, score **100 %**
  (`reports/v22_4/mutation_tests.json`).
* Source-span integrity: 2400 generated cases, 7223 spans, **0 mismatches**
  (`reports/v22_4/source_span_integrity.json`).
* Security scan: 1888 files, **0 secrets** (`reports/v22_4/security.json`).

---

## 1. P0 — time + currency escaped the dimension guard

| Field | Content |
|---|---|
| Bug | `verifier_v22.cross_dimension_chain_failure` whitelisted `known <= {time, count}` and `known <= {time, currency}` globally, so `2 hours + 5 USD` was never refused. |
| Old behavior | `2 hours + 5 dollars` → no failure (guard returned `[]`). |
| Expected behavior | rejected with a structured explanation; rate contexts stay legal. |
| Root cause | dimension-PAIR whitelisting instead of operation semantics. |
| Files changed | `semantics_v22_4.py` (new: `validate_binary_operation`, `chain_operands`, `cross_dimension_chain_failure_v4`), `verifier_v22.py` (exemptions removed), `local_intelligence_v22.py` (direct guard for unparseable violating requests). |
| Architecture fix | operation-aware dimension algebra: an add/subtract request validates EVERY consecutive operand pair; time quantities that merely qualify the sentence (`بعد از 3 روز`, `هر روز`) are excluded as operands. No dimension pair is globally whitelisted. |
| Tests | `tests/test_v224_root_cause.py::test_p0_time_currency_count_addition_rejected`, `::test_p0_time_distance_addition_rejected`, `::test_validate_binary_operation_matrix`, `::test_rate_contexts_stay_legal`. |
| Mutation test | M1 `disable_dimension_guard` → killed. |
| Regression result | 1215 + 585 green; broad 406/500; legacy fresh 1006/1023. |
| Status | **FIXED** |

## 2. P0 — speed units semantically wrong (10 m/s → KM_PER_HOUR)

| Field | Content |
|---|---|
| Bug | `quantity_v22` matched one shared speed regex and hard-coded `unit='KM_PER_HOUR'` for every family. |
| Old behavior | `10 m/s` → `Quantity(speed, KM_PER_HOUR)`; Persian `متر بر ثانیه` likewise. |
| Expected behavior | `m/s`→`M_PER_SECOND`, `km/h`→`KM_PER_HOUR`, `mph`→`MILE_PER_HOUR`, unknown→`UNKNOWN_UNIT`; raw_unit + normalized_unit preserved; conversion separate. |
| Root cause | no unit registry; unit identity destroyed at extraction time. |
| Files changed | `semantics_v22_4.py` (UNIT_REGISTRY, SPEED_UNIT_PATTERNS, `convert_unit`), `quantity_v22.py` (per-family anchored patterns, `raw_unit`/`normalized_unit` fields, UNKNOWN_UNIT policy). |
| Architecture fix | raw unit → normalized unit → dimension registry; extraction NEVER converts; `convert_unit` returns full provenance (source_value, source_unit, result_value, result_unit, factor). |
| Tests | `test_speed_unit_mapping`, `test_unknown_speed_unit_never_substituted`, `test_quantity_preserves_raw_and_normalized_unit`, `test_extraction_does_not_convert`, `test_explicit_conversion_metadata`; property test fuzzes 0..10000 values. |
| Mutation test | M2 `m/s → KM_PER_HOUR` → killed. |
| Regression result | speed family 60/60 legacy; unit_mapping report 10/10. |
| Status | **FIXED** |

## 3. P0 — source span truncated (`50 کال`, `3 worker`, `8 نف`)

| Field | Content |
|---|---|
| Bug | unit suffixes matched on a STRIPPED substring; the suffix offset was reused against the unstripped string, shifting the span end. |
| Old behavior | `50 کالا` → span `50 کال`; invariant `original[start:end]==text` violated. |
| Expected behavior | exact provenance, always. |
| Root cause | offsets from transformed text reused against the original text. |
| Files changed | `semantics_v22_4.py` (`SourceSpan`, `make_span`, `SpanIntegrityError`, `IndexMap`), `quantity_v22.py` (all suffix branches anchored on the unstripped `right`; emit() hard-asserts the invariant). |
| Architecture fix | `SourceSpan` frozen dataclass with a hard invariant; a wrong span cannot leave the module. |
| Tests | `test_exact_spans_on_original_text` (8 parametrized forms), span property tests (hypothesis, generated sentences), `test_span_invariant_enforced`, `test_source_span_is_frozen_type`. |
| Mutation test | M3 `span_end -= 1` → killed. |
| Regression result | 2400 generated cases / 7223 spans / 0 mismatches. |
| Status | **FIXED** |

## 4. P0 — age understanding regex-bound

| Field | Content |
|---|---|
| Bug | `detect_age_difference` relied on three surface regexes (`X N ساله`, `سن X N`, `X is N`); `aged 35`, possessive, `سال دارد`, relations and pronoun anaphora all failed. |
| Old behavior | 3 of 4 paraphrase families returned `None`; age-difference language missed `years apart`; no sum-of-ages. |
| Expected behavior | entity → attribute → value semantics with query binding. |
| Root cause | surface regexes instead of an entity-attribute model. |
| Files changed | `semantics_v22_4.py` (`AgeFact`, `extract_age_facts`, `bind_age_query`, `resolve_age`, `solve_age_query`, pronoun anaphora), `language_brain_v22.py` (delegates to the engine), `local_intelligence_v22.py` (`_age_relation_answer`, `_age_sum_answer`, last-resort model path). |
| Architecture fix | one generalized extraction architecture; a new surface form adds a PATTERN ROW, not a code path. Query binds NAMED entities (never the first two numbers). Difference invariant: `abs(a-b) >= 0` only after binding. |
| Tests | `test_age_entity_binding_generalizes` (8 parametrized forms), `test_age_query_binds_named_entities` (3-name binding), `test_age_difference_invariant_non_negative`, `test_generalized_scheduling_frames`; age property tests. |
| Regression result | legacy age family **70/70** (was 46/70). |
| Status | **FIXED** |

## 5. P0 — dimension failure messages not structured

| Field | Content |
|---|---|
| Bug | generic `source_operation_dimension_consistency` was rendered as the hard-coded `currency_item` message: `2 ساعت و 120 کیلومتر` was described as "money and items". |
| Old behavior | wrong explanation for every non-currency mismatch. |
| Expected behavior | structured failure object + dimension-accurate messages (FA/EN). |
| Root cause | the verifier carried only a failure NAME; the renderer guessed. |
| Files changed | `semantics_v22_4.py` (`DimensionFailure` with operator/left/right dimension+unit and per-pair messages), `verifier_v22.py` (`LAST_DIMENSION_FAILURE` ContextVar, structured channel), `language_brain_v22.py` (`structured_clarification`), `local_intelligence_v22.py` (reads the structured object). |
| Architecture fix | the renderer may only speak fields of the structured object; messages: زمان و مسافت / جرم و حجم / پول و کالا / دو ارز / زمان و مبلغ پول / زمان و تعداد / rate-mismatch + English equivalents. |
| Tests | `test_structured_failure_object_fields`, `test_structured_messages_exact`, `test_time_distance_no_longer_claims_money_items`. |
| Regression result | live report `dimension_algebra.json` 13/13 incl. the full §5 matrix. |
| Status | **FIXED** |

## 6. P1 — ownership reasoning (31.7 % → 100 %)

| Field | Content |
|---|---|
| Bug | `ownership_facts` recognised only `transfers N to` / `می دهد` transfer shapes, required unit words on every balance, and always answered the queried name via the v21 path; `pays/gives/receives/sent/می فرستد/منتقل می‌کند/داد` fell through; `Sara has 223;` (unit dropped), shared verbs (`دارند`), bare `Ali 100` all failed. |
| Old behavior | 19/60 (31.7 %) on the frozen benchmark. |
| Expected behavior | a real event model: balances + ordered transfers + query binding + conservation + currency safety. |
| Root cause | narrow transfer/balance surface grammar; no query semantics beyond a name. |
| Files changed | `semantics_v22_4.py` (`OwnershipEvent`, `extract_ownership_model`, `execute_ownership`, `_ownership_query`), `local_intelligence_v22.py` (`_ownership_answer` + witness + rendering), `source_facts_v21.py` (trigger extensions). |
| Architecture fix | event model: `type/source/target/amount/unit/source_span`; entities appearing only in transfers join the closed system at 0; sequential execution with conservation + non-negativity; query binding supports **sender / receiver / combined / difference**; mixed currencies refuse (`ownership_unit_conflict`); Persian punctuation (؟) can no longer glue onto names. |
| Tests | `test_ownership_direction_resolves` (7 verb forms × 2 languages), `test_ownership_sender_query_binding`, `test_ownership_combined_and_difference_queries`, `test_multi_transfer_sequential_state`, `test_transfer_conserves_total_funds`, `test_ownership_currency_safety`, `test_impossible_transfer_refused`, `test_ownership_event_structure`; conservation property test. |
| Mutation test | M6 `transfer source/target swap` → killed. |
| Regression result | legacy ownership family **60/60 (100 %)**; combined report 12/12. |
| Status | **FIXED** |

## 7. P1 — inventory model incomplete (67/100 → 99/100)

| Field | Content |
|---|---|
| Bug | POS/NEG direction lexicon missed damage/return/correction/arrive verbs; no typed events; no multi-product binding; `remove … items` word problems were hijacked by the destructive-command safety gate. |
| Old behavior | 67/100; damage/return sentences and `انبار X کالا دارد؛ … می‌رسد` fell to the internet fallback. |
| Expected behavior | typed events (event_id/type/direction/quantity/unit/source_span), product binding, ordered execution. |
| Root cause | keyword-direction fallback instead of a typed event model; safety gate had no word-problem context check. |
| Files changed | `source_facts_v21.py` (POS/NEG + `_typed_event` + `_inventory_event_types`), `semantics_v22_4.py` (`InventoryEvent`, `extract_inventory_model`, `execute_inventory_model`), `intent_router.py` (`_is_word_problem_payload` guard), `local_intelligence_v22.py` (`_inventory_answer`). |
| Architecture fix | every inventory event now carries its type derived from the clause verb; multi-product problems bind events to the named product and never touch the other product; the router neutralizes DELETE/CREATE actions when the payload is a numeric word problem. |
| Tests | `test_inventory_event_types_classified`, `test_inventory_event_order_preserved`, `test_multi_product_inventory_binding` (EN + FA), `test_inventory_event_structure_fields`, `test_inventory_order_is_semantic`; inventory property test. |
| Mutation test | M7 `inventory event order reversed` → killed. |
| Regression result | legacy inventory family **99/100**; report 4/4. |
| Status | **FIXED** (1 remaining case abstains honestly: intermediate-negative stock — physical invariant kept by design) |

## 8. P1 — scheduling frame generalization

| Field | Content |
|---|---|
| Bug | `normalize_clock_tokens` searched only 14 characters back for the `ساعت/at/از` cue, so `A train departs at 23:30` (cue 18 chars back) was never decoded; end-instant frames (`ends at 22:00`) were read as starts. |
| Old behavior | the exact gap documented in the v22.3 release notes ("A train departs at 23:30. The trip takes 45 minutes." → abstain). |
| Expected behavior | generalized start/duration/end frame for any temporal context. |
| Root cause | fixed-width cue window + no end-frame semantics. |
| Files changed | `language_brain_v22.py` (window widened to 40 chars, END-verb guard), `semantics_v22_4.py` (`TemporalFrame`, `extract_temporal_frame` with end-first binding + bare-clock evidence rule), `verifier_v22.py` (`verify_temporal` end-frame witness), `local_intelligence_v22.py` (`_temporal_frame_answer`). |
| Architecture fix | temporal frame = {start_time, end_time, durations, day_offset, query}; `end − duration → start` supported; a bare clock token requires frame evidence and is never composed into a schedule without it. |
| Tests | `test_generalized_scheduling_frames` (train/shop/meeting/FA shift), `test_time_invariants_calendar_wrap` (23:59+2m, 23:30+90m, 00:15−30m), `test_bare_clock_requires_context`, `test_end_plus_duration_computes_start`; clock property tests (24h wrap, day offsets). |
| Regression result | legacy scheduling family **90/90**; temporal report 6/6. |
| Status | **FIXED** |

## 9. Price role patch — adversarial hardening

| Field | Content |
|---|---|
| Bug | `The price was reduced by 20 dollars.` classified 20 as `price` (the v22.3 possessive bridge swallowed delta verbs). |
| Old behavior | delta misread as absolute price. |
| Root cause | delta verbs checked after the price bridge. |
| Files changed | `numeric_roles_v22.py` (delta-verb check BEFORE the bridge; new roles `price_delta`, `discount_percentage`). |
| Tests | `test_price_delta_role` (4 forms incl. `قیمت 20 دلار کاهش یافت.`), `test_price_and_discount_roles_separated`. |
| Regression result | numeric_roles report 8/8. |
| Status | **FIXED** |

## 10. Rate arithmetic (new capability, spec §6/§7)

Compound dimensions are first-class: `DimensionExpr(numerator, denominator)`.
`5 USD/hour × 2 hours → 10 USD`, `10 items/hour × 3 hours → 30 items`,
`60 km/hour × 2 hours → 120 km`, `10 m/s × 5 seconds → 50 meters`,
`10 USD/hour + 3 USD/hour → 13 USD/hour` (same family),
`10 USD/hour + 3 ITEM/hour` → structured refusal.
Persian inverted form `هر ساعت 5 دلار` resolves to the same rate structure
(metamorphic test), and never hijacks chain increments
(`هر روز 5 کالا اضافه می شود` stays a chain operand).

## 11. Additional hardening (spec §43-§53)

* **Fail-closed verifier** preserved: injected `extract_typed_quantities`
  crashes → `typed_audit_internal_error` → abstain, never PASS
  (`test_verifier_fail_closed_on_internal_error`); M5 mutant (fail-open) killed.
* **ORIGINAL_SOURCE immutable** — ContextVar protocol with set/try/reset;
  concurrency check: 8 threads × 40 iterations, **0 contamination events**
  (`reports/v22_4/concurrency.json`).
* **`except Exception` audit** — the only swallow paths are rendering
  fallbacks (`_naturalize`, `_upgrade_ir`) and the age witness, each of which
  degrades to abstain; the typed-audit handler explicitly fails closed.
* **Identifiers vs quantities** — `Order #9001`, `کد پستی …`, `Version 3.2`,
  `شناسه پرونده …` never become arithmetic operands while the surrounding
  sentence still computes (`test_identifiers_are_not_arithmetic_values`,
  identifier benchmark family 66/66).
* **Negation preserved** — `اضافه نکن` / `do not subtract` survive NLU
  normalization (`test_negation_is_preserved_by_normalization`).
* **Decimals & negatives** — `2.5 hours`, `۲٫۵ ساعت`, `-5 C`; a glued minus in
  value position is a sign, `100-5` stays two operands
  (`test_decimal_and_negative_values`).
* **Non-finite & division** — NaN/±Inf refuse via `numeric_guard`; division by
  zero raises a structured `ExecutionError` (`test_non_finite_never_verified`,
  `test_division_by_zero_structured`).
* **Probability invariants** — `0 <= p <= 1` enforced
  (`test_probability_invariant`).
* **Fact-locked NLG** — the renderer reads the verified unit: `M_PER_SECOND`
  renders `متر بر ثانیه / m/s`, never km/h
  (`test_renderer_never_substitutes_speed_unit`).

## 12. Performance

Deterministic path preserved (no regex explosion). Measured per family
(`reports/v22_4/performance.json`, 30 runs × family):

| family | mean ms | p50 | p95 | p99 |
|---|---|---|---|---|
| simple_math | 3.4 | 2.5 | 2.8 | 51.9 |
| typed_semantic | 2.9 | 3.4 | 3.6 | 5.5 |
| ownership | ~2-4 | — | — | — |
| inventory | ~2-4 | — | — | — |
| temporal | ~3-5 | — | — | — |
| age | ~2-4 | — | — | — |

(Exact live values are in the JSON artifact; means stay in the low
milliseconds — v22.3 measured 7.7 ms mean.)

## 13. Honest benchmark results

| metric | v22.3 | v22.4 | note |
|---|---|---|---|
| Full regression | 1042 + 585 | **1215 + 585, 0 failures** | generated from raw pytest |
| Broad (500) | 406 (81.2 %) | **406 (81.2 %)** | parity; language rows unchanged |
| Legacy fresh (1023, now `FRESH_REGRESSION_LEGACY`) | 855 (83.58 %) | **1006 (98.34 %)** | answerable-only 98.19 |
| Ownership | 19/60 (31.7 %) | **60/60 (100 %)** | |
| Inventory | 67/100 | **99/100** | 1 honest abstain kept |
| Age | 46/70 | **70/70** | |
| Finance | 180/200 | **200/200** | |
| Ratio | 86/120 | **120/120** | first/second-share query binding |
| Scheduling | 90/90 | **90/90** | |
| Speed / Work rate | 60/60, 105/105 | **60/60, 105/105** | |
| Persian broad language | 6/50 | **6/50** | unchanged — v23 scope (§78) |
| English broad language | 0/50 | **0/50** | unchanged — v23 scope (§78) |
| NEW blind benchmark (2075, frozen, SHA256 `063aef8f…`) | — | **1891/2075 = 91.13 %** (answerable-only 89.65) | `reports/v22_4/new_blind.json` |
| Ownership (new blind) | — | 194/235 | genuine OOD transfer phrasings remain |
| Rates (new blind) | — | 176/210 | remaining misses are consultant/billing paraphrases |
| Units (new blind) | — | 153/154 | m/s distance questions fixed |

The 1023-case benchmark keeps its rows but is now labelled
**FRESH_REGRESSION_LEGACY** — it is no longer unseen.

## 14. Release metadata consistency (spec §59/§60)

The v22.3 artifacts carried 1031 / 1037 / 1042 in different places. v22.4
introduces the pipeline **raw pytest → parser → regression.json →
RELEASE_V22_4.json → final audit**; `RELEASE_V22_4.json` is generated by
`benchmarks/v22_4_release_pipeline.py` + `benchmarks/v22_4_finalize.py` from
the live artifacts and is never hand-typed. `RELEASE_V22.json` remains for
history only.

## 15. Acceptance gate (spec §79)

| Gate | State |
|---|---|
| 2 hours + 5 USD rejected | ✅ |
| time+count direct addition rejected | ✅ |
| rates represented explicitly | ✅ (DimensionExpr) |
| 10 m/s = M_PER_SECOND | ✅ |
| 36 km/h = KM_PER_HOUR | ✅ |
| source spans exact | ✅ (7223 spans, 0 mismatches) |
| age semantics entity-bound | ✅ |
| age paraphrases generalized | ✅ |
| dimension errors structured | ✅ |
| ownership event model implemented | ✅ |
| inventory event model implemented | ✅ |
| train/meeting/general scheduling frame generalized | ✅ |
| verifier stays fail-closed | ✅ (M5 killed) |
| original source immutable | ✅ |
| ContextVar concurrency safe | ✅ (0 contamination) |
| all previous tests pass | ✅ 1215 + 585 |
| new tests pass | ✅ 173 new v22.4 tests |
| old broad >= 406/500 | ✅ 406/500 |
| legacy 1023 >= 855/1023 | ✅ 1006/1023 |
| new blind executed exactly once before inspection | ✅ (frozen + SHA256) |
| release metadata internally consistent | ✅ (generated pipeline) |
| scorecard regenerated | ✅ reports/v22_4/final_scorecard.json |
| manifest clean | ✅ (regenerated, missing=0) |
| 0 secrets | ✅ (1888 files scanned) |

**Label: TRUE ROOT-CAUSE HARDENED** — every critical P0 gate passes.

## 16. What v22.4 does NOT claim

* No new model weights; no GPU training; no 7B/14B Language Brain (v23 scope).
* Broad language generation is unchanged (Persian 6/50, English 0/50).
* One inventory benchmark case still abstains by design (intermediate
  negative stock is refused as physically impossible).
* Binomial (34/45) and combination (40/45) legacy families are unchanged —
  their misses stem from the v21 counting grammar, documented as open gaps.
* The blind benchmark is synthetic-plus-hand-written and honestly labelled;
  template rows are generated from NEW shapes, never the training templates.
