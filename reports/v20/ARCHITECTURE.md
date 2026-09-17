# v19 audit and v20 implementation

The uploaded v19 archive was preserved under a separate extraction and tested before modification. Baseline: 696 primary tests and 585 subtests passed in 26.93 seconds. The supplied baseline index remains exactly 80.19; this archive does not contain the executable rubric/weights that originally produced that index or the raw external 48-question benchmark. No new index is fabricated.

## Entry and existing execution path

`jarvis/gui/app.py` starts `jarvis/runtime/bootstrap.py:build_runtime`. Bootstrap creates config, hardware manager, SQLite memory, the hybrid fast brain, lazy NumPy transformer, knowledge store, RAG store, permission-aware tools and `JarvisAgent`. `jarvis/agent/core.py:respond` learns explicit memory facts, resolves conversation state, invokes local reasoning early, then uses `IntentRouter`, the planner/tools, the reasoning engine, knowledge/research or confidence fallback. Memory persistence is distinct from v20 per-task world state.

v19 used `LocalIntelligenceV19 -> SemanticIRParserV19 -> SemanticSolverV19`, reusing v18/v17/v16 solvers. The v19 parser tried family-specific regex functions before its legacy parser; numeric-role predictions could be replaced by regex bindings, and the multistep path enforced two steps. Its frame classifier did not actually determine the main parse dispatch. The full audit JSON includes every .npz array/shape/hash and 693 regex/numeric-index sites across runtime Python code, plus all training script paths. These sites are an inventory, not a claim that every regex is harmful. The model registry preserves the 13 existing NPZ checkpoints and the existing active v18 transformer.

## Active v20 path

`core.respond -> ReasoningEngine.local_intelligence / IntentRouter -> LocalIntelligenceV20 -> SemanticParser -> LearnedModel semantic_frame -> LearnedModel numeric_role -> SemanticIR v2 -> WorldState -> graph -> GraphExecutor -> Verifier -> bounded ReflectionLoop -> response`.

Both `intent_router.py` and `reasoning.py` import the v20 runtime class. Core's response metadata distinguishes a v20 IR result from a legacy result. `tests/test_v20_cognitive.py` proves that changing the trained operation prediction changes the final `agent.respond()` answer from 86 to 74, that world-state mutation changes execution, and that a corrupted ratio candidate is rejected and repaired. These are causal checks, not just file-existence checks.

All five v20 heads load NPZ checkpoints with `allow_pickle=False`, shape checks and checkpoint hashes. Runtime inference only needs NumPy. Training uses scipy/sklearn. These are small linear classifiers over hashed word/character/context features, NOT new transformer reasoning weights or a large language model. They were trained, not merely supplied as empty modules. The transformer/fast brain were not continually trained in this release.

### Unified semantic IR and task state

`cognitive_ir_v20.py` defines one dataclass schema with task, language, entities, slots, units, relations, constraints, operations, answer type, confidence, source and model evidence. WorldState deep-copies that schema; graph compilation consumes world state only. A new state is constructed for each solve and is not stored as chat memory. Mature v19 fallbacks keep their original schema to avoid breaking their tests; v20 does not claim that every legacy solver has been migrated.

The parser chooses a learned frame, tags numeric spans from context, and compiles complete supported frames. It does not assign roles by a global `nums[0]` ordering. Work observations use learned worker/time/output roles and clause distance from the output observation to identify base versus target entities. This relation binder is a deterministic clause heuristic, not a learned entity coreference model. Units and connectors are parsed with bounded grammar. Lexical relationship cues still identify older/younger, at-least/at-most, comparison type and explicit non-uniqueness questions. Low confidence or incomplete parses abstain to legacy solvers; a failed invariant after an accepted v20 IR does not fall back to an unverified answer.

### Execution, verification and reflection

The graph handles add/subtract, multiplication/division, percentages, powers, modulo, rounding, min/max, conditional branches, inventory/finance events and typed domain operations. Domain solvers cover ratio allocation, binomial and independent probability, dice, combinations/permutations, arithmetic/geometric/alternating/additive-recurrence sequences, ages, work scaling, speed and sequential scheduling. This is bounded mathematical interpretation; arbitrary world modelling, concurrent schedules and unrestricted natural-language planning are not claimed.

Verification uses independent Decimal evaluation, conservation equations, binomial dynamic programming, dice inclusion-exclusion, rational counting and separate sequence checks. It verifies unit labels and converted canonical values, but cannot universally detect a plausible wrong semantic parse. A numeric result can be internally consistent with incorrectly inferred text semantics; that remains a measured weakness.

Reflection records original IR, failed invariant, candidate, changed slots/plan and a new verdict. It predicts a repair stage with a trained head, reparses source slots or discards execution state, then executes and verifies again, with at most two retries. It never displays internal diagnostic traces in ordinary user answers. The verified final answer only carries concise self-check metadata.

### Code, Persian, knowledge and web

The new code head extends bounded recipe selection without an action-verb whitelist. Established generation/debugging/tracing/refactoring paths remain available. Ten project-owned pure Python recipes undergo AST syntax validation and fixed behavior tests in an isolated, timed interpreter. Arbitrary user code is never executed by this module. The new selector only supplements legacy answers; broad new code synthesis/debugging is NOT claimed.

Persian normalization handles Unicode digits, Arabic letter variants, diacritics, half-space, compounds and word numbers in a dedicated numeric interpretation path. It is not applied to raw code source. Formal, colloquial and mixed-language examples are included, but no broad Persian conversational transformer improvement is claimed.

The existing RAG already provided BM25, hashed vectors, reranking and confidence. v20 adds explicit self-contained task protection, absolute content-support filtering, exact normalized chunk deduplication and a small documented bilingual retrieval lexicon. The lexicon covers only twelve concepts, not unrestricted cross-language retrieval. Existing research already had query rewriting, source ranking and multi-source synthesis. v20 prevents identical mirrored passages from increasing multi-domain corroboration confidence. Live internet retrieval was not benchmarked; freshness decisions were tested offline.

## Dataset limitations and provenance

Main corpus: 1,643 unique input skeletons, including 1,166 new authored/composed rows and 477 capped legacy-train replays. At most one input is retained per normalized numeric skeleton. Near duplicates at character 3–5 gram TF-IDF cosine >=0.90 are clustered into one split before main-head training. The main dataset has zero exact template overlaps and zero test/train near duplicates at this threshold. Common mathematical concepts necessarily recur across splits: conceptual/semantic disjointness is not claimed.

Auxiliary code, repair and verification corpora have separate audits. Their simpler diagnostic labels and repeated recipe concepts make their internal classifier accuracy less indicative of broad intelligence. Verification-candidate rows are executable invariant examples; the arithmetic verifier itself is deterministic and is not falsely described as a learned truth classifier. The trained repair head consumes failure diagnostics.

The requested 60k–120k count was a quality-dependent suggestion. This release deliberately does not claim that volume, 100 separate datasets, broad generative-language training, or a new core-neural score. See phase_status.json for each requested phase and limitation.
