# JARVIS v20 — Tested cognitive upgrade

Based on v19 REAL_TRAINED_FINAL_READY. Existing source, checkpoints, desktop UI, memory and legacy solvers are retained.

- Added five trained NumPy runtime heads for semantic frames, numeric roles, graph operations, code intent and repair stage.
- Activated LocalIntelligenceV20 in both intent routing and reasoning; tagged actual v20 responses in core metadata.
- Added unified IR, task-local world state and bounded execution graphs with multiple operations and conditional branches.
- Added independent arithmetic, conservation, probability, sequence, code and response-constraint verification.
- Added bounded repair that changes failed candidates or slots and rechecks them.
- Added word-number and Unicode normalization, including fractional values and contextual time-unit handling.
- Added sequence, probability and domain solver coverage, plus ten syntax/behavior-checked code recipes.
- Hardened RAG against self-contained tasks, duplicate/unsupported chunks, and mirrored web corroboration.
- Added model-load failure recovery, checkpoint integrity metadata, training/dataset audits, fresh comparison, performance measurements and adversarial tests.

This is a tested incremental release, not a claim that every ambitious target in the mission has been achieved. Detailed phase limits, actual model metrics, benchmark failures and the unavailable official intelligence-index score are recorded in reports/v20.
