# v21 changes from v20.1

- Add independent source verification of slots, events, units, candidate values and rendered numeric answers, with bounded grammar coverage and two-attempt repair.
- Wire typed world state and Graph V2 operations into Runtime; add ownership, conservation and dependency checks.
- Add and train numeric binding, code task and rank-16 native Transformer output adapter. Keep failure classifier diagnostic-only and rejected operation classifier inactive.
- Add bounded AST code generation, syntax repair, explanation, refactoring and characterization tests.
- Add attributed episodic project recall and user semantic memory, respecting memory disable and forget.
- Add 20,000 synthetic reasoning and 10,000 synthetic code records, structural splits, full registry and near-duplicate audit. Broad linguistic quality target remains unmet.
- Add 56 tests, preserve all 18 old checkpoints, and retain initial and final benchmark evidence.
- Record incomplete language/general-code goals explicitly. See V21_README_FA.md and reports/v21/release_gate.json for measured results.
