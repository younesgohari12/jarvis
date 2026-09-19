# JARVIS v22.4.1 — RELEASE INTEGRITY AUDIT

**Release:** `v0.11.0-intelligence-v22.4.1`
**Label:** JARVIS v0.11.0 — Intelligence v22.4.1 — FINAL CLEAN HARDENED / RELEASE INTEGRITY LOCK
**Base release:** `v0.11.0-intelligence-v22.4` (commit `dc5c328`)
**Package:** `Jarvis_v0.11.0_Intelligence_v22.4.1_FINAL_CLEAN_HARDENED.zip`
**Nature:** release-integrity patch — **not** a new intelligence upgrade.

---

## 1. Base release

| Item | Value |
|---|---|
| Base tag | `v0.11.0-intelligence-v22.4` |
| Base commit | `dc5c32889296614bdf4b96179f565cdfaa8b70e8` |
| Target tag | `v0.11.0-intelligence-v22.4.1` |
| Frozen blind benchmark SHA-256 | `063aef8f732c845e77a34c6916749d732e96489bbb74edfe12a3a8afc13f7be1` |

v22.4 shipped the root-cause semantic hardening (operation-aware dimension
algebra, exact SourceSpan, entity-bound age semantics, ownership/inventory
event models, structured dimension failures, generalized temporal frames,
fail-closed verifier). This release changes **none** of it.

## 2. Scope

Four independent audit defects were fixed; nothing else:

| Issue | Defect | Fix |
|---|---|---|
| A | `.hypothesis` cache (~175 files) packaged inside the v22.4 ZIP | explicit release-exclusion policy + forbidden-path verification of the actual archive |
| B | `package_integrity.json` stale (1746 entries vs 1959 actual) — generated before final packaging | generated LAST, directly from the final ZIP via `ZipFile()`; every value measured |
| C | Mutation M5 false kill — `killed = returncode != 0` counted "no tests ran" (pytest usage error 4) as a kill | valid-kill semantics with parsed collected/failed/error counts; M5 test selection fixed; framework self-test; manual M5 reproduction |
| D | "635 hand-written" claim misleading — 570 of those rows came from `rng` loops | honest 3-class authoring sidecar (`authoring_metadata_version = 2`), frozen bytes untouched |

## 3. Runtime changes

**Runtime behavioral files changed: 0** (measured: `git diff v22.4 -- jarvis/`
plus working tree = empty). Evidence:

- Runtime parity harness (40 deterministic cases across all families):
  **40/40 byte-identical** responses vs a checked-out v22.4 worktree
  (`reports/v22_4_1/runtime_parity.json`).
- Full P0 smoke suite (the v22.4 hardening behaviors): **16/16 PASS**
  (`reports/v22_4_1/p0_smoke.json`).
- Regression: **1228 tests + 585 subtests, 0 failures** (v22.4 baseline was
  1215 + 585, 0 — held and raised by release-pipeline tests).

## 4. Packaging cleanup

The v22.4 packager excluded only `{__pycache__, .pytest_cache, .git, .venv,
node_modules, blind_chunks}`. v22.4.1 replaces it with an explicit
release-exclusion policy (`benchmarks/package_v22_4_1.py`):

- **Directories:** `.git`, `.venv`, `__pycache__`, `.pytest_cache`,
  `.hypothesis`, `.mypy_cache`, `.ruff_cache`, `.tox`, `.cache`, `tox`,
  `htmlcov`, `node_modules`, `blind_chunks`, `dist_v22_4_1`, `.idea`,
  `.vscode`, `runtime_data`
- **Files/suffixes:** `*.pyc`, `*.pyo`, `*.partial.json`, `*.tmp`, `*.bak`,
  `*.log`, `*.zip`, `*.orig`, `*.rej`, `*.db`, `.DS_Store`, `Thumbs.db`,
  `coverage.xml`, `.coverage`, `blind_run.log`, runtime databases
- Nothing legitimate is blindly deleted: the policy only affects what is
  *packaged*; all sources remain in the repository.
- The final archive is verified against a forbidden-path matcher
  (`/.hypothesis/`, `/.pytest_cache/`, `/__pycache__/`, `/.git/`, `/.venv/`,
  `/node_modules/` + suffix rules) — enforced both in the packager and in
  `tests/test_v224_1_release_pipeline.py::test_release_zip_integrity`.

Deterministic packaging (§36): sorted paths, normalized timestamps
(`2026-09-19 12:00:00`), normalized Unix permissions (`0644`), stable JSON
with `sort_keys`; two builds of one tree are verified **byte-identical**
(rebuild check runs inside the packager and is recorded in
`package_integrity.json.reproducibility`).

**External finalization artifacts policy.**
`reports/v22_4_1/package_integrity.json`, `package_inventory.json`,
`metadata_consistency.json` and `security_package.json` are generated **from
the final ZIP** and shipped beside it as release assets. A file inside a ZIP
cannot truthfully contain that ZIP's final SHA-256 (nor a scan of itself);
publishing these reports next to the artifact they authenticate is the only
self-consistent scheme (same precedent as v22.4's external
`Archive_Check.json`). They are excluded from the archive by explicit policy
and verified against it by the release tests.

## 5. Mutation framework correction

Defect: the v22.4 runner used `killed = proc.returncode != 0`, counting
usage errors, test-not-found, collection errors and timeouts as kills. M5's
own artifact betrayed it: `pytest_tail: "no tests ran in 1.17s"` yet
`killed: true` (pytest exit 4 = usage error, because the targeted node
`test_typed_audit_fail_closed` did not exist; the real test is
`test_typed_audit_crash_fails_closed`).

The v22.4.1 framework (`benchmarks/v22_4_1_mutation.py`) implements:

- **Statuses:** `KILLED_BY_ASSERTION`, `SURVIVED`, `INVALID_MUTANT`,
  `TEST_COLLECTION_ERROR`, `TEST_NOT_FOUND`, `INVALID_TEST_SELECTION`,
  `TIMEOUT`, `ENVIRONMENT_ERROR`.
- **Valid kill requires:** mutation applied AND `collected ≥ 1` AND
  `failed ≥ 1` AND `errors == 0` — i.e. a real assertion failure on a real
  collected test.
- **Parsing:** returncode, stdout, stderr, collected, passed, failed, error
  counts (§16).
- **Source safety (§57):** after every mutant the file is restored and
  `SHA256(before) == SHA256(after)` is asserted; the suite reports
  `all_source_restored: true`.

Results (target §18: 7 valid / 7 killed / 0 survived / 0 invalid):

| Mutant | Status | collected | failed | errors | restored |
|---|---|---|---|---|---|
| M1 disable dimension guard | KILLED_BY_ASSERTION | 2 | 1 | 0 | ✓ |
| M2 speed unit flattened | KILLED_BY_ASSERTION | 2 | 1 | 0 | ✓ |
| M3 span end shifted | KILLED_BY_ASSERTION | 2 | 1 | 0 | ✓ |
| M4 age abs removed | KILLED_BY_ASSERTION | 38 | 1 | 0 | ✓ |
| M5 verifier fail-open | KILLED_BY_ASSERTION | 2 | 1 | 0 | ✓ |
| M6 transfer direction swapped | KILLED_BY_ASSERTION | 1 | 1 | 0 | ✓ |
| M7 inventory order reversed | KILLED_BY_ASSERTION | 2 | 1 | 0 | ✓ |

(Exact per-mutant counts are in `reports/v22_4_1/mutation_tests.json`.)

**M5 manual reproduction (§56):** applied the fail-open mutation by hand,
ran `tests/test_v221_audit_fixes.py::test_typed_audit_crash_fails_closed`:
pre-mutation `1 passed`; mutated run `collected 1, failed 1, errors 0`
(assertion `verification.passed is False` /
`'typed_audit_internal_error' in failed_checks` fails when the verifier
falls open); source restored. Evidence:
`reports/v22_4_1/m5_manual_reproduction.json`.

**Framework self-test (§55):** six controlled scenarios classify exactly as
required — assertion failure → KILLED; all pass → SURVIVED; invalid node →
TEST_NOT_FOUND; collection error → TEST_COLLECTION_ERROR; timeout → TIMEOUT;
empty selection → INVALID_TEST_SELECTION. Only the first counts as a kill
(`reports/v22_4_1/mutation_framework_validation.json`).

## 6. Benchmark authoring correction

`build_v22_4_blind.py` was re-executed in instrumented form (seeded rng,
traced `row()` calls recording the creating function for every row). The
regenerated rows were compared field-by-field with the frozen JSONL:
**2075/2075 identical, same order** — the mapping is a faithful description
of the frozen benchmark, not a reinterpretation.

Corrected classification (`benchmarks/v22_4_blind_authoring_metadata_v2.json`,
sidecar — benchmark bytes untouched, SHA re-verified):

| Class | Rows | Definition |
|---|---|---|
| `literal_hand_written` | **65** | explicit individual rows (explicit case lists; every question/value authored) |
| `authored_template_generated` | **570** | human-designed templates instantiated by loops with randomized values/entities (batch2/batch3) |
| `synthetic_template_generated` | **1440** | generalized programmatic template families (`generated_*`) |
| **Total** | **2075** | 65 + 570 + 1440 = 2075 ✓ |

The old claim "635 hand-written" = 65 literal + 570 template instances.
Release language from now on (§22): *2075-case frozen benchmark with 65
literal individually-authored cases, 570 human-designed template instances,
and 1440 synthetic template-generated cases.* No question text, expected
answer, or benchmark byte was modified; the frozen SHA-256 is unchanged.

## 7. Benchmark accounting

Raw chunks `chunk_00..chunk_04` independently aggregated
(`reports/v22_4_1/benchmark_accounting.json`; per-chunk SHA-256 recorded as
provenance — raw chunks are not shipped in the ZIP):

| Metric | Aggregated | Claimed | Match |
|---|---|---|---|
| total | 2075 | 2075 | ✓ |
| correct_total | 1891 | 1891 | ✓ |
| answerable | 1749 | 1749 | ✓ |
| correct_answerable | 1568 | 1568 | ✓ |
| score | 91.13% | 91.13% | ✓ |
| answerable-only | 89.65% | 89.65% | ✓ |

Per-chunk partitions and family totals are also internally consistent
(`sum(chunk.total) == manifest.total` etc. — enforced by
`test_benchmark_accounting_is_consistent`). The frozen score stays attached
to the original SHA-256; the blind benchmark is now **V22_4_FROZEN_EVALUATION**
— no runtime patching against it in this release (§27/§28).

## 8. Regression

Raw pytest run of `tests/` (`reports/v22_4_1/regression.json`, measured —
never hand-typed):

| | v22.4 baseline | v22.4.1 |
|---|---|---|
| tests | 1215 | **1228** |
| subtests | 585 | **585** |
| failures | 0 | **0** |

The increase (+13) is the new release-pipeline test module
(`tests/test_v224_1_release_pipeline.py`: forbidden-path matcher, ZIP
integrity, benchmark accounting, benchmark immutability, authoring sidecar
consistency, mutation classification semantics, metadata namespacing, P0
smoke report). Baseline held; nothing dropped.

## 9. Runtime parity

40 deterministic cases (FA/EN × arithmetic, rates, ownership, inventory,
temporal day-wrap, age relations, finance, ratio, work-rate, hard negatives,
OOD, identifiers, mixed-language) executed in **both** the v22.4 reference
worktree and this tree; normalized responses (answer text + intent):

**diff_count = 0 → parity confirmed.** (`reports/v22_4_1/runtime_parity.json`)

## 10. Package integrity

`reports/v22_4_1/package_integrity.json` is generated **last**, directly from
the final ZIP (`ZipFile(final_zip)` — §44), with the required schema: zip
name, **zip_sha256**, size bytes, package root `Jarvis_v0.11.0/`, archive
file count, manifest entries, missing/mismatch/extra = 0/0/0, CRC OK,
forbidden cache files = 0, rebuild reproducibility, and the two-scan security
summary. The manifest-vs-archive invariant is exact:

```
set(SHA256SUMS.json files) == set(ZIP contents) - {SHA256SUMS.json}
```

Git tree count ≠ package count by design (§11): the package intentionally
excludes dev caches, logs, raw blind chunks, partial reports and the external
finalization artifacts. Order enforced (§9): source → tests → reports →
cache cleanup → file list → SHA256SUMS → ZIP → verify ZIP →
package_integrity → metadata consistency.

## 11. Security

Two scans (§40), same pattern set (GitHub PAT, API keys, AWS keys, private
key blocks, bot tokens, Slack tokens, Google API keys, hardcoded passwords —
reporting only `(file, secret_type)`, never values, §41):

| Scan | Files | Secrets | Result |
|---|---|---|---|
| Source tree (`security_source.json`) | 1818 | 0 | PASS |
| Final package (`security_package.json` + `package_integrity.json.security_scan`) | = archive files | 0 | PASS — authoritative for release |

## 12. Performance

Warm measurement with explicit warmup (§30,
`reports/v22_4_1/performance_warm.json`). Warm ownership:
**mean 5.89 ms, p50 5.89 ms, p95 6.02 ms, p99 6.03 ms**.

**Ownership p99 investigation (§29, read-only):** the v22.4 report
(mean ≈ 7.65 ms, p99 ≈ 59 ms) mixed cold first-call initialization into the
distribution. Cold harness shows import ≈ 300 ms and first-call costs an
order of magnitude above steady state; with warmup, the tail collapses
(p99/p50 ≈ 1.02). GC-on vs GC-off distributions show no GC-dominated tail.
**Conclusion: cold-start noise, no pathological input, no algorithmic issue,
no Runtime change required.** All families: warm p99 ≤ 6.1 ms.

## 13. Honest remaining model gaps

Unchanged and published (§62) — they become **v23** targets, not v22.4.1
patches:

| Area | Result |
|---|---|
| Persian broad (language generation) | 6/50 |
| English broad (language generation) | 0/50 |
| New Blind — Work Rate | 52/90 |
| New Blind — Ownership | 194/235 |
| New Blind — Rates | 176/210 |
| New Blind — Finance | 124/150 |
| New Blind — Age | 139/163 |

Strong families remain published alongside (hard negatives 241/243,
identifiers 82/86, inventory 185/187, mixed-language 120/120, OOD 81/81,
probability 90/90, ratio 100/110, scheduling 154/156, units 153/154).
The blind benchmark is frozen evidence: never to be fed into v23 training
(§63) — only failure-family abstractions are handed forward.

## 14. Final release gate

All §60 acceptance gates are green (machine-checked in
`RELEASE_V22_4_1.json.gates` and `reports/v22_4_1/final_scorecard.json`):

- runtime parity with v22.4 confirmed (40/40) — `runtime_changed = false`
- 1228 tests + 585 subtests, 0 failures (≥ 1215 baseline held)
- all P0 smoke tests pass (16/16)
- `.hypothesis` and all forbidden caches absent from the package (verified
  from the archive itself)
- manifest exactly matches ZIP; 0 missing / 0 extra / 0 mismatch; CRC clean
- `package_integrity.json` generated from the final ZIP (not before, not
  hand-typed)
- mutation framework correctly classifies failures; M5 has a real collected
  failing test; **7 valid / 7 killed / 0 survived / 0 invalid**
- frozen blind SHA unchanged; 1891/2075 and 1568/1749 accounting reproduced
  from raw chunks
- authoring labels corrected; no misleading "635 hand-written" claim
- 0 secrets in source scan and final package scan
- release metadata internally consistent; historical metrics namespaced
- single `Jarvis_v0.11.0/` root in the archive

**Release status: FINAL** — the deterministic v22 core is locked; next
development phase is v23 (GPU training and real Language Brain work), which
must start from `V23_TRAINING_HANDOFF.md` and must not train on the frozen
blind benchmark.
