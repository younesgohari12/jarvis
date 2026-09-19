"""JARVIS v22.4.1 — honest benchmark authoring reclassification (spec §19-§24).

The v22.4 manifest claimed `hand_written = 635`, but inspection of
benchmarks/build_v22_4_blind.py shows that rows inside hand_written_batch2()
and hand_written_batch3() are produced by rng.randint/rng.sample loops.
This script creates a deterministic, verifiable reclassification:

  literal_hand_written        — explicit individual rows (explicit `cases`
                                lists; every question text/value authored)
  authored_template_generated — human-designed templates instantiated by
                                loops with randomized values/entities
  synthetic_template_generated — generalized programmatic template families

The frozen benchmark bytes are NEVER touched: the builder is re-executed in
instrumented form, the regenerated rows are compared row-by-row against the
frozen JSONL (faithfulness proof), and the labels are stored in a SIDECAR
file (spec §21 preference), preserving original_benchmark_sha256.

Output:
  benchmarks/v22_4_blind_authoring_metadata_v2.json   (sidecar, row_id → class)
  reports/v22_4_1/benchmark_authoring_metadata.json   (summary report)
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
import tempfile

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE)
sys.path.insert(0, os.path.join(BASE, 'benchmarks'))

import build_v22_4_blind as builder  # noqa: E402

OUT_DIR = os.path.join(BASE, 'reports', 'v22_4_1')
os.makedirs(OUT_DIR, exist_ok=True)
SIDECAR = os.path.join(BASE, 'benchmarks', 'v22_4_blind_authoring_metadata_v2.json')
FROZEN = os.path.join(BASE, 'benchmarks', 'v22_4_blind_1600.jsonl')
FROZEN_SHA = '063aef8f732c845e77a34c6916749d732e96489bbb74edfe12a3a8afc13f7be1'

LITERAL_CREATORS = {
    'hand_written_rates', 'hand_written_ownership', 'hand_written_inventory',
    'hand_written_scheduling', 'hand_written_units', 'hand_written_age',
    'hand_written_hard_negatives', 'hand_written_identifiers', 'hand_written_ood',
}
AUTHORED_TEMPLATE_CREATORS = {'hand_written_batch2', 'hand_written_batch3'}
# everything else (generated_*) -> synthetic_template_generated

CLASS_DEFS = {
    'literal_hand_written':
        'Question text and values are explicitly written as individual '
        'benchmark rows in the source; no programmatic generation loop.',
    'authored_template_generated':
        'A human manually designed the semantic/surface template, but '
        'multiple instances were produced by loops with randomized '
        'values/entities (rng.randint / rng.sample / rng.choice).',
    'synthetic_template_generated':
        'Programmatically generated benchmark family based primarily on '
        'generalized templates.',
}


def main() -> int:
    frozen_bytes = open(FROZEN, 'rb').read()
    frozen_sha = hashlib.sha256(frozen_bytes).hexdigest()
    assert frozen_sha == FROZEN_SHA, \
        f'frozen benchmark SHA mismatch: {frozen_sha}'
    frozen_rows = [json.loads(line) for line in frozen_bytes.decode('utf-8').splitlines()]

    # ---------------------------------------------------------------
    # instrumented re-execution: record the creator of every row
    # ---------------------------------------------------------------
    creator_of: dict[str, str] = {}   # row_id -> creator function name
    authoring_flag: dict[str, str] = {}
    original_row = builder.row
    original_out = builder.OUT
    original_manifest = os.path.join(BASE, 'benchmarks', 'v22_4_blind_manifest.json')

    def traced_row(family, kind, language, question, expected,
                   authoring='hand_written', forbidden=None, **extra):
        import inspect
        creator = inspect.currentframe().f_back.f_code.co_name
        n_before = len(builder.rows)
        original_row(family, kind, language, question, expected,
                     authoring=authoring, forbidden=forbidden, **extra)
        if len(builder.rows) > n_before:          # row() appends in place
            r = builder.rows[-1]
            row_id = r['id']
            creator_of[row_id] = creator
            authoring_flag[row_id] = authoring

    # snapshot files the builder would overwrite; redirect the JSONL output
    with tempfile.TemporaryDirectory() as tmp:
        builder.OUT = os.path.join(tmp, 'regenerated.jsonl')
        builder.row = traced_row
        manifest_backup = open(original_manifest, 'rb').read()
        try:
            builder.build()
        finally:
            builder.row = original_row
            builder.OUT = original_out
            # restore the frozen manifest byte-for-byte no matter what
            open(original_manifest, 'wb').write(manifest_backup)

    regen_rows = [json.loads(line) for line in
                  open(builder.OUT, encoding='utf-8').read().splitlines()]

    # ---------------------------------------------------------------
    # faithfulness proof: regenerated rows == frozen rows, row by row
    # ---------------------------------------------------------------
    fields = ('id', 'family', 'kind', 'language', 'authoring', 'question',
              'expected', 'forbidden')
    mismatches = []
    for i, (a, b) in enumerate(zip(regen_rows, frozen_rows)):
        for f in fields:
            if a.get(f) != b.get(f):
                mismatches.append({'index': i, 'field': f,
                                   'regenerated': a.get(f), 'frozen': b.get(f)})
    order_match = (len(regen_rows) == len(frozen_rows) and not mismatches)

    # ---------------------------------------------------------------
    # classify every row
    # ---------------------------------------------------------------
    def classify(creator: str, legacy_flag: str) -> str:
        if creator in LITERAL_CREATORS:
            return 'literal_hand_written'
        if creator in AUTHORED_TEMPLATE_CREATORS:
            return 'authored_template_generated'
        if creator.startswith('generated_'):
            return 'synthetic_template_generated'
        # fallback: trust the in-row legacy flag conservatively
        return ('literal_hand_written' if legacy_flag == 'hand_written'
                else 'synthetic_template_generated')

    row_classes = {r['id']: classify(creator_of.get(r['id'], '?'), r['authoring'])
                   for r in frozen_rows}
    counts = {c: sum(1 for v in row_classes.values() if v == c) for c in CLASS_DEFS}
    total = len(frozen_rows)

    # per-family breakdown for the honest report
    per_family = {}
    for r in frozen_rows:
        fam = r['family']
        cls = row_classes[r['id']]
        slot = per_family.setdefault(fam, {'literal_hand_written': 0,
                                           'authored_template_generated': 0,
                                           'synthetic_template_generated': 0})
        slot[cls] += 1

    # ---------------------------------------------------------------
    # sidecar (spec §21: benchmark bytes untouched)
    # ---------------------------------------------------------------
    sidecar = {
        'authoring_metadata_version': 2,
        'benchmark': 'v22_4_blind',
        'original_benchmark_sha256': frozen_sha,
        'benchmark_bytes_modified': False,
        'sidecar_policy': ('authoring labels live in this sidecar so the '
                           'frozen v22.4 blind benchmark bytes (and their '
                           'SHA256) remain identical'),
        'class_definitions': CLASS_DEFS,
        'creator_function_map': {
            'literal_hand_written': sorted(LITERAL_CREATORS),
            'authored_template_generated': sorted(AUTHORED_TEMPLATE_CREATORS),
            'synthetic_template_generated': 'generated_* functions in build_v22_4_blind.py',
        },
        'faithfulness_verification': {
            'method': 'instrumented deterministic re-execution of '
                      'build_v22_4_blind.build() (seeded rng), row-by-row '
                      'comparison with the frozen JSONL',
            'rows_compared': len(frozen_rows),
            'order_and_fields_match': order_match,
            'field_mismatches': mismatches[:20],
            'faithful': order_match,
        },
        'counts': counts,
        'total': total,
        'sum_check': {'literal_plus_authored_plus_synthetic': sum(counts.values()),
                      'equals_total': sum(counts.values()) == total},
        'per_family': per_family,
        'row_classes': row_classes,
    }
    with open(SIDECAR, 'w', encoding='utf-8') as f:
        json.dump(sidecar, f, ensure_ascii=False, indent=2)

    # ---------------------------------------------------------------
    # post-write immutability re-check (spec §54)
    # ---------------------------------------------------------------
    sha_after = hashlib.sha256(open(FROZEN, 'rb').read()).hexdigest()
    immutable = sha_after == FROZEN_SHA

    report = {
        'release': 'v0.11.0-intelligence-v22.4.1',
        'issue': 'D — benchmark authoring label honesty (spec §19-§24)',
        'old_label': {'hand_written': 635, 'template_generated': 1440,
                      'problem': 'the 635 included ~570 rows produced by '
                                 'rng loops inside hand_written_batch2/3'},
        'corrected_counts': counts,
        'total': total,
        'x_plus_y_plus_z_equals_total': sum(counts.values()) == total,
        'original_benchmark_sha256': frozen_sha,
        'benchmark_sha_verified_again': sha_after == FROZEN_SHA,
        'benchmark_bytes_unchanged': immutable,
        'authoring_metadata_version': 2,
        'sidecar_file': 'benchmarks/v22_4_blind_authoring_metadata_v2.json',
        'faithful_reconstruction': order_match,
        'per_family': per_family,
        'claim_language': (f'{total}-case frozen benchmark with '
                           f'{counts["literal_hand_written"]} literal '
                           f'individually-authored cases, '
                           f'{counts["authored_template_generated"]} '
                           f'human-designed template instances, and '
                           f'{counts["synthetic_template_generated"]} '
                           f'synthetic template-generated cases'),
    }
    with open(os.path.join(OUT_DIR, 'benchmark_authoring_metadata.json'),
              'w', encoding='utf-8') as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print(json.dumps({k: report[k] for k in
                      ('corrected_counts', 'total', 'x_plus_y_plus_z_equals_total',
                       'benchmark_bytes_unchanged', 'faithful_reconstruction',
                       'claim_language')}, ensure_ascii=False, indent=1))
    return 0 if (order_match and immutable and
                 sum(counts.values()) == total) else 1


if __name__ == '__main__':
    sys.exit(main())
