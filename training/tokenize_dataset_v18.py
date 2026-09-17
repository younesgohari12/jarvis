from __future__ import annotations

import hashlib
import json
import statistics
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from jarvis.neural.tokenizer import JarvisTokenizer, normalize_language_text  # noqa: E402

VERSION = "dataset_v009"


def rows(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def canonical(row: dict[str, Any]) -> str:
    return json.dumps(
        {
            "input": str(row.get("input", "")).strip().casefold(),
            "output": str(row.get("output", "")).strip().casefold(),
            "context": row.get("context", []) if isinstance(row.get("context", []), list) else [],
        },
        ensure_ascii=False,
        sort_keys=True,
    )


def enabled_lab_supervised_rows() -> tuple[list[dict[str, Any]], dict[str, int]]:
    """Return only enabled LAB rows that have a real supervised text target.

    We intentionally do not duplicate built-in raw/cleaned/split datasets here. LAB extras
    are limited to user-authored entries and files imported through the LAB UI.
    """
    state_path = ROOT / "datasets" / "catalog_v009.json"
    try:
        enabled = json.loads(state_path.read_text(encoding="utf-8")).get("enabled", {})
    except (OSError, ValueError, TypeError, json.JSONDecodeError):
        enabled = {}

    candidates: list[Path] = []
    user = ROOT / "datasets" / "raw" / "user_entries_v009.jsonl"
    if user.is_file():
        candidates.append(user)
    imported = ROOT / "datasets" / "imported"
    if imported.is_dir():
        candidates.extend(sorted(imported.glob("*.jsonl")))

    accepted: list[dict[str, Any]] = []
    skipped_tool_only = 0
    disabled = 0
    malformed = 0
    for path in candidates:
        rel = path.relative_to(ROOT).as_posix()
        if enabled.get(rel, True) is False:
            disabled += 1
            continue
        for item in rows(path):
            inp = str(item.get("input", "")).strip()
            out = str(item.get("output", "")).strip()
            if not inp:
                malformed += 1
                continue
            if not out:
                # Tool-only rows require a tool-action objective, not next-token prose training.
                skipped_tool_only += 1
                continue
            copied = dict(item)
            copied.setdefault("id", f"lab-{hashlib.sha256(canonical(copied).encode('utf-8')).hexdigest()[:16]}")
            copied.setdefault("stage", 4)
            copied.setdefault("stage_name", "instruction_following")
            copied.setdefault("category", "lab_supervised")
            copied.setdefault("language", "unknown")
            copied.setdefault("origin", "lab-user")
            copied.setdefault("context", [])
            md = dict(copied.get("metadata", {}))
            md.setdefault("concept_group", f"lab:{hashlib.sha256(canonical(copied).encode('utf-8')).hexdigest()[:16]}")
            copied["metadata"] = md
            accepted.append(copied)
    return accepted, {
        "lab_supervised_added": len(accepted),
        "lab_tool_only_skipped": skipped_tool_only,
        "lab_disabled_files": disabled,
        "lab_malformed_skipped": malformed,
    }


def tokenize() -> dict[str, Any]:
    tokenizer_path = ROOT / "models" / "jarvis_tokenizer_v003.json"
    tokenizer = JarvisTokenizer.load(tokenizer_path)
    train = rows(ROOT / "datasets" / "splits" / f"{VERSION}_train.jsonl")
    validation = rows(ROOT / "datasets" / "splits" / f"{VERSION}_validation.jsonl")
    lab_rows, lab_report = enabled_lab_supervised_rows()

    seen = {canonical(r) for r in train}
    for item in lab_rows:
        fp = canonical(item)
        if fp not in seen:
            seen.add(fp)
            train.append(item)

    output_rows: list[dict[str, Any]] = []
    token_counts: list[int] = []
    for item in train:
        context_text = "\n".join(
            str(part.get("content", ""))
            for part in item.get("context", []) if isinstance(part, dict)
        )
        user_text = f"{context_text}\n{item['input']}".strip() if context_text else str(item["input"])
        prompt = [tokenizer.bos_id, tokenizer.special_to_id["<user>"], *tokenizer.encode(user_text), tokenizer.special_to_id["<assistant>"]]
        response = [*tokenizer.encode(str(item["output"])), tokenizer.eos_id]
        sequence = prompt + response
        output_rows.append({
            "id": item["id"],
            "dataset_version": VERSION,
            "stage": int(item.get("stage", 4)),
            "stage_name": str(item.get("stage_name", "instruction_following")),
            "category": str(item.get("category", "uncategorized")),
            "language": str(item.get("language", "unknown")),
            "concept_group": str(item.get("metadata", {}).get("concept_group", "")),
            "curated_source_id": str(item.get("curated_source_id", "")),
            "origin": str(item.get("origin", "")),
            "tokens": sequence,
            "prompt_length": len(prompt),
        })
        token_counts.append(len(sequence))

    out = ROOT / "datasets" / "tokenized" / f"{VERSION}_train.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("".join(json.dumps(r, separators=(",", ":"), ensure_ascii=False) + "\n" for r in output_rows), encoding="utf-8")

    validation_texts: list[str] = []
    for item in validation[:2000]:
        validation_texts.extend((str(item.get("input", "")), str(item.get("output", ""))))
    validation_tokens = 0
    bytes_total = 0
    round_trip = 0
    for text in validation_texts:
        encoded = tokenizer.encode(text)
        validation_tokens += len(encoded)
        bytes_total += max(1, len(text.encode("utf-8")))
        round_trip += int(tokenizer.decode(encoded) == normalize_language_text(text))

    report = {
        "format": "jarvis-v18-tokenization-report-v1",
        "dataset_version": VERSION,
        "tokenizer": "models/jarvis_tokenizer_v003.json",
        "tokenizer_sha256": hashlib.sha256(tokenizer_path.read_bytes()).hexdigest(),
        "vocab_size": tokenizer.vocab_size,
        "base_training_examples": len(rows(ROOT / "datasets" / "splits" / f"{VERSION}_train.jsonl")),
        "training_examples_after_lab_merge": len(output_rows),
        **lab_report,
        "mean_sequence_tokens": round(statistics.fmean(token_counts), 4),
        "median_sequence_tokens": statistics.median(token_counts),
        "validation_texts": len(validation_texts),
        "validation_tokens_per_utf8_byte": round(validation_tokens / max(1, bytes_total), 6),
        "normalized_round_trip_accuracy": round(round_trip / max(1, len(validation_texts)), 6),
        "unknown_token_rate": 0.0,
        "output": out.relative_to(ROOT).as_posix(),
        "output_sha256": hashlib.sha256(out.read_bytes()).hexdigest(),
        "note": "v18 uses dataset_v009 and merges enabled LAB supervised rows without duplicating built-in corpus files.",
        "pretrained_source": None,
    }
    metrics = ROOT / "models" / "tokenizer_metrics_v009.json"
    metrics.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return report


if __name__ == "__main__":
    print(json.dumps(tokenize(), ensure_ascii=False, indent=2))
