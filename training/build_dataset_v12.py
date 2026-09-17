from __future__ import annotations

import hashlib
import json
import random
from collections import Counter
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
VERSION = "dataset_v004"
SEED = 12092026


def stable(value: str, length: int = 12) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()[:length]


def split_for_group(group: str) -> str:
    bucket = int(stable(group, 8), 16) % 100
    return "train" if bucket < 80 else "validation" if bucket < 90 else "test"


def canonical(row: dict[str, Any]) -> str:
    context = row.get("context", [])
    return json.dumps(
        {
            "input": str(row.get("input", "")).strip().casefold(),
            "output": str(row.get("output", "")).strip().casefold(),
            "context": context if isinstance(context, list) else [],
        },
        ensure_ascii=False,
        sort_keys=True,
    )


def load_base() -> list[dict[str, Any]]:
    path = ROOT / "datasets" / "raw" / "seed_v003.jsonl"
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        copied = dict(row)
        copied["dataset_version"] = VERSION
        copied["inherited_from"] = "dataset_v003"
        metadata = dict(copied.get("metadata", {}))
        metadata.setdefault("concept_group", f"v3:{stable(canonical(copied))}")
        copied["metadata"] = metadata
        # Preserve the reviewed v003 split to keep historical holdouts stable.
        copied["split"] = str(row.get("split") or split_for_group(str(metadata["concept_group"])))
        rows.append(copied)
    return rows


def load_curated() -> tuple[list[dict[str, Any]], dict[str, Any]]:
    manifest_path = ROOT / "datasets" / "curated_100_v12_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if int(manifest.get("source_dataset_count", 0)) != 100:
        raise RuntimeError("Curated v12 pack must contain exactly 100 source datasets")
    rows: list[dict[str, Any]] = []
    for source in manifest.get("sources", []):
        path = ROOT / str(source["file"])
        expected = str(source["sha256"])
        observed = hashlib.sha256(path.read_bytes()).hexdigest()
        if observed != expected:
            raise RuntimeError(f"Curated dataset checksum mismatch: {path.name}")
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            metadata = dict(row.get("metadata", {}))
            concept = str(metadata.get("concept_group", "")).strip()
            if not concept:
                raise ValueError(f"Missing concept group in {path.name}")
            row["dataset_version"] = VERSION
            row["split"] = split_for_group(concept)
            row["curated_source_id"] = str(source["id"])
            row["pretrained_source"] = None
            metadata["source_dataset"] = str(source["id"])
            metadata["source_file"] = path.relative_to(ROOT).as_posix()
            row["metadata"] = metadata
            rows.append(row)
    return rows, manifest


STAGE_NAMES = {
    1: "language_foundations", 2: "conversation", 3: "general_knowledge",
    4: "instruction_following", 5: "action_recognition", 6: "entity_extraction",
    7: "tool_calling", 8: "argument_extraction", 9: "context_memory",
    10: "planning", 11: "reasoning", 12: "recovery_verification",
    13: "search_decision", 14: "jarvis_personality",
}


def _lab_target(row: dict[str, Any]) -> str:
    output = str(row.get("output", "")).strip()
    if output:
        return output
    expected = row.get("expected")
    if expected not in (None, ""):
        return expected if isinstance(expected, str) else json.dumps(expected, ensure_ascii=False, sort_keys=True)
    action = row.get("action") or row.get("tool")
    if action:
        payload = {"action": action}
        for key in ("arguments", "args", "parameters"):
            if row.get(key) not in (None, ""):
                payload["arguments"] = row[key]
                break
        return json.dumps(payload, ensure_ascii=False, sort_keys=True)
    return ""


def load_lab_imports() -> tuple[list[dict[str, Any]], dict[str, Any]]:
    imported_root = ROOT / "datasets" / "imported"
    state_path = ROOT / "datasets" / "catalog_v003.json"
    enabled: dict[str, bool] = {}
    try:
        payload = json.loads(state_path.read_text(encoding="utf-8"))
        enabled = {str(k): bool(v) for k, v in payload.get("enabled", {}).items()}
    except (OSError, ValueError, TypeError, json.JSONDecodeError):
        pass
    rows: list[dict[str, Any]] = []
    files = skipped = 0
    if not imported_root.is_dir():
        return rows, {"enabled_files": 0, "examples": 0, "skipped": 0}
    for path in sorted(imported_root.rglob("*.jsonl")):
        relative = path.relative_to(ROOT).as_posix()
        if not enabled.get(relative, True):
            continue
        files += 1
        for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            if not line.strip():
                continue
            try:
                source = json.loads(line)
                if not isinstance(source, dict):
                    raise ValueError("row is not an object")
                input_text = str(source.get("input", "")).strip()
                target = _lab_target(source)
                if not input_text or not target:
                    raise ValueError("missing input/target")
            except (ValueError, TypeError, json.JSONDecodeError):
                skipped += 1
                continue
            stage = int(source.get("stage", 4) or 4)
            if stage not in STAGE_NAMES:
                stage = 4
            language = str(source.get("language", "unknown") or "unknown")
            category = str(source.get("category", "lab_import") or "lab_import")
            metadata = dict(source.get("metadata", {})) if isinstance(source.get("metadata", {}), dict) else {}
            concept = str(metadata.get("concept_group", "")).strip() or f"lab:{stable(relative + ':' + input_text.casefold(), 16)}"
            metadata.update({"concept_group": concept, "source_file": relative, "lab_import": True})
            row_id = str(source.get("id", "")).strip() or f"lab-{stable(relative + ':' + str(line_number), 16)}"
            row = {
                "id": row_id, "stage": stage, "stage_name": STAGE_NAMES[stage],
                "category": category, "language": language, "input": input_text,
                "output": target, "context": source.get("context", []) if isinstance(source.get("context", []), list) else [],
                "metadata": metadata, "origin": "lab-import-v12", "dataset_version": VERSION,
                "split": split_for_group(concept), "curated_source_id": "", "pretrained_source": None,
            }
            rows.append(row)
    return rows, {"enabled_files": files, "examples": len(rows), "skipped": skipped}


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def build() -> dict[str, Any]:
    random.seed(SEED)
    base = load_base()
    curated, curated_manifest = load_curated()
    lab_imports, lab_stats = load_lab_imports()
    seen: set[str] = set()
    combined: list[dict[str, Any]] = []
    duplicates = 0
    for row in [*base, *curated, *lab_imports]:
        digest = hashlib.sha256(canonical(row).encode("utf-8")).hexdigest()
        if digest in seen:
            duplicates += 1
            continue
        seen.add(digest)
        combined.append(row)
    combined.sort(key=lambda row: (str(row["split"]), str(row["id"])))
    splits = {name: [row for row in combined if row["split"] == name] for name in ("train", "validation", "test")}
    groups = {
        name: {str(row.get("metadata", {}).get("concept_group", "")) for row in values}
        for name, values in splits.items()
    }
    leakage = sum(
        len(groups[a] & groups[b])
        for a, b in (("train", "validation"), ("train", "test"), ("validation", "test"))
    )
    if leakage:
        raise RuntimeError(f"Concept leakage detected: {leakage}")

    raw_path = ROOT / "datasets" / "raw" / "seed_v004.jsonl"
    write_jsonl(raw_path, combined)
    write_jsonl(ROOT / "datasets" / "cleaned" / "dataset_v004.jsonl", combined)
    write_jsonl(ROOT / "datasets" / "normalized" / "dataset_v004.jsonl", combined)
    for name, values in splits.items():
        write_jsonl(ROOT / "datasets" / "splits" / f"dataset_v004_{name}.jsonl", values)

    languages = Counter(str(row.get("language", "unknown")) for row in combined)
    categories = Counter(str(row.get("category", "uncategorized")) for row in combined)
    origins = Counter(str(row.get("origin", "unknown")) for row in combined)
    stages = Counter(int(row.get("stage", 0)) for row in combined)
    curated_kept = sum(1 for row in combined if row.get("origin") == "jarvis-curated-v12")
    manifest: dict[str, Any] = {
        "format": "jarvis-dataset-manifest-v4",
        "dataset_version": VERSION,
        "seed": SEED,
        "total_examples": len(combined),
        "inherited_v003_examples": sum(1 for row in combined if row.get("inherited_from") == "dataset_v003"),
        "curated_source_dataset_count": int(curated_manifest["source_dataset_count"]),
        "curated_examples_kept": curated_kept,
        "lab_imports": lab_stats,
        "duplicates_removed_during_merge": duplicates,
        "split_counts": {name: len(values) for name, values in splits.items()},
        "concept_group_counts": {name: len(values) for name, values in groups.items()},
        "cross_split_concept_leakage": leakage,
        "languages": dict(sorted(languages.items())),
        "origins": dict(sorted(origins.items())),
        "stages": {str(k): v for k, v in sorted(stages.items())},
        "categories": dict(sorted(categories.items())),
        "provenance": {
            "base": "dataset_v003 project-authored",
            "curated": "100 project-authored deterministic/editorial datasets",
            "lab_imports": "enabled datasets/imported/*.jsonl are merged into v004 training data",
            "external_model_outputs": False,
            "pretrained_source": None,
        },
        "quality_gates": {
            "exact_deduplication": True,
            "concept_group_split": True,
            "cross_split_leakage_required": 0,
            "curated_source_checksums_verified": True,
        },
        "files": {},
    }
    for path in [
        raw_path,
        ROOT / "datasets" / "splits" / "dataset_v004_train.jsonl",
        ROOT / "datasets" / "splits" / "dataset_v004_validation.jsonl",
        ROOT / "datasets" / "splits" / "dataset_v004_test.jsonl",
        ROOT / "datasets" / "curated_100_v12_manifest.json",
    ]:
        manifest["files"][path.relative_to(ROOT).as_posix()] = {
            "bytes": path.stat().st_size,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }
    out = ROOT / "datasets" / "manifest_v004.json"
    out.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return manifest


if __name__ == "__main__":
    print(json.dumps(build(), ensure_ascii=False, indent=2))
