from __future__ import annotations

import hashlib
import json
import re
import shutil
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from jarvis.neural.tokenizer import JarvisTokenizer
from jarvis.neural.transformer import JarvisTransformer


@dataclass(frozen=True, slots=True)
class DatasetRecord:
    path: str
    samples: int
    categories: tuple[str, ...]
    languages: tuple[str, ...]
    duplicates: int
    malformed: int
    enabled: bool


class DatasetCatalog:
    """Streaming JSONL validation and safe lifecycle management for LAB data."""

    _IMPORT_NAME = re.compile(r"[^A-Za-z0-9_.-]+")

    def __init__(self, project_root: Path) -> None:
        self.root = project_root.resolve()
        self.datasets = self.root / "datasets"
        self.imported = self.datasets / "imported"
        self.state_path = self.datasets / "catalog_v009.json"

    def _state(self) -> dict[str, bool]:
        try:
            payload = json.loads(self.state_path.read_text(encoding="utf-8"))
            values = payload.get("enabled", {})
            return {str(key): bool(value) for key, value in values.items()}
        except (OSError, ValueError, TypeError, json.JSONDecodeError):
            return {}

    def _write_state(self, values: dict[str, bool]) -> None:
        payload = {
            "format": "jarvis-dataset-catalog-v1",
            "dataset_version": "dataset_v009",
            "enabled": dict(sorted(values.items())),
            "updated_at_utc": datetime.now(UTC).isoformat(),
        }
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.state_path.with_suffix(".tmp")
        temporary.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        temporary.replace(self.state_path)

    def validate(self, path: Path) -> DatasetRecord:
        resolved = path.resolve()
        if resolved.suffix.casefold() != ".jsonl" or not resolved.is_file():
            raise ValueError("Dataset must be an existing JSONL file")
        try:
            relative = resolved.relative_to(self.root).as_posix()
        except ValueError as exc:
            raise ValueError("Dataset is outside the JARVIS project") from exc
        samples = malformed = duplicates = 0
        categories: set[str] = set()
        languages: set[str] = set()
        fingerprints: set[str] = set()
        with resolved.open("r", encoding="utf-8", errors="strict") as handle:
            for line in handle:
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                    if not isinstance(row, dict):
                        raise ValueError("sample is not an object")
                    input_text = str(row.get("input", "")).strip()
                    output_text = str(row.get("output", "")).strip()
                    if not input_text or not (
                        output_text or row.get("action") or row.get("tool") or row.get("expected")
                    ):
                        raise ValueError("missing input/target")
                except (ValueError, TypeError, json.JSONDecodeError):
                    malformed += 1
                    continue
                samples += 1
                categories.add(str(row.get("category", "uncategorized")))
                languages.add(str(row.get("language", "unknown")))
                canonical = json.dumps(
                    {
                        "input": input_text.casefold(),
                        "output": output_text.casefold(),
                        "action": row.get("action") or row.get("tool"),
                    },
                    ensure_ascii=False,
                    sort_keys=True,
                )
                digest = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
                if digest in fingerprints:
                    duplicates += 1
                fingerprints.add(digest)
        return DatasetRecord(
            relative, samples, tuple(sorted(categories)), tuple(sorted(languages)),
            duplicates, malformed, self._state().get(relative, True),
        )

    def scan(self) -> tuple[DatasetRecord, ...]:
        state = self._state()
        records: list[DatasetRecord] = []
        for path in sorted(self.datasets.rglob("*.jsonl")):
            relative = path.relative_to(self.datasets)
            if relative.parts and relative.parts[0] in {"benchmarks", "tokenized"}:
                continue
            record = self.validate(path)
            records.append(
                DatasetRecord(
                    record.path, record.samples, record.categories, record.languages,
                    record.duplicates, record.malformed, state.get(record.path, True),
                )
            )
        return tuple(records)

    def import_jsonl(self, source: Path) -> DatasetRecord:
        source = source.resolve()
        if source.suffix.casefold() != ".jsonl" or not source.is_file():
            raise ValueError("Only JSONL datasets can be imported")
        samples = malformed = 0
        with source.open("r", encoding="utf-8", errors="strict") as handle:
            for line in handle:
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                    if not isinstance(row, dict):
                        raise ValueError("sample is not an object")
                    input_text = str(row.get("input", "")).strip()
                    has_target = bool(
                        str(row.get("output", "")).strip()
                        or row.get("action") or row.get("tool") or row.get("expected")
                    )
                    if not input_text or not has_target:
                        raise ValueError("missing input/target")
                    samples += 1
                except (ValueError, TypeError, json.JSONDecodeError):
                    malformed += 1
        if samples == 0 or malformed:
            raise ValueError(f"Import rejected: samples={samples}, malformed={malformed}")
        self.imported.mkdir(parents=True, exist_ok=True)
        safe = self._IMPORT_NAME.sub("_", source.stem).strip("._") or "dataset"
        with source.open("rb") as handle:
            digest = hashlib.file_digest(handle, "sha256").hexdigest()[:10]
        destination = self.imported / f"{safe}_{digest}.jsonl"
        shutil.copy2(source, destination)
        return self.validate(destination)

    def set_enabled(self, relative_path: str, enabled: bool) -> None:
        target = (self.root / relative_path).resolve()
        target.relative_to(self.datasets.resolve())
        if not target.is_file():
            raise ValueError("Dataset does not exist")
        values = self._state()
        values[relative_path] = bool(enabled)
        self._write_state(values)

    def delete_imported(self, relative_path: str) -> None:
        target = (self.root / relative_path).resolve()
        try:
            target.relative_to(self.imported.resolve())
        except ValueError as exc:
            raise PermissionError("Only datasets imported through LAB can be deleted") from exc
        if not target.is_file():
            raise ValueError("Dataset does not exist")
        target.unlink()
        values = self._state()
        values.pop(relative_path, None)
        self._write_state(values)


@dataclass(frozen=True, slots=True)
class ModelRecord:
    key: str
    profile: str
    path: str
    parameters: int
    status: str
    activatable: bool


class ModelCatalog:
    """Validated import/activation with deletion constrained to LAB-owned files."""

    def __init__(self, project_root: Path) -> None:
        self.root = project_root.resolve()
        self.models = self.root / "models"
        self.lab_models = self.models / "lab_runs"
        self.active_path = self.models / "active_model_v8.json"

    def scan(self) -> tuple[ModelRecord, ...]:
        records: list[ModelRecord] = []
        registry_path = self.models / "model_registry_v2.json"
        if registry_path.is_file():
            registry = json.loads(registry_path.read_text(encoding="utf-8"))
            for row in registry.get("profiles", []):
                weight = str(row.get("weights", ""))
                path = self.models / weight if weight else None
                records.append(
                    ModelRecord(
                        str(row.get("id", row.get("profile", "model"))),
                        str(row.get("profile", "unknown")),
                        path.relative_to(self.root).as_posix() if path else "",
                        int(row.get("parameters", 0)), str(row.get("status", "unknown")),
                        bool(path and path.is_file()),
                    )
                )
        if self.lab_models.is_dir():
            for path in sorted(self.lab_models.rglob("*.npz")):
                try:
                    metadata = JarvisTransformer.peek_metadata(path)
                except Exception:
                    records.append(ModelRecord(
                        path.stem, "unknown", path.relative_to(self.root).as_posix(), 0,
                        "invalid", False,
                    ))
                    continue
                records.append(ModelRecord(
                    f"lab:{path.stem}", "custom", path.relative_to(self.root).as_posix(),
                    int(metadata.get("parameter_count", 0)), "validated", True,
                ))
        return tuple(records)

    def import_model(self, source: Path, tokenizer_path: Path) -> ModelRecord:
        source = source.resolve()
        if source.suffix.casefold() != ".npz" or not source.is_file():
            raise ValueError("Only JARVIS NPZ models can be imported")
        metadata = JarvisTransformer.peek_metadata(source)
        model = JarvisTransformer.load(source)
        tokenizer = JarvisTokenizer.load(tokenizer_path)
        if model.config.vocab_size != tokenizer.vocab_size:
            raise ValueError("Model vocabulary is incompatible with the active tokenizer")
        digest = JarvisTransformer.file_sha256(source)
        destination = self.lab_models / "imported" / f"{source.stem}_{digest[:10]}.npz"
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
        return ModelRecord(
            f"lab:{destination.stem}", "custom", destination.relative_to(self.root).as_posix(),
            int(metadata.get("parameter_count", 0)), "validated", True,
        )

    def activate(self, relative_path: str) -> None:
        target = (self.root / relative_path).resolve()
        target.relative_to(self.models.resolve())
        metadata = JarvisTransformer.peek_metadata(target)
        model = JarvisTransformer.load(target)
        tokenizer = JarvisTokenizer.load(self.models / "jarvis_tokenizer_v003.json")
        if model.config.vocab_size != tokenizer.vocab_size:
            raise ValueError("Model vocabulary is incompatible with the active tokenizer")
        payload = {
            "format": "jarvis-active-model-v1",
            "path": target.relative_to(self.root).as_posix(),
            "parameter_count": int(metadata.get("parameter_count", 0)),
            "sha256": JarvisTransformer.file_sha256(target),
            "activated_at_utc": datetime.now(UTC).isoformat(),
        }
        temporary = self.active_path.with_suffix(".tmp")
        temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        temporary.replace(self.active_path)

    def reset_activation(self) -> None:
        self.active_path.unlink(missing_ok=True)

    def delete_lab_model(self, relative_path: str) -> None:
        target = (self.root / relative_path).resolve()
        try:
            target.relative_to(self.lab_models.resolve())
        except ValueError as exc:
            raise PermissionError("Bundled release models cannot be deleted from LAB") from exc
        if not target.is_file():
            raise ValueError("Model does not exist")
        target.unlink()
