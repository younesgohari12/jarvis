from __future__ import annotations

import json
import os
import platform
from dataclasses import dataclass
from pathlib import Path
from typing import Any


class ConfigError(RuntimeError):
    """Raised when required configuration is missing or invalid."""


@dataclass(frozen=True, slots=True)
class WindowConfig:
    width: int
    height: int
    min_width: int
    min_height: int


@dataclass(frozen=True, slots=True)
class MemoryConfig:
    recent_messages: int
    short_term_messages: int
    resume_session_hours: int
    max_history_rows: int
    max_sessions: int
    retention_days: int


@dataclass(frozen=True, slots=True)
class FileLimits:
    max_file_bytes: int
    max_text_preview_bytes: int
    max_zip_entries: int
    max_zip_member_bytes: int
    max_zip_total_preview_bytes: int


@dataclass(frozen=True, slots=True)
class KnowledgeConfig:
    max_documents: int
    minimum_score: float
    seed_version: int


@dataclass(frozen=True, slots=True)
class InternetConfig:
    enabled_by_default: bool
    timeout_seconds: float
    max_response_bytes: int
    max_search_results: int
    fetch_top_pages: int
    max_redirects: int


@dataclass(frozen=True, slots=True)
class LearningConfig:
    max_corrections: int
    max_feedback_rows: int


@dataclass(frozen=True, slots=True)
class GUIConfig:
    performance: str
    animation: bool
    timestamps: bool
    theme: str
    language: str


@dataclass(frozen=True, slots=True)
class TrainingConfig:
    epochs: int
    learning_rate: float
    learning_rate_decay: float
    l2: float
    seed: int
    validation_ratio: float
    test_ratio: float


@dataclass(frozen=True, slots=True)
class BrainConfig:
    format: str
    feature_size: int
    embedding_method: str
    embedding_size: int
    hidden_size: int
    layers: int
    context_length: int
    confidence_threshold: float
    margin_threshold: float
    lexical_ood_threshold: float
    maximum_normalized_entropy: float
    neural_weight: float
    lexical_weight: float
    temperature: float
    training: TrainingConfig

    @property
    def nominal_parameter_count(self) -> int:
        return (
            self.feature_size * self.embedding_size
            + self.embedding_size * self.hidden_size
            + self.hidden_size * 48
        )


@dataclass(frozen=True, slots=True)
class RuntimePaths:
    root: Path
    data_dir: Path
    database: Path
    knowledge_database: Path
    rag_database: Path
    log_file: Path
    model: Path
    neural_model: Path
    tokenizer: Path
    dataset_manifest_v5: Path
    dataset_manifest_v6: Path
    dataset_manifest_v7: Path
    failure_queue: Path
    logs_dir: Path
    responses: Path
    knowledge_seed: Path
    context_rules: Path
    intent_signatures: Path
    websites: Path
    apps: Path
    scenario_dataset: Path
    app_index: Path
    personalities: Path
    dataset_train: Path
    dataset_validation: Path
    dataset_test: Path


@dataclass(frozen=True, slots=True)
class AppConfig:
    app_name: str
    version: str
    default_personality: str
    window: WindowConfig
    memory: MemoryConfig
    files: FileLimits
    knowledge: KnowledgeConfig
    internet: InternetConfig
    learning: LearningConfig
    gui: GUIConfig
    brain: BrainConfig
    paths: RuntimePaths


def _read_json(path: Path) -> dict[str, Any]:
    try:
        with path.open("r", encoding="utf-8") as handle:
            value = json.load(handle)
    except (OSError, json.JSONDecodeError) as exc:
        raise ConfigError(f"Cannot load configuration {path.name}: {exc}") from exc
    if not isinstance(value, dict):
        raise ConfigError(f"Configuration root must be an object: {path.name}")
    return value


def _positive_int(value: Any, label: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise ConfigError(f"{label} must be a positive integer")
    return value


def _bounded_float(value: Any, label: str, minimum: float, maximum: float) -> float:
    result = float(value)
    if not minimum <= result <= maximum:
        raise ConfigError(f"{label} must be between {minimum} and {maximum}")
    return result


def _preferred_data_directory() -> Path:
    override = os.environ.get("JARVIS_DATA_DIR")
    if override:
        return Path(override).expanduser().resolve()
    system = platform.system()
    if system == "Windows":
        parent = os.environ.get("LOCALAPPDATA") or os.environ.get("APPDATA")
        if parent:
            return Path(parent) / "Jarvis_v0.7"
    if system == "Darwin":
        return Path.home() / "Library" / "Application Support" / "Jarvis_v0.7"
    xdg = os.environ.get("XDG_DATA_HOME")
    parent = Path(xdg).expanduser() if xdg else Path.home() / ".local" / "share"
    return parent / "jarvis_v0_7"


def _create_data_directory(root: Path) -> Path:
    preferred = _preferred_data_directory()
    try:
        preferred.mkdir(parents=True, exist_ok=True)
        return preferred
    except OSError as primary_error:
        fallback = root / "runtime_data"
        try:
            fallback.mkdir(parents=True, exist_ok=True)
            return fallback
        except OSError as fallback_error:
            raise ConfigError(
                "Cannot create Jarvis data directory in the user profile or project: "
                f"{primary_error}; {fallback_error}"
            ) from fallback_error


def _active_neural_model(root: Path) -> Path:
    """Return a LAB-activated project model, or the immutable release default."""
    release = root / "models" / "jarvis_nano_v18.npz"
    fallback_v12 = root / "models" / "jarvis_nano_v12.npz"
    fallback_v08 = root / "models" / "jarvis_nano_v08.npz"
    default = release if release.is_file() else (fallback_v12 if fallback_v12.is_file() else (fallback_v08 if fallback_v08.is_file() else root / "models" / "jarvis_nano_v07.npz"))
    selection = root / "models" / "active_model_v8.json"
    try:
        payload = _read_json(selection)
        if payload.get("format") != "jarvis-active-model-v1":
            return default
        candidate = (root / str(payload["path"])).resolve()
        candidate.relative_to((root / "models").resolve())
        if candidate.suffix.casefold() != ".npz" or not candidate.is_file():
            return default
        return candidate
    except (ConfigError, KeyError, TypeError, ValueError, OSError):
        # A stale LAB selection never prevents the production GUI from opening.
        return default


def load_config(root: Path) -> AppConfig:
    root = root.resolve()
    app = _read_json(root / "config" / "app.json")
    raw_brain = _read_json(root / "config" / "brain.json")
    raw_training = raw_brain.get("training", {})
    try:
        window_data = app["window"]
        memory_data = app["memory"]
        files_data = app["files"]
        knowledge_data = app["knowledge"]
        internet_data = app["internet"]
        learning_data = app["learning"]
        gui_data = app["gui"]

        window = WindowConfig(
            width=_positive_int(window_data["width"], "window.width"),
            height=_positive_int(window_data["height"], "window.height"),
            min_width=_positive_int(window_data["min_width"], "window.min_width"),
            min_height=_positive_int(window_data["min_height"], "window.min_height"),
        )
        memory = MemoryConfig(
            recent_messages=_positive_int(memory_data["recent_messages"], "memory.recent_messages"),
            short_term_messages=_positive_int(
                memory_data["short_term_messages"], "memory.short_term_messages"
            ),
            resume_session_hours=_positive_int(
                memory_data["resume_session_hours"], "memory.resume_session_hours"
            ),
            max_history_rows=_positive_int(memory_data["max_history_rows"], "memory.max_history_rows"),
            max_sessions=_positive_int(memory_data["max_sessions"], "memory.max_sessions"),
            retention_days=_positive_int(memory_data["retention_days"], "memory.retention_days"),
        )
        files = FileLimits(
            max_file_bytes=_positive_int(files_data["max_file_bytes"], "files.max_file_bytes"),
            max_text_preview_bytes=_positive_int(
                files_data["max_text_preview_bytes"], "files.max_text_preview_bytes"
            ),
            max_zip_entries=_positive_int(files_data["max_zip_entries"], "files.max_zip_entries"),
            max_zip_member_bytes=_positive_int(
                files_data["max_zip_member_bytes"], "files.max_zip_member_bytes"
            ),
            max_zip_total_preview_bytes=_positive_int(
                files_data["max_zip_total_preview_bytes"], "files.max_zip_total_preview_bytes"
            ),
        )
        knowledge = KnowledgeConfig(
            max_documents=_positive_int(knowledge_data["max_documents"], "knowledge.max_documents"),
            minimum_score=_bounded_float(
                knowledge_data["minimum_score"], "knowledge.minimum_score", 0.0, 1.0
            ),
            seed_version=_positive_int(knowledge_data["seed_version"], "knowledge.seed_version"),
        )
        internet = InternetConfig(
            enabled_by_default=bool(internet_data["enabled_by_default"]),
            timeout_seconds=_bounded_float(
                internet_data["timeout_seconds"], "internet.timeout_seconds", 0.5, 30.0
            ),
            max_response_bytes=_positive_int(
                internet_data["max_response_bytes"], "internet.max_response_bytes"
            ),
            max_search_results=_positive_int(
                internet_data["max_search_results"], "internet.max_search_results"
            ),
            fetch_top_pages=_positive_int(
                internet_data["fetch_top_pages"], "internet.fetch_top_pages"
            ),
            max_redirects=_positive_int(
                internet_data["max_redirects"], "internet.max_redirects"
            ),
        )
        learning = LearningConfig(
            max_corrections=_positive_int(
                learning_data["max_corrections"], "learning.max_corrections"
            ),
            max_feedback_rows=_positive_int(
                learning_data["max_feedback_rows"], "learning.max_feedback_rows"
            ),
        )
        gui = GUIConfig(
            performance=str(gui_data["performance"]).upper(),
            animation=bool(gui_data["animation"]),
            timestamps=bool(gui_data["timestamps"]),
            theme=str(gui_data["theme"]).upper(),
            language=str(gui_data["language"]).upper(),
        )
        training = TrainingConfig(
            epochs=_positive_int(raw_training["epochs"], "training.epochs"),
            learning_rate=float(raw_training["learning_rate"]),
            learning_rate_decay=float(raw_training["learning_rate_decay"]),
            l2=float(raw_training["l2"]),
            seed=int(raw_training["seed"]),
            validation_ratio=_bounded_float(
                raw_training["validation_ratio"], "training.validation_ratio", 0.05, 0.35
            ),
            test_ratio=_bounded_float(raw_training["test_ratio"], "training.test_ratio", 0.05, 0.35),
        )
        brain = BrainConfig(
            format=str(raw_brain["format"]),
            feature_size=_positive_int(raw_brain["feature_size"], "brain.feature_size"),
            embedding_method=str(raw_brain["embedding_method"]),
            embedding_size=_positive_int(raw_brain["embedding_size"], "brain.embedding_size"),
            hidden_size=_positive_int(raw_brain["hidden_size"], "brain.hidden_size"),
            layers=_positive_int(raw_brain["layers"], "brain.layers"),
            context_length=_positive_int(raw_brain["context_length"], "brain.context_length"),
            confidence_threshold=_bounded_float(
                raw_brain["confidence_threshold"], "brain.confidence_threshold", 0.0, 1.0
            ),
            margin_threshold=_bounded_float(
                raw_brain["margin_threshold"], "brain.margin_threshold", 0.0, 1.0
            ),
            lexical_ood_threshold=_bounded_float(
                raw_brain["lexical_ood_threshold"], "brain.lexical_ood_threshold", 0.0, 1.0
            ),
            maximum_normalized_entropy=_bounded_float(
                raw_brain["maximum_normalized_entropy"],
                "brain.maximum_normalized_entropy",
                0.0,
                1.0,
            ),
            neural_weight=_bounded_float(raw_brain["neural_weight"], "brain.neural_weight", 0.0, 1.0),
            lexical_weight=_bounded_float(
                raw_brain["lexical_weight"], "brain.lexical_weight", 0.0, 1.0
            ),
            temperature=_bounded_float(raw_brain["temperature"], "brain.temperature", 0.05, 5.0),
            training=training,
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise ConfigError(f"Invalid configuration value: {exc}") from exc

    if training.validation_ratio + training.test_ratio >= 0.7:
        raise ConfigError("Validation and test ratios leave too little training data")
    if abs((brain.neural_weight + brain.lexical_weight) - 1.0) > 0.001:
        raise ConfigError("Neural and lexical weights must sum to 1.0")
    if gui.performance not in {"AUTO", "LOW", "BALANCED", "ULTRA"}:
        raise ConfigError("gui.performance must be AUTO, LOW, BALANCED, or ULTRA")
    if gui.language not in {"AUTO", "PERSIAN", "ENGLISH"}:
        raise ConfigError("gui.language must be AUTO, PERSIAN, or ENGLISH")

    data_dir = _create_data_directory(root)
    training_dir = root / "data" / "training"
    paths = RuntimePaths(
        root=root,
        data_dir=data_dir,
        database=data_dir / "jarvis.db",
        knowledge_database=data_dir / "knowledge.db",
        rag_database=data_dir / "rag.db",
        log_file=data_dir / "jarvis.log",
        model=root / "models" / "hybrid_brain_v7.jv.gz",
        neural_model=_active_neural_model(root),
        tokenizer=root / "models" / "jarvis_tokenizer_v003.json",
        dataset_manifest_v5=root / "datasets" / "manifest_v001.json",
        dataset_manifest_v6=root / "datasets" / "manifest_v002.json",
        dataset_manifest_v7=root / "datasets" / "manifest_v003.json",
        failure_queue=data_dir / "failure_queue.jsonl",
        logs_dir=data_dir / "logs",
        responses=root / "data" / "responses_v4.json",
        knowledge_seed=root / "data" / "knowledge_v4.json",
        context_rules=root / "data" / "context_rules.json",
        intent_signatures=root / "data" / "intent_signatures.json",
        websites=root / "data" / "entities" / "websites.json",
        apps=root / "data" / "entities" / "apps.json",
        scenario_dataset=training_dir / "scenarios_v6.json",
        app_index=data_dir / "app_index.json",
        personalities=root / "jarvis" / "personalities",
        dataset_train=training_dir / "train.json",
        dataset_validation=training_dir / "validation.json",
        dataset_test=training_dir / "test.json",
    )
    return AppConfig(
        app_name=str(app.get("app_name", "Jarvis")),
        version=str(app.get("version", "0.11.0")),
        default_personality=str(app.get("default_personality", "Normal")),
        window=window,
        memory=memory,
        files=files,
        knowledge=knowledge,
        internet=internet,
        learning=learning,
        gui=gui,
        brain=brain,
        paths=paths,
    )
