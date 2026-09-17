from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

DEFAULT_OUTPUT = ROOT.parent / "Jarvis_v0.11.0_ready.zip"
ARCHIVE_ROOT = "Jarvis_v0.11.0"
FIXED_TIMESTAMP = (2026, 8, 31, 0, 0, 0)
IGNORED_DIRECTORIES = {
    ".git", ".venv", ".venv-runtime", ".venv-training", "__pycache__",
    "runtime_data", "logs", ".pytest_cache", ".mypy_cache", "exports",
    "lab_runs", "checkpoints", "checkpoints_v7_lab",
}
IGNORED_SUFFIXES = {".pyc", ".pyo", ".db", ".zip", ".tmp", ".log"}

ROOT_FILES = {
    ".gitignore", "main.py", "training_studio.py", "setup.ps1", "RUN_JARVIS.bat",
    "requirements.txt", "requirements-runtime.txt", "requirements-training.txt",
    "requirements-documents.txt", "requirements-automation.txt", "README.md",
    "ARCHITECTURE.md", "TRAINING.md", "BENCHMARK.md", "CHANGELOG_v0.9.md",
    "CHANGELOG_v0.10.md", "CHANGELOG_v0.11.md", "V19_RELEASE_NOTES.md",
    "RELEASE_V19_FINAL.json",
}
MODEL_FILES = {
    "hybrid_brain_v7.jv.gz", "jarvis_nano_v07.npz", "jarvis_nano_v08.npz",
    "dialogue_adapter_v8.json",
    "persian_fluency_v9.json", "persian_fluency_metrics_v9.json",
    "dialogue_followup_v10.json", "dialogue_followup_metrics_v10.json",
    "bilingual_fluency_v11.json", "bilingual_fluency_metrics_v11.json",
    "cognitive_skills_v11.json", "cognitive_skills_metrics_v11.json",
    "jarvis_tokenizer_v003.json", "model_registry_v2.json",
    "tokenizer_metrics_v003.json", "jarvis_nano_v07_training_metrics.json",
    "training_metrics_v7.json", "training_metrics_fast_v7.json",
    "model_evaluation_v7.json", "evaluation_latest.json",
    "agent_evaluation_v7.json", "agent_evaluation_v8.json", "agent_evaluation_v9.json",
    "benchmark_v07_latest.json",
    "quantization_benchmark_v7.json", "alignment_experiment_v7.json",
    "grounding_evaluation_v071.json", "alignment_experiment_v8_rejected.json",
    "numeric_role_v19.npz", "semantic_frame_v19.npz",
    "execution_pattern_v19.npz", "code_intent_v19.npz",
}
DATASET_FILES = {
    "manifest_v001.json", "manifest_v002.json", "manifest_v003.json",
    "raw/seed_v003.jsonl", "splits/dataset_v003_train.jsonl",
    "splits/dataset_v003_validation.jsonl", "splits/dataset_v003_test.jsonl",
    "tokenized/dataset_v003_train.jsonl",
    "benchmarks/unseen_v003_1000.jsonl", "corrections/learning_queue.jsonl",
    "numeric_roles_v19/numeric_role_24000_real.jsonl",
    "multistep_v19/execution_graph_12000_real.jsonl",
    "code_v19/code_intelligence_5000_real.jsonl",
    "semantic_ir_v19/semantic_frame_18000_real.jsonl",
}
REQUIRED_STAGES = (
    "stage1_language_foundations", "stage2_conversation",
    "stage3_general_knowledge", "stage4_instruction_following",
    "stage5_action_recognition", "stage6_entity_extraction",
    "stage7_tool_calling", "stage8_argument_extraction",
    "stage9_context_memory", "stage10_planning", "stage11_reasoning",
    "stage12_recovery_verification", "stage13_search_decision",
    "stage14_jarvis_personality",
)
REQUIRED_FILES = (
    "main.py", "training_studio.py", "RUN_JARVIS.bat", "README.md", "CHANGELOG_v0.11.md",
    "config/app.json", "config/brain.json", "configs/nano_v7.json",
    "configs/core.json", "configs/pro.json", "models/hybrid_brain_v7.jv.gz",
    "models/jarvis_nano_v08.npz", "models/dialogue_adapter_v8.json",
    "models/persian_fluency_v9.json", "models/persian_fluency_metrics_v9.json",
    "models/dialogue_followup_v10.json", "models/dialogue_followup_metrics_v10.json",
    "models/bilingual_fluency_v11.json", "models/bilingual_fluency_metrics_v11.json",
    "models/cognitive_skills_v11.json", "models/cognitive_skills_metrics_v11.json",
    "models/jarvis_tokenizer_v003.json",
    "models/model_registry_v2.json", "models/model_evaluation_v7.json",
    "models/agent_evaluation_v9.json", "models/quantization_benchmark_v7.json",
    "models/grounding_evaluation_v071.json",
    "datasets/manifest_v003.json", "datasets/benchmarks/unseen_v003_1000.jsonl",
    "training/train_model_v7.py", "training/train_tokenizer_v7.py",
    "training/evaluate_agent_v7.py", "tests/test_v07_smart_agent.py",
    "tests/test_v071_grounded_intelligence.py",
    "tests/test_v072_context_research_ui.py", "tests/test_v08_hardening.py",
    "tests/test_v09_persian_fluency.py", "training/train_persian_fluency_v9.py",
    "tests/test_v094_challenge_reasoning.py", "jarvis/agent/challenge_reasoner.py",
    "jarvis/agent/conversation_chain.py",
    "tests/test_v095_regressions.py",
    "tests/test_v10_context_model.py", "training/train_followup_model_v10.py",
    "jarvis/agent/followup_model.py",
    "tests/test_v11_bilingual_cognitive_research.py",
    "training/train_bilingual_fluency_v11.py", "training/train_cognitive_skills_v11.py",
    "jarvis/agent/bilingual_fluency.py", "jarvis/agent/cognitive_model.py",
    "data/training/bilingual_fluency_v11_supplement.json",
    "data/training/cognitive_skills_v11.json",
    "jarvis/agent/dialogue_subject.py", "jarvis/gui/chat_surface.py",
    "jarvis/agent/semantic_ir_v19.py", "jarvis/agent/semantic_models_v19.py",
    "jarvis/agent/local_intelligence_v19.py", "jarvis/nlu/semantic_slots_v19.py",
    "models/numeric_role_v19.npz", "models/semantic_frame_v19.npz",
    "models/execution_pattern_v19.npz", "models/code_intent_v19.npz",
    "datasets/numeric_roles_v19/numeric_role_24000_real.jsonl",
    "datasets/multistep_v19/execution_graph_12000_real.jsonl",
    "datasets/code_v19/code_intelligence_5000_real.jsonl",
    "datasets/semantic_ir_v19/semantic_frame_18000_real.jsonl",
    "reports/v19_real_training_report.json", "reports/v19_regression_report.json",
    "tests/test_v19_intelligence.py", "tests/v19_holdout_cases.py",
    "training/train_v19_real.py", "V19_RELEASE_NOTES.md", "RELEASE_V19_FINAL.json",
    "skills/file_workflow/skill.json", "plugins/desktop_core/plugin.json",
)


def _json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise RuntimeError(f"Expected JSON object: {path}")
    return payload


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _included(relative: Path) -> bool:
    # Final v19 packaging keeps the complete working project so older v15-v18
    # trained assets and datasets required by regression tests are not dropped.
    # Only superseded prototype v19 datasets are excluded.
    value = relative.as_posix()
    superseded = {
        "datasets/numeric_roles_v19/numeric_role_20000.jsonl",
        "datasets/multistep_v19/execution_graph_10000.jsonl",
        "datasets/semantic_ir_v19/slot_binding_ratio_5000.jsonl",
    }
    return value not in superseded


def release_files() -> list[Path]:
    files: list[Path] = []
    for path in ROOT.rglob("*"):
        relative = path.relative_to(ROOT)
        if not path.is_file() or any(part in IGNORED_DIRECTORIES for part in relative.parts):
            continue
        if path.suffix.casefold() in IGNORED_SUFFIXES:
            continue
        if _included(relative):
            files.append(path)
    return sorted(files, key=lambda item: item.relative_to(ROOT).as_posix().casefold())


def validate_release() -> None:
    missing = [name for name in REQUIRED_FILES if not (ROOT / name).is_file()]
    missing.extend(
        f"models/checkpoints_v7/{stage}/checkpoint.json"
        for stage in REQUIRED_STAGES
        if not (ROOT / "models" / "checkpoints_v7" / stage / "checkpoint.json").is_file()
    )
    if missing:
        raise RuntimeError("Missing release files: " + ", ".join(missing))

    from jarvis.agent.bilingual_fluency import BilingualFluencyEngine
    from jarvis.agent.cognitive_model import CognitiveSkillModel
    from jarvis.agent.fluency import PersianFluencyEngine
    from jarvis.agent.followup_model import FollowupIntentModel
    from jarvis.neural.tokenizer import JarvisTokenizer
    from jarvis.neural.transformer import JarvisTransformer

    model_path = ROOT / "models" / "jarvis_nano_v08.npz"
    model = JarvisTransformer.load(model_path)
    tokenizer = JarvisTokenizer.load(ROOT / "models" / "jarvis_tokenizer_v003.json")
    if model.parameter_count != 23_077_376:
        raise RuntimeError("Nano parameter count is invalid")
    if model.config.max_seq_len != 2048 or tokenizer.vocab_size != 4096:
        raise RuntimeError("Nano context/tokenizer shape is invalid")
    if not model.quantized_runtime or model.metadata.get("model_format") != "jarvis-numpy-int8-v2":
        raise RuntimeError("Release model is not resident-INT8 v2")
    if model.config.architecture != "decoder_only_transformer_mha_rope_swiglu":
        raise RuntimeError("Release model architecture metadata is not MHA")
    if model.metadata.get("model_id") != "jarvis_nano_v08":
        raise RuntimeError("Release model ID is invalid")
    if model.metadata.get("pretrained_source") not in (None, ""):
        raise RuntimeError("External pretrained provenance is forbidden")
    fluency = PersianFluencyEngine(ROOT / "models" / "persian_fluency_v9.json")
    fluency_metrics = _json(ROOT / "models" / "persian_fluency_metrics_v9.json")
    if not fluency.ready or fluency.version != "0.9.0":
        raise RuntimeError("Persian fluency model is unavailable or stale")
    if not fluency_metrics.get("accepted_for_release"):
        raise RuntimeError("Persian fluency model did not pass its quality gate")
    if fluency_metrics.get("classification_accuracy", 0.0) < 0.85:
        raise RuntimeError("Persian fluency held-out accuracy is below the release floor")
    followup = FollowupIntentModel(ROOT / "models" / "dialogue_followup_v10.json")
    followup_metrics = _json(ROOT / "models" / "dialogue_followup_metrics_v10.json")
    if not followup.ready or followup.training_examples < 94:
        raise RuntimeError("Dialogue follow-up model is unavailable or incomplete")
    if followup_metrics.get("pretrained_source") not in (None, ""):
        raise RuntimeError("Dialogue follow-up provenance is invalid")
    if not followup_metrics.get("accepted_for_release"):
        raise RuntimeError("Dialogue follow-up model did not pass its quality gate")
    if followup_metrics.get("classification_accuracy", 0.0) < 0.875:
        raise RuntimeError("Dialogue follow-up held-out accuracy is below the release floor")
    bilingual = BilingualFluencyEngine(ROOT / "models" / "bilingual_fluency_v11.json")
    bilingual_metrics = _json(ROOT / "models" / "bilingual_fluency_metrics_v11.json")
    if not bilingual.ready or bilingual_metrics.get("dataset_examples") != 105:
        raise RuntimeError("Bilingual fluency model is unavailable or incomplete")
    if bilingual_metrics.get("pretrained_source") not in (None, ""):
        raise RuntimeError("Bilingual fluency provenance is invalid")
    if not bilingual_metrics.get("accepted_for_release"):
        raise RuntimeError("Bilingual fluency model did not pass its quality gate")
    if bilingual_metrics.get("classification_accuracy", 0.0) < 0.85:
        raise RuntimeError("Bilingual fluency held-out accuracy is below the release floor")
    if bilingual_metrics.get("generation_acceptance", 0.0) != 1.0:
        raise RuntimeError("Bilingual fluency generation acceptance is incomplete")
    cognitive = CognitiveSkillModel(ROOT / "models" / "cognitive_skills_v11.json")
    cognitive_metrics = _json(ROOT / "models" / "cognitive_skills_metrics_v11.json")
    if not cognitive.ready or cognitive.training_examples != 96:
        raise RuntimeError("Cognitive skill model is unavailable or incomplete")
    if cognitive_metrics.get("pretrained_source") not in (None, ""):
        raise RuntimeError("Cognitive skill provenance is invalid")
    if not cognitive_metrics.get("accepted_for_release"):
        raise RuntimeError("Cognitive skill model did not pass its quality gate")
    if cognitive_metrics.get("classification_accuracy", 0.0) < 0.875:
        raise RuntimeError("Cognitive skill held-out accuracy is below the release floor")

    manifest = _json(ROOT / "datasets" / "manifest_v003.json")
    if manifest.get("split_counts") != {"train": 4583, "validation": 535, "test": 411}:
        raise RuntimeError("dataset_v003 split counts changed")
    if int(manifest.get("cross_split_concept_leakage", -1)) != 0:
        raise RuntimeError("dataset_v003 concept leakage is non-zero")
    if int(manifest.get("unseen_benchmark_training_overlap", -1)) != 0:
        raise RuntimeError("Unseen benchmark overlaps training")
    benchmark_rows = sum(
        1 for line in (ROOT / "datasets" / "benchmarks" / "unseen_v003_1000.jsonl")
        .read_text(encoding="utf-8").splitlines() if line
    )
    if benchmark_rows != 1000:
        raise RuntimeError("Unseen benchmark must contain exactly 1000 prompts")

    evaluation = _json(ROOT / "models" / "agent_evaluation_v9.json")
    if evaluation.get("dataset_cases") != 1000 or evaluation.get("error_count") != 0:
        raise RuntimeError("Agent evaluation is stale or failing")
    if evaluation.get("format") != "jarvis-agent-evaluation-v9":
        raise RuntimeError("Agent evaluation format is stale")
    if evaluation.get("evaluation_center", {}).get("execution_success", "missing") is not None:
        raise RuntimeError("Unsupported execution metric must remain null")

    registry = _json(ROOT / "models" / "model_registry_v2.json")
    profiles = {str(row.get("profile")): row for row in registry.get("profiles", [])}
    if registry.get("active_profile") != "nano" or set(profiles) != {"nano", "core", "pro"}:
        raise RuntimeError("Model profile registry is invalid")
    nano = profiles["nano"]
    if nano.get("weights") != model_path.name or nano.get("id") != "jarvis_nano_v08":
        raise RuntimeError("Nano registry does not select the v0.8 model")
    if nano.get("sha256") != _sha256(model_path):
        raise RuntimeError("Nano file SHA-256 does not match registry")
    if any(
        profiles[name].get("status") != "config_only_not_bundled"
        for name in ("core", "pro")
    ):
        raise RuntimeError("Core/Pro status is invalid")

    # v19 semantic models are trained artifacts, not placeholder datasets.
    from jarvis.agent.semantic_models_v19 import (
        NumericRoleTaggerV19, SemanticFrameClassifierV19,
        ExecutionPatternClassifierV19, CodeIntentClassifierV19,
    )
    v19_models = (
        ("numeric_role_v19.npz", NumericRoleTaggerV19),
        ("semantic_frame_v19.npz", SemanticFrameClassifierV19),
        ("execution_pattern_v19.npz", ExecutionPatternClassifierV19),
        ("code_intent_v19.npz", CodeIntentClassifierV19),
    )
    for filename, model_cls in v19_models:
        model_v19 = model_cls(ROOT / "models" / filename)
        if not model_v19.ready or model_v19.parameter_count < 100_000:
            raise RuntimeError(f"v19 trained model is unavailable: {filename}")
        if model_v19.metadata.get("dataset_version") != "dataset_v019_real":
            raise RuntimeError(f"v19 model dataset metadata is stale: {filename}")
        if model_v19.metadata.get("holdout_strategy") != "template_id_disjoint":
            raise RuntimeError(f"v19 model holdout metadata is invalid: {filename}")

    v19_report = _json(ROOT / "reports" / "v19_real_training_report.json")
    if not v19_report.get("passed"):
        raise RuntimeError("v19 training quality gates did not pass")
    observed = v19_report.get("quality_gates", {}).get("observed", {})
    required_v19 = {
        "numeric_role_v19": 0.95,
        "semantic_frame_v19": 0.96,
        "execution_pattern_v19": 0.99,
        "code_intent_v19": 0.95,
    }
    for model_id, floor in required_v19.items():
        if float(observed.get(model_id, 0.0)) < floor:
            raise RuntimeError(f"v19 quality gate below final release floor: {model_id}")

    regression = _json(ROOT / "reports" / "v19_regression_report.json")
    if regression.get("failures") != 0 or regression.get("standard_tests_passed") != 696:
        raise RuntimeError("v19 regression report is incomplete or failing")
    if regression.get("subtests_passed") != 585:
        raise RuntimeError("v19 subtest regression count is incomplete")

    for index, stage in enumerate(REQUIRED_STAGES, 1):
        checkpoint = _json(ROOT / "models" / "checkpoints_v7" / stage / "checkpoint.json")
        completed = checkpoint.get("training_stages", [])
        if not isinstance(completed, list) or not completed or int(completed[-1]) != index:
            raise RuntimeError(f"Checkpoint stage mismatch: {stage}")
        if checkpoint.get("pretrained_source") not in (None, ""):
            raise RuntimeError(f"Checkpoint provenance is invalid: {stage}")


def build_archive(output: Path = DEFAULT_OUTPUT) -> tuple[Path, int, str, int]:
    validate_release()
    output = output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    files = release_files()
    with zipfile.ZipFile(
        output, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=1, allowZip64=True,
    ) as archive:
        for path in files:
            relative = path.relative_to(ROOT).as_posix()
            info = zipfile.ZipInfo(f"{ARCHIVE_ROOT}/{relative}", FIXED_TIMESTAMP)
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o100644 << 16
            with archive.open(info, "w", force_zip64=True) as destination, path.open("rb") as source:
                shutil.copyfileobj(source, destination, length=1024 * 1024)
    with zipfile.ZipFile(output) as archive:
        broken = archive.testzip()
        names = archive.namelist()
        if broken:
            raise RuntimeError(f"Corrupt ZIP member: {broken}")
        if not names or any(not name.startswith(f"{ARCHIVE_ROOT}/") for name in names):
            raise RuntimeError("Archive root is invalid")
    return output, output.stat().st_size, _sha256(output), len(files)


def verify_extracted(archive_path: Path) -> dict[str, Any]:
    with tempfile.TemporaryDirectory(prefix="jarvis-v011-release-") as temporary:
        temporary_root = Path(temporary)
        with zipfile.ZipFile(archive_path) as archive:
            archive.extractall(temporary_root)
        extracted = temporary_root / ARCHIVE_ROOT
        environment = os.environ.copy()
        environment["JARVIS_DATA_DIR"] = str(temporary_root / "runtime")
        completed = subprocess.run(
            [sys.executable, "main.py", "--self-test"], cwd=extracted,
            env=environment, capture_output=True, text=True, encoding="utf-8",
            errors="replace", timeout=180, check=False,
        )
        if completed.returncode != 0:
            tail = "\n".join((completed.stdout + "\n" + completed.stderr).splitlines()[-80:])
            raise RuntimeError(f"Extracted release self-test failed:\n{tail}")
        match = re.search(r'"tests_run"\s*:\s*(\d+)', completed.stdout)
        tests_run = int(match.group(1)) if match else 0
        if tests_run < 656 or '"successful": true' not in completed.stdout:
            raise RuntimeError("Extracted release did not report the complete successful test suite")
        v19_regression = subprocess.run(
            [sys.executable, "-m", "pytest", "-q",
             "tests/test_v06_desktop.py", "tests/test_v18_intelligence.py",
             "tests/test_v19_intelligence.py"], cwd=extracted, env=environment,
            capture_output=True, text=True, encoding="utf-8", errors="replace",
            timeout=120, check=False,
        )
        if v19_regression.returncode != 0:
            tail = "\n".join((v19_regression.stdout + "\n" + v19_regression.stderr).splitlines()[-80:])
            raise RuntimeError(f"Extracted v19 regression failed:\n{tail}")
        version = subprocess.run(
            [sys.executable, "main.py", "--version"], cwd=extracted,
            env=environment, capture_output=True, text=True, encoding="utf-8",
            errors="replace", timeout=30, check=False,
        )
        if version.returncode != 0 or "0.11.0" not in version.stdout:
            raise RuntimeError("Extracted release version check failed")
        return {"tests_run": tests_run, "successful": True, "version": version.stdout.strip()}


def main() -> int:
    parser = argparse.ArgumentParser(description="Build and verify deterministic JARVIS v0.11.0 ZIP")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--skip-extracted-test", action="store_true")
    args = parser.parse_args()
    output, size, digest, count = build_archive(args.output)
    verification = None if args.skip_extracted_test else verify_extracted(output)
    print(json.dumps({
        "archive": str(output), "files": count, "size_bytes": size,
        "sha256": digest, "verification": verification,
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
