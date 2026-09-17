from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from jarvis.neural.config import TransformerConfig  # noqa: E402
from jarvis.neural.tokenizer import JarvisTokenizer  # noqa: E402
from jarvis.neural.transformer import JarvisTransformer  # noqa: E402
from training.numpy_trainer import CurriculumRunner  # noqa: E402


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def train(steps_per_stage: int = 48, sequence_length: int = 64) -> dict[str, Any]:
    config = TransformerConfig.load(ROOT / "configs" / "nano_v7.json")
    tokenizer_path = ROOT / "models" / "jarvis_tokenizer_v003.json"
    tokenized_path = ROOT / "datasets" / "tokenized" / "dataset_v003_train.jsonl"
    tokenizer = JarvisTokenizer.load(tokenizer_path)
    if tokenizer.vocab_size != config.vocab_size:
        raise RuntimeError(
            f"Tokenizer vocab {tokenizer.vocab_size} does not match Nano {config.vocab_size}"
        )
    if not tokenized_path.is_file():
        raise RuntimeError("Run training/train_tokenizer_v7.py before training")
    model = JarvisTransformer.initialize(config)
    runner = CurriculumRunner(
        ROOT,
        model,
        steps_per_stage=max(1, int(steps_per_stage)),
        sequence_length=max(8, int(sequence_length)),
        log_callback=lambda event: print(json.dumps(event, ensure_ascii=False), flush=True),
        retain_full_checkpoints=1,
        dataset_version="dataset_v003",
        tokenized_file=tokenized_path,
        checkpoint_root=ROOT / "models" / "checkpoints_v7",
        metrics_filename="jarvis_nano_v07_training_metrics.json",
        save_stage_weights=False,
    )
    metrics = runner.run()
    if metrics.get("stopped"):
        return metrics
    final_checkpoint = (
        ROOT / "models" / "checkpoints_v7" / "stage14_jarvis_personality" / "model.npz"
    )
    destination = ROOT / "models" / "jarvis_nano_v07.npz"
    shutil.copy2(final_checkpoint, destination)
    metadata = JarvisTransformer.peek_metadata(destination)
    registry = {
        "format": "jarvis-model-registry-v2",
        "active_profile": "nano",
        "runtime_policy": "fast_brain_eager_smart_brain_lazy",
        "profiles": [
            {
                "id": "jarvis_nano_v07",
                "name": config.name,
                "profile": "nano",
                "status": "trained",
                "architecture": config.architecture,
                "parameters": model.parameter_count,
                "context_length": config.max_seq_len,
                "vocab_size": config.vocab_size,
                "dataset_version": "dataset_v003",
                "training_steps": metrics["training_steps"],
                "training_date_utc": datetime.now(UTC).isoformat(),
                "weights": destination.name,
                "tokenizer": tokenizer_path.name,
                "quantization": metadata.get("quantization"),
                "model_format": metadata.get("model_format"),
                "integrity": metadata.get("integrity"),
                "pretrained_source": None,
                "sha256": _sha256(destination),
                "size_bytes": destination.stat().st_size,
            },
            {
                "id": "jarvis_core_v07",
                "name": "JARVIS Core v0.7",
                "profile": "core",
                "status": "config_only_not_bundled",
                "parameters": TransformerConfig.load(ROOT / "configs" / "core.json").parameter_count,
                "context_length": 4096,
                "config": "../configs/core.json",
                "pretrained_source": None,
            },
            {
                "id": "jarvis_pro_v07",
                "name": "JARVIS Pro v0.7",
                "profile": "pro",
                "status": "config_only_not_bundled",
                "parameters": TransformerConfig.load(ROOT / "configs" / "pro.json").parameter_count,
                "context_length": 4096,
                "config": "../configs/pro.json",
                "pretrained_source": None,
            },
        ],
    }
    registry_path = ROOT / "models" / "model_registry_v2.json"
    registry_path.write_text(json.dumps(registry, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    release = {
        **metrics,
        "weights": str(destination.relative_to(ROOT)),
        "weights_size_bytes": destination.stat().st_size,
        "weights_sha256": _sha256(destination),
        "tokenizer": str(tokenizer_path.relative_to(ROOT)),
        "tokenizer_sha256": _sha256(tokenizer_path),
        "model_format": metadata.get("model_format"),
        "integrity_verified": bool(metadata.get("integrity")),
        "pretrained_source": None,
    }
    (ROOT / "models" / "training_metrics_v7.json").write_text(
        json.dumps(release, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({"event": "v07_release_model_ready", **release}, ensure_ascii=False), flush=True)
    return release


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Train JARVIS Nano v0.7 from random initialization with NumPy on CPU"
    )
    parser.add_argument("--steps-per-stage", type=int, default=48)
    parser.add_argument("--sequence-length", type=int, default=64)
    args = parser.parse_args()
    train(args.steps_per_stage, args.sequence_length)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

