from __future__ import annotations

import argparse
import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from jarvis.neural.config import TransformerConfig  # noqa: E402
from jarvis.neural.tokenizer import JarvisTokenizer  # noqa: E402
from jarvis.neural.transformer import JarvisTransformer  # noqa: E402
from training.numpy_trainer import CurriculumRunner  # noqa: E402


def _ensure_inputs(config: TransformerConfig) -> None:
    dataset = ROOT / "datasets" / "tokenized" / "dataset_v002_train.jsonl"
    tokenizer_path = ROOT / "models" / "jarvis_tokenizer_v002.json"
    if not dataset.is_file():
        from training.build_dataset_v6 import build
        from training.train_tokenizer import train

        build()
        train()
    elif not tokenizer_path.is_file():
        from training.train_tokenizer import train

        train()
    tokenizer = JarvisTokenizer.load(tokenizer_path)
    if tokenizer.vocab_size != config.vocab_size:
        raise RuntimeError(
            f"Tokenizer vocab {tokenizer.vocab_size} does not match model {config.vocab_size}"
        )


def train(args: argparse.Namespace) -> dict[str, object]:
    config = TransformerConfig.load(ROOT / "configs" / f"{args.profile}.json")
    _ensure_inputs(config)
    if args.resume:
        model = JarvisTransformer.load(Path(args.resume), dequantize=True)
        if model.config.to_dict() != config.to_dict():
            raise RuntimeError("Checkpoint architecture does not match selected profile")
    else:
        model = JarvisTransformer.initialize(config)
    steps = args.steps_per_stage or config.training.steps_per_stage
    runner = CurriculumRunner(
        ROOT,
        model,
        steps_per_stage=steps,
        sequence_length=args.sequence_length or config.training.sequence_length,
        retain_full_checkpoints=14,
        log_callback=lambda event: print(json.dumps(event, ensure_ascii=False), flush=True),
    )
    metrics = runner.run()
    if metrics.get("stopped"):
        return metrics
    source = ROOT / "models" / "checkpoints" / "stage14_jarvis_personality" / "model.npz"
    destination = ROOT / "models" / "jarvis_nano_v06.npz"
    shutil.copy2(source, destination)
    tokenizer_path = ROOT / "models" / "jarvis_tokenizer_v002.json"
    registry = {
        "format": "jarvis-model-registry-v1",
        "active_model": "jarvis_nano_v06",
        "models": [
            {
                "id": "jarvis_nano_v06",
                "name": config.name,
                "profile": config.profile,
                "architecture": config.architecture,
                "parameters": model.parameter_count,
                "dataset_version": "dataset_v002",
                "training_date": datetime.now(timezone.utc).isoformat(),
                "training_stages": list(range(1, 15)),
                "validation_metrics": {
                    "loss": metrics["final_validation_loss"],
                    "perplexity": metrics["final_perplexity"],
                },
                "checkpoint_parent": "stage13_search_decision",
                "weights": destination.name,
                "tokenizer": tokenizer_path.name,
                "quantization": "symmetric_per_tensor_int8",
                "pretrained_source": None,
                "sha256": model.file_sha256(destination),
                "size_bytes": destination.stat().st_size,
            },
            {
                "id": "jarvis_base",
                "name": "Jarvis Base",
                "profile": "base",
                "architecture": "decoder_only_transformer",
                "parameters": TransformerConfig.load(ROOT / "configs" / "base.json").parameter_count,
                "status": "config_only_not_trained",
                "config": "../configs/base.json",
                "pretrained_source": None,
            },
            {
                "id": "jarvis_smart",
                "name": "Jarvis Smart",
                "profile": "smart",
                "architecture": "decoder_only_transformer",
                "parameters": TransformerConfig.load(ROOT / "configs" / "smart.json").parameter_count,
                "status": "config_only_not_trained",
                "config": "../configs/smart.json",
                "pretrained_source": None,
            },
        ],
    }
    (ROOT / "models" / "model_registry.json").write_text(
        json.dumps(registry, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    summary = {
        **metrics,
        "model_file": str(destination.relative_to(ROOT)),
        "model_size_bytes": destination.stat().st_size,
        "model_sha256": model.file_sha256(destination),
    }
    print(json.dumps({"event": "release_model_ready", **summary}, ensure_ascii=False), flush=True)
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Train a JARVIS Transformer from random initialization and project data"
    )
    parser.add_argument("--profile", choices=("nano", "base", "smart"), default="nano")
    parser.add_argument("--steps-per-stage", type=int)
    parser.add_argument("--sequence-length", type=int)
    parser.add_argument("--resume", type=Path)
    args = parser.parse_args()
    if args.profile != "nano":
        try:
            import torch  # type: ignore[import-not-found]  # noqa: F401
        except ImportError:
            parser.error("Base/Smart training requires requirements-training.txt and PyTorch")
        parser.error("Use training/torch_train.py for Base or Smart profiles")
    train(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
