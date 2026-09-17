from __future__ import annotations

import argparse
import json
import math
import sys
import time
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from jarvis.neural.transformer import JarvisTransformer  # noqa: E402
from training.numpy_trainer import NumpyTransformerTrainer  # noqa: E402


STAGE_WEIGHTS = {
    1: 0.04, 2: 0.22, 3: 0.28, 4: 0.08, 5: 0.03, 6: 0.03, 7: 0.03,
    8: 0.03, 9: 0.05, 10: 0.04, 11: 0.07, 12: 0.02, 13: 0.02, 14: 0.06,
}


def _load_sequences(path: Path) -> dict[int, list[tuple[list[int], int]]]:
    grouped: dict[int, list[tuple[list[int], int]]] = defaultdict(list)
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line:
            continue
        row = json.loads(line)
        grouped[int(row["stage"])].append(
            ([int(token) for token in row["tokens"]], int(row["prompt_length"]))
        )
    if set(grouped) != set(STAGE_WEIGHTS):
        raise RuntimeError(f"Alignment dataset is missing stages: {sorted(set(STAGE_WEIGHTS) - set(grouped))}")
    return dict(grouped)


def align(
    source: Path,
    output: Path,
    *,
    steps: int = 400,
    sequence_length: int = 64,
    learning_rate: float = 0.00008,
) -> dict[str, Any]:
    model = JarvisTransformer.load(source, dequantize=True)
    sequences = _load_sequences(
        ROOT / "datasets" / "tokenized" / "dataset_v003_train.jsonl"
    )
    length = max(8, min(int(sequence_length), model.config.max_seq_len))
    total_steps = max(1, int(steps))
    trainer = NumpyTransformerTrainer(
        model,
        learning_rate=float(learning_rate),
        minimum_learning_rate=max(0.00001, float(learning_rate) * 0.2),
        weight_decay=0.005,
        gradient_clip=model.config.training.gradient_clip,
        warmup_steps=min(20, max(1, total_steps // 10)),
        total_steps=total_steps,
    )
    rng = np.random.default_rng(model.config.seed + 700)
    stages = np.asarray(sorted(STAGE_WEIGHTS), dtype=np.int64)
    probabilities = np.asarray([STAGE_WEIGHTS[int(stage)] for stage in stages], dtype=np.float64)
    probabilities /= probabilities.sum()

    def batch() -> tuple[np.ndarray, np.ndarray]:
        inputs: list[int] = []
        labels: list[int] = []
        while len(inputs) < length:
            stage = int(rng.choice(stages, p=probabilities))
            rows = sequences[stage]
            tokens, prompt_length = rows[int(rng.integers(0, len(rows)))]
            inputs.extend(tokens[:-1])
            labels.extend(
                token if index >= max(0, prompt_length - 2) else -100
                for index, token in enumerate(tokens[1:])
            )
        maximum = len(inputs) - length
        candidates = [
            start for start in range(maximum + 1)
            if any(token >= 0 for token in labels[start : start + length])
        ]
        start = candidates[int(rng.integers(0, len(candidates)))]
        return (
            np.asarray(inputs[start : start + length], dtype=np.int64)[None, :],
            np.asarray(labels[start : start + length], dtype=np.int64)[None, :],
        )

    validation_batches = [batch() for _ in range(24)]
    initial_validation = float(np.mean([
        trainer.evaluate_batch(inputs, targets) for inputs, targets in validation_batches
    ]))
    losses: list[float] = []
    started = time.perf_counter()
    for step in range(1, total_steps + 1):
        inputs, targets = batch()
        loss, rate, gradient_norm = trainer.train_batch(inputs, targets)
        losses.append(loss)
        if step == 1 or step % 25 == 0 or step == total_steps:
            print(json.dumps({
                "event": "alignment_step", "step": step, "steps": total_steps,
                "mean_recent_loss": round(float(np.mean(losses[-25:])), 6),
                "learning_rate": rate, "gradient_norm": round(gradient_norm, 6),
                "elapsed_seconds": round(time.perf_counter() - started, 3),
            }, ensure_ascii=False), flush=True)
    final_validation = float(np.mean([
        trainer.evaluate_batch(inputs, targets) for inputs, targets in validation_batches
    ]))
    previous_step = int(model.metadata.get("step", 0) or 0)
    model.save_quantized(output, {
        **model.metadata,
        "training_complete": True,
        "dataset_version": "dataset_v003",
        "pretrained_source": None,
        "backend": "numpy_cpu_full_gradient",
        "alignment": "balanced_conversation_knowledge_rehearsal",
        "alignment_steps": total_steps,
        "step": previous_step + total_steps,
        "trained_at_utc": datetime.now(UTC).isoformat(),
    })
    result = {
        "format": "jarvis-alignment-metrics-v1",
        "source": str(source), "output": str(output),
        "parameters": model.parameter_count,
        "steps": total_steps,
        "initial_validation_loss": round(initial_validation, 6),
        "final_validation_loss": round(final_validation, 6),
        "final_perplexity": round(math.exp(min(20.0, final_validation)), 6),
        "mean_training_loss": round(float(np.mean(losses)), 6),
        "seconds": round(time.perf_counter() - started, 3),
        "device": "cpu", "pretrained_source": None,
    }
    metrics_path = output.with_name(output.stem + "_alignment_metrics.json")
    metrics_path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"event": "alignment_complete", **result}, ensure_ascii=False), flush=True)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description="Balanced alignment for the project-owned JARVIS v0.7 model")
    parser.add_argument("--source", type=Path, default=ROOT / "models" / "jarvis_nano_v07.npz")
    parser.add_argument("--output", type=Path, default=ROOT / "models" / "jarvis_nano_v07_aligned.npz")
    parser.add_argument("--steps", type=int, default=400)
    parser.add_argument("--sequence-length", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=0.00008)
    args = parser.parse_args()
    align(args.source, args.output, steps=args.steps, sequence_length=args.sequence_length, learning_rate=args.learning_rate)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
