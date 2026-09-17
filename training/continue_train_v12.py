from __future__ import annotations

import argparse
import json
import math
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from jarvis.neural.transformer import JarvisTransformer  # noqa: E402
from training.numpy_trainer import NumpyTransformerTrainer  # noqa: E402

FOCUS_STAGES = (1, 2, 3, 4, 11)


def load_sequences(path: Path) -> tuple[dict[int, list[tuple[list[int], int]]], dict[int, list[tuple[list[int], int]]]]:
    curated = {stage: [] for stage in FOCUS_STAGES}
    rehearsal = {stage: [] for stage in FOCUS_STAGES}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        stage = int(row.get("stage", 0))
        if stage not in curated:
            continue
        sample = ([int(token) for token in row["tokens"]], int(row["prompt_length"]))
        if str(row.get("curated_source_id", "")).strip():
            curated[stage].append(sample)
        else:
            rehearsal[stage].append(sample)
    missing = [stage for stage in FOCUS_STAGES if not curated[stage] or not rehearsal[stage]]
    if missing:
        raise RuntimeError(f"Missing curated/rehearsal data for stages: {missing}")
    return curated, rehearsal


def make_batch(sample: tuple[list[int], int], length: int, pad_id: int = 0) -> tuple[np.ndarray, np.ndarray]:
    tokens, prompt_length = sample
    inputs = tokens[:-1][:length]
    labels = tokens[1:length + 1]
    assistant_boundary = max(0, prompt_length - 2)
    labels = [token if index >= assistant_boundary else -100 for index, token in enumerate(labels)]
    if not any(token >= 0 for token in labels):
        # Keep the response tail if a very long prompt consumed the crop.
        start = max(0, min(len(tokens) - length - 1, assistant_boundary - max(1, length // 3)))
        inputs = tokens[start:start + length]
        raw_labels = tokens[start + 1:start + length + 1]
        labels = [token if (start + index) >= assistant_boundary else -100 for index, token in enumerate(raw_labels)]
    if len(inputs) < length:
        padding = length - len(inputs)
        inputs.extend([pad_id] * padding)
        labels.extend([-100] * padding)
    return np.asarray(inputs, dtype=np.int64)[None, :], np.asarray(labels, dtype=np.int64)[None, :]


def mean_eval(trainer: NumpyTransformerTrainer, samples: list[tuple[list[int], int]], length: int, limit: int = 12) -> float:
    chosen = samples[: max(1, min(limit, len(samples)))]
    return float(np.mean([trainer.evaluate_batch(*make_batch(row, length)) for row in chosen]))


def train(args: argparse.Namespace) -> dict[str, Any]:
    source = ROOT / args.source
    output = ROOT / args.output
    tokenized = ROOT / "datasets" / "tokenized" / "dataset_v004_train.jsonl"
    model = JarvisTransformer.load(source, dequantize=True)
    length = min(int(args.sequence_length), model.config.max_seq_len)
    curated, rehearsal = load_sequences(tokenized)
    total_steps = int(args.steps_per_stage) * len(FOCUS_STAGES)
    trainer = NumpyTransformerTrainer(
        model,
        learning_rate=float(args.learning_rate),
        minimum_learning_rate=max(0.000006, float(args.learning_rate) / 8.0),
        weight_decay=float(args.weight_decay),
        gradient_clip=float(args.gradient_clip),
        warmup_steps=min(12, max(1, total_steps // 12)),
        total_steps=total_steps,
    )
    rng = np.random.default_rng(model.config.seed + 1200)
    before = {
        str(stage): {
            "curated": round(mean_eval(trainer, curated[stage], length), 6),
            "rehearsal": round(mean_eval(trainer, rehearsal[stage], length), 6),
        }
        for stage in FOCUS_STAGES
    }
    stage_metrics: list[dict[str, Any]] = []
    all_losses: list[float] = []
    started = time.perf_counter()
    for stage in FOCUS_STAGES:
        losses: list[float] = []
        gradients: list[float] = []
        for local_step in range(1, int(args.steps_per_stage) + 1):
            use_curated = float(rng.random()) < float(args.curated_ratio)
            pool = curated[stage] if use_curated else rehearsal[stage]
            sample = pool[int(rng.integers(0, len(pool)))]
            loss, rate, gradient_norm = trainer.train_batch(*make_batch(sample, length))
            losses.append(float(loss)); all_losses.append(float(loss)); gradients.append(float(gradient_norm))
            if local_step == 1 or local_step % 16 == 0 or local_step == int(args.steps_per_stage):
                print(json.dumps({
                    "event": "continual_training_step", "stage": stage,
                    "local_step": local_step, "global_step": trainer.step,
                    "recent_loss": round(float(np.mean(losses[-16:])), 6),
                    "learning_rate": rate, "gradient_norm": round(float(gradient_norm), 6),
                }, ensure_ascii=False), flush=True)
        stage_metrics.append({
            "stage": stage,
            "steps": len(losses),
            "mean_loss": round(float(np.mean(losses)), 6),
            "mean_gradient_norm": round(float(np.mean(gradients)), 6),
        })
    after = {
        str(stage): {
            "curated": round(mean_eval(trainer, curated[stage], length), 6),
            "rehearsal": round(mean_eval(trainer, rehearsal[stage], length), 6),
        }
        for stage in FOCUS_STAGES
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    metadata = {
        **model.metadata,
        "model_id": "jarvis_nano_v12_candidate",
        "training_complete": True,
        "dataset_version": "dataset_v004",
        "pretrained_source": None,
        "backend": "numpy_cpu_full_gradient",
        "continual_training": "curated_100_v12_with_v003_rehearsal",
        "continual_focus_stages": list(FOCUS_STAGES),
        "continual_steps": total_steps,
        "step": int(model.metadata.get("step", 0) or 0) + total_steps,
        "trained_at_utc": datetime.now(UTC).isoformat(),
    }
    model.save_quantized(output, metadata)
    curated_before = float(np.mean([before[str(s)]["curated"] for s in FOCUS_STAGES]))
    curated_after = float(np.mean([after[str(s)]["curated"] for s in FOCUS_STAGES]))
    rehearsal_before = float(np.mean([before[str(s)]["rehearsal"] for s in FOCUS_STAGES]))
    rehearsal_after = float(np.mean([after[str(s)]["rehearsal"] for s in FOCUS_STAGES]))
    report = {
        "format": "jarvis-continual-training-v12-v1",
        "source": args.source,
        "output": args.output,
        "parameters": model.parameter_count,
        "dataset_version": "dataset_v004",
        "focus_stages": list(FOCUS_STAGES),
        "steps_per_stage": int(args.steps_per_stage),
        "total_steps": total_steps,
        "sequence_length": length,
        "curated_ratio": float(args.curated_ratio),
        "learning_rate": float(args.learning_rate),
        "weight_decay": float(args.weight_decay),
        "gradient_clip": float(args.gradient_clip),
        "mean_training_loss": round(float(np.mean(all_losses)), 6),
        "probe_loss_before": before,
        "probe_loss_after": after,
        "probe_summary": {
            "curated_mean_before": round(curated_before, 6),
            "curated_mean_after": round(curated_after, 6),
            "curated_improvement_pct": round(100.0 * (curated_before - curated_after) / max(curated_before, 1e-9), 3),
            "rehearsal_mean_before": round(rehearsal_before, 6),
            "rehearsal_mean_after": round(rehearsal_after, 6),
            "rehearsal_change_pct": round(100.0 * (rehearsal_after - rehearsal_before) / max(rehearsal_before, 1e-9), 3),
        },
        "stage_metrics": stage_metrics,
        "duration_seconds": round(time.perf_counter() - started, 3),
    }
    report_path = ROOT / "models" / "continual_training_v12.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
    return report


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    p.add_argument("--source", default="models/jarvis_nano_v08.npz")
    p.add_argument("--output", default="models/jarvis_nano_v12_candidate.npz")
    p.add_argument("--steps-per-stage", type=int, default=48)
    p.add_argument("--sequence-length", type=int, default=64)
    p.add_argument("--learning-rate", type=float, default=0.00008)
    p.add_argument("--weight-decay", type=float, default=0.004)
    p.add_argument("--gradient-clip", type=float, default=0.8)
    p.add_argument("--curated-ratio", type=float, default=0.72)
    return p


if __name__ == "__main__":
    train(parser().parse_args())
