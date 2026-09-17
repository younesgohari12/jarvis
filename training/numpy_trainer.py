from __future__ import annotations

import json
import math
import os
import platform
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import numpy as np

from jarvis.neural.config import TransformerConfig
from jarvis.neural.transformer import JarvisTransformer


LogCallback = Callable[[dict[str, Any]], None]


@dataclass(slots=True)
class TrainingControl:
    stop: threading.Event = field(default_factory=threading.Event)
    pause: threading.Event = field(default_factory=threading.Event)
    save_checkpoint: threading.Event = field(default_factory=threading.Event)

    def wait_if_paused(self) -> None:
        while self.pause.is_set() and not self.stop.is_set():
            time.sleep(0.1)


def _rms_forward(
    value: np.ndarray, weight: np.ndarray, epsilon: float,
) -> tuple[np.ndarray, np.ndarray]:
    inverse = 1.0 / np.sqrt(np.mean(value * value, axis=-1, keepdims=True) + epsilon)
    return value * inverse * weight, inverse


def _rms_backward(
    gradient: np.ndarray,
    value: np.ndarray,
    weight: np.ndarray,
    inverse: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    scaled = gradient * weight
    projection = np.mean(scaled * value, axis=-1, keepdims=True)
    input_gradient = scaled * inverse - value * (inverse ** 3) * projection
    axes = tuple(range(gradient.ndim - 1))
    weight_gradient = np.sum(gradient * value * inverse, axis=axes)
    return input_gradient, weight_gradient


def _silu_forward(value: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    sigmoid = 1.0 / (1.0 + np.exp(-np.clip(value, -30.0, 30.0)))
    return value * sigmoid, sigmoid


def _silu_backward(gradient: np.ndarray, value: np.ndarray, sigmoid: np.ndarray) -> np.ndarray:
    return gradient * (sigmoid + value * sigmoid * (1.0 - sigmoid))


def _rope(
    value: np.ndarray, positions: np.ndarray, theta: float,
) -> tuple[np.ndarray, tuple[np.ndarray, np.ndarray]]:
    head_dim = value.shape[-1]
    inverse = 1.0 / (theta ** (np.arange(0, head_dim, 2, dtype=np.float32) / head_dim))
    angles = positions.astype(np.float32)[:, None] * inverse[None, :]
    cosine = np.cos(angles)[None, :, None, :]
    sine = np.sin(angles)[None, :, None, :]
    even = value[..., 0::2]
    odd = value[..., 1::2]
    output = np.empty_like(value)
    output[..., 0::2] = even * cosine - odd * sine
    output[..., 1::2] = even * sine + odd * cosine
    return output, (cosine, sine)


def _rope_backward(
    gradient: np.ndarray, tables: tuple[np.ndarray, np.ndarray],
) -> np.ndarray:
    cosine, sine = tables
    even_gradient = gradient[..., 0::2]
    odd_gradient = gradient[..., 1::2]
    output = np.empty_like(gradient)
    output[..., 0::2] = even_gradient * cosine + odd_gradient * sine
    output[..., 1::2] = -even_gradient * sine + odd_gradient * cosine
    return output


def _softmax(value: np.ndarray) -> np.ndarray:
    shifted = value - np.max(value, axis=-1, keepdims=True)
    exponent = np.exp(shifted)
    return exponent / np.sum(exponent, axis=-1, keepdims=True)


class NumpyTransformerTrainer:
    """Full-gradient CPU trainer for the exact JARVIS NumPy Transformer.

    This backend is intentionally dependency-light and is used for the bundled
    from-scratch checkpoint. The optional PyTorch backend is recommended for
    larger profiles and CUDA training.
    """

    def __init__(
        self,
        model: JarvisTransformer,
        *,
        learning_rate: float,
        minimum_learning_rate: float,
        weight_decay: float,
        gradient_clip: float,
        warmup_steps: int,
        total_steps: int,
    ) -> None:
        if model.config.n_heads != model.config.n_kv_heads:
            raise ValueError("NumPy training currently requires full multi-head attention")
        self.model = model
        self.learning_rate = learning_rate
        self.minimum_learning_rate = minimum_learning_rate
        self.weight_decay = weight_decay
        self.gradient_clip = gradient_clip
        self.warmup_steps = warmup_steps
        self.total_steps = max(1, total_steps)
        self.step = 0
        self.first_moment = {
            name: np.zeros_like(value, dtype=np.float32)
            for name, value in model.parameters.items()
        }
        self.second_moment = {
            name: np.zeros_like(value, dtype=np.float32)
            for name, value in model.parameters.items()
        }

    def _forward(
        self, input_ids: np.ndarray, target_ids: np.ndarray,
    ) -> tuple[float, dict[str, Any]]:
        p = self.model.parameters
        c = self.model.config
        batch, length = input_ids.shape
        positions = np.arange(length, dtype=np.float32)
        causal = np.triu(np.full((length, length), -1e9, dtype=np.float32), 1)
        x = p["tok_embeddings"][input_ids]
        layers: list[dict[str, Any]] = []
        for layer in range(c.n_layers):
            prefix = f"layers.{layer}"
            layer_input = x
            n1, inverse1 = _rms_forward(x, p[f"{prefix}.attn_norm"], c.rms_epsilon)
            query_linear = n1 @ p[f"{prefix}.wq"]
            key_linear = n1 @ p[f"{prefix}.wk"]
            value_linear = n1 @ p[f"{prefix}.wv"]
            query_raw = query_linear.reshape(batch, length, c.n_heads, c.head_dim)
            key_raw = key_linear.reshape(batch, length, c.n_heads, c.head_dim)
            value = value_linear.reshape(batch, length, c.n_heads, c.head_dim)
            query, query_tables = _rope(query_raw, positions, c.rope_theta)
            key, key_tables = _rope(key_raw, positions, c.rope_theta)
            scores = np.einsum("bthd,bshd->bhts", query, key, optimize=True)
            scores = scores / math.sqrt(c.head_dim) + causal[None, None, :, :]
            probabilities = _softmax(scores)
            attention_heads = np.einsum(
                "bhts,bshd->bthd", probabilities, value, optimize=True
            )
            attention = attention_heads.reshape(batch, length, c.d_model)
            middle = x + attention @ p[f"{prefix}.wo"]
            n2, inverse2 = _rms_forward(
                middle, p[f"{prefix}.ffn_norm"], c.rms_epsilon
            )
            gate_linear = n2 @ p[f"{prefix}.w1"]
            gate, sigmoid = _silu_forward(gate_linear)
            up = n2 @ p[f"{prefix}.w3"]
            hidden = gate * up
            x = middle + hidden @ p[f"{prefix}.w2"]
            layers.append(
                {
                    "input": layer_input,
                    "n1": n1,
                    "inverse1": inverse1,
                    "query": query,
                    "key": key,
                    "value": value,
                    "query_tables": query_tables,
                    "key_tables": key_tables,
                    "probabilities": probabilities,
                    "attention": attention,
                    "middle": middle,
                    "n2": n2,
                    "inverse2": inverse2,
                    "gate_linear": gate_linear,
                    "gate": gate,
                    "sigmoid": sigmoid,
                    "up": up,
                    "hidden": hidden,
                }
            )
        final_input = x
        normalized, final_inverse = _rms_forward(x, p["final_norm"], c.rms_epsilon)
        output_weight = p["tok_embeddings"].T if c.tie_embeddings else p["lm_head"]
        logits = normalized @ output_weight
        flat_logits = logits.reshape(-1, c.vocab_size)
        flat_targets = target_ids.reshape(-1)
        valid = flat_targets >= 0
        if not np.any(valid):
            raise ValueError("Training batch has no assistant target tokens")
        maximum = np.max(flat_logits, axis=1, keepdims=True)
        exponent = np.exp(flat_logits - maximum)
        probabilities = exponent / np.sum(exponent, axis=1, keepdims=True)
        selected = probabilities[np.arange(flat_targets.size)[valid], flat_targets[valid]]
        loss = -float(np.mean(np.log(np.maximum(selected, 1e-12))))
        return loss, {
            "input_ids": input_ids,
            "targets": flat_targets,
            "valid_targets": valid,
            "layers": layers,
            "final_input": final_input,
            "final_inverse": final_inverse,
            "normalized": normalized,
            "probabilities": probabilities,
        }

    def _backward(self, cache: dict[str, Any]) -> dict[str, np.ndarray]:
        p = self.model.parameters
        c = self.model.config
        gradients = {name: np.zeros_like(value) for name, value in p.items()}
        probabilities = cache["probabilities"]
        targets = cache["targets"]
        valid = cache["valid_targets"]
        invalid = ~valid
        probabilities[invalid] = 0.0
        probabilities[np.arange(targets.size)[valid], targets[valid]] -= 1.0
        probabilities /= int(np.sum(valid))
        normalized = cache["normalized"]
        flat_normalized = normalized.reshape(-1, c.d_model)
        output_gradient = probabilities
        if c.tie_embeddings:
            gradients["tok_embeddings"] += output_gradient.T @ flat_normalized
            normalized_gradient = output_gradient @ p["tok_embeddings"]
        else:
            gradients["lm_head"] = flat_normalized.T @ output_gradient
            normalized_gradient = output_gradient @ p["lm_head"].T
        normalized_gradient = normalized_gradient.reshape(normalized.shape)
        x_gradient, gradients["final_norm"] = _rms_backward(
            normalized_gradient,
            cache["final_input"],
            p["final_norm"],
            cache["final_inverse"],
        )
        scale = math.sqrt(c.head_dim)
        for layer in reversed(range(c.n_layers)):
            prefix = f"layers.{layer}"
            data = cache["layers"][layer]

            middle_gradient = x_gradient.copy()
            hidden_gradient = x_gradient @ p[f"{prefix}.w2"].T
            gradients[f"{prefix}.w2"] = data["hidden"].reshape(-1, c.ffn_hidden).T @ x_gradient.reshape(-1, c.d_model)
            gate_gradient = hidden_gradient * data["up"]
            up_gradient = hidden_gradient * data["gate"]
            gate_linear_gradient = _silu_backward(
                gate_gradient, data["gate_linear"], data["sigmoid"]
            )
            gradients[f"{prefix}.w1"] = data["n2"].reshape(-1, c.d_model).T @ gate_linear_gradient.reshape(-1, c.ffn_hidden)
            gradients[f"{prefix}.w3"] = data["n2"].reshape(-1, c.d_model).T @ up_gradient.reshape(-1, c.ffn_hidden)
            n2_gradient = (
                gate_linear_gradient @ p[f"{prefix}.w1"].T
                + up_gradient @ p[f"{prefix}.w3"].T
            )
            rms_gradient, gradients[f"{prefix}.ffn_norm"] = _rms_backward(
                n2_gradient, data["middle"], p[f"{prefix}.ffn_norm"], data["inverse2"]
            )
            middle_gradient += rms_gradient

            layer_input_gradient = middle_gradient.copy()
            attention_gradient = middle_gradient @ p[f"{prefix}.wo"].T
            gradients[f"{prefix}.wo"] = data["attention"].reshape(-1, c.d_model).T @ middle_gradient.reshape(-1, c.d_model)
            attention_heads_gradient = attention_gradient.reshape(
                *attention_gradient.shape[:-1], c.n_heads, c.head_dim
            )
            probabilities = data["probabilities"]
            value = data["value"]
            probability_gradient = np.einsum(
                "bthd,bshd->bhts", attention_heads_gradient, value, optimize=True
            )
            value_gradient = np.einsum(
                "bhts,bthd->bshd", probabilities, attention_heads_gradient, optimize=True
            )
            score_gradient = probabilities * (
                probability_gradient
                - np.sum(probability_gradient * probabilities, axis=-1, keepdims=True)
            )
            query_gradient = np.einsum(
                "bhts,bshd->bthd", score_gradient, data["key"], optimize=True
            ) / scale
            key_gradient = np.einsum(
                "bhts,bthd->bshd", score_gradient, data["query"], optimize=True
            ) / scale
            query_gradient = _rope_backward(query_gradient, data["query_tables"])
            key_gradient = _rope_backward(key_gradient, data["key_tables"])
            query_gradient = query_gradient.reshape(*data["n1"].shape[:-1], c.d_model)
            key_gradient = key_gradient.reshape(*data["n1"].shape[:-1], c.d_model)
            value_gradient = value_gradient.reshape(*data["n1"].shape[:-1], c.d_model)
            n1_flat = data["n1"].reshape(-1, c.d_model)
            gradients[f"{prefix}.wq"] = n1_flat.T @ query_gradient.reshape(-1, c.d_model)
            gradients[f"{prefix}.wk"] = n1_flat.T @ key_gradient.reshape(-1, c.d_model)
            gradients[f"{prefix}.wv"] = n1_flat.T @ value_gradient.reshape(-1, c.d_model)
            n1_gradient = (
                query_gradient @ p[f"{prefix}.wq"].T
                + key_gradient @ p[f"{prefix}.wk"].T
                + value_gradient @ p[f"{prefix}.wv"].T
            )
            rms_gradient, gradients[f"{prefix}.attn_norm"] = _rms_backward(
                n1_gradient, data["input"], p[f"{prefix}.attn_norm"], data["inverse1"]
            )
            x_gradient = layer_input_gradient + rms_gradient
        np.add.at(gradients["tok_embeddings"], cache["input_ids"], x_gradient)
        return gradients

    def _scheduled_learning_rate(self) -> float:
        if self.step < self.warmup_steps:
            return self.learning_rate * (self.step + 1) / max(1, self.warmup_steps)
        progress = (self.step - self.warmup_steps) / max(
            1, self.total_steps - self.warmup_steps
        )
        cosine = 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))
        return self.minimum_learning_rate + (
            self.learning_rate - self.minimum_learning_rate
        ) * cosine

    def train_batch(self, input_ids: np.ndarray, target_ids: np.ndarray) -> tuple[float, float, float]:
        loss, cache = self._forward(input_ids, target_ids)
        gradients = self._backward(cache)
        global_norm = math.sqrt(
            sum(float(np.sum(gradient * gradient)) for gradient in gradients.values())
        )
        if global_norm > self.gradient_clip > 0:
            factor = self.gradient_clip / max(global_norm, 1e-12)
            for gradient in gradients.values():
                gradient *= factor
        rate = self._scheduled_learning_rate()
        self.step += 1
        beta1, beta2 = 0.9, 0.95
        correction1 = 1.0 - beta1 ** self.step
        correction2 = 1.0 - beta2 ** self.step
        for name, value in self.model.parameters.items():
            gradient = gradients[name]
            if value.ndim > 1 and self.weight_decay:
                gradient = gradient + self.weight_decay * value
            first = self.first_moment[name]
            second = self.second_moment[name]
            first *= beta1
            first += (1.0 - beta1) * gradient
            second *= beta2
            second += (1.0 - beta2) * gradient * gradient
            value -= rate * (first / correction1) / (np.sqrt(second / correction2) + 1e-8)
        return loss, rate, global_norm

    def evaluate_batch(self, input_ids: np.ndarray, target_ids: np.ndarray) -> float:
        loss, _ = self._forward(input_ids, target_ids)
        return loss


class CurriculumRunner:
    STAGE_NAMES = {
        1: "language_foundations", 2: "conversation", 3: "general_knowledge",
        4: "instruction_following", 5: "action_recognition", 6: "entity_extraction",
        7: "tool_calling", 8: "argument_extraction", 9: "context_memory",
        10: "planning", 11: "reasoning", 12: "recovery_verification",
        13: "search_decision", 14: "jarvis_personality",
    }

    def __init__(
        self,
        root: Path,
        model: JarvisTransformer,
        *,
        steps_per_stage: int,
        sequence_length: int,
        log_callback: LogCallback | None = None,
        control: TrainingControl | None = None,
        retain_full_checkpoints: int = 1,
        dataset_version: str = "dataset_v002",
        tokenized_file: Path | None = None,
        checkpoint_root: Path | None = None,
        metrics_filename: str = "jarvis_nano_training_metrics.json",
        save_stage_weights: bool = True,
    ) -> None:
        self.root = root
        self.model = model
        self.steps_per_stage = max(1, steps_per_stage)
        self.sequence_length = max(8, min(sequence_length, model.config.max_seq_len))
        self.log_callback = log_callback
        self.control = control or TrainingControl()
        self.retain_full_checkpoints = max(1, retain_full_checkpoints)
        self.dataset_version = str(dataset_version)
        self.tokenized_file = tokenized_file or (
            root / "datasets" / "tokenized" / "dataset_v002_train.jsonl"
        )
        self.checkpoint_root = checkpoint_root or (root / "models" / "checkpoints")
        self.metrics_filename = metrics_filename
        self.save_stage_weights = bool(save_stage_weights)
        total = self.steps_per_stage * len(self.STAGE_NAMES)
        t = model.config.training
        self.trainer = NumpyTransformerTrainer(
            model,
            learning_rate=t.learning_rate,
            minimum_learning_rate=t.minimum_learning_rate,
            weight_decay=t.weight_decay,
            gradient_clip=t.gradient_clip,
            warmup_steps=min(t.warmup_steps, max(1, total // 4)),
            total_steps=total,
        )
        self.rng = np.random.default_rng(model.config.seed + 17)
        self.stage_sequences = self._load_sequences()
        self.log_directory = root / "logs" / "training"
        self.log_directory.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        self.jsonl_path = self.log_directory / f"train_{stamp}.jsonl"
        self.text_path = self.log_directory / f"train_{stamp}.log"

    def _load_sequences(self) -> dict[int, list[tuple[list[int], int]]]:
        path = self.tokenized_file
        values: dict[int, list[tuple[list[int], int]]] = {
            stage: [] for stage in self.STAGE_NAMES
        }
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line:
                continue
            row = json.loads(line)
            values[int(row["stage"])].append(
                ([int(token) for token in row["tokens"]], int(row["prompt_length"]))
            )
        missing = [stage for stage, rows in values.items() if not rows]
        if missing:
            raise RuntimeError(f"Missing tokenized curriculum stages: {missing}")
        return values

    def _batch(self, stage: int) -> tuple[np.ndarray, np.ndarray]:
        inputs: list[int] = []
        labels: list[int] = []
        while len(inputs) < self.sequence_length:
            # Rehearsal prevents later curriculum stages from erasing earlier
            # language and conversation behavior.
            selected_stage = (
                stage
                if stage == 1 or float(self.rng.random()) < 0.72
                else int(self.rng.integers(1, stage + 1))
            )
            sequences = self.stage_sequences[selected_stage]
            sequence, prompt_length = sequences[int(self.rng.integers(0, len(sequences)))]
            item_inputs = sequence[:-1]
            item_labels = sequence[1:]
            assistant_boundary = max(0, prompt_length - 2)
            item_labels = [
                token if index >= assistant_boundary else -100
                for index, token in enumerate(item_labels)
            ]
            inputs.extend(item_inputs)
            labels.extend(item_labels)
        start = 0
        maximum_start = len(inputs) - self.sequence_length
        if maximum_start > 0:
            # A crop is valid only when it retains at least one assistant label.
            candidates = [
                value
                for value in range(maximum_start + 1)
                if any(token >= 0 for token in labels[value : value + self.sequence_length])
            ]
            start = candidates[int(self.rng.integers(0, len(candidates)))]
        return (
            np.asarray(inputs[start : start + self.sequence_length], dtype=np.int64)[None, :],
            np.asarray(labels[start : start + self.sequence_length], dtype=np.int64)[None, :],
        )

    def _emit(self, event: dict[str, Any]) -> None:
        payload = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "backend": "numpy_cpu_full_gradient",
            **event,
        }
        with self.jsonl_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, ensure_ascii=False) + "\n")
        with self.text_path.open("a", encoding="utf-8") as handle:
            fields = (
                "model_config", "dataset_version", "hardware", "parameters", "stage", "stage_name", "step",
                "loss", "validation_loss", "learning_rate", "gradient_norm",
                "tokens_per_second", "checkpoint", "device", "error",
            )
            details = " ".join(
                f"{name}={payload[name]}" for name in fields if name in payload
            )
            handle.write(f"[{payload['timestamp']}] {payload.get('event')} {details}\n")
        if self.log_callback:
            self.log_callback(payload)

    def _checkpoint(self, stage: int, *, final: bool = False) -> Path:
        name = self.STAGE_NAMES[stage]
        directory = self.checkpoint_root / f"stage{stage}_{name}"
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / "model.npz"
        metadata = {
            "training_complete": final,
            "dataset_version": self.dataset_version,
            "training_stages": list(range(1, stage + 1)),
            "checkpoint_parent": None if stage == 1 else f"stage{stage - 1}_{self.STAGE_NAMES[stage - 1]}",
            "backend": "numpy_cpu_full_gradient",
            "optimizer": "AdamW",
            "step": self.trainer.step,
            "trained_at_utc": datetime.now(timezone.utc).isoformat(),
        }
        if final or self.save_stage_weights:
            self.model.save_quantized(path, metadata)
        manifest = {
            **metadata,
            "model_file": path.name if path.is_file() else None,
            "size_bytes": path.stat().st_size if path.is_file() else 0,
            "sha256": self.model.file_sha256(path) if path.is_file() else None,
        }
        (directory / "checkpoint.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        # Release builds keep full weights for the latest requested stages while
        # preserving honest manifests for earlier curriculum boundaries.
        if path.is_file() and not final and stage <= len(self.STAGE_NAMES) - self.retain_full_checkpoints:
            archive = self.checkpoint_root / "retired_full_weights"
            archive.mkdir(parents=True, exist_ok=True)
            path.replace(archive / f"stage{stage}_{name}.npz")
            manifest["model_file"] = f"../retired_full_weights/stage{stage}_{name}.npz"
            (directory / "checkpoint.json").write_text(
                json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
            )
        return directory

    def run(self) -> dict[str, Any]:
        started = time.perf_counter()
        stage_metrics: list[dict[str, Any]] = []
        initial_input, initial_target = self._batch(1)
        initial_loss = self.trainer.evaluate_batch(initial_input, initial_target)
        self._emit(
            {
                "event": "training_start",
                "model_config": self.model.config.to_dict(),
                "dataset_version": self.dataset_version,
                "hardware": {
                    "platform": platform.platform(),
                    "cpu_count": os.cpu_count() or 1,
                    "device": "cpu",
                    "cuda": False,
                },
                "parameters": self.model.parameter_count,
                "initial_loss": round(initial_loss, 6),
            }
        )
        for stage, name in self.STAGE_NAMES.items():
            losses: list[float] = []
            stage_started = time.perf_counter()
            for local_step in range(1, self.steps_per_stage + 1):
                self.control.wait_if_paused()
                if self.control.stop.is_set():
                    self._emit({"event": "training_stopped", "stage": stage, "step": self.trainer.step})
                    return {"stopped": True, "stage_metrics": stage_metrics}
                input_ids, target_ids = self._batch(stage)
                step_started = time.perf_counter()
                loss, rate, gradient_norm = self.trainer.train_batch(input_ids, target_ids)
                duration = max(time.perf_counter() - step_started, 1e-9)
                losses.append(loss)
                event = {
                    "event": "step",
                    "stage": stage,
                    "stage_name": name,
                    "stage_step": local_step,
                    "step": self.trainer.step,
                    "loss": round(loss, 6),
                    "learning_rate": rate,
                    "gradient_norm": round(gradient_norm, 6),
                    "tokens_per_second": round(self.sequence_length / duration, 3),
                    "elapsed_seconds": round(time.perf_counter() - started, 3),
                }
                self._emit(event)
                if self.control.save_checkpoint.is_set():
                    self._checkpoint(stage)
                    self.control.save_checkpoint.clear()
            validation_input, validation_target = self._batch(stage)
            validation_loss = self.trainer.evaluate_batch(validation_input, validation_target)
            checkpoint = self._checkpoint(stage, final=stage == len(self.STAGE_NAMES))
            metric = {
                "stage": stage,
                "name": name,
                "steps": self.steps_per_stage,
                "mean_train_loss": round(float(np.mean(losses)), 6),
                "last_train_loss": round(losses[-1], 6),
                "validation_loss": round(validation_loss, 6),
                "seconds": round(time.perf_counter() - stage_started, 3),
                "checkpoint": str(checkpoint.relative_to(self.root)),
            }
            stage_metrics.append(metric)
            self._emit({"event": "stage_complete", **metric})
        elapsed = time.perf_counter() - started
        final_loss = stage_metrics[-1]["validation_loss"]
        metrics = {
            "format": "jarvis-training-metrics-v1",
            "model": self.model.config.name,
            "architecture": self.model.config.architecture,
            "parameters": self.model.parameter_count,
            "dataset_version": self.dataset_version,
            "pretrained_source": None,
            "initial_loss": round(initial_loss, 6),
            "final_validation_loss": final_loss,
            "final_perplexity": round(float(math.exp(min(20.0, final_loss))), 6),
            "training_steps": self.trainer.step,
            "training_seconds": round(elapsed, 3),
            "device": "cpu",
            "backend": "numpy_cpu_full_gradient",
            "stage_metrics": stage_metrics,
            "completed_at_utc": datetime.now(timezone.utc).isoformat(),
            "jsonl_log": str(self.jsonl_path.relative_to(self.root)),
        }
        (self.root / "models" / self.metrics_filename).write_text(
            json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        self._emit({"event": "training_complete", **metrics})
        return metrics
