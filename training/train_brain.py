from __future__ import annotations

import argparse
import base64
import gzip
import json
import math
import os
import random
import sys
import tempfile
from array import array
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from jarvis.utils.text import hashed_features  # noqa: E402
from training.build_dataset import build as build_dataset  # noqa: E402


@dataclass(slots=True)
class Sample:
    text: str
    target: int
    features: dict[int, float]


@dataclass(slots=True)
class Network:
    feature_size: int
    embedding_size: int
    hidden_size: int
    output_size: int
    embedding: list[float]
    hidden: list[float]
    hidden_bias: list[float]
    output: list[float]
    output_bias: list[float]


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object in {path}")
    return value


def load_split(
    path: Path,
    intent_index: dict[str, int],
    feature_size: int,
    augment: bool = False,
) -> list[Sample]:
    payload = read_json(path)
    samples: list[Sample] = []
    seen: set[tuple[int, str]] = set()
    for item in payload.get("intents", []):
        target = intent_index[str(item["tag"])]
        route_type = str(item.get("route_type", "conversation"))
        for raw in item.get("examples", []):
            text = str(raw).strip()
            variants = [text]
            if augment:
                plain = text.rstrip("?!؟.!، ")
                if plain != text:
                    variants.append(plain)
                if route_type == "tool":
                    if any("\u0600" <= char <= "\u06ff" for char in text):
                        variants.append(f"لطفاً {plain}")
                    else:
                        variants.append(f"please {plain}")
            for variant in variants:
                key = (target, variant.casefold())
                if variant and key not in seen:
                    seen.add(key)
                    samples.append(Sample(variant, target, hashed_features(variant, feature_size)))
    return samples


def initialize(config: dict[str, Any], output_size: int) -> Network:
    feature_size = int(config["feature_size"])
    embedding_size = int(config["embedding_size"])
    hidden_size = int(config["hidden_size"])
    rng = random.Random(int(config["training"]["seed"]))
    embedding_limit = 0.018
    hidden_limit = math.sqrt(6.0 / (embedding_size + hidden_size))
    output_limit = math.sqrt(6.0 / (hidden_size + output_size))
    return Network(
        feature_size,
        embedding_size,
        hidden_size,
        output_size,
        [rng.uniform(-embedding_limit, embedding_limit) for _ in range(feature_size * embedding_size)],
        [rng.uniform(-hidden_limit, hidden_limit) for _ in range(embedding_size * hidden_size)],
        [0.0] * hidden_size,
        [rng.uniform(-output_limit, output_limit) for _ in range(hidden_size * output_size)],
        [0.0] * output_size,
    )


def forward(network: Network, features: dict[int, float]) -> tuple[list[float], list[float], list[float]]:
    pooled = [0.0] * network.embedding_size
    for feature_index, feature_value in features.items():
        offset = feature_index * network.embedding_size
        for embedding_index in range(network.embedding_size):
            pooled[embedding_index] += network.embedding[offset + embedding_index] * feature_value
    hidden = network.hidden_bias.copy()
    for embedding_index, value in enumerate(pooled):
        offset = embedding_index * network.hidden_size
        for hidden_index in range(network.hidden_size):
            hidden[hidden_index] += network.hidden[offset + hidden_index] * value
    hidden = [math.tanh(value) for value in hidden]
    logits = network.output_bias.copy()
    for hidden_index, value in enumerate(hidden):
        offset = hidden_index * network.output_size
        for output_index in range(network.output_size):
            logits[output_index] += network.output[offset + output_index] * value
    peak = max(logits)
    probabilities = [math.exp(value - peak) for value in logits]
    total = sum(probabilities) or 1.0
    return pooled, hidden, [value / total for value in probabilities]


def train_epoch(
    network: Network,
    samples: list[Sample],
    learning_rate: float,
    l2: float,
    rng: random.Random,
) -> float:
    rng.shuffle(samples)
    loss = 0.0
    for sample in samples:
        pooled, hidden_values, probabilities = forward(network, sample.features)
        loss -= math.log(max(1e-12, probabilities[sample.target]))
        output_gradient = probabilities
        output_gradient[sample.target] -= 1.0

        hidden_gradient = [0.0] * network.hidden_size
        for hidden_index in range(network.hidden_size):
            offset = hidden_index * network.output_size
            gradient = sum(
                output_gradient[out] * network.output[offset + out]
                for out in range(network.output_size)
            )
            hidden_gradient[hidden_index] = gradient * (1.0 - hidden_values[hidden_index] ** 2)

        pooled_gradient = [0.0] * network.embedding_size
        for embedding_index in range(network.embedding_size):
            offset = embedding_index * network.hidden_size
            pooled_gradient[embedding_index] = sum(
                hidden_gradient[hidden_index] * network.hidden[offset + hidden_index]
                for hidden_index in range(network.hidden_size)
            )

        for hidden_index, hidden_value in enumerate(hidden_values):
            offset = hidden_index * network.output_size
            for output_index in range(network.output_size):
                index = offset + output_index
                gradient = output_gradient[output_index] * hidden_value + l2 * network.output[index]
                network.output[index] -= learning_rate * gradient
        for output_index in range(network.output_size):
            network.output_bias[output_index] -= learning_rate * output_gradient[output_index]

        for embedding_index, pooled_value in enumerate(pooled):
            offset = embedding_index * network.hidden_size
            for hidden_index in range(network.hidden_size):
                index = offset + hidden_index
                gradient = hidden_gradient[hidden_index] * pooled_value + l2 * network.hidden[index]
                network.hidden[index] -= learning_rate * gradient
        for hidden_index in range(network.hidden_size):
            network.hidden_bias[hidden_index] -= learning_rate * hidden_gradient[hidden_index]

        for feature_index, feature_value in sample.features.items():
            offset = feature_index * network.embedding_size
            for embedding_index in range(network.embedding_size):
                index = offset + embedding_index
                gradient = pooled_gradient[embedding_index] * feature_value + l2 * network.embedding[index]
                network.embedding[index] -= learning_rate * gradient
    return loss / max(1, len(samples))


def neural_accuracy(network: Network, samples: list[Sample]) -> float:
    if not samples:
        return 0.0
    correct = 0
    for sample in samples:
        _, _, probabilities = forward(network, sample.features)
        correct += int(max(range(len(probabilities)), key=probabilities.__getitem__) == sample.target)
    return correct / len(samples)


def build_centroids(samples: list[Sample], output_size: int) -> list[dict[int, float]]:
    sums: list[dict[int, float]] = [{} for _ in range(output_size)]
    counts = [0] * output_size
    for sample in samples:
        counts[sample.target] += 1
        target = sums[sample.target]
        for index, value in sample.features.items():
            target[index] = target.get(index, 0.0) + value
    centroids: list[dict[int, float]] = []
    for target, vector in enumerate(sums):
        divisor = max(1, counts[target])
        averaged = {index: value / divisor for index, value in vector.items()}
        norm = math.sqrt(sum(value * value for value in averaged.values())) or 1.0
        centroids.append({index: value / norm for index, value in averaged.items()})
    return centroids


def quantize(values: list[float]) -> tuple[float, str]:
    maximum = max((abs(value) for value in values), default=0.0)
    scale = maximum / 127.0 if maximum else 1.0
    values_int8 = array(
        "b", (max(-127, min(127, round(value / scale))) for value in values)
    )
    return scale, base64.b64encode(values_int8.tobytes()).decode("ascii")


def write_model(path: Path, payload: dict[str, Any]) -> None:
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as raw:
        with gzip.GzipFile(fileobj=raw, mode="wb", compresslevel=9, mtime=0) as compressed:
            compressed.write(encoded)


def evaluate_saved(path: Path, split_samples: dict[str, list[Sample]], intents: list[str]) -> dict[str, float]:
    from jarvis.brain.model import HybridNeuralBrain
    from jarvis.config import load_config

    previous = os.environ.get("JARVIS_DATA_DIR")
    with tempfile.TemporaryDirectory(prefix="jarvis-v04-eval-") as temporary:
        os.environ["JARVIS_DATA_DIR"] = temporary
        try:
            config = load_config(ROOT)
            brain = HybridNeuralBrain.load(path, config.brain)
            metrics: dict[str, float] = {}
            for split, samples in split_samples.items():
                correct = 0
                known_correct = 0
                unknown_total = 0
                unknown_correct = 0
                for sample in samples:
                    expected = intents[sample.target]
                    prediction = brain.classify(sample.text)
                    correct += int(prediction.intent == expected)
                    if expected == "unknown":
                        unknown_total += 1
                        unknown_correct += int(prediction.intent == "unknown")
                    else:
                        known_correct += int(prediction.intent == expected)
                metrics[f"{split}_hybrid_accuracy"] = round(correct / max(1, len(samples)), 6)
                known_count = sum(int(intents[s.target] != "unknown") for s in samples)
                metrics[f"{split}_known_accuracy"] = round(known_correct / max(1, known_count), 6)
                if unknown_total:
                    metrics[f"{split}_unknown_recall"] = round(unknown_correct / unknown_total, 6)
            return metrics
        finally:
            if previous is None:
                os.environ.pop("JARVIS_DATA_DIR", None)
            else:
                os.environ["JARVIS_DATA_DIR"] = previous


def train(verbose: bool = True, epoch_override: int | None = None) -> dict[str, Any]:
    manifest = build_dataset()
    config = read_json(ROOT / "config" / "brain.json")
    training_config = config["training"]
    epochs = int(epoch_override or training_config["epochs"])
    seed = int(training_config["seed"])
    feature_size = int(config["feature_size"])
    train_payload = read_json(ROOT / "data" / "training" / "train.json")
    intent_items = train_payload["intents"]
    intents = [str(item["tag"]) for item in intent_items]
    route_types = [str(item.get("route_type", "conversation")) for item in intent_items]
    intent_index = {intent: index for index, intent in enumerate(intents)}
    splits = {
        "train": load_split(ROOT / "data" / "training" / "train.json", intent_index, feature_size, True),
        "validation": load_split(ROOT / "data" / "training" / "validation.json", intent_index, feature_size),
        "test": load_split(ROOT / "data" / "training" / "test.json", intent_index, feature_size),
    }
    network = initialize(config, len(intents))
    rng = random.Random(seed + 1)
    learning_rate = float(training_config["learning_rate"])
    decay = float(training_config["learning_rate_decay"])
    l2 = float(training_config["l2"])
    best_validation = 0.0
    final_loss = 0.0
    for epoch in range(1, epochs + 1):
        final_loss = train_epoch(network, splits["train"], learning_rate, l2, rng)
        learning_rate *= decay
        if epoch == 1 or epoch % 4 == 0 or epoch == epochs:
            train_acc = neural_accuracy(network, splits["train"])
            val_acc = neural_accuracy(network, splits["validation"])
            best_validation = max(best_validation, val_acc)
            if verbose:
                print(
                    f"epoch {epoch:03d}/{epochs} loss={final_loss:.4f} "
                    f"neural_train={train_acc:.3f} neural_validation={val_acc:.3f}"
                )

    centroids = build_centroids(splits["train"], len(intents))
    embedding_scale, embedding_data = quantize(network.embedding)
    hidden_scale, hidden_data = quantize(network.hidden)
    output_scale, output_data = quantize(network.output)
    parameter_count = (
        len(network.embedding)
        + len(network.hidden)
        + len(network.hidden_bias)
        + len(network.output)
        + len(network.output_bias)
    )
    metadata: dict[str, Any] = {
        "dataset_manifest": manifest,
        "training_examples_after_augmentation": len(splits["train"]),
        "validation_examples": len(splits["validation"]),
        "test_examples": len(splits["test"]),
        "intent_count": len(intents),
        "epochs": epochs,
        "best_float_neural_validation_accuracy": round(best_validation, 6),
        "final_float_neural_train_accuracy": round(neural_accuracy(network, splits["train"]), 6),
        "final_float_neural_validation_accuracy": round(
            neural_accuracy(network, splits["validation"]), 6
        ),
        "final_float_neural_test_accuracy": round(neural_accuracy(network, splits["test"]), 6),
        "final_loss": round(final_loss, 6),
        "quantization": "symmetric_int8",
        "trained_at_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        "python": f"{sys.version_info.major}.{sys.version_info.minor}",
    }
    payload: dict[str, Any] = {
        "format": config["format"],
        "feature_size": network.feature_size,
        "embedding_size": network.embedding_size,
        "hidden_size": network.hidden_size,
        "intents": intents,
        "route_types": route_types,
        "parameter_count": parameter_count,
        "embedding": {"scale": embedding_scale, "data": embedding_data},
        "hidden": {"scale": hidden_scale, "data": hidden_data},
        "hidden_bias": [round(value, 8) for value in network.hidden_bias],
        "output": {"scale": output_scale, "data": output_data},
        "output_bias": [round(value, 8) for value in network.output_bias],
        "lexical_centroids": [
            [[index, round(value, 7)] for index, value in sorted(centroid.items())]
            for centroid in centroids
        ],
        "metadata": metadata,
    }
    metadata["release"] = "0.7.0"
    metadata["pretrained_source"] = None
    metadata["project_dataset_manifest"] = "datasets/manifest_v003.json"
    model_path = ROOT / "models" / "hybrid_brain_v7.jv.gz"
    write_model(model_path, payload)
    hybrid_metrics = evaluate_saved(model_path, splits, intents)
    metadata.update(hybrid_metrics)
    payload["metadata"] = metadata
    write_model(model_path, payload)
    (ROOT / "models" / "training_metrics_fast_v7.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    if verbose:
        print(json.dumps(metadata, ensure_ascii=False, indent=2))
        print(f"model_size_bytes={model_path.stat().st_size}")
        print(f"parameter_count={parameter_count}")
    return metadata


def main() -> int:
    parser = argparse.ArgumentParser(description="Train the JARVIS v0.7 Fast Brain")
    parser.add_argument("--epochs", type=int, help="override configured epochs")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()
    if args.epochs is not None and args.epochs < 1:
        parser.error("--epochs must be positive")
    train(not args.quiet, args.epochs)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
