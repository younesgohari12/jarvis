from __future__ import annotations

import argparse
import hashlib
import json
import sys
import tempfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from jarvis.agent.fluency import PersianFluencyEngine, train_centroids  # noqa: E402


def _load(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("format") != "jarvis-persian-fluency-dataset-v1":
        raise ValueError("Unsupported Persian fluency dataset")
    rows = payload.get("examples", [])
    if not isinstance(rows, list):
        raise ValueError("Dataset examples must be a list")
    cleaned = [row for row in rows if isinstance(row, dict)]
    fingerprints = {
        hashlib.sha256(str(row.get("prompt", "")).strip().encode("utf-8")).hexdigest()
        for row in cleaned
    }
    if len(fingerprints) != len(cleaned):
        raise ValueError("Duplicate fluency prompts are not allowed")
    return cleaned


def _split(rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    grouped: defaultdict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row.get("label", ""))].append(row)
    train: list[dict[str, Any]] = []
    held_out: list[dict[str, Any]] = []
    for label in sorted(grouped):
        values = grouped[label]
        if len(values) < 5:
            raise ValueError(f"Label {label!r} needs at least five examples")
        held_out.append(values[-1])
        train.extend(values[:-1])
    return train, held_out


def _evaluate(payload: dict[str, Any], held_out: list[dict[str, Any]]) -> dict[str, Any]:
    with tempfile.TemporaryDirectory(prefix="jarvis-fluency-v9-") as temporary:
        path = Path(temporary) / "model.json"
        path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
        engine = PersianFluencyEngine(path)
        predictions: list[dict[str, Any]] = []
        correct = 0
        generation_passed = 0
        forced_intents = {
            "text": "writing_request",
            "formal_message": "writing_request",
            "caption": "writing_request",
            "story": "short_story",
            "advice": "advice_request",
            "support": "user_low_mood",
            "rewrite": "rewrite_request",
        }
        for row in held_out:
            prompt = str(row["prompt"])
            expected = str(row["label"])
            predicted, confidence = engine.predict(prompt)
            correct += int(predicted == expected)
            candidate = engine.compose(prompt, forced_intents[expected], "fa")
            accepted = bool(candidate and candidate.text and len(candidate.text) >= 45)
            generation_passed += int(accepted)
            predictions.append({
                "prompt": prompt,
                "expected": expected,
                "predicted": predicted,
                "confidence": confidence,
                "generation_accepted": accepted,
            })
    cases = len(held_out)
    return {
        "held_out_cases": cases,
        "classification_accuracy": round(correct / max(1, cases), 6),
        "generation_acceptance": round(generation_passed / max(1, cases), 6),
        "predictions": predictions,
    }


def train(dataset: Path, output: Path, metrics_path: Path) -> dict[str, Any]:
    rows = _load(dataset)
    train_rows, held_out = _split(rows)
    evaluation_payload = train_centroids(train_rows)
    evaluation = _evaluate(evaluation_payload, held_out)
    final_payload = train_centroids(rows)
    dataset_bytes = dataset.read_bytes()
    final_payload.update({
        "dataset": dataset.relative_to(ROOT).as_posix(),
        "dataset_sha256": hashlib.sha256(dataset_bytes).hexdigest(),
        "split_policy": "one held-out prompt per label; final artifact retrained on all reviewed rows",
        "quality_gate": {
            "minimum_classification_accuracy": 0.85,
            "minimum_generation_acceptance": 1.0,
        },
    })
    accepted = (
        evaluation["classification_accuracy"] >= 0.85
        and evaluation["generation_acceptance"] >= 1.0
    )
    metrics = {
        "format": "jarvis-persian-fluency-training-metrics-v1",
        "version": "0.9.0",
        "dataset_examples": len(rows),
        "train_examples": len(train_rows),
        "held_out_examples": len(held_out),
        "label_counts": dict(sorted(Counter(str(row["label"]) for row in rows).items())),
        **evaluation,
        "accepted_for_release": accepted,
        "pretrained_source": None,
    }
    if not accepted:
        raise RuntimeError(
            "Persian fluency candidate failed its quality gate: "
            f"accuracy={evaluation['classification_accuracy']}, "
            f"generation={evaluation['generation_acceptance']}"
        )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(final_payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    metrics_path.write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return metrics


def main() -> int:
    parser = argparse.ArgumentParser(description="Train the reviewed Persian fluency selector")
    parser.add_argument(
        "--dataset", type=Path,
        default=ROOT / "data" / "training" / "persian_fluency_v9.json",
    )
    parser.add_argument(
        "--output", type=Path,
        default=ROOT / "models" / "persian_fluency_v9.json",
    )
    parser.add_argument(
        "--metrics", type=Path,
        default=ROOT / "models" / "persian_fluency_metrics_v9.json",
    )
    args = parser.parse_args()
    result = train(args.dataset, args.output, args.metrics)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
