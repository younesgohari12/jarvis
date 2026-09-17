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

from jarvis.agent.fluency import train_centroids  # noqa: E402
from jarvis.agent.followup_model import FollowupIntentModel  # noqa: E402


def _load(path: Path) -> list[dict[str, str]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("format") != "jarvis-dialogue-followup-dataset-v1":
        raise ValueError("Unsupported follow-up dataset")
    rows = payload.get("examples")
    if not isinstance(rows, list):
        raise ValueError("Dataset examples must be a list")
    cleaned = [
        {"prompt": str(row.get("prompt", "")).strip(), "label": str(row.get("label", "")).strip()}
        for row in rows if isinstance(row, dict)
    ]
    if any(not row["prompt"] or not row["label"] for row in cleaned):
        raise ValueError("Every row needs prompt and label")
    fingerprints = {hashlib.sha256(row["prompt"].encode("utf-8")).hexdigest() for row in cleaned}
    if len(fingerprints) != len(cleaned):
        raise ValueError("Duplicate prompts are not allowed")
    return cleaned


def _split(rows: list[dict[str, str]]) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    grouped: defaultdict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[row["label"]].append(row)
    if any(len(values) < 8 for values in grouped.values()):
        raise ValueError("Every follow-up label needs at least eight reviewed examples")
    train: list[dict[str, str]] = []
    held_out: list[dict[str, str]] = []
    for label in sorted(grouped):
        held_out.append(grouped[label][-1])
        train.extend(grouped[label][:-1])
    return train, held_out


def _payload(rows: list[dict[str, str]], dataset: Path) -> dict[str, Any]:
    payload = train_centroids(rows)
    payload.update({
        "format": FollowupIntentModel.FORMAT,
        "version": "1.0.0",
        "pretrained_source": None,
        "training_examples": len(rows),
        "dataset": dataset.relative_to(ROOT).as_posix(),
        "dataset_sha256": hashlib.sha256(dataset.read_bytes()).hexdigest(),
        "training_method": "word_and_character_tfidf_centroid",
    })
    return payload


def _evaluate(payload: dict[str, Any], rows: list[dict[str, str]]) -> dict[str, Any]:
    with tempfile.TemporaryDirectory(prefix="jarvis-followup-v10-") as temporary:
        path = Path(temporary) / "model.json"
        path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
        model = FollowupIntentModel(path)
        predictions = []
        correct = 0
        for row in rows:
            predicted, confidence, margin = model.predict(row["prompt"])
            correct += int(predicted == row["label"])
            predictions.append({
                "prompt": row["prompt"], "expected": row["label"],
                "predicted": predicted, "confidence": confidence, "margin": margin,
            })
    return {
        "held_out_cases": len(rows),
        "classification_accuracy": round(correct / max(1, len(rows)), 6),
        "predictions": predictions,
    }


def train(dataset: Path, output: Path, metrics_path: Path) -> dict[str, Any]:
    rows = _load(dataset)
    train_rows, held_out = _split(rows)
    evaluation = _evaluate(_payload(train_rows, dataset), held_out)
    accepted = evaluation["classification_accuracy"] >= 0.875
    metrics = {
        "format": "jarvis-dialogue-followup-metrics-v1",
        "version": "1.0.0",
        "dataset_examples": len(rows),
        "train_examples": len(train_rows),
        "held_out_examples": len(held_out),
        "label_counts": dict(sorted(Counter(row["label"] for row in rows).items())),
        **evaluation,
        "minimum_accuracy": 0.875,
        "accepted_for_release": accepted,
        "pretrained_source": None,
    }
    if not accepted:
        raise RuntimeError(f"Follow-up model failed quality gate: {evaluation['classification_accuracy']}")
    final_payload = _payload(rows, dataset)
    final_payload["split_policy"] = "one held-out paraphrase per label; final artifact retrained on all rows"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(final_payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    metrics_path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return metrics


def main() -> int:
    parser = argparse.ArgumentParser(description="Train the project-owned dialogue follow-up model")
    parser.add_argument("--dataset", type=Path, default=ROOT / "data" / "training" / "dialogue_followup_v10.json")
    parser.add_argument("--output", type=Path, default=ROOT / "models" / "dialogue_followup_v10.json")
    parser.add_argument("--metrics", type=Path, default=ROOT / "models" / "dialogue_followup_metrics_v10.json")
    args = parser.parse_args()
    print(json.dumps(train(args.dataset, args.output, args.metrics), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
