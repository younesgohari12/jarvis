from __future__ import annotations

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

from jarvis.agent.cognitive_model import CognitiveSkillModel  # noqa: E402
from jarvis.agent.fluency import train_centroids  # noqa: E402


def _rows() -> tuple[list[dict[str, str]], Path]:
    source = ROOT / "data" / "training" / "cognitive_skills_v11.json"
    payload = json.loads(source.read_text(encoding="utf-8"))
    rows = [
        {"prompt": str(row["prompt"]).strip(), "label": str(row["label"]).strip(), "response": ""}
        for row in payload.get("examples", [])
    ]
    fingerprints = {hashlib.sha256(row["prompt"].casefold().encode("utf-8")).hexdigest() for row in rows}
    if len(rows) != len(fingerprints):
        raise ValueError("Duplicate cognitive prompts are not allowed")
    return rows, source


def _split(rows: list[dict[str, str]]) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    grouped: defaultdict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[row["label"]].append(row)
    expected = {"syllogism", "probability", "relative_day", "clock_angle", "trailing_zeros", "sequence", "mislabeled_boxes", "general"}
    if set(grouped) != expected:
        raise ValueError(f"Cognitive label coverage mismatch: {set(grouped) ^ expected}")
    train: list[dict[str, str]] = []
    held_out: list[dict[str, str]] = []
    for label in sorted(grouped):
        if len(grouped[label]) < 10:
            raise ValueError(f"Label {label} needs at least ten examples")
        held_out.extend(grouped[label][-2:])
        train.extend(grouped[label][:-2])
    return train, held_out


def _payload(rows: list[dict[str, str]], source: Path) -> dict[str, Any]:
    payload = train_centroids(rows)
    payload.update({
        "format": CognitiveSkillModel.FORMAT,
        "version": "1.1.0",
        "pretrained_source": None,
        "training_examples": len(rows),
        "sources": [source.relative_to(ROOT).as_posix()],
        "source_sha256": {source.name: hashlib.sha256(source.read_bytes()).hexdigest()},
        "training_method": "word_and_character_tfidf_centroid_bilingual",
    })
    return payload


def _evaluate(payload: dict[str, Any], held_out: list[dict[str, str]]) -> dict[str, Any]:
    with tempfile.TemporaryDirectory(prefix="jarvis-cognitive-v11-") as temporary:
        path = Path(temporary) / "model.json"
        path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
        model = CognitiveSkillModel(path)
        predictions = []
        correct = 0
        for row in held_out:
            predicted, confidence, margin = model.predict(row["prompt"])
            correct += int(predicted == row["label"])
            predictions.append({
                "prompt": row["prompt"], "expected": row["label"], "predicted": predicted,
                "confidence": confidence, "margin": margin,
            })
    return {
        "held_out_cases": len(held_out),
        "classification_accuracy": round(correct / len(held_out), 6),
        "predictions": predictions,
    }


def train(output: Path, metrics_path: Path) -> dict[str, Any]:
    rows, source = _rows()
    train_rows, held_out = _split(rows)
    evaluation = _evaluate(_payload(train_rows, source), held_out)
    accepted = evaluation["classification_accuracy"] >= 0.875
    metrics = {
        "format": "jarvis-cognitive-skills-metrics-v1",
        "version": "1.1.0",
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
        raise RuntimeError(f"Cognitive model failed quality gate: {evaluation}")
    output.write_text(json.dumps(_payload(rows, source), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    metrics_path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return metrics


if __name__ == "__main__":
    result = train(
        ROOT / "models" / "cognitive_skills_v11.json",
        ROOT / "models" / "cognitive_skills_metrics_v11.json",
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
