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

from jarvis.agent.bilingual_fluency import BilingualFluencyEngine  # noqa: E402
from jarvis.agent.fluency import train_centroids  # noqa: E402


def _rows() -> tuple[list[dict[str, str]], list[Path]]:
    base_path = ROOT / "data" / "training" / "persian_fluency_v9.json"
    extra_path = ROOT / "data" / "training" / "bilingual_fluency_v11_supplement.json"
    base = json.loads(base_path.read_text(encoding="utf-8"))
    extra = json.loads(extra_path.read_text(encoding="utf-8"))
    rows: list[dict[str, str]] = []
    for row in base.get("examples", []):
        rows.append({
            "prompt": str(row["prompt"]).strip(), "label": "fa:" + str(row["label"]).strip(),
            "response": str(row.get("response", "")).strip(),
        })
    for row in extra.get("examples", []):
        language = str(row["language"]).strip()
        rows.append({
            "prompt": str(row["prompt"]).strip(),
            "label": language + ":" + str(row["label"]).strip(),
            "response": str(row.get("response", "")).strip(),
        })
    fingerprints = {hashlib.sha256(row["prompt"].casefold().encode("utf-8")).hexdigest() for row in rows}
    if len(fingerprints) != len(rows):
        raise ValueError("Duplicate bilingual prompts are not allowed")
    return rows, [base_path, extra_path]


def _split(rows: list[dict[str, str]]) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    grouped: defaultdict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[row["label"]].append(row)
    if set(grouped) != {f"{language}:{label}" for language in ("fa", "en") for label in ("text", "formal_message", "caption", "story", "advice", "support", "rewrite")}:
        raise ValueError("Bilingual label coverage is incomplete")
    train: list[dict[str, str]] = []
    held_out: list[dict[str, str]] = []
    for label in sorted(grouped):
        if len(grouped[label]) < 4:
            raise ValueError(f"Label {label} needs at least four examples")
        held_out.append(grouped[label][-1])
        train.extend(grouped[label][:-1])
    return train, held_out


def _payload(rows: list[dict[str, str]], sources: list[Path]) -> dict[str, Any]:
    payload = train_centroids(rows)
    payload.update({
        "format": BilingualFluencyEngine.FORMAT,
        "version": "1.1.0",
        "pretrained_source": None,
        "training_examples": len(rows),
        "sources": [path.relative_to(ROOT).as_posix() for path in sources],
        "source_sha256": {path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in sources},
        "training_method": "word_and_character_tfidf_centroid_bilingual",
    })
    return payload


def _evaluate(payload: dict[str, Any], held_out: list[dict[str, str]]) -> dict[str, Any]:
    with tempfile.TemporaryDirectory(prefix="jarvis-bilingual-v11-") as temporary:
        model_path = Path(temporary) / "model.json"
        model_path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
        model = BilingualFluencyEngine(model_path)
        predictions = []
        correct = accepted = 0
        routed = {
            "text": "writing_request", "formal_message": "writing_request",
            "caption": "writing_request", "story": "short_story", "advice": "advice_request",
            "support": "user_low_mood", "rewrite": "rewrite_request",
        }
        for row in held_out:
            language, task = row["label"].split(":", 1)
            predicted, confidence = model.predict(row["prompt"])
            correct += int(predicted == row["label"])
            candidate = model.compose(row["prompt"], routed[task], language)
            generation_ok = bool(candidate and candidate.text and len(candidate.text) >= 45)
            accepted += int(generation_ok)
            predictions.append({
                "prompt": row["prompt"], "expected": row["label"], "predicted": predicted,
                "confidence": confidence, "generation_accepted": generation_ok,
            })
    return {
        "held_out_cases": len(held_out),
        "classification_accuracy": round(correct / len(held_out), 6),
        "generation_acceptance": round(accepted / len(held_out), 6),
        "predictions": predictions,
    }


def train(output: Path, metrics_path: Path) -> dict[str, Any]:
    rows, sources = _rows()
    train_rows, held_out = _split(rows)
    evaluation = _evaluate(_payload(train_rows, sources), held_out)
    accepted = evaluation["classification_accuracy"] >= 0.85 and evaluation["generation_acceptance"] == 1.0
    metrics = {
        "format": "jarvis-bilingual-fluency-metrics-v1", "version": "1.1.0",
        "dataset_examples": len(rows), "train_examples": len(train_rows),
        "held_out_examples": len(held_out),
        "label_counts": dict(sorted(Counter(row["label"] for row in rows).items())),
        **evaluation, "minimum_accuracy": 0.85, "accepted_for_release": accepted,
        "pretrained_source": None,
    }
    if not accepted:
        raise RuntimeError(f"Bilingual model failed quality gate: {evaluation}")
    output.write_text(json.dumps(_payload(rows, sources), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    metrics_path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return metrics


if __name__ == "__main__":
    result = train(
        ROOT / "models" / "bilingual_fluency_v11.json",
        ROOT / "models" / "bilingual_fluency_metrics_v11.json",
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
