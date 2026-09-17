from __future__ import annotations

import json
from pathlib import Path

from jarvis.agent.fluency import PersianFluencyEngine


class CognitiveSkillModel:
    """Project-trained bilingual classifier for bounded reasoning skills."""

    FORMAT = "jarvis-cognitive-skills-v1"

    def __init__(self, model_path: Path) -> None:
        self.model_path = model_path
        self.ready = False
        self.version = ""
        self.training_examples = 0
        self._idf: dict[str, float] = {}
        self._centroids: dict[str, dict[str, float]] = {}
        try:
            payload = json.loads(model_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return
        if payload.get("format") != self.FORMAT or payload.get("pretrained_source") not in (None, ""):
            return
        idf = payload.get("idf")
        centroids = payload.get("centroids")
        if not isinstance(idf, dict) or not isinstance(centroids, dict):
            return
        self._idf = {str(key): float(value) for key, value in idf.items()}
        self._centroids = {
            str(label): {str(key): float(value) for key, value in vector.items()}
            for label, vector in centroids.items() if isinstance(vector, dict)
        }
        self.version = str(payload.get("version", ""))
        self.training_examples = int(payload.get("training_examples", 0) or 0)
        self.ready = bool(self._idf and self._centroids and self.training_examples)

    def predict(self, text: str) -> tuple[str, float, float]:
        if not self.ready:
            return "", 0.0, 0.0
        vector = PersianFluencyEngine.weighted_vector(text, self._idf, allowed=set(self._idf))
        ranked = sorted(
            (
                (PersianFluencyEngine.similarity(vector, centroid), label)
                for label, centroid in self._centroids.items()
            ),
            reverse=True,
        )
        if not ranked:
            return "", 0.0, 0.0
        best_score, best_label = ranked[0]
        second_score = ranked[1][0] if len(ranked) > 1 else 0.0
        margin = max(0.0, best_score - second_score)
        confidence = max(0.0, min(0.99, best_score * 0.72 + margin * 0.9))
        return best_label, round(confidence, 4), round(margin, 4)


__all__ = ["CognitiveSkillModel"]
