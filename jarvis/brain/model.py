from __future__ import annotations

import base64
import gzip
import json
import math
from array import array
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from jarvis.config import BrainConfig
from jarvis.utils.text import cosine_sparse, hashed_features


class BrainLoadError(RuntimeError):
    """Raised when neural weights are missing, damaged, or incompatible."""


@dataclass(frozen=True, slots=True)
class IntentPrediction:
    intent: str
    route_type: str
    confidence: float
    margin: float
    normalized_entropy: float
    neural_confidence: float
    lexical_similarity: float
    alternatives: tuple[tuple[str, float], ...]
    is_unknown: bool
    unknown_reason: str = ""


class HybridNeuralBrain:
    """Sparse learned embeddings + hidden neural layer + calibrated lexical ensemble."""

    def __init__(self, payload: dict[str, Any], config: BrainConfig) -> None:
        self.feature_size = int(payload["feature_size"])
        self.embedding_size = int(payload["embedding_size"])
        self.hidden_size = int(payload["hidden_size"])
        self.intents = tuple(str(value) for value in payload["intents"])
        self.route_types = tuple(str(value) for value in payload["route_types"])
        self.parameter_count = int(payload["parameter_count"])
        self.metadata = dict(payload.get("metadata", {}))
        self.temperature = config.temperature
        self.confidence_threshold = config.confidence_threshold
        self.margin_threshold = config.margin_threshold
        self.lexical_ood_threshold = config.lexical_ood_threshold
        self.maximum_normalized_entropy = config.maximum_normalized_entropy
        self.neural_weight = config.neural_weight
        self.lexical_weight = config.lexical_weight

        try:
            self._embedding = self._decode_int8(payload["embedding"]["data"])
            self._embedding_scale = float(payload["embedding"]["scale"])
            self._hidden = self._decode_int8(payload["hidden"]["data"])
            self._hidden_scale = float(payload["hidden"]["scale"])
            self._hidden_bias = tuple(float(value) for value in payload["hidden_bias"])
            self._output = self._decode_int8(payload["output"]["data"])
            self._output_scale = float(payload["output"]["scale"])
            self._output_bias = tuple(float(value) for value in payload["output_bias"])
            raw_centroids = payload["lexical_centroids"]
            self._centroids = tuple(
                {int(index): float(value) for index, value in centroid}
                for centroid in raw_centroids
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise BrainLoadError(f"Invalid brain payload: {exc}") from exc

        output_size = len(self.intents)
        expected = {
            "embedding": self.feature_size * self.embedding_size,
            "hidden": self.embedding_size * self.hidden_size,
            "output": self.hidden_size * output_size,
        }
        actual = {
            "embedding": len(self._embedding),
            "hidden": len(self._hidden),
            "output": len(self._output),
        }
        if actual != expected:
            raise BrainLoadError(f"Brain matrix shape mismatch: expected {expected}, got {actual}")
        if len(self._hidden_bias) != self.hidden_size or len(self._output_bias) != output_size:
            raise BrainLoadError("Brain bias vector shape mismatch")
        if len(self.route_types) != output_size or len(self._centroids) != output_size:
            raise BrainLoadError("Intent metadata shape mismatch")
        if (
            self.feature_size != config.feature_size
            or self.embedding_size != config.embedding_size
            or self.hidden_size != config.hidden_size
        ):
            raise BrainLoadError("Brain weights are incompatible with config/brain.json")

    @staticmethod
    def _decode_int8(encoded: str) -> array[int]:
        try:
            return array("b", base64.b64decode(encoded, validate=True))
        except (ValueError, TypeError) as exc:
            raise BrainLoadError(f"Invalid int8 weight block: {exc}") from exc

    @classmethod
    def load(cls, path: Path, config: BrainConfig) -> "HybridNeuralBrain":
        if not path.is_file():
            raise BrainLoadError(f"Trained JARVIS brain was not found: {path.name}")
        try:
            with gzip.open(path, "rt", encoding="utf-8") as handle:
                payload = json.load(handle)
        except (OSError, json.JSONDecodeError) as exc:
            raise BrainLoadError(f"Cannot read trained JARVIS brain: {exc}") from exc
        if not isinstance(payload, dict) or payload.get("format") != config.format:
            raise BrainLoadError("Unsupported JARVIS brain format")
        return cls(payload, config)

    @staticmethod
    def _softmax(logits: list[float], temperature: float = 1.0) -> list[float]:
        adjusted = [value / max(0.05, temperature) for value in logits]
        peak = max(adjusted)
        exponentials = [math.exp(value - peak) for value in adjusted]
        total = sum(exponentials) or 1.0
        return [value / total for value in exponentials]

    def _neural_probabilities(self, features: dict[int, float]) -> list[float]:
        pooled = [0.0] * self.embedding_size
        for feature_index, feature_value in features.items():
            offset = feature_index * self.embedding_size
            scaled = feature_value * self._embedding_scale
            for embedding_index in range(self.embedding_size):
                pooled[embedding_index] += self._embedding[offset + embedding_index] * scaled

        hidden = list(self._hidden_bias)
        for embedding_index, embedding_value in enumerate(pooled):
            offset = embedding_index * self.hidden_size
            scaled = embedding_value * self._hidden_scale
            for hidden_index in range(self.hidden_size):
                hidden[hidden_index] += self._hidden[offset + hidden_index] * scaled
        hidden = [math.tanh(value) for value in hidden]

        logits = list(self._output_bias)
        output_size = len(self.intents)
        for hidden_index, hidden_value in enumerate(hidden):
            offset = hidden_index * output_size
            scaled = hidden_value * self._output_scale
            for output_index in range(output_size):
                logits[output_index] += self._output[offset + output_index] * scaled
        return self._softmax(logits, self.temperature)

    def classify(self, text: str) -> IntentPrediction:
        features = hashed_features(text, self.feature_size)
        if not features:
            return IntentPrediction(
                "unknown", "unknown", 0.0, 0.0, 1.0, 0.0, 0.0, (), True, "empty_input"
            )

        neural = self._neural_probabilities(features)
        lexical_scores = [max(0.0, cosine_sparse(features, centroid)) for centroid in self._centroids]
        lexical_probabilities = self._softmax([score * 7.0 for score in lexical_scores])
        combined = [
            self.neural_weight * neural[index] + self.lexical_weight * lexical_probabilities[index]
            for index in range(len(self.intents))
        ]
        total = sum(combined) or 1.0
        combined = [value / total for value in combined]
        ranking_indices = sorted(range(len(combined)), key=combined.__getitem__, reverse=True)
        top_index = ranking_indices[0]
        runner_up = combined[ranking_indices[1]] if len(ranking_indices) > 1 else 0.0
        confidence = combined[top_index]
        margin = confidence - runner_up
        entropy = -sum(value * math.log(max(value, 1e-12)) for value in combined)
        normalized_entropy = entropy / max(1e-9, math.log(len(combined)))
        lexical_best = lexical_scores[top_index]
        neural_confidence = neural[top_index]
        predicted_intent = self.intents[top_index]

        unknown_reason = ""
        if predicted_intent == "unknown":
            unknown_reason = "model_unknown_class"
        elif lexical_best < self.lexical_ood_threshold and neural_confidence < 0.72:
            unknown_reason = "out_of_distribution"
        elif confidence < self.confidence_threshold:
            unknown_reason = "low_confidence"
        elif margin < self.margin_threshold and lexical_best < 0.22:
            unknown_reason = "ambiguous_intent"
        elif normalized_entropy > self.maximum_normalized_entropy and lexical_best < 0.2:
            unknown_reason = "high_entropy"

        is_unknown = bool(unknown_reason)
        intent = "unknown" if is_unknown else predicted_intent
        route_type = "unknown" if is_unknown else self.route_types[top_index]
        alternatives = tuple(
            (self.intents[index], combined[index]) for index in ranking_indices[:4]
        )
        return IntentPrediction(
            intent=intent,
            route_type=route_type,
            confidence=confidence,
            margin=margin,
            normalized_entropy=normalized_entropy,
            neural_confidence=neural_confidence,
            lexical_similarity=lexical_best,
            alternatives=alternatives,
            is_unknown=is_unknown,
            unknown_reason=unknown_reason,
        )


# Backward-compatible import name for v0.1 extensions.
TinyNeuralBrain = HybridNeuralBrain
