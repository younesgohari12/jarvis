from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable

try:
    import numpy as np
except ImportError as exc:  # pragma: no cover - exercised by startup fallback
    raise ImportError(
        "Jarvis neural runtime needs NumPy. Run setup.ps1 or install requirements-runtime.txt."
    ) from exc

from jarvis.neural.config import TransformerConfig


class NeuralModelError(RuntimeError):
    pass


@dataclass(frozen=True, slots=True)
class GenerationResult:
    text: str
    token_ids: tuple[int, ...]
    prompt_tokens: int
    generated_tokens: int
    elapsed_seconds: float
    tokens_per_second: float


def _softmax(value: np.ndarray, axis: int = -1) -> np.ndarray:
    shifted = value - np.max(value, axis=axis, keepdims=True)
    result = np.exp(shifted, dtype=np.float32)
    return result / np.sum(result, axis=axis, keepdims=True)


def _silu(value: np.ndarray) -> np.ndarray:
    clipped = np.clip(value, -30.0, 30.0)
    return value / (1.0 + np.exp(-clipped))


def _rms_norm(value: np.ndarray, weight: np.ndarray, epsilon: float) -> np.ndarray:
    scale = np.reciprocal(
        np.sqrt(np.mean(np.square(value, dtype=np.float32), axis=-1, keepdims=True) + epsilon)
    )
    return value * scale * weight


def _rope_tables(
    positions: np.ndarray, head_dim: int, theta: float,
) -> tuple[np.ndarray, np.ndarray]:
    inverse = 1.0 / (theta ** (np.arange(0, head_dim, 2, dtype=np.float32) / head_dim))
    angles = positions.astype(np.float32)[:, None] * inverse[None, :]
    return np.cos(angles), np.sin(angles)


def _apply_rope(value: np.ndarray, positions: np.ndarray, theta: float) -> np.ndarray:
    # value: [batch, time, heads, head_dim]
    cosine, sine = _rope_tables(positions, value.shape[-1], theta)
    cosine = cosine[None, :, None, :]
    sine = sine[None, :, None, :]
    even = value[..., 0::2]
    odd = value[..., 1::2]
    output = np.empty_like(value)
    output[..., 0::2] = even * cosine - odd * sine
    output[..., 1::2] = even * sine + odd * cosine
    return output


class JarvisTransformer:
    """NumPy inference for the JARVIS decoder-only Transformer.

    The architecture and all weights are project-owned. The loader accepts the
    integrity-checked INT8 v2 format and the legacy project v1 format, and rejects
    metadata that declares an external pretrained source.
    """

    MODEL_FORMAT = "jarvis-numpy-int8-v2"
    LEGACY_MODEL_FORMAT = "jarvis-numpy-int8-v1"

    def __init__(
        self,
        config: TransformerConfig,
        parameters: dict[str, np.ndarray],
        metadata: dict[str, Any] | None = None,
        quantization_scales: dict[str, float] | None = None,
    ) -> None:
        self.config = config
        self.parameters = parameters
        self.metadata = dict(metadata or {})
        self.quantization_scales = dict(quantization_scales or {})
        self._validate()

    @classmethod
    def initialize(cls, config: TransformerConfig) -> "JarvisTransformer":
        rng = np.random.default_rng(config.seed)
        parameters: dict[str, np.ndarray] = {
            "tok_embeddings": rng.normal(
                0.0, config.initializer_std, (config.vocab_size, config.d_model)
            ).astype(np.float32),
            "final_norm": np.ones(config.d_model, dtype=np.float32),
        }
        kv_width = config.n_kv_heads * config.head_dim
        for layer in range(config.n_layers):
            prefix = f"layers.{layer}"
            parameters[f"{prefix}.attn_norm"] = np.ones(config.d_model, dtype=np.float32)
            parameters[f"{prefix}.wq"] = rng.normal(
                0.0, config.initializer_std, (config.d_model, config.d_model)
            ).astype(np.float32)
            for name in ("wk", "wv"):
                parameters[f"{prefix}.{name}"] = rng.normal(
                    0.0, config.initializer_std, (config.d_model, kv_width)
                ).astype(np.float32)
            parameters[f"{prefix}.wo"] = rng.normal(
                0.0, config.initializer_std / math.sqrt(2 * config.n_layers),
                (config.d_model, config.d_model),
            ).astype(np.float32)
            parameters[f"{prefix}.ffn_norm"] = np.ones(config.d_model, dtype=np.float32)
            for name in ("w1", "w3"):
                parameters[f"{prefix}.{name}"] = rng.normal(
                    0.0, config.initializer_std,
                    (config.d_model, config.ffn_hidden),
                ).astype(np.float32)
            parameters[f"{prefix}.w2"] = rng.normal(
                0.0, config.initializer_std / math.sqrt(2 * config.n_layers),
                (config.ffn_hidden, config.d_model),
            ).astype(np.float32)
        if not config.tie_embeddings:
            parameters["lm_head"] = rng.normal(
                0.0, config.initializer_std, (config.d_model, config.vocab_size)
            ).astype(np.float32)
        return cls(
            config,
            parameters,
            {
                "initialization": "random_normal",
                "seed": config.seed,
                "pretrained_source": None,
                "training_complete": False,
            },
        )

    @property
    def parameter_count(self) -> int:
        return sum(int(value.size) for value in self.parameters.values())

    @property
    def memory_bytes(self) -> int:
        return sum(int(value.nbytes) for value in self.parameters.values()) + 4 * len(
            self.quantization_scales
        )

    @property
    def quantized_runtime(self) -> bool:
        return bool(self.quantization_scales)

    def _linear(self, value: np.ndarray, name: str) -> np.ndarray:
        """Matrix multiply while keeping resident weights in their INT8 form."""
        result = value @ self.parameters[name]
        scale = self.quantization_scales.get(name)
        if scale is not None:
            result = result * np.float32(scale)
        return result.astype(np.float32, copy=False)

    def _embedding(self, tokens: np.ndarray) -> np.ndarray:
        selected = self.parameters["tok_embeddings"][tokens]
        scale = self.quantization_scales.get("tok_embeddings")
        if scale is not None:
            return selected.astype(np.float32) * np.float32(scale)
        return selected.astype(np.float32, copy=False)

    def _project_head(self, value: np.ndarray) -> np.ndarray:
        name = "tok_embeddings" if self.config.tie_embeddings else "lm_head"
        result = value @ self.parameters[name].T if self.config.tie_embeddings else value @ self.parameters[name]
        scale = self.quantization_scales.get(name)
        if scale is not None:
            result = result * np.float32(scale)
        return result.astype(np.float32, copy=False)

    def _expected_shapes(self) -> dict[str, tuple[int, ...]]:
        config = self.config
        kv_width = config.n_kv_heads * config.head_dim
        expected: dict[str, tuple[int, ...]] = {
            "tok_embeddings": (config.vocab_size, config.d_model),
            "final_norm": (config.d_model,),
        }
        for layer in range(config.n_layers):
            prefix = f"layers.{layer}"
            expected.update(
                {
                    f"{prefix}.attn_norm": (config.d_model,),
                    f"{prefix}.wq": (config.d_model, config.d_model),
                    f"{prefix}.wk": (config.d_model, kv_width),
                    f"{prefix}.wv": (config.d_model, kv_width),
                    f"{prefix}.wo": (config.d_model, config.d_model),
                    f"{prefix}.ffn_norm": (config.d_model,),
                    f"{prefix}.w1": (config.d_model, config.ffn_hidden),
                    f"{prefix}.w2": (config.ffn_hidden, config.d_model),
                    f"{prefix}.w3": (config.d_model, config.ffn_hidden),
                }
            )
        if not config.tie_embeddings:
            expected["lm_head"] = (config.d_model, config.vocab_size)
        return expected

    def _validate(self) -> None:
        expected = self._expected_shapes()
        if set(expected) != set(self.parameters):
            missing = sorted(set(expected) - set(self.parameters))
            extra = sorted(set(self.parameters) - set(expected))
            raise NeuralModelError(f"Model parameter mismatch; missing={missing}, extra={extra}")
        for name, shape in expected.items():
            if self.parameters[name].shape != shape:
                raise NeuralModelError(
                    f"Wrong shape for {name}: {self.parameters[name].shape}, expected {shape}"
                )
        if self.parameter_count != self.config.parameter_count:
            raise NeuralModelError("Model parameter count does not match config")
        if self.metadata.get("pretrained_source") not in (None, ""):
            raise NeuralModelError("External pretrained weights are forbidden")

    def forward(self, token_ids: Iterable[int] | np.ndarray) -> np.ndarray:
        tokens = np.asarray(token_ids, dtype=np.int64)
        if tokens.ndim == 1:
            tokens = tokens[None, :]
        if tokens.ndim != 2 or tokens.shape[1] == 0:
            raise ValueError("token_ids must have shape [batch, time]")
        if tokens.shape[1] > self.config.max_seq_len:
            tokens = tokens[:, -self.config.max_seq_len :]
        if np.any(tokens < 0) or np.any(tokens >= self.config.vocab_size):
            raise ValueError("token id outside model vocabulary")
        config = self.config
        batch, length = tokens.shape
        positions = np.arange(length, dtype=np.float32)
        x = self._embedding(tokens)
        causal = np.triu(np.full((length, length), -1e9, dtype=np.float32), 1)
        repeat = config.n_heads // config.n_kv_heads
        for layer in range(config.n_layers):
            prefix = f"layers.{layer}"
            normalized = _rms_norm(
                x, self.parameters[f"{prefix}.attn_norm"], config.rms_epsilon
            )
            query = self._linear(normalized, f"{prefix}.wq").reshape(
                batch, length, config.n_heads, config.head_dim
            )
            key = self._linear(normalized, f"{prefix}.wk").reshape(
                batch, length, config.n_kv_heads, config.head_dim
            )
            value = self._linear(normalized, f"{prefix}.wv").reshape(
                batch, length, config.n_kv_heads, config.head_dim
            )
            query = _apply_rope(query, positions, config.rope_theta)
            key = _apply_rope(key, positions, config.rope_theta)
            if repeat > 1:
                key = np.repeat(key, repeat, axis=2)
                value = np.repeat(value, repeat, axis=2)
            scores = np.einsum("bthd,bshd->bhts", query, key, optimize=True)
            scores = scores / math.sqrt(config.head_dim) + causal[None, None, :, :]
            probabilities = _softmax(scores)
            attention = np.einsum(
                "bhts,bshd->bthd", probabilities, value, optimize=True
            ).reshape(batch, length, config.d_model)
            x = x + self._linear(attention, f"{prefix}.wo")
            normalized = _rms_norm(
                x, self.parameters[f"{prefix}.ffn_norm"], config.rms_epsilon
            )
            gate = _silu(self._linear(normalized, f"{prefix}.w1"))
            up = self._linear(normalized, f"{prefix}.w3")
            x = x + self._linear(gate * up, f"{prefix}.w2")
        x = _rms_norm(x, self.parameters["final_norm"], config.rms_epsilon)
        return self._project_head(x)

    def next_token_logits(self, token_ids: list[int]) -> np.ndarray:
        if not token_ids:
            raise ValueError("At least one prompt token is required")
        return self.forward(token_ids)[0, -1].astype(np.float32, copy=False)

    def init_kv_cache(self) -> list[dict[str, np.ndarray]]:
        """Create an empty per-layer KV cache for autoregressive decoding."""
        shape = (1, 0, self.config.n_kv_heads, self.config.head_dim)
        return [
            {
                "key": np.empty(shape, dtype=np.float32),
                "value": np.empty(shape, dtype=np.float32),
            }
            for _ in range(self.config.n_layers)
        ]

    def decode_step(
        self,
        token_id: int,
        position: int,
        cache: list[dict[str, np.ndarray]],
    ) -> np.ndarray:
        """Decode one token and append its keys/values to the supplied cache."""
        if not 0 <= token_id < self.config.vocab_size:
            raise ValueError("token id outside model vocabulary")
        if len(cache) != self.config.n_layers:
            raise ValueError("KV cache layer count does not match model")
        config = self.config
        x = self._embedding(np.asarray([[token_id]], dtype=np.int64))
        positions = np.asarray([position], dtype=np.float32)
        repeat = config.n_heads // config.n_kv_heads
        for layer in range(config.n_layers):
            prefix = f"layers.{layer}"
            normalized = _rms_norm(
                x, self.parameters[f"{prefix}.attn_norm"], config.rms_epsilon
            )
            query = self._linear(normalized, f"{prefix}.wq").reshape(
                1, 1, config.n_heads, config.head_dim
            )
            key = self._linear(normalized, f"{prefix}.wk").reshape(
                1, 1, config.n_kv_heads, config.head_dim
            )
            value = self._linear(normalized, f"{prefix}.wv").reshape(
                1, 1, config.n_kv_heads, config.head_dim
            )
            query = _apply_rope(query, positions, config.rope_theta)
            key = _apply_rope(key, positions, config.rope_theta)
            cached = cache[layer]
            cached["key"] = np.concatenate((cached["key"], key), axis=1)[
                :, -config.max_seq_len :
            ]
            cached["value"] = np.concatenate((cached["value"], value), axis=1)[
                :, -config.max_seq_len :
            ]
            all_keys = cached["key"]
            all_values = cached["value"]
            if repeat > 1:
                all_keys = np.repeat(all_keys, repeat, axis=2)
                all_values = np.repeat(all_values, repeat, axis=2)
            scores = np.einsum("bthd,bshd->bhts", query, all_keys, optimize=True)
            scores = scores / math.sqrt(config.head_dim)
            probabilities = _softmax(scores)
            attention = np.einsum(
                "bhts,bshd->bthd", probabilities, all_values, optimize=True
            ).reshape(1, 1, config.d_model)
            x = x + self._linear(attention, f"{prefix}.wo")
            normalized = _rms_norm(
                x, self.parameters[f"{prefix}.ffn_norm"], config.rms_epsilon
            )
            gate = _silu(self._linear(normalized, f"{prefix}.w1"))
            up = self._linear(normalized, f"{prefix}.w3")
            x = x + self._linear(gate * up, f"{prefix}.w2")
        x = _rms_norm(x, self.parameters["final_norm"], config.rms_epsilon)
        return self._project_head(x)[0, 0]

    def prefill_kv_cache(
        self, token_ids: list[int], should_cancel: Callable[[], bool] | None = None,
    ) -> tuple[np.ndarray, list[dict[str, np.ndarray]]]:
        if not token_ids:
            raise ValueError("At least one prompt token is required")
        cache = self.init_kv_cache()
        logits: np.ndarray | None = None
        start = max(0, len(token_ids) - self.config.max_seq_len)
        for position, token_id in enumerate(token_ids[start:], start=start):
            if should_cancel is not None and should_cancel():
                break
            logits = self.decode_step(int(token_id), position, cache)
        if logits is None:
            logits = np.zeros(self.config.vocab_size, dtype=np.float32)
        return logits, cache

    def sample(
        self,
        prompt_ids: list[int],
        *,
        max_new_tokens: int = 48,
        temperature: float = 0.75,
        top_k: int = 40,
        top_p: float = 0.92,
        repetition_penalty: float = 1.08,
        eos_id: int = 2,
        seed: int | None = None,
        should_cancel: Callable[[], bool] | None = None,
        suppress_token_ids: tuple[int, ...] = (),
    ) -> list[int]:
        rng = np.random.default_rng(seed)
        tokens = list(prompt_ids[-self.config.max_seq_len :])
        logits, cache = self.prefill_kv_cache(tokens, should_cancel)
        position = len(prompt_ids)
        generated: list[int] = []
        for _ in range(max(0, max_new_tokens)):
            if should_cancel is not None and should_cancel():
                break
            logits = logits.copy()
            for suppressed in suppress_token_ids:
                if 0 <= int(suppressed) < logits.size:
                    logits[int(suppressed)] = -1e30
            if repetition_penalty > 1.0:
                for token in set(tokens[-64:]):
                    logits[token] = (
                        logits[token] / repetition_penalty
                        if logits[token] >= 0
                        else logits[token] * repetition_penalty
                    )
            if temperature <= 0:
                token = int(np.argmax(logits))
            else:
                logits = logits / max(temperature, 1e-4)
                candidates = np.arange(logits.size)
                if 0 < top_k < logits.size:
                    keep = np.argpartition(logits, -top_k)[-top_k:]
                    candidates = candidates[keep]
                    logits = logits[keep]
                order = np.argsort(logits)[::-1]
                ordered_logits = logits[order]
                probabilities = _softmax(ordered_logits)
                if 0.0 < top_p < 1.0:
                    cumulative = np.cumsum(probabilities)
                    count = max(1, int(np.searchsorted(cumulative, top_p)) + 1)
                    order = order[:count]
                    probabilities = probabilities[:count]
                    probabilities = probabilities / probabilities.sum()
                token = int(rng.choice(candidates[order], p=probabilities))
            generated.append(token)
            tokens.append(token)
            if token == eos_id:
                break
            logits = self.decode_step(token, position, cache)
            position += 1
            if len(tokens) > self.config.max_seq_len:
                tokens = tokens[-self.config.max_seq_len :]
        return generated

    def save_quantized(self, path: Path, metadata: dict[str, Any]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        arrays: dict[str, np.ndarray] = {}
        parameter_checksums: dict[str, str] = {}
        for name, value in self.parameters.items():
            if value.ndim == 1:
                encoded = value.astype(np.float16)
                arrays[f"f::{name}"] = encoded
                parameter_checksums[name] = hashlib.sha256(encoded.tobytes()).hexdigest()
                continue
            existing_scale = self.quantization_scales.get(name)
            if existing_scale is not None and value.dtype == np.int8:
                scale = existing_scale
                quantized = value
            else:
                maximum = float(np.max(np.abs(value)))
                scale = maximum / 127.0 if maximum else 1.0
                quantized = np.clip(np.rint(value / scale), -127, 127).astype(np.int8)
            encoded_scale = np.asarray(scale, dtype=np.float32)
            arrays[f"q::{name}"] = quantized
            arrays[f"s::{name}"] = encoded_scale
            parameter_checksums[name] = hashlib.sha256(
                quantized.tobytes() + encoded_scale.tobytes()
            ).hexdigest()
        aggregate = hashlib.sha256()
        for name, checksum in sorted(parameter_checksums.items()):
            aggregate.update(name.encode("utf-8"))
            aggregate.update(checksum.encode("ascii"))
        model_metadata = {
            **self.metadata,
            **metadata,
            "model_format": self.MODEL_FORMAT,
            "config": self.config.to_dict(),
            "parameter_count": self.parameter_count,
            "quantization": "symmetric_per_tensor_int8",
            "pretrained_source": None,
            "integrity": {
                "algorithm": "sha256",
                "weights_sha256": aggregate.hexdigest(),
                "parameter_sha256": parameter_checksums,
            },
        }
        arrays["__metadata__"] = np.frombuffer(
            json.dumps(model_metadata, ensure_ascii=False, sort_keys=True).encode("utf-8"),
            dtype=np.uint8,
        )
        np.savez_compressed(path, **arrays)

    @classmethod
    def load(cls, path: Path, *, dequantize: bool = False) -> "JarvisTransformer":
        try:
            with np.load(path, allow_pickle=False) as archive:
                metadata = json.loads(bytes(archive["__metadata__"]).decode("utf-8"))
                model_format = metadata.get("model_format")
                if model_format not in {cls.MODEL_FORMAT, cls.LEGACY_MODEL_FORMAT}:
                    raise NeuralModelError("Unsupported JARVIS neural model file")
                config = TransformerConfig.from_dict(dict(metadata["config"]))
                parameters: dict[str, np.ndarray] = {}
                quantization_scales: dict[str, float] = {}
                observed_checksums: dict[str, str] = {}
                for key in archive.files:
                    if key.startswith("f::"):
                        name = key[3:]
                        raw = archive[key]
                        observed_checksums[name] = hashlib.sha256(raw.tobytes()).hexdigest()
                        parameters[name] = raw.astype(np.float32) if dequantize else raw.copy()
                    elif key.startswith("q::"):
                        name = key[3:]
                        raw = archive[key]
                        scale = archive[f"s::{name}"]
                        observed_checksums[name] = hashlib.sha256(
                            raw.tobytes() + scale.tobytes()
                        ).hexdigest()
                        if dequantize:
                            parameters[name] = raw.astype(np.float32) * float(scale)
                        else:
                            parameters[name] = raw.copy()
                            quantization_scales[name] = float(scale)
                if model_format == cls.MODEL_FORMAT:
                    integrity = metadata.get("integrity", {})
                    expected = integrity.get("parameter_sha256", {}) if isinstance(integrity, dict) else {}
                    if expected != observed_checksums:
                        raise NeuralModelError("JARVIS model parameter checksum failed")
                    aggregate = hashlib.sha256()
                    for name, checksum in sorted(observed_checksums.items()):
                        aggregate.update(name.encode("utf-8"))
                        aggregate.update(checksum.encode("ascii"))
                    if aggregate.hexdigest() != integrity.get("weights_sha256"):
                        raise NeuralModelError("JARVIS model aggregate checksum failed")
        except (OSError, KeyError, ValueError, json.JSONDecodeError) as exc:
            raise NeuralModelError(f"Cannot load JARVIS neural model: {exc}") from exc
        return cls(config, parameters, metadata, quantization_scales)

    @classmethod
    def peek_metadata(cls, path: Path) -> dict[str, Any]:
        try:
            with np.load(path, allow_pickle=False) as archive:
                metadata = json.loads(bytes(archive["__metadata__"]).decode("utf-8"))
        except (OSError, KeyError, ValueError, json.JSONDecodeError) as exc:
            raise NeuralModelError(f"Cannot read JARVIS neural metadata: {exc}") from exc
        if metadata.get("model_format") not in {cls.MODEL_FORMAT, cls.LEGACY_MODEL_FORMAT}:
            raise NeuralModelError("Unsupported JARVIS neural model file")
        if metadata.get("pretrained_source") not in (None, ""):
            raise NeuralModelError("External pretrained weights are forbidden")
        return dict(metadata)

    @staticmethod
    def file_sha256(path: Path) -> str:
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
        return digest.hexdigest()
