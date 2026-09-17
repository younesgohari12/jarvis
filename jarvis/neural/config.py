from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass(frozen=True, slots=True)
class TrainingConfig:
    sequence_length: int = 32
    micro_batch_size: int = 1
    gradient_accumulation: int = 4
    learning_rate: float = 8e-4
    minimum_learning_rate: float = 8e-5
    weight_decay: float = 0.01
    gradient_clip: float = 1.0
    warmup_steps: int = 8
    steps_per_stage: int = 8


@dataclass(frozen=True, slots=True)
class TransformerConfig:
    name: str
    profile: str
    format: str
    architecture: str
    vocab_size: int
    d_model: int
    n_layers: int
    n_heads: int
    n_kv_heads: int
    ffn_hidden: int
    max_seq_len: int
    rope_theta: float = 10_000.0
    rms_epsilon: float = 1e-5
    tie_embeddings: bool = True
    initializer_std: float = 0.02
    seed: int = 5_052_026
    training: TrainingConfig = field(default_factory=TrainingConfig)

    def __post_init__(self) -> None:
        positive = {
            "vocab_size": self.vocab_size,
            "d_model": self.d_model,
            "n_layers": self.n_layers,
            "n_heads": self.n_heads,
            "n_kv_heads": self.n_kv_heads,
            "ffn_hidden": self.ffn_hidden,
            "max_seq_len": self.max_seq_len,
        }
        for name, value in positive.items():
            if value <= 0:
                raise ValueError(f"{name} must be positive")
        if self.d_model % self.n_heads:
            raise ValueError("d_model must be divisible by n_heads")
        if self.n_heads % self.n_kv_heads:
            raise ValueError("n_heads must be divisible by n_kv_heads")
        if (self.d_model // self.n_heads) % 2:
            raise ValueError("RoPE requires an even attention head dimension")
        if self.format not in {"jarvis-transformer-v1", "jarvis-transformer-v2"}:
            raise ValueError(f"Unsupported model format: {self.format}")

    @property
    def head_dim(self) -> int:
        return self.d_model // self.n_heads

    @property
    def parameter_count(self) -> int:
        embedding = self.vocab_size * self.d_model
        attention = self.n_layers * (
            self.d_model * self.d_model * 2
            + self.d_model * (self.n_kv_heads * self.head_dim) * 2
        )
        feed_forward = self.n_layers * self.d_model * self.ffn_hidden * 3
        norms = (self.n_layers * 2 + 1) * self.d_model
        output = 0 if self.tie_embeddings else self.d_model * self.vocab_size
        return embedding + attention + feed_forward + norms + output

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "profile": self.profile,
            "format": self.format,
            "architecture": self.architecture,
            "vocab_size": self.vocab_size,
            "d_model": self.d_model,
            "n_layers": self.n_layers,
            "n_heads": self.n_heads,
            "n_kv_heads": self.n_kv_heads,
            "ffn_hidden": self.ffn_hidden,
            "max_seq_len": self.max_seq_len,
            "rope_theta": self.rope_theta,
            "rms_epsilon": self.rms_epsilon,
            "tie_embeddings": self.tie_embeddings,
            "initializer_std": self.initializer_std,
            "seed": self.seed,
            "training": {
                field_name: getattr(self.training, field_name)
                for field_name in self.training.__dataclass_fields__
            },
        }

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "TransformerConfig":
        payload = dict(value)
        payload["training"] = TrainingConfig(**dict(payload.get("training", {})))
        return cls(**payload)

    @classmethod
    def load(cls, path: Path) -> "TransformerConfig":
        with path.open("r", encoding="utf-8") as handle:
            value = json.load(handle)
        if not isinstance(value, dict):
            raise ValueError("Transformer config root must be an object")
        return cls.from_dict(value)
