from __future__ import annotations

import math
from typing import Any

from jarvis.neural.config import TransformerConfig

try:  # PyTorch is training-only and intentionally absent from the light runtime.
    import torch
    import torch.nn as nn
    import torch.nn.functional as functional
except ImportError:  # pragma: no cover - depends on optional training environment
    torch = None  # type: ignore[assignment]
    nn = None  # type: ignore[assignment]
    functional = None  # type: ignore[assignment]


def torch_available() -> bool:
    return torch is not None


if torch is not None:

    class RMSNorm(nn.Module):
        def __init__(self, width: int, epsilon: float) -> None:
            super().__init__()
            self.weight = nn.Parameter(torch.ones(width))
            self.epsilon = epsilon

        def forward(self, value: torch.Tensor) -> torch.Tensor:
            normalized = value * torch.rsqrt(
                value.float().pow(2).mean(-1, keepdim=True) + self.epsilon
            )
            return normalized.to(value.dtype) * self.weight


    def _rope(
        value: torch.Tensor, positions: torch.Tensor, theta: float,
    ) -> torch.Tensor:
        head_dim = value.shape[-1]
        inverse = 1.0 / (
            theta
            ** (torch.arange(0, head_dim, 2, device=value.device, dtype=torch.float32) / head_dim)
        )
        angles = positions.float()[:, None] * inverse[None, :]
        cosine = angles.cos()[None, :, None, :].to(value.dtype)
        sine = angles.sin()[None, :, None, :].to(value.dtype)
        even, odd = value[..., 0::2], value[..., 1::2]
        return torch.stack((even * cosine - odd * sine, even * sine + odd * cosine), dim=-1).flatten(-2)


    class CausalAttention(nn.Module):
        def __init__(self, config: TransformerConfig) -> None:
            super().__init__()
            self.config = config
            kv_width = config.n_kv_heads * config.head_dim
            self.wq = nn.Linear(config.d_model, config.d_model, bias=False)
            self.wk = nn.Linear(config.d_model, kv_width, bias=False)
            self.wv = nn.Linear(config.d_model, kv_width, bias=False)
            self.wo = nn.Linear(config.d_model, config.d_model, bias=False)

        def forward(
            self,
            value: torch.Tensor,
            cache: tuple[torch.Tensor, torch.Tensor] | None = None,
        ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
            batch, length, _ = value.shape
            past_length = 0 if cache is None else cache[0].shape[1]
            positions = torch.arange(
                past_length, past_length + length, device=value.device
            )
            query = self.wq(value).view(
                batch, length, self.config.n_heads, self.config.head_dim
            )
            key = self.wk(value).view(
                batch, length, self.config.n_kv_heads, self.config.head_dim
            )
            values = self.wv(value).view(
                batch, length, self.config.n_kv_heads, self.config.head_dim
            )
            query = _rope(query, positions, self.config.rope_theta)
            key = _rope(key, positions, self.config.rope_theta)
            if cache is not None:
                key = torch.cat((cache[0], key), dim=1)
                values = torch.cat((cache[1], values), dim=1)
            new_cache = (key, values)
            repeat = self.config.n_heads // self.config.n_kv_heads
            attention_key = key.repeat_interleave(repeat, dim=2) if repeat > 1 else key
            attention_value = values.repeat_interleave(repeat, dim=2) if repeat > 1 else values
            query = query.transpose(1, 2)
            attention_key = attention_key.transpose(1, 2)
            attention_value = attention_value.transpose(1, 2)
            scores = query @ attention_key.transpose(-2, -1) / math.sqrt(self.config.head_dim)
            total_length = attention_key.shape[-2]
            query_positions = torch.arange(
                past_length, past_length + length, device=value.device
            )[:, None]
            key_positions = torch.arange(total_length, device=value.device)[None, :]
            scores = scores.masked_fill(
                key_positions[None, None, :, :] > query_positions[None, None, :, :],
                torch.finfo(scores.dtype).min,
            )
            probabilities = functional.softmax(scores.float(), dim=-1).to(scores.dtype)
            output = (probabilities @ attention_value).transpose(1, 2).contiguous()
            return self.wo(output.view(batch, length, self.config.d_model)), new_cache


    class SwiGLU(nn.Module):
        def __init__(self, config: TransformerConfig) -> None:
            super().__init__()
            self.w1 = nn.Linear(config.d_model, config.ffn_hidden, bias=False)
            self.w2 = nn.Linear(config.ffn_hidden, config.d_model, bias=False)
            self.w3 = nn.Linear(config.d_model, config.ffn_hidden, bias=False)

        def forward(self, value: torch.Tensor) -> torch.Tensor:
            return self.w2(functional.silu(self.w1(value)) * self.w3(value))


    class TransformerBlock(nn.Module):
        def __init__(self, config: TransformerConfig) -> None:
            super().__init__()
            self.attn_norm = RMSNorm(config.d_model, config.rms_epsilon)
            self.attention = CausalAttention(config)
            self.ffn_norm = RMSNorm(config.d_model, config.rms_epsilon)
            self.feed_forward = SwiGLU(config)

        def forward(
            self,
            value: torch.Tensor,
            cache: tuple[torch.Tensor, torch.Tensor] | None = None,
        ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
            attention, new_cache = self.attention(self.attn_norm(value), cache)
            value = value + attention
            return value + self.feed_forward(self.ffn_norm(value)), new_cache


    class JarvisTorchTransformer(nn.Module):
        """Project-owned decoder Transformer; never loads external model weights."""

        def __init__(self, config: TransformerConfig) -> None:
            super().__init__()
            self.config = config
            self.tok_embeddings = nn.Embedding(config.vocab_size, config.d_model)
            self.layers = nn.ModuleList(
                TransformerBlock(config) for _ in range(config.n_layers)
            )
            self.final_norm = RMSNorm(config.d_model, config.rms_epsilon)
            self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
            if config.tie_embeddings:
                self.lm_head.weight = self.tok_embeddings.weight
            self.apply(self._initialize)

        def _initialize(self, module: nn.Module) -> None:
            if isinstance(module, (nn.Linear, nn.Embedding)):
                nn.init.normal_(module.weight, mean=0.0, std=self.config.initializer_std)

        def forward(
            self,
            input_ids: torch.Tensor,
            targets: torch.Tensor | None = None,
            cache: list[tuple[torch.Tensor, torch.Tensor] | None] | None = None,
        ) -> dict[str, Any]:
            value = self.tok_embeddings(input_ids)
            supplied_cache = cache or [None] * len(self.layers)
            new_cache: list[tuple[torch.Tensor, torch.Tensor]] = []
            for layer, layer_cache in zip(self.layers, supplied_cache, strict=True):
                value, produced = layer(value, layer_cache)
                new_cache.append(produced)
            logits = self.lm_head(self.final_norm(value))
            loss = None
            if targets is not None:
                loss = functional.cross_entropy(
                    logits.reshape(-1, logits.shape[-1]), targets.reshape(-1),
                    ignore_index=-100,
                )
            return {"logits": logits, "loss": loss, "cache": new_cache}


else:

    class JarvisTorchTransformer:  # pragma: no cover - simple optional dependency guard
        def __init__(self, _config: TransformerConfig) -> None:
            raise RuntimeError(
                "PyTorch is not installed. Run setup.ps1 -Training for CUDA/large-model training."
            )


__all__ = ["JarvisTorchTransformer", "torch_available"]
