from __future__ import annotations

import gc
import json
import statistics
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from jarvis.neural.tokenizer import JarvisTokenizer  # noqa: E402
from jarvis.neural.transformer import JarvisTransformer  # noqa: E402


def _measure(
    model_path: Path, tokenizer: JarvisTokenizer, precision: str,
    reference: np.ndarray | None,
) -> tuple[dict[str, Any], np.ndarray]:
    started = time.perf_counter()
    model = JarvisTransformer.load(model_path, dequantize=precision != "int8")
    if precision == "fp16":
        for name in model.parameters:
            model.parameters[name] = model.parameters[name].astype(np.float16)
    load_ms = (time.perf_counter() - started) * 1000.0
    prompt = [
        tokenizer.bos_id, tokenizer.special_to_id["<user>"],
        *tokenizer.encode("فرق TCP و UDP چیه؟"),
        tokenizer.special_to_id["<assistant>"],
    ]
    latencies: list[float] = []
    rates: list[float] = []
    logits: np.ndarray | None = None
    for seed in range(2):
        run_started = time.perf_counter()
        logits, _cache = model.prefill_kv_cache(prompt)
        latencies.append((time.perf_counter() - run_started) * 1000.0)
        generation_started = time.perf_counter()
        generated = model.sample(prompt, max_new_tokens=8, seed=seed)
        rates.append(len(generated) / max(1e-9, time.perf_counter() - generation_started))
    assert logits is not None
    result: dict[str, Any] = {
        "precision": precision,
        "resident_weight_bytes": model.memory_bytes,
        "resident_weight_mb": round(model.memory_bytes / (1024 * 1024), 3),
        "load_latency_ms": round(load_ms, 3),
        "first_token_latency_ms": round(statistics.median(latencies), 3),
        "tokens_per_second": round(statistics.median(rates), 3),
        "runtime_quantized": model.quantized_runtime,
        "maximum_logit_error_vs_fp32": (
            0.0 if reference is None
            else round(float(np.max(np.abs(logits - reference))), 7)
        ),
    }
    del model
    gc.collect()
    return result, logits


def benchmark() -> dict[str, Any]:
    model_path = ROOT / "models" / "jarvis_nano_v07.npz"
    tokenizer = JarvisTokenizer.load(ROOT / "models" / "jarvis_tokenizer_v003.json")
    fp32, reference = _measure(model_path, tokenizer, "fp32", None)
    fp16, _ = _measure(model_path, tokenizer, "fp16", reference)
    int8, _ = _measure(model_path, tokenizer, "int8", reference)
    result = {
        "format": "jarvis-quantization-benchmark-v1",
        "model": "jarvis_nano_v07.npz",
        "parameters": 23_077_376,
        "device": "cpu",
        "profiles": [fp32, fp16, int8],
        "selected_runtime": "int8",
        "selection_reason": "lowest resident weight memory with project loader retaining INT8 matrices",
        "pretrained_source": None,
        "benchmarked_at_utc": datetime.now(UTC).isoformat(),
    }
    output = ROOT / "models" / "quantization_benchmark_v7.json"
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return result


if __name__ == "__main__":
    print(json.dumps(benchmark(), ensure_ascii=False, indent=2))
