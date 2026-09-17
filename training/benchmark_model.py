from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from jarvis.neural.tokenizer import JarvisTokenizer  # noqa: E402
from jarvis.neural.transformer import JarvisTransformer  # noqa: E402
from jarvis.runtime.hardware import HardwareManager  # noqa: E402


def benchmark(model_path: Path, tokenizer_path: Path, rounds: int = 4) -> dict[str, object]:
    hardware = HardwareManager()
    before = hardware.sample().app_ram_mb
    load_started = time.perf_counter()
    model = JarvisTransformer.load(model_path)
    tokenizer = JarvisTokenizer.load(tokenizer_path)
    load_ms = (time.perf_counter() - load_started) * 1000.0
    after = hardware.sample().app_ram_mb
    prompts = ("سلام", "hello", "what is your name?", "حالت چطوره؟")
    first_token: list[float] = []
    rates: list[float] = []
    generated = 0
    for index in range(max(1, rounds)):
        ids = [
            tokenizer.bos_id, tokenizer.special_to_id["<user>"],
            *tokenizer.encode(prompts[index % len(prompts)]),
            tokenizer.special_to_id["<assistant>"],
        ]
        started = time.perf_counter()
        _logits, cache = model.prefill_kv_cache(ids)
        first_token.append((time.perf_counter() - started) * 1000.0)
        generation_started = time.perf_counter()
        output = model.sample(ids, max_new_tokens=16, temperature=0.65, top_k=24, seed=index)
        generation_seconds = max(time.perf_counter() - generation_started, 1e-9)
        rates.append(len(output) / generation_seconds)
        generated += len(output)
    evaluation_path = ROOT / "models" / "evaluation_latest.json"
    evaluation = json.loads(evaluation_path.read_text(encoding="utf-8")) if evaluation_path.is_file() else {}
    result: dict[str, object] = {
        "format": "jarvis-benchmark-v1",
        "model_size_bytes": model_path.stat().st_size,
        "parameters": model.parameter_count,
        "quantization": model.metadata.get("quantization", "symmetric_per_tensor_int8"),
        "runtime_weight_dtype": "int8" if model.quantized_runtime else "float32",
        "runtime_quantized": model.quantized_runtime,
        "resident_weight_bytes": model.memory_bytes,
        "resident_weight_mb": round(model.memory_bytes / (1024 * 1024), 3),
        "load_latency_ms": round(load_ms, 3),
        "ram_delta_mb": round(max(0.0, after - before), 3),
        "vram_mb": 0 if not hardware.info.gpu_available else None,
        "device": "cpu",
        "tokens_per_second": round(statistics.median(rates), 3),
        "first_token_latency_ms": round(statistics.median(first_token), 3),
        "generated_tokens": generated,
        "conversation_score": evaluation.get("persian_conversation_score"),
        "tool_score": evaluation.get("tool_accuracy"),
        "context_score": evaluation.get("context_accuracy"),
        "pretrained_source": None,
        "benchmarked_at_utc": datetime.now(timezone.utc).isoformat(),
        "platform": os.name,
    }
    (ROOT / "models" / "benchmark_v07_latest.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description="Measure real JARVIS inference performance")
    parser.add_argument("--model", type=Path, default=ROOT / "models" / "jarvis_nano_v07.npz")
    parser.add_argument("--tokenizer", type=Path, default=ROOT / "models" / "jarvis_tokenizer_v003.json")
    parser.add_argument("--rounds", type=int, default=4)
    args = parser.parse_args()
    print(json.dumps(benchmark(args.model, args.tokenizer, args.rounds), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
