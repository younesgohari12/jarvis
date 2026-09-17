from __future__ import annotations

import argparse
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from jarvis.neural.tokenizer import JarvisTokenizer  # noqa: E402
from jarvis.neural.transformer import JarvisTransformer  # noqa: E402


def rows(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def tokenized_case(tokenizer: JarvisTokenizer, row: dict[str, Any], max_len: int) -> tuple[list[int], list[int]]:
    context_text = "\n".join(str(x.get("content", "")) for x in row.get("context", []) if isinstance(x, dict))
    user = f"{context_text}\n{row['input']}".strip() if context_text else str(row["input"])
    prompt = [tokenizer.bos_id, tokenizer.special_to_id["<user>"], *tokenizer.encode(user), tokenizer.special_to_id["<assistant>"]]
    response = [*tokenizer.encode(str(row["output"])), tokenizer.eos_id]
    seq = (prompt + response)[: max_len + 1]
    x = seq[:-1]
    labels = seq[1:]
    boundary = max(0, len(prompt) - 2)
    mask = [i >= boundary for i in range(len(labels))]
    return x, [token if valid else -100 for token, valid in zip(labels, mask)]


def score(model: JarvisTransformer, tokenizer: JarvisTokenizer, data: list[dict[str, Any]], max_len: int) -> dict[str, Any]:
    losses: list[float] = []
    correct = total = 0
    by_language: dict[str, list[float]] = defaultdict(list)
    by_stage: dict[str, list[float]] = defaultdict(list)
    for row in data:
        x, labels = tokenized_case(tokenizer, row, max_len)
        logits = model.forward(np.asarray(x, dtype=np.int64))[0]
        targets = np.asarray(labels, dtype=np.int64)
        valid = targets >= 0
        if not np.any(valid):
            continue
        selected_logits = logits[valid]
        selected_targets = targets[valid]
        maximum = np.max(selected_logits, axis=-1, keepdims=True)
        exp = np.exp(selected_logits - maximum)
        probs = exp / np.sum(exp, axis=-1, keepdims=True)
        p = probs[np.arange(selected_targets.size), selected_targets]
        loss = -float(np.mean(np.log(np.maximum(p, 1e-12))))
        losses.append(loss)
        by_language[str(row.get("language", "unknown"))].append(loss)
        by_stage[str(row.get("stage", 0))].append(loss)
        pred = np.argmax(selected_logits, axis=-1)
        correct += int(np.sum(pred == selected_targets)); total += int(selected_targets.size)
    mean = float(np.mean(losses)) if losses else float("inf")
    return {
        "cases": len(losses),
        "cross_entropy": round(mean, 6),
        "perplexity": round(math.exp(min(20.0, mean)), 6),
        "token_accuracy": round(correct / max(1, total), 6),
        "by_language_cross_entropy": {k: round(float(np.mean(v)), 6) for k, v in sorted(by_language.items())},
        "by_stage_cross_entropy": {k: round(float(np.mean(v)), 6) for k, v in sorted(by_stage.items(), key=lambda kv: int(kv[0]))},
    }


def main(args: argparse.Namespace) -> dict[str, Any]:
    tokenizer = JarvisTokenizer.load(ROOT / "models" / "jarvis_tokenizer_v003.json")
    test = rows(ROOT / "datasets" / "splits" / "dataset_v004_test.jsonl")
    curated = [row for row in test if row.get("origin") == "jarvis-curated-v12"]
    # Deterministic bounded holdout for a fast CPU gate.
    curated = sorted(curated, key=lambda row: str(row.get("id", "")))[: int(args.maximum_cases)]
    baseline = JarvisTransformer.load(ROOT / args.baseline)
    candidate = JarvisTransformer.load(ROOT / args.candidate)
    base_score = score(baseline, tokenizer, curated, int(args.sequence_length))
    cand_score = score(candidate, tokenizer, curated, int(args.sequence_length))
    improvement = 100.0 * (base_score["cross_entropy"] - cand_score["cross_entropy"]) / max(base_score["cross_entropy"], 1e-9)
    result = {
        "format": "jarvis-v12-quality-gate-v1",
        "dataset": "datasets/splits/dataset_v004_test.jsonl",
        "scope": "curated_v12_holdout_only",
        "baseline": args.baseline,
        "candidate": args.candidate,
        "baseline_metrics": base_score,
        "candidate_metrics": cand_score,
        "cross_entropy_improvement_pct": round(improvement, 3),
        "candidate_improves_curated_holdout": cand_score["cross_entropy"] < base_score["cross_entropy"],
    }
    path = ROOT / "models" / "evaluation_v12_quality_gate.json"
    path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return result


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    p.add_argument("--baseline", default="models/jarvis_nano_v08.npz")
    p.add_argument("--candidate", default="models/jarvis_nano_v12_candidate.npz")
    p.add_argument("--maximum-cases", type=int, default=120)
    p.add_argument("--sequence-length", type=int, default=64)
    return p


if __name__ == "__main__":
    main(parser().parse_args())
