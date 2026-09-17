from __future__ import annotations

import argparse
import json
import math
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from jarvis.neural.tokenizer import JarvisTokenizer  # noqa: E402
from jarvis.neural.transformer import JarvisTransformer  # noqa: E402


def _group(row: dict[str, Any]) -> str:
    stage = int(row.get("stage", 0) or 0)
    stage_groups = {
        1: "language", 2: f"conversation_{row.get('language', 'mixed')}",
        3: "knowledge", 4: "instruction", 5: "action", 6: "entity",
        7: "tool", 8: "arguments", 9: "context", 10: "planning",
        11: "reasoning", 12: "recovery", 13: "search_decision",
        14: "personality",
    }
    if stage in stage_groups:
        return stage_groups[stage]
    category = str(row.get("category", ""))
    if category in {"persian_conversation", "english_conversation", "conversation"}:
        return f"conversation_{row.get('language', 'mixed')}"
    if category == "tool_calling":
        return "tool"
    if category in {"context", "multi_step_tasks"}:
        return "context"
    return category or "other"


def evaluate(
    model_path: Path,
    tokenizer_path: Path,
    *,
    maximum_cases: int = 128,
) -> dict[str, Any]:
    model = JarvisTransformer.load(model_path)
    tokenizer = JarvisTokenizer.load(tokenizer_path)
    rows = [
        json.loads(line)
        for line in (ROOT / "datasets" / "splits" / "dataset_v003_test.jsonl")
        .read_text(encoding="utf-8").splitlines()
        if line
    ][: max(1, maximum_cases)]
    losses: list[float] = []
    correct = defaultdict(int)
    totals = defaultdict(int)
    sequence_limit = min(model.config.max_seq_len, 64)
    for row in rows:
        prompt = [
            tokenizer.bos_id,
            tokenizer.special_to_id["<user>"],
            *tokenizer.encode(str(row["input"])),
            tokenizer.special_to_id["<assistant>"],
        ]
        response = [*tokenizer.encode(str(row["output"])), tokenizer.eos_id]
        tokens = (prompt + response)[-sequence_limit:]
        prompt_in_crop = max(1, len(prompt) - max(0, len(prompt) + len(response) - sequence_limit))
        if len(tokens) < 2:
            continue
        logits = model.forward(tokens[:-1])[0]
        targets = np.asarray(tokens[1:], dtype=np.int64)
        start = max(0, prompt_in_crop - 2)
        selected_logits = logits[start:]
        selected_targets = targets[start:]
        if selected_targets.size == 0:
            continue
        maximum = np.max(selected_logits, axis=1, keepdims=True)
        log_sum_exp = np.log(np.sum(np.exp(selected_logits - maximum), axis=1)) + maximum[:, 0]
        token_loss = log_sum_exp - selected_logits[
            np.arange(selected_targets.size), selected_targets
        ]
        losses.extend(float(value) for value in token_loss)
        group = _group(row)
        predictions = np.argmax(selected_logits, axis=1)
        correct[group] += int(np.sum(predictions == selected_targets))
        totals[group] += int(selected_targets.size)

    mean_loss = float(np.mean(losses)) if losses else float("inf")

    def score(*groups: str) -> float | None:
        numerator = sum(correct[group] for group in groups)
        denominator = sum(totals[group] for group in groups)
        return round(numerator / denominator, 6) if denominator else None

    tool_score = score("tool")
    result = {
        "format": "jarvis-evaluation-v1",
        "model": str(model_path.relative_to(ROOT)) if model_path.is_relative_to(ROOT) else str(model_path),
        "dataset_version": "dataset_v003",
        "cases": len(rows),
        "metric_method": "teacher_forced_next_token_accuracy",
        "evaluated_token_counts": dict(sorted(totals.items())),
        "cross_entropy": round(mean_loss, 6),
        "perplexity": round(math.exp(min(20.0, mean_loss)), 6),
        "persian_conversation_score": score("conversation_fa"),
        "english_conversation_score": score("conversation_en"),
        "tool_accuracy": tool_score,
        "action_accuracy": score("action"),
        "argument_accuracy": score("arguments"),
        "entity_accuracy": score("entity"),
        "context_accuracy": score("context"),
        "knowledge_accuracy": score("knowledge"),
        "instruction_accuracy": score("instruction"),
        "planning_accuracy": score("planning"),
        "reasoning_accuracy": score("reasoning"),
        "recovery_accuracy": score("recovery"),
        "search_decision_accuracy": score("search_decision"),
        "personality_accuracy": score("personality"),
        "token_accuracy": round(sum(correct.values()) / max(1, sum(totals.values())), 6),
        "evaluated_at_utc": datetime.now(timezone.utc).isoformat(),
        "pretrained_source": None,
    }
    encoded = json.dumps(result, ensure_ascii=False, indent=2) + "\n"
    (ROOT / "models" / "model_evaluation_v7.json").write_text(encoded, encoding="utf-8")
    (ROOT / "models" / "evaluation_latest.json").write_text(encoded, encoding="utf-8")
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate a project-owned JARVIS checkpoint")
    parser.add_argument("--model", type=Path, default=ROOT / "models" / "jarvis_nano_v07.npz")
    parser.add_argument("--tokenizer", type=Path, default=ROOT / "models" / "jarvis_tokenizer_v003.json")
    parser.add_argument("--maximum-cases", type=int, default=128)
    args = parser.parse_args()
    print(json.dumps(evaluate(args.model, args.tokenizer, maximum_cases=args.maximum_cases), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
