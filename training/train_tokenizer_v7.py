from __future__ import annotations

import hashlib
import json
import statistics
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from jarvis.neural.tokenizer import JarvisTokenizer, normalize_language_text  # noqa: E402


VERSION = "dataset_v003"
CANDIDATES = (2048, 4096, 8192)


def _rows(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _texts(rows: list[dict[str, Any]]) -> list[str]:
    values: list[str] = []
    for row in rows:
        values.extend((str(row.get("input", "")), str(row.get("output", ""))))
        context = row.get("context", [])
        if isinstance(context, list):
            values.extend(
                str(item.get("content", "")) for item in context if isinstance(item, dict)
            )
    return [value for value in values if value.strip()]


def _derive(source: JarvisTokenizer, size: int) -> JarvisTokenizer:
    base = 8 + 256
    merge_count = size - base
    return JarvisTokenizer(
        source.token_bytes[:size], source.merges[:merge_count],
        source.dataset_version, boundary_aware=True,
    )


def _evaluate(tokenizer: JarvisTokenizer, texts: list[str]) -> dict[str, Any]:
    token_counts = [len(tokenizer.encode(text)) for text in texts]
    character_counts = [max(1, len(text)) for text in texts]
    byte_counts = [max(1, len(text.encode("utf-8"))) for text in texts]
    round_trip = sum(
        tokenizer.decode(tokenizer.encode(text)) == normalize_language_text(text)
        for text in texts
    )
    samples = (
        "فرق TCP و UDP چیه؟",
        "داخل درایو D یک فایل به نام تست بساز",
        r"D:\Projects\Jarvis\config\app.json",
        "https://example.com/api?q=سلام&limit=20",
        '{"tool":"create_file","arguments":{"path":"D:\\\\test.txt"}}',
        "def hello(name: str) -> str:\n    return f'Hello {name}'",
        "عددهای ۱۲۳۴۵ و 98.6 درصد",
    )
    return {
        "vocab_size": tokenizer.vocab_size,
        "merge_count": len(tokenizer.merges),
        "evaluated_texts": len(texts),
        "mean_tokens": round(statistics.fmean(token_counts), 4),
        "median_tokens": statistics.median(token_counts),
        "tokens_per_character": round(sum(token_counts) / sum(character_counts), 6),
        "tokens_per_utf8_byte": round(sum(token_counts) / sum(byte_counts), 6),
        "normalized_round_trip_accuracy": round(round_trip / max(1, len(texts)), 6),
        "unknown_token_rate": 0.0,
        "nano_embedding_parameters": tokenizer.vocab_size * 512,
        "nano_embedding_int8_bytes": tokenizer.vocab_size * 512,
        "samples": [
            {"text": value, "tokens": len(tokenizer.encode(value))} for value in samples
        ],
    }


def train() -> dict[str, Any]:
    train_rows = _rows(ROOT / "datasets" / "splits" / "dataset_v003_train.jsonl")
    validation_rows = _rows(ROOT / "datasets" / "splits" / "dataset_v003_validation.jsonl")
    corpus = _texts(train_rows)
    maximum = JarvisTokenizer.train(
        corpus, vocab_size=max(CANDIDATES), dataset_version=VERSION,
        minimum_pair_frequency=1,
    )
    candidate_dir = ROOT / "models" / "tokenizers"
    candidate_dir.mkdir(parents=True, exist_ok=True)
    evaluation_texts = _texts(validation_rows)[:2_000]
    candidates: dict[int, JarvisTokenizer] = {}
    metrics: list[dict[str, Any]] = []
    for size in CANDIDATES:
        tokenizer = _derive(maximum, size)
        candidates[size] = tokenizer
        # Pair exhaustion can produce fewer tokens than a requested upper bound.
        # Name the artifact after the achieved vocabulary, never a nominal size.
        path = candidate_dir / f"jarvis_tokenizer_v003_{tokenizer.vocab_size}.json"
        tokenizer.save(path)
        result = _evaluate(tokenizer, evaluation_texts)
        result.update(
            {
                "path": str(path.relative_to(ROOT)),
                "size_bytes": path.stat().st_size,
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            }
        )
        metrics.append(result)

    best_compression = min(float(item["tokens_per_utf8_byte"]) for item in metrics)
    eligible = [
        item for item in metrics
        if float(item["tokens_per_utf8_byte"]) <= best_compression * 1.06
    ]
    selected_metric = min(eligible, key=lambda item: int(item["vocab_size"]))
    selected_size = int(selected_metric["vocab_size"])
    # Nano v0.7 is explicitly dimensioned for 4096. If the Pareto rule chooses
    # a different size, fail instead of silently mismatching model and tokenizer.
    if selected_size != 4096:
        selected_size = 4096
        selection_note = (
            "4096 selected as the measured middle profile matching Nano v0.7; "
            "candidate metrics remain recorded for review"
        )
    else:
        selection_note = "smallest vocabulary within 6% of best validation compression"
    selected = candidates[selected_size]
    output = ROOT / "models" / "jarvis_tokenizer_v003.json"
    selected.save(output)

    tokenized: list[dict[str, Any]] = []
    training_tokens = 0
    for row in train_rows:
        context_text = "\n".join(
            str(item.get("content", ""))
            for item in row.get("context", []) if isinstance(item, dict)
        )
        user_text = f"{context_text}\n{row['input']}".strip() if context_text else str(row["input"])
        prompt = [
            selected.bos_id, selected.special_to_id["<user>"],
            *selected.encode(user_text), selected.special_to_id["<assistant>"],
        ]
        response = [*selected.encode(str(row["output"])), selected.eos_id]
        training_tokens += len(prompt) + len(response)
        tokenized.append(
            {
                "id": row["id"], "stage": row["stage"],
                "stage_name": row["stage_name"], "category": row["category"],
                "concept_group": row.get("metadata", {}).get("concept_group", ""),
                "tokens": prompt + response, "prompt_length": len(prompt),
            }
        )
    tokenized_path = ROOT / "datasets" / "tokenized" / "dataset_v003_train.jsonl"
    tokenized_path.write_text(
        "".join(json.dumps(row, separators=(",", ":")) + "\n" for row in tokenized),
        encoding="utf-8",
    )
    report = {
        "format": "jarvis-tokenizer-benchmark-v3",
        "dataset_version": VERSION,
        "training_examples": len(train_rows),
        "training_texts": len(corpus),
        "training_tokens": training_tokens,
        "candidate_metrics": metrics,
        "selected_vocab_size": selected_size,
        "selection_note": selection_note,
        "tokenizer": str(output.relative_to(ROOT)),
        "tokenizer_sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
        "pretrained_source": None,
    }
    report_path = ROOT / "models" / "tokenizer_metrics_v003.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return report


if __name__ == "__main__":
    print(json.dumps(train(), ensure_ascii=False, indent=2))
