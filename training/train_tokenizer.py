from __future__ import annotations

import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from jarvis.neural.config import TransformerConfig  # noqa: E402
from jarvis.neural.tokenizer import JarvisTokenizer  # noqa: E402


def _rows(path: Path) -> list[dict[str, object]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def train() -> dict[str, object]:
    config = TransformerConfig.load(ROOT / "configs" / "nano.json")
    source = ROOT / "datasets" / "splits" / "dataset_v002_train.jsonl"
    rows = _rows(source)
    texts: list[str] = []
    for row in rows:
        texts.extend((str(row["input"]), str(row["output"])))
        context = row.get("context")
        if isinstance(context, list):
            texts.extend(str(item.get("content", "")) for item in context if isinstance(item, dict))
    tokenizer = JarvisTokenizer.train(
        texts,
        vocab_size=config.vocab_size,
        dataset_version="dataset_v002",
        minimum_pair_frequency=1,
    )
    if tokenizer.vocab_size != config.vocab_size:
        raise RuntimeError(
            f"Tokenizer produced {tokenizer.vocab_size} tokens, expected {config.vocab_size}"
        )
    output = ROOT / "models" / "jarvis_tokenizer_v002.json"
    tokenizer.save(output)
    encoded_rows: list[dict[str, object]] = []
    token_total = 0
    for row in rows:
        prompt = [
            tokenizer.bos_id,
            tokenizer.special_to_id["<user>"],
            *tokenizer.encode(str(row["input"])),
            tokenizer.special_to_id["<assistant>"],
        ]
        response = [*tokenizer.encode(str(row["output"])), tokenizer.eos_id]
        token_total += len(prompt) + len(response)
        encoded_rows.append(
            {
                "id": row["id"],
                "stage": row["stage"],
                "category": row["category"],
                "tokens": prompt + response,
                "prompt_length": len(prompt),
            }
        )
    tokenized_path = ROOT / "datasets" / "tokenized" / "dataset_v002_train.jsonl"
    tokenized_path.write_text(
        "".join(json.dumps(row, separators=(",", ":")) + "\n" for row in encoded_rows),
        encoding="utf-8",
    )
    stats = {
        "format": JarvisTokenizer.FORMAT,
        "dataset_version": "dataset_v002",
        "vocab_size": tokenizer.vocab_size,
        "merge_count": len(tokenizer.merges),
        "training_examples": len(rows),
        "training_tokens": token_total,
        "average_tokens_per_example": round(token_total / max(1, len(rows)), 3),
        "pretrained_source": None,
    }
    (ROOT / "models" / "tokenizer_metrics_v002.json").write_text(
        json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return stats


if __name__ == "__main__":
    print(json.dumps(train(), ensure_ascii=False, indent=2))
