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

VERSION = "dataset_v005"


def rows(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def tokenize() -> dict[str, Any]:
    tokenizer_path = ROOT / "models" / "jarvis_tokenizer_v003.json"
    tokenizer = JarvisTokenizer.load(tokenizer_path)
    train = rows(ROOT / "datasets" / "splits" / "dataset_v005_train.jsonl")
    validation = rows(ROOT / "datasets" / "splits" / "dataset_v005_validation.jsonl")
    output_rows = []
    token_counts=[]
    bytes_total=0
    round_trip=0
    total_texts=0
    for row in train:
        context_text = "\n".join(
            str(item.get("content", ""))
            for item in row.get("context", []) if isinstance(item, dict)
        )
        user_text = f"{context_text}\n{row['input']}".strip() if context_text else str(row["input"])
        prompt = [tokenizer.bos_id, tokenizer.special_to_id["<user>"], *tokenizer.encode(user_text), tokenizer.special_to_id["<assistant>"]]
        response = [*tokenizer.encode(str(row["output"])), tokenizer.eos_id]
        output_rows.append({
            "id": row["id"], "stage": row["stage"], "stage_name": row["stage_name"],
            "category": row["category"], "language": row.get("language", "unknown"),
            "concept_group": row.get("metadata", {}).get("concept_group", ""),
            "curated_source_id": row.get("curated_source_id", ""), "origin": row.get("origin", ""),
            "tokens": prompt + response, "prompt_length": len(prompt),
        })
        token_counts.append(len(prompt)+len(response))
    out = ROOT / "datasets" / "tokenized" / "dataset_v005_train.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("".join(json.dumps(r,separators=(",",":"),ensure_ascii=False)+"\n" for r in output_rows),encoding="utf-8")

    validation_texts=[]
    for row in validation[:2000]:
        validation_texts.extend((str(row.get("input","")), str(row.get("output",""))))
    validation_tokens=0
    for text in validation_texts:
        encoded=tokenizer.encode(text)
        validation_tokens += len(encoded)
        bytes_total += max(1,len(text.encode("utf-8")))
        round_trip += int(tokenizer.decode(encoded)==normalize_language_text(text))
        total_texts += 1
    report={
        "format":"jarvis-v13-tokenization-report-v1",
        "dataset_version":VERSION,
        "tokenizer":"models/jarvis_tokenizer_v003.json",
        "tokenizer_sha256":hashlib.sha256(tokenizer_path.read_bytes()).hexdigest(),
        "vocab_size":tokenizer.vocab_size,
        "training_examples":len(train),
        "mean_sequence_tokens":round(statistics.fmean(token_counts),4),
        "median_sequence_tokens":statistics.median(token_counts),
        "validation_texts":total_texts,
        "validation_tokens_per_utf8_byte":round(validation_tokens/max(1,bytes_total),6),
        "normalized_round_trip_accuracy":round(round_trip/max(1,total_texts),6),
        "unknown_token_rate":0.0,
        "output":out.relative_to(ROOT).as_posix(),
        "output_sha256":hashlib.sha256(out.read_bytes()).hexdigest(),
        "note":"Reuses the release tokenizer to keep v0.8 embedding compatibility during continual training.",
        "pretrained_source":None,
    }
    (ROOT/"models"/"tokenizer_metrics_v005.json").write_text(json.dumps(report,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    return report

if __name__=="__main__":
    print(json.dumps(tokenize(),ensure_ascii=False,indent=2))
