from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

import numpy as np
from scipy.sparse import csr_matrix
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

from jarvis.agent.semantic_router_v18 import SemanticIntentRouterV18

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "datasets" / "splits"
MODEL = ROOT / "models" / "semantic_router_v18.npz"
REPORT = ROOT / "training" / "semantic_router_v18_metrics.json"
FEATURE_SIZE = 262_144
SEED = 18092026
LABELS = SemanticIntentRouterV18.DEFAULT_LABELS


def rows(split: str) -> list[dict]:
    path = DATA / f"dataset_v009_{split}.jsonl"
    out = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            item = json.loads(line)
            if item.get("origin") == "jarvis-routing-v18":
                out.append(item)
    return out


def vectorize(items: Iterable[dict]) -> tuple[csr_matrix, np.ndarray]:
    items = list(items)
    indptr = [0]
    indices: list[int] = []
    values: list[float] = []
    y: list[int] = []
    label_to_id = {label: i for i, label in enumerate(LABELS)}
    for item in items:
        idx, val = SemanticIntentRouterV18.sparse_features(item["input"], FEATURE_SIZE)
        order = np.argsort(idx)
        idx = idx[order]
        val = val[order]
        indices.extend(int(v) for v in idx)
        values.extend(float(v) for v in val)
        indptr.append(len(indices))
        y.append(label_to_id[item["output"]])
    x = csr_matrix(
        (np.asarray(values, dtype=np.float32), np.asarray(indices, dtype=np.int32), np.asarray(indptr, dtype=np.int32)),
        shape=(len(items), FEATURE_SIZE),
        dtype=np.float32,
    )
    return x, np.asarray(y, dtype=np.int64)


def evaluate(clf: SGDClassifier, x: csr_matrix, y: np.ndarray) -> dict:
    pred = clf.predict(x)
    cm = confusion_matrix(y, pred, labels=np.arange(len(LABELS)))
    per_class = {}
    for i, label in enumerate(LABELS):
        denom = int(cm[i].sum())
        per_class[label] = float(cm[i, i] / denom) if denom else None
    return {
        "examples": int(len(y)),
        "accuracy": float(accuracy_score(y, pred)),
        "per_class_accuracy": per_class,
        "confusion_matrix": cm.tolist(),
    }


def main() -> None:
    train_rows, val_rows, test_rows = rows("train"), rows("validation"), rows("test")
    x_train, y_train = vectorize(train_rows)
    x_val, y_val = vectorize(val_rows)
    x_test, y_test = vectorize(test_rows)

    clf = SGDClassifier(
        loss="log_loss",
        penalty="l2",
        alpha=2e-6,
        learning_rate="optimal",
        max_iter=180,
        tol=1e-6,
        random_state=SEED,
        fit_intercept=True,
        class_weight="balanced",
        average=True,
    )
    clf.fit(x_train, y_train)

    # sklearn keeps class order numeric because y is integer encoded.
    weights = np.asarray(clf.coef_, dtype=np.float32)
    bias = np.asarray(clf.intercept_, dtype=np.float32)
    if weights.shape != (len(LABELS), FEATURE_SIZE):
        raise RuntimeError(f"unexpected weight shape: {weights.shape}")

    metrics = {
        "format": "jarvis-semantic-router-v18-training-report",
        "feature_size": FEATURE_SIZE,
        "labels": list(LABELS),
        "parameter_count": int(weights.size + bias.size),
        "training_examples": len(train_rows),
        "validation": evaluate(clf, x_val, y_val),
        "test": evaluate(clf, x_test, y_test),
        "train": evaluate(clf, x_train, y_train),
        "seed": SEED,
        "learner": "SGDClassifier(log_loss)+hashed word/character ngrams",
        "concept_group_split": True,
    }
    if metrics["test"]["accuracy"] < 0.96:
        raise RuntimeError(f"semantic router quality gate failed: test_accuracy={metrics['test']['accuracy']:.4f}")
    if min(v for v in metrics["test"]["per_class_accuracy"].values() if v is not None) < 0.90:
        raise RuntimeError("semantic router per-class gate failed")

    MODEL.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        MODEL,
        format=np.asarray(SemanticIntentRouterV18.FORMAT),
        labels=np.asarray(LABELS),
        weights=weights,
        bias=bias,
        temperature=np.asarray(1.0, dtype=np.float32),
        parameter_count=np.asarray(metrics["parameter_count"], dtype=np.int64),
        training_examples=np.asarray(len(train_rows), dtype=np.int64),
        dataset_version=np.asarray("dataset_v009"),
    )
    REPORT.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
