from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
from jarvis.utils.text import normalize_text


@dataclass(frozen=True, slots=True)
class FramePrediction:
    label: str
    confidence: float
    margin: float


@dataclass(frozen=True, slots=True)
class SlotPrediction:
    label: str
    confidence: float
    margin: float


class _HashedLinear:
    FORMAT = ""
    PERSON = b"jrv18mdl"

    def __init__(self, model_path: Path | None = None) -> None:
        self.labels: tuple[str, ...] = ()
        self.weights: np.ndarray | None = None
        self.bias: np.ndarray | None = None
        self.feature_size = 0
        self.metadata: dict[str, object] = {}
        if model_path and model_path.is_file():
            try:
                with np.load(model_path, allow_pickle=False) as p:
                    if str(p["format"].item()) != self.FORMAT:
                        return
                    self.labels = tuple(str(x) for x in p["labels"].tolist())
                    self.weights = p["weights"].astype(np.float32)
                    self.bias = p["bias"].astype(np.float32)
                    self.feature_size = int(self.weights.shape[1])
                    for k in ("parameter_count", "training_examples", "dataset_version"):
                        if k in p:
                            v = p[k].item()
                            self.metadata[k] = int(v) if k != "dataset_version" else str(v)
            except (OSError, ValueError, KeyError):
                self.weights = None; self.bias = None; self.feature_size = 0

    @property
    def ready(self) -> bool:
        return self.weights is not None and self.bias is not None and self.feature_size > 0

    @property
    def parameter_count(self) -> int:
        return int(self.weights.size + self.bias.size) if self.ready else 0

    @classmethod
    def _hash(cls, s: str) -> int:
        return int.from_bytes(hashlib.blake2b(s.encode("utf-8"), digest_size=8, person=cls.PERSON).digest(), "little")

    @classmethod
    def _vector(cls, feats: Iterable[str], size: int) -> tuple[np.ndarray, np.ndarray]:
        counts: dict[int, float] = {}
        for feat in feats:
            h = cls._hash(feat); idx = h % size; sign = -1.0 if (cls._hash("s:"+feat) & 1) else 1.0
            counts[idx] = counts.get(idx, 0.0) + sign
        if not counts:
            return np.empty(0, np.int32), np.empty(0, np.float32)
        idx = np.fromiter(counts.keys(), dtype=np.int32)
        val = np.fromiter((counts[int(i)] for i in idx), dtype=np.float32)
        n = float(np.linalg.norm(val))
        if n: val /= n
        return idx, val

    def _predict(self, idx: np.ndarray, val: np.ndarray) -> FramePrediction | None:
        if not self.ready or idx.size == 0:
            return None
        scores = self.weights[:, idx] @ val + self.bias
        scores = scores - np.max(scores)
        probs = np.exp(np.clip(scores, -30, 30)); probs /= max(float(probs.sum()), 1e-9)
        order = np.argsort(-probs)
        a = int(order[0]); b = int(order[1]) if len(order) > 1 else a
        return FramePrediction(self.labels[a], float(probs[a]), float(probs[a]-probs[b]))


class SemanticFrameClassifierV18(_HashedLinear):
    FORMAT = "jarvis-semantic-frame-v18"
    PERSON = b"jrv18frm"

    @classmethod
    def sparse_features(cls, text: str, size: int) -> tuple[np.ndarray, np.ndarray]:
        value = normalize_text(text).casefold().strip()
        tokens = re.findall(r"[A-Za-z\u0600-\u06ff0-9_.%]+", value)
        feats: list[str] = [f"w:{x}" for x in tokens]
        feats += [f"b:{a}|{b}" for a,b in zip(tokens,tokens[1:])]
        padded = f"  {value}  "
        for n in (3,4,5): feats += [f"c{n}:{padded[i:i+n]}" for i in range(max(0,len(padded)-n+1))]
        if re.search(r"\d", value): feats.append("shape:num")
        if re.search(r"[%٪]", value): feats.append("shape:percent")
        if re.search(r"(?:km|کیلومتر|hours?|ساعت|دقیقه|minutes?)", value, re.I): feats.append("shape:unit")
        if re.search(r"(?:python|پایتون|function|تابع)", value, re.I): feats.append("shape:code")
        return cls._vector(feats,size)

    def predict(self, text: str) -> FramePrediction | None:
        if not self.ready: return None
        return self._predict(*self.sparse_features(text,self.feature_size))


class NumericSlotTaggerV18(_HashedLinear):
    FORMAT = "jarvis-numeric-slot-tagger-v18"
    PERSON = b"jrv18slt"

    @classmethod
    def sparse_features(cls, text: str, start: int, end: int, size: int) -> tuple[np.ndarray, np.ndarray]:
        value = normalize_text(text).casefold()
        left = value[max(0,start-55):start]
        raw = value[start:end]
        right = value[end:min(len(value),end+55)]
        window = f"{left}<num>{right}"
        tokens = re.findall(r"[A-Za-z\u0600-\u06ff0-9_%٪.]+|<num>", window)
        feats = [f"w:{x}" for x in tokens] + [f"b:{a}|{b}" for a,b in zip(tokens,tokens[1:])]
        near_left = re.findall(r"[A-Za-z\u0600-\u06ff]+", left)[-4:]
        near_right = re.findall(r"[A-Za-z\u0600-\u06ff]+", right)[:4]
        feats += [f"L{i}:{x}" for i,x in enumerate(reversed(near_left),1)]
        feats += [f"R{i}:{x}" for i,x in enumerate(near_right,1)]
        if re.match(r"\s*(?:%|٪|درصد|percent)", right, re.I): feats.append("unit:percent")
        if re.match(r"\s*(?:km|کیلومتر)", right, re.I): feats.append("unit:distance")
        if re.match(r"\s*(?:hours?|ساعت|minutes?|دقیقه)", right, re.I): feats.append("unit:time")
        if re.match(r"\s*(?:workers?|کارگر)", right, re.I): feats.append("unit:workers")
        return cls._vector(feats,size)

    def predict_number(self, text: str, start: int, end: int) -> SlotPrediction | None:
        p = self._predict(*self.sparse_features(text,start,end,self.feature_size)) if self.ready else None
        return SlotPrediction(p.label,p.confidence,p.margin) if p else None


__all__=["FramePrediction","SlotPrediction","SemanticFrameClassifierV18","NumericSlotTaggerV18"]
