from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
from jarvis.utils.text import normalize_text


@dataclass(frozen=True, slots=True)
class PredictionV19:
    label: str
    confidence: float
    margin: float


class _HashedLinearV19:
    FORMAT = ""
    PERSON = b"jrv19mdl"

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
                    for key in ("parameter_count", "training_examples", "dataset_version", "holdout_strategy"):
                        if key in p:
                            value = p[key].item()
                            self.metadata[key] = int(value) if key in {"parameter_count", "training_examples"} else str(value)
            except (OSError, ValueError, KeyError):
                self.weights = None
                self.bias = None
                self.feature_size = 0

    @property
    def ready(self) -> bool:
        return self.weights is not None and self.bias is not None and self.feature_size > 0

    @property
    def parameter_count(self) -> int:
        return int(self.weights.size + self.bias.size) if self.ready else 0

    @classmethod
    def _hash(cls, value: str) -> int:
        return int.from_bytes(hashlib.blake2b(value.encode("utf-8"), digest_size=8, person=cls.PERSON).digest(), "little")

    @classmethod
    def _vector(cls, feats: Iterable[str], size: int) -> tuple[np.ndarray, np.ndarray]:
        counts: dict[int, float] = {}
        for feat in feats:
            h = cls._hash(feat)
            idx = h % size
            sign = -1.0 if (cls._hash("s:" + feat) & 1) else 1.0
            counts[idx] = counts.get(idx, 0.0) + sign
        if not counts:
            return np.empty(0, np.int32), np.empty(0, np.float32)
        idx = np.fromiter(counts.keys(), dtype=np.int32)
        val = np.fromiter((counts[int(i)] for i in idx), dtype=np.float32)
        norm = float(np.linalg.norm(val))
        if norm:
            val /= norm
        return idx, val

    def _predict(self, idx: np.ndarray, val: np.ndarray) -> PredictionV19 | None:
        if not self.ready or idx.size == 0:
            return None
        scores = self.weights[:, idx] @ val + self.bias
        scores = scores - np.max(scores)
        probs = np.exp(np.clip(scores, -30, 30))
        probs /= max(float(probs.sum()), 1e-9)
        order = np.argsort(-probs)
        a = int(order[0])
        b = int(order[1]) if len(order) > 1 else a
        return PredictionV19(self.labels[a], float(probs[a]), float(probs[a] - probs[b]))


class SemanticFrameClassifierV19(_HashedLinearV19):
    FORMAT = "jarvis-semantic-frame-v19"
    PERSON = b"jrv19frm"

    @classmethod
    def sparse_features(cls, text: str, size: int) -> tuple[np.ndarray, np.ndarray]:
        value = normalize_text(text).casefold().strip()
        tokens = re.findall(r"[A-Za-z\u0600-\u06ff0-9_.%٪:+-]+", value)
        feats: list[str] = [f"w:{x}" for x in tokens]
        feats += [f"b:{a}|{b}" for a, b in zip(tokens, tokens[1:])]
        feats += [f"t:{i}:{x}" for i, x in enumerate(tokens[:8])]
        padded = f"  {value}  "
        for n in (3, 4):
            feats += [f"c{n}:{padded[i:i+n]}" for i in range(max(0, len(padded) - n + 1))]
        shape_rules = (
            ("shape:percent", r"[%٪]|درصد|percent"),
            ("shape:ratio", r"نسبت|ratio|\d\s*:\s*\d"),
            ("shape:split", r"تقسیم|پخش|سهم|split|divide|distribute|allocate"),
            ("shape:sequence", r"دنباله|تصاعد\s+هندسی|جمله\s+\d|sequence|geometric|\bgp\b|common\s+ratio|term"),
            ("shape:prob", r"احتمال|شانس|probability|chance|exactly"),
            ("shape:system", r"صدا|روشنایی|volume|brightness"),
            ("shape:work", r"کارگر|worker"),
            ("shape:age", r"سن|ساله|سالشه|years? old|age|sister|brother|خواهر|برادر"),
            ("shape:code", r"python|پایتون|تابع|function|flask|api|کد|code"),
        )
        for name, pat in shape_rules:
            if re.search(pat, value, re.I):
                feats.append(name)
        if re.search(r"(?:دنباله|تصاعد\s+هندسی|sequence|geometric|\bgp\b|common\s+ratio|term|جمله\s+\d)", value, re.I) and re.search(r"(?:نسبت|ratio)", value, re.I):
            feats.append("shape:sequence_ratio")
        if re.search(r"(?:نسبت|ratio)", value, re.I) and re.search(r"(?:تقسیم|پخش|سهم|split|divide|distribute|allocate)", value, re.I):
            feats.append("shape:ratio_split")
        return cls._vector(feats, size)

    def predict(self, text: str) -> PredictionV19 | None:
        return self._predict(*self.sparse_features(text, self.feature_size)) if self.ready else None


class NumericRoleTaggerV19(_HashedLinearV19):
    FORMAT = "jarvis-numeric-role-v19"
    PERSON = b"jrv19num"

    @classmethod
    def sparse_features(cls, text: str, start: int, end: int, size: int) -> tuple[np.ndarray, np.ndarray]:
        value = normalize_text(text).casefold()
        left = value[max(0, start - 80):start]
        right = value[end:min(len(value), end + 80)]
        whole_tokens = re.findall(r"[A-Za-z\u0600-\u06ff]+", value)
        left_tokens = re.findall(r"[A-Za-z\u0600-\u06ff]+", left)[-7:]
        right_tokens = re.findall(r"[A-Za-z\u0600-\u06ff]+", right)[:7]
        feats: list[str] = []
        feats += [f"L{i}:{tok}" for i, tok in enumerate(reversed(left_tokens), 1)]
        feats += [f"R{i}:{tok}" for i, tok in enumerate(right_tokens, 1)]
        feats += [f"LB:{a}|{b}" for a, b in zip(left_tokens, left_tokens[1:])]
        feats += [f"RB:{a}|{b}" for a, b in zip(right_tokens, right_tokens[1:])]
        masked = value[:start] + "<NUM>" + value[end:]
        masked = re.sub(r"\d+(?:[.,]\d+)?", "<OTHER_NUM>", masked)
        feats += [f"GW:{a}|{b}" for a,b in zip(whole_tokens,whole_tokens[1:])]
        # Global semantic cues let the model distinguish e.g. ratio from sequence while
        # the local context determines the number's role. This is still order-independent.
        for tok in sorted(set(whole_tokens)):
            if tok in {
                "کارگر","worker","workers","ساعت","hour","hours","تولید","produce","قطعه","کالا","units","items",
                "نسبت","ratio","تقسیم","split","سهم","share","درصد","percent","مبلغ","قیمت","money","price",
                "سن","ساله","سالشه","age","years","خواهر","برادر","sister","brother","بعد","later",
                "احتمال","probability","chance","دقیقاً","دقیقا","exactly","بار","trials","attempts",
                "کیلومتر","km","سرعت","speed","فاصله","distance","جمله","دنباله","term","sequence",
                "اضافه","add","کم","subtract","کاهش","increase","decrease","بعدش","then"
            }:
                feats.append(f"G:{tok}")
        unit_rules = (
            ("unit:percent", r"^\s*(?:%|٪|درصد|percent)"),
            ("unit:workers", r"^\s*(?:کارگر|workers?)"),
            ("unit:time", r"^\s*(?:ساعت|hours?|دقیقه|minutes?)"),
            ("unit:distance", r"^\s*(?:کیلومتر|km|متر|meters?)"),
            ("unit:age", r"^\s*(?:سال(?:ه)?|years?(?:\s+old)?)"),
            ("unit:items", r"^\s*(?:قطعه|کالا|واحد|items?|units?|pieces?)"),
        )
        for name, pat in unit_rules:
            if re.search(pat, right, re.I):
                feats.append(name)
        if re.search(r"(?:نسبت|ratio)\s*$", left, re.I):
            feats.append("ratio:left-cue")
        if re.search(r"(?:به|to|:)\s*$", left, re.I):
            feats.append("ratio:second-cue")
        if re.search(r"(?:دقیقاً|دقیقا|exactly)\s*$", left, re.I):
            feats.append("prob:k-cue")
        # Explicit semantic proximity cues are features, not final labels. The trained
        # classifier learns how much to trust them and how they interact with global context.
        cue_features = (
            ("cue:right-workers", right, r"^\s*(?:کارگر|نفر\s+کارگر|workers?)"),
            ("cue:left-workers", left, r"(?:تعداد\s+کارگر(?:ها)?|نیروی\s+کار|worker\s+count)\s*$"),
            ("cue:right-hours", right, r"^\s*(?:ساعت|hours?)"),
            ("cue:right-output", right, r"^\s*(?:قطعه|کالا|واحد|items?|units?|pieces?)"),
            ("cue:left-output", left, r"(?:خروجی|تولید|output|production)\s*$"),
            ("cue:left-money", left, r"(?:مبلغ|قیمت|هزینه|price|amount|cost)\s*$"),
            ("cue:right-percent", right, r"^\s*(?:%|٪|درصد|percent)"),
            ("cue:left-prob", left, r"(?:احتمال|شانس|probability|chance|rate)\s*$"),
            ("cue:right-age", right, r"^\s*(?:سال(?:ه)?|years?(?:\s+old)?)"),
            ("cue:left-age", left, r"(?:سن|age(?:\s+is)?|reached\s+age)\s*$"),
            ("cue:later", left+"<NUM>"+right, r"(?:بعد\s+از\s*<NUM>|<NUM>\s*سال\s*(?:بعد|دیگر)|after\s*<NUM>\s*years?|<NUM>\s*years?\s*(?:later|from\s+now))"),
            ("cue:difference", left+"<NUM>"+right, r"(?:<NUM>\s*سال\s*(?:بزرگ|کوچک)|older\s+by\s*<NUM>|<NUM>\s*years?\s*(?:older|younger))"),
            ("cue:left-count", left, r"(?:تعداد(?:\s+کل)?|count|total\s+count|list\s+has|از)\s*$"),
            ("cue:right-count", right, r"^\s*(?:مورد|عضو|entries|records|items)"),
            ("cue:left-index", left, r"(?:شماره|شاخص|index|position|item)\s*$"),
        )
        for name, scope, pat in cue_features:
            if re.search(pat, scope, re.I): feats.append(name)
        # Structural ratio position: identify whether this number is first or second term.
        around = value[max(0,start-35):min(len(value),end+35)]
        if re.search(r"(?:نسبت|ratio).{0,18}<NUM>", value[:start]+"<NUM>"+value[end:], re.I): feats.append("ratio:term-near")
        if re.search(r"<NUM>\s*(?:به|to|:)\s*\d", value[:start]+"<NUM>"+value[end:], re.I): feats.append("ratio:first-term")
        if re.search(r"\d\s*(?:به|to|:)\s*<NUM>", value[:start]+"<NUM>"+value[end:], re.I): feats.append("ratio:second-term")
        return cls._vector(feats, size)

    def predict_number(self, text: str, start: int, end: int) -> PredictionV19 | None:
        return self._predict(*self.sparse_features(text, start, end, self.feature_size)) if self.ready else None


class ExecutionPatternClassifierV19(_HashedLinearV19):
    FORMAT = "jarvis-execution-pattern-v19"
    PERSON = b"jrv19exe"

    @classmethod
    def sparse_features(cls, text: str, size: int) -> tuple[np.ndarray, np.ndarray]:
        value = normalize_text(text).casefold()
        masked = re.sub(r"\d+(?:[.,]\d+)?", "<num>", value)
        tokens = re.findall(r"<num>|[A-Za-z\u0600-\u06ff%٪]+", masked)
        feats = [f"w:{x}" for x in tokens]
        feats += [f"b:{a}|{b}" for a, b in zip(tokens, tokens[1:])]
        feats += [f"tri:{a}|{b}|{c}" for a, b, c in zip(tokens, tokens[1:], tokens[2:])]
        if re.search(r"درصد|percent|[%٪]", value, re.I): feats.append("shape:percent")
        if re.search(r"بعد|سپس|then|after|followed", value, re.I): feats.append("shape:chain")
        parts = re.split(r"(?:بعد(?:ش)?|سپس|then|afterwards|followed by)", value, maxsplit=1, flags=re.I)
        first = parts[0]
        second = parts[1] if len(parts)>1 else value
        if re.search(r"(?:درصد|percent|[%٪])", first, re.I) and re.search(r"(?:کم|کاهش|تخفیف|فروش|reduce|decrease|discount|sold|remove|subtract)", first, re.I): feats.append("sem:first:pct_remove")
        if re.search(r"(?:درصد|percent|[%٪])", first, re.I) and re.search(r"(?:زیاد|افزایش|رشد|اضافه|increase|grow|add|plus)", first, re.I): feats.append("sem:first:pct_add")
        if re.search(r"(?:تخفیف|discount)", first, re.I): feats.append("sem:first:pct_remove")
        if re.search(r"(?:فروش|sold)", first, re.I) and re.search(r"(?:درصد|percent|[%٪])", first, re.I): feats.append("sem:first:pct_remove")
        if re.search(r"(?:اضافه|بیفزا|زیاد|شارژ|وارد|add|plus|increase|restock|fee)", second, re.I): feats.append("sem:second:add")
        if re.search(r"(?:کم|کاهش|منهای|بردار|subtract|minus|remove|take\s+away)", second, re.I): feats.append("sem:second:subtract")
        if re.search(r"(?:ضرب|برابر|multiply|times)", second, re.I): feats.append("sem:second:multiply")
        if re.search(r"(?:تقسیم|divide)", second, re.I): feats.append("sem:second:divide")
        if re.search(r"(?:موجودی|inventory|فروش|sold|restock|انبار)", value, re.I): feats.append("domain:inventory")
        if re.search(r"(?:قیمت|مبلغ|هزینه|price|cost|fee|shipping|تخفیف|discount)", value, re.I): feats.append("domain:money")
        first_kind = ""
        second_kind = ""
        if re.search(r"(?:درصد|percent|[%٪])", first, re.I):
            if re.search(r"(?:کم|کاهش|تخفیف|فروش|reduce|decrease|discount|sold|remove|subtract)|-\s*<num>|-\s*\d", first, re.I): first_kind="pct_remove"
            elif re.search(r"(?:زیاد|افزایش|رشد|اضافه|increase|grow|add|plus)|\+\s*<num>|\+\s*\d|apply\s*\+", first, re.I): first_kind="pct_add"
        if re.search(r"(?:ضرب|برابر|multiply|times)", second, re.I): second_kind="multiply"
        elif re.search(r"(?:تقسیم|divide)", second, re.I): second_kind="divide"
        elif re.search(r"(?:کم|کاهش|منهای|بردار|subtract|minus|remove|take\s+away)|-\s*<num>|-\s*\d", second, re.I): second_kind="subtract"
        elif re.search(r"(?:اضافه|بیفزا|زیاد|شارژ|وارد|add|plus|increase|restock|fee)|\+\s*<num>|\+\s*\d", second, re.I): second_kind="add"
        # Non-percentage first steps.
        if not first_kind:
            if re.search(r"(?:اضافه|بیفزا|add|plus)", first, re.I): first_kind="add"
            elif re.search(r"(?:کم|منهای|بردار|subtract|minus|remove|take\s+away)", first, re.I): first_kind="subtract"
        if first_kind and second_kind:
            feats.extend([f"graph:{first_kind}:{second_kind}"] * 10)
        return cls._vector(feats, size)

    def predict(self, text: str) -> PredictionV19 | None:
        return self._predict(*self.sparse_features(text, self.feature_size)) if self.ready else None


class CodeIntentClassifierV19(_HashedLinearV19):
    FORMAT = "jarvis-code-intent-v19"
    PERSON = b"jrv19cod"

    @classmethod
    def sparse_features(cls, text: str, size: int) -> tuple[np.ndarray, np.ndarray]:
        value = normalize_text(text).casefold()
        tokens = re.findall(r"[A-Za-z\u0600-\u06ff0-9_]+", value)
        feats = [f"w:{x}" for x in tokens]
        feats += [f"b:{a}|{b}" for a, b in zip(tokens, tokens[1:])]
        padded = f" {value} "
        feats += [f"c4:{padded[i:i+4]}" for i in range(max(0, len(padded)-3))]
        cue_map = (
            ("intent:positive", r"مثبت|positive|greater\s+than\s+zero|بزرگ.?تر\s+از\s+صفر"),
            ("intent:even", r"زوج|even|divisible\s+by\s+(?:two|2)"),
            ("intent:max", r"بیشترین|بزرگ.?ترین|ماکزیمم|maximum|largest|biggest|max[- ]?value"),
            ("intent:flask", r"flask|endpoint|rest\s+api|health[- ]?check|api"),
            ("intent:sort", r"مرتب|صعودی|sort|ascending|low\s+to\s+high|order\s+array"),
            ("intent:sum", r"مجموع|جمع\s+همه|sum|total\s+(?:of|array|list)|sums\s+all"),
        )
        for name,pat in cue_map:
            if re.search(pat,value,re.I): feats.append(name)
        return cls._vector(feats, size)

    def predict(self, text: str) -> PredictionV19 | None:
        return self._predict(*self.sparse_features(text, self.feature_size)) if self.ready else None


__all__ = [
    "PredictionV19", "SemanticFrameClassifierV19", "NumericRoleTaggerV19",
    "ExecutionPatternClassifierV19", "CodeIntentClassifierV19",
]
