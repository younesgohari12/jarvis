from __future__ import annotations

import hashlib
import math
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from jarvis.utils.text import normalize_text


@dataclass(frozen=True, slots=True)
class SemanticIntentPrediction:
    intent: str
    confidence: float
    margin: float
    alternatives: tuple[tuple[str, float], ...]
    source: str = "semantic_router_v15"


class SemanticIntentRouterV15:
    """Sparse learned intent router specialized for cognitive/language requests.

    Runtime inference is dependency-free (NumPy only). The model is an online-
    trained hashed linear classifier whose features are word n-grams and Unicode
    character n-grams. Deterministic semantic guards are layered in front of the
    learned scores to prevent regex/number collisions and prioritize code syntax.
    """

    FORMAT = "jarvis-semantic-router-v15"
    DEFAULT_LABELS = (
        "coding", "code_trace", "translation", "rewrite", "probability",
        "logic", "word_problem", "math", "constraint_writing",
        "fresh_information", "desktop_command", "general_question",
    )

    _CODE_SYNTAX = re.compile(
        r"```|\b(?:def|class|for|while|if|elif|else|return|print|import|from|try|except|lambda)\b"
        r"|\w+\s*=\s*[^=]|[{};]|\[[^\]]*\]",
        re.I,
    )
    _TRACE_CUE = re.compile(
        r"(?:خروجی|اجرا|چه\s+چاپ|چی\s+چاپ|ردیابی|مرحله.?به.?مرحله|trace|output|"
        r"what\s+(?:does|will).*(?:print|output)|what\s+is\s+printed|run\s+this)",
        re.I,
    )
    _CODING_CUE = re.compile(
        r"(?:(?<!\w)کد(?!\w)|(?<!\w)تابع(?!\w)|(?<!\w)برنامه(?!\w)|(?<!\w)اسکریپت(?!\w)|(?<!\w)الگوریتم(?!\w)|"
        r"\b(?:python|javascript|typescript|java|c\+\+|c#|sql|regex|function|code|script|implement|programming)\b)",
        re.I,
    )
    _TRANSLATION_CUE = re.compile(
        r"(?:ترجمه(?:\s*اش|\s*اشو|\s*اش\s+کن)?|(?:به|رو|را)\s+(?:انگلیسی|فارسی)\s+(?:کن|بگو)|"
        r"(?:انگلیسی|فارسی)\s*اش\s+کن|translate|translation|say\s+(?:it|this)\s+in\s+(?:english|persian|farsi))",
        re.I,
    )
    _REWRITE_CUE = re.compile(
        r"(?:بازنویسی|روان.?تر|رسمی(?:.?تر)?(?:\s+کن|\s+بنویس)?|صمیمی.?تر|حرفه.?ای.?تر|ویرایش|اصلاح\s+کن|بهترش\s+کن|"
        r"rewrite|rephrase|polish|make\s+(?:it|this)\s+(?:formal|clearer|shorter|professional|friendlier))",
        re.I,
    )
    _PROB_CUE = re.compile(r"(?:احتمال|شانس|رخداد|پیشامد|سکه|تاس|probab|chance|coin|dice|die)", re.I)
    _LOGIC_CUE = re.compile(
        r"(?:منطق|استنتاج|نتیجه\s+می.?شود|حتماً|تناقض|اگر\s+همه|هیچ\s+.+نیست|(?<!به.)ترتیب|قبل\s+از|بعد\s+از|"
        r"\blogic\b|deduc|infer|syllog|necessarily|all\s+.+\s+are|no\s+.+\s+are|\bordering\b)", re.I,
    )
    _WORD_PROBLEM_CUE = re.compile(
        r"(?:سرعت|مسافت|زمان|کارگر|کار\s+را|مخزن|سن|سال\s+دیگر|نسبت|تخفیف|قیمت|صفحه|روزانه|"
        r"\bspeed\b|\bdistance\b|\brate\b|\bworker|work\s+problem|\bage\b|\bdiscount\b|pages?\s+per\s+day|\bratio\b)", re.I,
    )
    _MATH_CUE = re.compile(
        r"(?:حساب\s+کن|محاسبه|حل\s+کن|معادله|فاکتوریل|عدد\s+اول|مقسوم.?علیه|بخش.?پذیر|ب\.م\.م|ک\.م\.م|درصد|میانگین|دنباله|"
        r"\bcalculate\b|\bcompute\b|\bsolve\b|\bequation\b|\bfactorial\b|\bprime\b|divisib|\bdivisor\b|\bgcd\b|\blcm\b|\bpercent\b|\baverage\b|\bsequence\b|\bcombination\b|\bpermutation\b)", re.I,
    )
    _FRESH_CUE = re.compile(
        r"(?:امروز|الان|آخرین|جدیدترین|قیمت\s+فعلی|خبر|آب.?وهوا|بورس|نتیجه\s+بازی|"
        r"\btoday\b|\bcurrent\b|\blatest\b|\bnews\b|\bweather\b|\bstock\b|live\s+score|price\s+now)", re.I,
    )
    _DESKTOP_CUE = re.compile(
        r"(?:باز\s+کن|ببند|صدا|روشنایی|فایل|پوشه|مرورگر|کلیپ.?بورد|خاموش|ریستارت|"
        r"\bopen\b|\bclose\b|\bvolume\b|\bbrightness\b|\bfile\b|\bfolder\b|\bbrowser\b|\bclipboard\b|\bshutdown\b|\brestart\b)", re.I,
    )
    _CONSTRAINT_CUE = re.compile(
        r"(?:دقیقاً\s*\d+\s*جمله|حداکثر\s*\d+\s*(?:کلمه|واژه)|فقط\s+(?:فارسی|انگلیسی|جواب)|بدون\s+توضیح|"
        r"شامل\s+(?:کلمه|واژه)|بدون\s+استفاده\s+از|exactly\s+\d+\s+sentences?|at\s+most\s+\d+\s+words?|"
        r"only\s+(?:persian|english)|no\s+explanation|must\s+contain)", re.I,
    )

    def __init__(self, model_path: Path | None = None) -> None:
        self.model_path = model_path
        self.labels = self.DEFAULT_LABELS
        self.feature_size = 0
        self.weights: np.ndarray | None = None
        self.bias: np.ndarray | None = None
        self.temperature = 1.0
        self.metadata: dict[str, object] = {}
        if model_path is not None and model_path.is_file():
            try:
                with np.load(model_path, allow_pickle=False) as payload:
                    fmt = str(payload["format"].item())
                    if fmt != self.FORMAT:
                        return
                    labels = tuple(str(value) for value in payload["labels"].tolist())
                    weights = payload["weights"].astype(np.float32)
                    bias = payload["bias"].astype(np.float32)
                    if weights.ndim != 2 or weights.shape[0] != len(labels) or bias.shape != (len(labels),):
                        return
                    self.labels = labels
                    self.feature_size = int(weights.shape[1])
                    self.weights = weights
                    self.bias = bias
                    self.temperature = float(payload["temperature"].item())
                    if "parameter_count" in payload:
                        self.metadata["parameter_count"] = int(payload["parameter_count"].item())
                    if "training_examples" in payload:
                        self.metadata["training_examples"] = int(payload["training_examples"].item())
                    if "dataset_version" in payload:
                        self.metadata["dataset_version"] = str(payload["dataset_version"].item())
            except (OSError, ValueError, KeyError):
                self.weights = None
                self.bias = None
                self.feature_size = 0

    @property
    def ready(self) -> bool:
        return self.weights is not None and self.bias is not None and self.feature_size > 0

    @property
    def parameter_count(self) -> int:
        if self.weights is None or self.bias is None:
            return 0
        return int(self.weights.size + self.bias.size)

    @staticmethod
    def _stable_hash(value: str) -> int:
        return int.from_bytes(hashlib.blake2b(value.encode("utf-8"), digest_size=8, person=b"jrv15rtr").digest(), "little")

    @classmethod
    def sparse_features(cls, text: str, feature_size: int) -> tuple[np.ndarray, np.ndarray]:
        value = normalize_text(text).casefold().strip()
        value = re.sub(r"\s+", " ", value)
        counts: dict[int, float] = {}
        tokens = re.findall(r"[A-Za-z\u0600-\u06ff0-9_+#.-]+", value)
        feats: list[str] = []
        feats.extend(f"w1:{tok}" for tok in tokens)
        feats.extend(f"w2:{a}|{b}" for a, b in zip(tokens, tokens[1:]))
        padded = f"  {value}  "
        for n in (3, 4, 5):
            feats.extend(f"c{n}:{padded[i:i+n]}" for i in range(max(0, len(padded)-n+1)))
        # Structural signals are critical for separating code from equations.
        if "```" in text:
            feats.append("shape:code_fence")
        if re.search(r"\b(?:def|print|for|while|return|import|class)\b", value):
            feats.append("shape:code_keyword")
        if re.search(r"\b[a-zA-Z_]\w*\s*=\s*[^=]", text):
            feats.append("shape:assignment")
        if re.search(r"\d", value):
            feats.append("shape:has_number")
        if text.rstrip().endswith(("?", "؟")):
            feats.append("shape:question")
        for feat in feats:
            idx = cls._stable_hash(feat) % feature_size
            sign = -1.0 if (cls._stable_hash("sign:" + feat) & 1) else 1.0
            counts[idx] = counts.get(idx, 0.0) + sign
        if not counts:
            return np.empty(0, dtype=np.int32), np.empty(0, dtype=np.float32)
        indices = np.fromiter(counts.keys(), dtype=np.int32)
        values = np.fromiter((counts[int(i)] for i in indices), dtype=np.float32)
        norm = float(np.linalg.norm(values))
        if norm > 0:
            values /= norm
        return indices, values

    @staticmethod
    def _softmax(scores: np.ndarray) -> np.ndarray:
        shifted = scores - np.max(scores)
        exp = np.exp(np.clip(shifted, -30.0, 30.0))
        return exp / max(float(exp.sum()), 1e-9)

    def _learned_predict(self, text: str) -> SemanticIntentPrediction | None:
        if not self.ready or self.weights is None or self.bias is None:
            return None
        indices, values = self.sparse_features(text, self.feature_size)
        if indices.size == 0:
            return None
        scores = self.bias.copy()
        scores += (self.weights[:, indices] * values[None, :]).sum(axis=1)
        probs = self._softmax(scores / max(0.2, self.temperature))
        order = np.argsort(-probs)
        top = int(order[0]); second = int(order[1]) if len(order) > 1 else top
        alternatives = tuple((self.labels[int(i)], float(probs[int(i)])) for i in order[:4])
        return SemanticIntentPrediction(
            self.labels[top], float(probs[top]), float(probs[top] - probs[second]), alternatives,
        )

    @classmethod
    def _guard_prediction(cls, text: str) -> SemanticIntentPrediction | None:
        value = normalize_text(text)
        # Code trace outranks math only when programming evidence accompanies the
        # trace request. A bare algebraic assignment such as x=3 is not enough to
        # suppress mathematical routing.
        # Explicit OS/app commands outrank lexical words that are also product
        # names (e.g. «وی اس کد رو باز کن» must not become a coding request).
        if cls._DESKTOP_CUE.search(value) and re.search(
            r"(?:باز\s+کن|ببند|بگذار|تنظیم|زیاد|کم|روشن|خاموش|open|close|set|turn|increase|decrease)",
            value, re.I,
        ):
            return SemanticIntentPrediction("desktop_command", 0.997, 0.90, (("desktop_command", 0.997),), "semantic_guard_v15")

        # Explicit language transformation commands own their payload even if it
        # contains words such as «کد»/code. This prevents the payload noun from
        # stealing an explicit rewrite/translation request.
        translation_command = re.search(
            r"(?:ترجمه(?:\s*اش|\s*اشو)?\s*کن|به\s+(?:انگلیسی|فارسی)\s+(?:ترجمه\s*)?(?:کن|بگو)|"
            r"(?:انگلیسی|فارسی)\s*اش\s*کن|translate\s+(?:this|it|the\s+following)|say\s+(?:it|this)\s+in\s+(?:english|persian|farsi))",
            value, re.I,
        )
        if translation_command:
            return SemanticIntentPrediction("translation", 0.996, 0.88, (("translation", 0.996),), "semantic_guard_v17")
        if cls._REWRITE_CUE.search(value):
            return SemanticIntentPrediction("rewrite", 0.995, 0.86, (("rewrite", 0.995),), "semantic_guard_v17")

        code_syntax = bool(cls._CODE_SYNTAX.search(text))
        strong_code_syntax = bool(re.search(
            r"```|\b(?:def|class|for|while|if|elif|else|return|print|import|from|try|except|lambda|range)\b|[{};]",
            text, re.I,
        ))
        if code_syntax and cls._TRACE_CUE.search(value):
            return SemanticIntentPrediction("code_trace", 0.998, 0.90, (("code_trace", 0.998),), "semantic_guard_v15")
        if cls._CODING_CUE.search(value) and re.search(r"(?:بنویس|بساز|پیاده.?سازی|\bwrite\b|\bcreate\b|\bimplement\b|\bfunction\b|(?<!\w)تابع(?!\w)|(?<!\w)کد(?!\w))", value, re.I):
            return SemanticIntentPrediction("coding", 0.995, 0.86, (("coding", 0.995),), "semantic_guard_v15")
        translation_command = re.search(
            r"(?:ترجمه(?:\s*اش|\s*اشو)?\s*کن|به\s+(?:انگلیسی|فارسی)\s+(?:ترجمه\s*)?(?:کن|بگو)|"
            r"(?:انگلیسی|فارسی)\s*اش\s*کن|translate\s+(?:this|it|the\s+following)|say\s+(?:it|this)\s+in\s+(?:english|persian|farsi))",
            value, re.I,
        )
        if translation_command:
            return SemanticIntentPrediction("translation", 0.996, 0.88, (("translation", 0.996),), "semantic_guard_v15")
        if cls._REWRITE_CUE.search(value):
            return SemanticIntentPrediction("rewrite", 0.994, 0.84, (("rewrite", 0.994),), "semantic_guard_v15")
        # Freshness language must be explicit; ordinary factual questions stay local.
        if cls._FRESH_CUE.search(value):
            return SemanticIntentPrediction("fresh_information", 0.992, 0.80, (("fresh_information", 0.992),), "semantic_guard_v15")
        word_problem_task = cls._WORD_PROBLEM_CUE.search(value) and re.search(
            r"(?:\d|چقدر|چند|محاسبه|یافتن|باقی|می.?شود|what|how\s+(?:many|long|much)|find|calculate)",
            value, re.I,
        )
        if word_problem_task:
            return SemanticIntentPrediction("word_problem", 0.976, 0.66, (("word_problem", 0.976),), "semantic_guard_v15")
        probability_task = cls._PROB_CUE.search(value) and re.search(
            r"(?:\d|دقیقاً|حداقل|حداکثر|چقدر|چند|احتمال\s+(?:اینکه|وقوع|آمدن)|"
            r"exactly|at\s+least|at\s+most|probability\s+of|chance\s+of|how\s+likely)", value, re.I
        )
        if probability_task:
            return SemanticIntentPrediction("probability", 0.993, 0.82, (("probability", 0.993),), "semantic_guard_v15")
        if cls._LOGIC_CUE.search(value) and not cls._DESKTOP_CUE.search(value):
            return SemanticIntentPrediction("logic", 0.985, 0.72, (("logic", 0.985),), "semantic_guard_v15")
        if cls._WORD_PROBLEM_CUE.search(value) and re.search(r"\d", value) and re.search(r"(?:چقدر|چند|محاسبه|یافتن|باقی|می.?شود|what|how\s+many|find|calculate)", value, re.I):
            return SemanticIntentPrediction("word_problem", 0.976, 0.66, (("word_problem", 0.976),), "semantic_guard_v15")
        math_task = cls._MATH_CUE.search(value) and re.search(
            r"(?:\d|حساب\s+کن|محاسبه|حل\s+کن|به\s+دست\s+آور|آیا\s+\d|کدام\s+عدد|"
            r"calculate|compute|solve|find|is\s+\d+|which\s+number|what\s+is\s+\d)", value, re.I
        )
        if math_task and not strong_code_syntax and not cls._DESKTOP_CUE.search(value):
            return SemanticIntentPrediction("math", 0.972, 0.62, (("math", 0.972),), "semantic_guard_v15")
        definition_question = re.search(
            r"(?:چیست(?:[؟?]|\s|$)|چه\s+تفاوتی|فرق\s+.+\s+(?:چیست|چیه)|چگونه\s+کار\s+می.?کند|"
            r"what\s+(?:is|does)|how\s+does|difference\s+between)", value, re.I
        )
        if definition_question and not cls._FRESH_CUE.search(value):
            return SemanticIntentPrediction("general_question", 0.965, 0.58, (("general_question", 0.965),), "semantic_guard_v15")
        if cls._CONSTRAINT_CUE.search(value) and re.search(r"(?:بنویس|پاسخ|جواب|write|answer|describe|توضیح)", value, re.I):
            return SemanticIntentPrediction("constraint_writing", 0.96, 0.55, (("constraint_writing", 0.96),), "semantic_guard_v15")
        return None

    def predict(self, text: str) -> SemanticIntentPrediction | None:
        # Leave social conversation to the mature conversational router. The
        # learned cognitive classifier is intentionally not trained to own
        # greetings/acknowledgements, so forcing a label here can steal them.
        value = normalize_text(text).strip()
        if re.fullmatch(
            r"(?:سلام|درود|صبح\s+بخیر|شب\s+بخیر|خوبی|حالت\s+چطوره|چه\s+خبر|"
            r"hello|hi|hey|good\s+(?:morning|evening|night)|how\s+are\s+you)[؟?!.,، ]*",
            value, re.I,
        ):
            return None
        # Explicit device/file/app actions remain owned by the mature action
        # router. This avoids learned labels stealing commands such as
        # “delete this file” or “rename the folder”.
        if re.search(
            r"(?:(?:delete|remove|rename|move|copy|open|close|launch|start|restart|shutdown)\b.{0,50}\b(?:file|folder|app|application|window|tab|browser|system)|"
            r"\b(?:file|folder|app|application|window|tab|browser|system)\b.{0,50}\b(?:delete|remove|rename|move|copy|open|close|launch|start|restart|shutdown)\b|"
            r"(?:حذف|پاک|تغییر\s+نام|منتقل|کپی|باز|ببند|اجرا|ریستارت|خاموش).{0,50}(?:فایل|پوشه|برنامه|پنجره|تب|مرورگر|سیستم)|"
            r"(?:فایل|پوشه|برنامه|پنجره|تب|مرورگر|سیستم).{0,50}(?:حذف|پاک|تغییر\s+نام|منتقل|کپی|باز|ببند|اجرا|ریستارت|خاموش))",
            value, re.I,
        ):
            return None

        # Explicit search/research commands belong to the action/web router.
        # The semantic cognitive model only helps when dispatch is uncertain; it
        # must not reinterpret an explicit request to search the web as writing.
        if re.search(
            r"(?:سرچ\s*(?:کن|بزن)?|جستجو\s*(?:کن)?|گوگل\s*کن|بگرد|تحقیق\s*کن|"
            r"\bsearch(?:\s+for)?\b|\blook\s+up\b|\bgoogle\b|\bresearch\b)",
            value, re.I,
        ):
            return None
        guarded = self._guard_prediction(text)
        if guarded is not None:
            return guarded
        learned = self._learned_predict(text)
        if learned is None:
            return None
        # Learned-only routing uses a strict threshold to avoid stealing desktop or fresh queries.
        # Learned predictions never bypass uncertainty checks.  High-risk routes
        # (desktop/fresh) use an even stricter threshold so a weak semantic score
        # cannot steal a request from the deterministic router or reasoning fallback.
        if learned.intent in {"desktop_command", "fresh_information"}:
            if learned.confidence < 0.78 or learned.margin < 0.22:
                return None
            return learned
        if learned.intent == "general_question":
            if learned.confidence < 0.70 or learned.margin < 0.18:
                return None
            return learned
        if learned.confidence < 0.60 or learned.margin < 0.14:
            return None
        return learned


__all__ = ["SemanticIntentPrediction", "SemanticIntentRouterV15"]
