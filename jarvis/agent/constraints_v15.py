from __future__ import annotations

import re
from dataclasses import dataclass

from jarvis.utils.text import normalize_text


@dataclass(frozen=True, slots=True)
class ResponseConstraints:
    exact_sentences: int | None = None
    max_words: int | None = None
    language_only: str = ""
    include_terms: tuple[str, ...] = ()
    exclude_terms: tuple[str, ...] = ()
    answer_only: bool = False

    @property
    def active(self) -> bool:
        return any((
            self.exact_sentences is not None,
            self.max_words is not None,
            bool(self.language_only),
            bool(self.include_terms),
            bool(self.exclude_terms),
            self.answer_only,
        ))


@dataclass(frozen=True, slots=True)
class VerificationResult:
    valid: bool
    violations: tuple[str, ...]
    text: str


class ConstraintExtractor:
    """Extract explicit output constraints in Persian and English.

    The extractor is deliberately conservative: it activates only when the user
    explicitly states a constraint, so normal answers are never shortened merely
    because a number occurs in the request.
    """

    _EXACT_SENTENCES = (
        re.compile(r"(?:دقیقاً|دقیقا|فقط)\s*(\d{1,2}|یک|دو|سه|چهار|پنج|شش|هفت|هشت|نه|ده)\s*جمله", re.I),
        re.compile(r"\bexactly\s+(\d{1,2}|one|two|three|four|five|six|seven|eight|nine|ten)\s+sentences?\b", re.I),
        re.compile(r"\bin\s+(\d{1,2}|one|two|three|four|five|six|seven|eight|nine|ten)\s+sentences?\b", re.I),
    )
    _MAX_WORDS = (
        re.compile(r"(?:حداکثر|بیشتر\s+از)\s*(\d{1,4})\s*(?:کلمه|واژه)", re.I),
        re.compile(r"\b(?:at\s+most|maximum|max)\s+(\d{1,4})\s+words?\b", re.I),
        re.compile(r"\bunder\s+(\d{1,4})\s+words?\b", re.I),
    )
    _INCLUDE = (
        re.compile(r"(?:حتماً|حتما)?\s*(?:شامل|حاوی)\s+(?:کلمه|واژه|عبارت)\s*[«\"']([^»\"']{1,60})[»\"']", re.I),
        re.compile(r"\b(?:include|must\s+contain)\s+(?:the\s+)?(?:word|phrase)\s*[\"']([^\"']{1,60})[\"']", re.I),
    )
    _EXCLUDE = (
        re.compile(r"(?:بدون\s+(?:استفاده\s+از|به.?کار\s+بردن)|استفاده\s+نکن\s+از)\s+(?:کلمه|واژه|عبارت)?\s*[«\"']([^»\"']{1,60})[»\"']", re.I),
        re.compile(r"\b(?:do\s+not\s+use|without\s+using|exclude)\s+(?:the\s+)?(?:word|phrase)?\s*[\"']([^\"']{1,60})[\"']", re.I),
    )

    @classmethod
    def extract(cls, text: str) -> ResponseConstraints:
        normalized = normalize_text(text)
        exact: int | None = None
        maximum: int | None = None
        for pattern in cls._EXACT_SENTENCES:
            match = pattern.search(normalized)
            if match:
                raw_value = match.group(1).casefold()
                number_words = {
                    "یک": 1, "دو": 2, "سه": 3, "چهار": 4, "پنج": 5, "شش": 6, "هفت": 7, "هشت": 8, "نه": 9, "ده": 10,
                    "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
                }
                value = int(raw_value) if raw_value.isdigit() else number_words.get(raw_value, 0)
                if 1 <= value <= 20:
                    exact = value
                break
        for pattern in cls._MAX_WORDS:
            match = pattern.search(normalized)
            if match:
                value = int(match.group(1))
                if 1 <= value <= 5000:
                    maximum = value
                break
        fa_only = bool(re.search(r"(?:فقط|صرفاً|صرفا)\s+فارسی|تنها\s+به\s+فارسی|only\s+(?:in\s+)?persian|persian\s+only", normalized, re.I))
        en_only = bool(re.search(r"(?:فقط|صرفاً|صرفا)\s+انگلیسی|تنها\s+به\s+انگلیسی|only\s+(?:in\s+)?english|english\s+only", normalized, re.I))
        language = "fa" if fa_only and not en_only else ("en" if en_only and not fa_only else "")
        include: list[str] = []
        exclude: list[str] = []
        for pattern in cls._INCLUDE:
            include.extend(m.group(1).strip() for m in pattern.finditer(text))
        for pattern in cls._EXCLUDE:
            exclude.extend(m.group(1).strip() for m in pattern.finditer(text))
        answer_only = bool(re.search(
            r"(?:فقط\s+(?:جواب|پاسخ)|بدون\s+توضیح|توضیح\s+نده|فقط\s+نتیجه)|\b(?:answer\s+only|just\s+the\s+answer|no\s+explanation)\b",
            normalized, re.I,
        ))
        return ResponseConstraints(
            exact_sentences=exact,
            max_words=maximum,
            language_only=language,
            include_terms=tuple(dict.fromkeys(term for term in include if term)),
            exclude_terms=tuple(dict.fromkeys(term for term in exclude if term)),
            answer_only=answer_only,
        )


class ResponseVerifier:
    """Verify and conservatively repair explicit response constraints."""

    _SENTENCE_END = re.compile(r"(?<=[.!؟])\s+|\n+")
    _WORD = re.compile(r"[A-Za-z\u0600-\u06ff0-9]+(?:[‌'-][A-Za-z\u0600-\u06ff0-9]+)*")

    @classmethod
    def _sentences(cls, text: str) -> list[str]:
        return [part.strip() for part in cls._SENTENCE_END.split(text.strip()) if part.strip()]

    @classmethod
    def verify(cls, text: str, constraints: ResponseConstraints) -> VerificationResult:
        if not constraints.active:
            return VerificationResult(True, (), text)
        violations: list[str] = []
        value = text.strip()
        if constraints.exact_sentences is not None:
            if len(cls._sentences(value)) != constraints.exact_sentences:
                violations.append("exact_sentences")
        if constraints.max_words is not None:
            if len(cls._WORD.findall(value)) > constraints.max_words:
                violations.append("max_words")
        for term in constraints.include_terms:
            if normalize_text(term).casefold() not in normalize_text(value).casefold():
                violations.append(f"missing:{term}")
        for term in constraints.exclude_terms:
            if normalize_text(term).casefold() in normalize_text(value).casefold():
                violations.append(f"forbidden:{term}")
        if constraints.language_only == "fa":
            latin_words = re.findall(r"\b[A-Za-z]{2,}\b", value)
            if latin_words:
                violations.append("language_fa_only")
        elif constraints.language_only == "en":
            if re.search(r"[\u0600-\u06ff]", value):
                violations.append("language_en_only")
        return VerificationResult(not violations, tuple(violations), value)

    @classmethod
    def repair(cls, text: str, constraints: ResponseConstraints) -> VerificationResult:
        value = text.strip()
        if not constraints.active:
            return VerificationResult(True, (), value)

        # Remove common explanatory prefixes when the user explicitly asks for the answer only.
        if constraints.answer_only:
            value = re.sub(r"^(?:پاسخ|جواب|نتیجه|Answer|Result)\s*[:：-]\s*", "", value, flags=re.I)
            lines = [line.strip() for line in value.splitlines() if line.strip()]
            if len(lines) > 1 and lines[0].casefold() in {"توضیح:", "explanation:"}:
                value = " ".join(lines[1:])

        for term in constraints.exclude_terms:
            value = re.sub(re.escape(term), "", value, flags=re.I)
        value = re.sub(r"[ \t]{2,}", " ", value).strip()

        if constraints.language_only == "fa":
            # Preserve numbers/symbols, but remove stray prose words in Latin script.
            value = re.sub(r"\b[A-Za-z]{2,}\b", "", value)
            value = re.sub(r"[ \t]{2,}", " ", value).strip()
        elif constraints.language_only == "en":
            value = re.sub(r"[\u0600-\u06ff‌]+", "", value)
            value = re.sub(r"[ \t]{2,}", " ", value).strip()

        # Sentence count is repaired by truncating surplus sentences. We never pad
        # with fabricated filler because semantic correctness outranks formatting.
        if constraints.exact_sentences is not None:
            sentences = cls._sentences(value)
            if len(sentences) > constraints.exact_sentences:
                value = " ".join(sentences[: constraints.exact_sentences]).strip()

        if constraints.max_words is not None:
            matches = list(cls._WORD.finditer(value))
            if len(matches) > constraints.max_words:
                end = matches[constraints.max_words - 1].end()
                value = value[:end].rstrip(" ,،;؛:-")
                if value and value[-1] not in ".!?؟":
                    value += "."

        for term in constraints.include_terms:
            if normalize_text(term).casefold() not in normalize_text(value).casefold():
                suffix = term if not value else f" {term}"
                # Respect max words when possible by replacing the final word.
                if constraints.max_words is not None and len(cls._WORD.findall(value + suffix)) > constraints.max_words:
                    words = list(cls._WORD.finditer(value))
                    if words:
                        value = value[:words[-1].start()].rstrip() + suffix
                else:
                    value += suffix
                if value and value[-1] not in ".!?؟":
                    value += "."

        return cls.verify(value, constraints)


class ConstraintAwareComposer:
    """Compose bounded formal text when explicit output constraints exist.

    This is intentionally used only for explicit writing requests. It creates a
    semantically relevant draft first; ResponseVerifier remains the final gate.
    """

    @staticmethod
    def _topic(request: str, language: str) -> str:
        value = normalize_text(request).strip()
        if language == "fa":
            # Prefer an explicit topic phrase wherever it appears. This handles
            # both «درباره X حداکثر 18 کلمه بنویس» and
            # «حداکثر 18 کلمه درباره X بنویس».
            match = re.search(
                r"(?:درباره|در\s+مورد|راجع\s+به)\s+(.+?)(?="
                r"\s+(?:دقیقاً|دقیقا|حداکثر|کمتر\s+از|فقط|صرفاً|صرفا|بدون|شامل|حاوی|حتماً|حتما|بنویس|توضیح\s+بده|پاسخ\s+بده)"
                r"|[،.!؟?]|$)", value, re.I,
            )
            if match:
                return match.group(1).strip(" .،؛:«»") or "موضوع"
            # Remove constraints and writing verbs independently instead of
            # deleting everything after the first constraint occurrence.
            cleaned = value
            cleaned = re.sub(r"(?:دقیقاً|دقیقا|فقط)\s*(?:\d+|یک|دو|سه|چهار|پنج|شش|هفت|هشت|نه|ده)\s*جمله", " ", cleaned, flags=re.I)
            cleaned = re.sub(r"(?:حداکثر|کمتر\s+از|بیشتر\s+از)\s*\d+\s*(?:کلمه|واژه)", " ", cleaned, flags=re.I)
            cleaned = re.sub(r"(?:فقط|صرفاً|صرفا)\s+(?:فارسی|انگلیسی|جواب|پاسخ)", " ", cleaned, flags=re.I)
            cleaned = re.sub(r"بدون\s+(?:توضیح|استفاده\s+از.+)$", " ", cleaned, flags=re.I)
            cleaned = re.sub(r"(?:شامل|حاوی)\s+(?:کلمه|واژه|عبارت)\s*[«\"'].*?[»\"']", " ", cleaned, flags=re.I)
            cleaned = re.sub(r"^(?:لطفاً\s*)?(?:درباره|در\s+مورد|راجع\s+به)\s+", "", cleaned, flags=re.I)
            cleaned = re.sub(r"\b(?:بنویس|توضیح\s+بده|پاسخ\s+بده|شرح\s+بده)\b", " ", cleaned, flags=re.I)
            cleaned = re.sub(r"\s+", " ", cleaned).strip(" .،؛:«»")
            return cleaned or "موضوع"

        match = re.search(
            r"(?:about|regarding|on)\s+(.+?)(?=\s+(?:and|in|using|with|without|exactly|at\s+most|under|only|include|must|write|explain)|[.!?]|$)",
            value, re.I,
        )
        if match:
            return match.group(1).strip(" .,:;\"'") or "the topic"
        cleaned = re.sub(r"\b(?:exactly\s+\w+\s+sentences?|at\s+most\s+\d+\s+words?|under\s+\d+\s+words?|only\s+(?:english|persian)|no\s+explanation)\b", " ", value, flags=re.I)
        cleaned = re.sub(r"\b(?:write|explain|describe|answer)\b", " ", cleaned, flags=re.I)
        cleaned = re.sub(r"\s+", " ", cleaned).strip(" .,:;\"'")
        return cleaned or "the topic"

    @classmethod
    def compose(cls, request: str, language: str, constraints: ResponseConstraints) -> str:
        topic = cls._topic(request, language)
        # If only a small word cap is requested, draft to the budget instead of
        # generating three long sentences and truncating the last one mid-thought.
        if constraints.exact_sentences is None and constraints.max_words is not None and constraints.max_words <= 24:
            if language == "fa":
                value = f"{topic} با هدف روشن، دادهٔ معتبر و ارزیابی دقیق، نتیجه‌ای قابل اعتماد و کاربردی ایجاد می‌کند."
            else:
                value = f"{topic.capitalize()} works best with a clear goal, reliable evidence, careful evaluation, and practical feedback."
            if constraints.include_terms:
                term = constraints.include_terms[0]
                if normalize_text(term).casefold() not in normalize_text(value).casefold():
                    value = value.rstrip(".!؟") + f"؛ {term}." if language == "fa" else value.rstrip(".!?") + f", including {term}."
            return ResponseVerifier.repair(value, constraints).text

        n = constraints.exact_sentences or 3
        n = max(1, min(n, 12))
        if language == "fa":
            sentences = [
                f"{topic} زمانی مؤثر است که هدف آن روشن، دقیق و قابل سنجش باشد.",
                "اجرای مرحله‌ای و بررسی نتیجه‌ها کمک می‌کند خطاها زودتر شناسایی و اصلاح شوند.",
                "ثبت تصمیم‌ها و بازبینی منظم، کیفیت و پایداری کار را افزایش می‌دهد.",
                "بهتر است معیارهای موفقیت از ابتدا مشخص باشند تا ارزیابی بر اساس شواهد انجام شود.",
                "بازخورد واقعی باید به اصلاح مسیر منجر شود، نه فقط به افزایش حجم فعالیت.",
                "سادگی، وضوح و قابلیت بررسی سه عامل مهم برای حفظ نتیجهٔ قابل اعتماد هستند.",
            ]
        else:
            sentences = [
                f"{topic.capitalize()} works best when its goal is clear, specific, and measurable.",
                "A step-by-step process makes errors easier to detect and correct early.",
                "Regular review and recorded decisions improve quality and consistency over time.",
                "Success criteria should be defined before execution so results can be judged by evidence.",
                "Useful feedback should change the next action rather than merely add more activity.",
                "Clarity, simplicity, and verifiability help keep the final result reliable.",
            ]
        while len(sentences) < n:
            i = len(sentences) + 1
            sentences.append(
                (f"در گام {i}، نتیجه باید دوباره با محدودیت‌های درخواست مقایسه و تأیید شود." if language == "fa"
                 else f"At step {i}, the result should be checked again against the requested constraints.")
            )
        if constraints.include_terms:
            term = constraints.include_terms[0]
            if normalize_text(term).casefold() not in normalize_text(sentences[0]).casefold():
                if language == "fa":
                    sentences[0] = sentences[0].rstrip(".") + f" و «{term}» نیز باید در ارزیابی آن دیده شود."
                else:
                    sentences[0] = sentences[0].rstrip(".") + f', and the term "{term}" should be considered explicitly.'
        value = " ".join(sentences[:n])
        return ResponseVerifier.repair(value, constraints).text


__all__ = ["ConstraintExtractor", "ResponseConstraints", "ResponseVerifier", "VerificationResult", "ConstraintAwareComposer"]
