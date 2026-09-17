from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from jarvis.agent.challenge_reasoner import ChallengeReasoner
from jarvis.agent.deliberation import content_terms
from jarvis.utils.text import normalize_text, tokenize


@dataclass(frozen=True, slots=True)
class ContextualQuery:
    original: str
    query: str
    subject: str = ""
    used_context: bool = False
    mode: str = ""


class DialogueSubjectResolver:
    """Carries one explicit information subject across short follow-up questions."""

    _RESEARCH_WORDS = re.compile(
        r"(?:تحقیق\s*(?:کن|کنید)?|بررسی\s+(?:عمیق|کامل)\s*(?:کن)?|"
        r"\b(?:research|investigate)\b)",
        re.I,
    )
    _ABOUT_PREFIX = re.compile(
        r"^(?:لطفا\s+|لطفاً\s+|please\s+)?"
        r"(?:درباره(?:ی)?|راجع\s+به|در\s+مورد|پیرامون|about|on)\s+",
        re.I,
    )
    _FOLLOW_UP = re.compile(
        r"(?:^(?:چقدر|چند|کی|کجا|کدام|چه\s+(?:تیمی|سالی|کشوری|تعداد|مقدار)|"
        r"الان|فعلا|فعلاً|قدش|سنش|اسمش|تیمش|ملیتش|درآمدش|رکوردش)\b|"
        r"\b(?:او|وی|ایشون|ایشان|اون|آن|این\s+(?:فرد|شخص)|خودش|"
        r"he|she|they|his|her|their|that\s+(?:person|player))\b|"
        r"\b(?:how\s+many|how\s+much|when|where|which\s+team|how\s+old)\b)",
        re.I,
    )
    _INFORMATION_QUESTION = re.compile(
        r"(?:[؟?]$|^(?:چقدر|چند|کی|کجا|چه|آیا|الان|فعلا|فعلاً)|"
        r"\b(?:what|when|where|who|how|which|is|does|did|has|have)\b)",
        re.I,
    )
    _DESKTOP_ACTION = re.compile(
        r"(?:باز\s*(?:کن)?|ببند|اجرا\s*(?:کن)?|حذف\s*(?:کن)?|پاک\s*(?:کن)?|"
        r"بساز|ایجاد\s*(?:کن)?|منتقل\s*(?:کن)?|کپی\s*(?:کن)?|"
        r"\b(?:open|close|launch|delete|remove|create|move|copy|run)\b)",
        re.I,
    )
    _GREETING_PREFIX = re.compile(
        r"^(?:سلام|درود|hello|hi|hey)\s*[،,!]*\s+(.+)$",
        re.I,
    )
    _WELLBEING = re.compile(
        r"^(?:خوبی|حالت\s+چطوره|چه\s+خبر|how\s+are\s+you|what'?s\s+up)[؟?!.,،]*$",
        re.I,
    )
    _LANGUAGE_QUESTION = re.compile(
        r"(?:معنی|معنا|معادل|ترجمه|مخفف).{0,90}(?:انگلیسی|فارسی)|"
        r"(?:انگلیسی|فارسی).{0,90}(?:معنی|معنا|معادل|ترجمه|مخفف|چیست|چیه|چی\s*می.?شه)|"
        r"\b(?:translate|translation|meaning|equivalent|abbreviation)\b",
        re.I,
    )
    _LANGUAGE_FOLLOW_UP = re.compile(
        r"^(?:خب\s+)?(?:و\s+)?(?:کلمه|واژه)\s+.{1,45}?(?:چطور|چی|چه\s+می.?شه)[؟?!.,،]*$|"
        r"(?:انگلیسیش|فارسیش|معنیش|ترجمه.?اش|ترجمه.?ش|مخففش)",
        re.I,
    )
    _LANGUAGE_TERM = re.compile(
        r"(?:کلمه|واژه)(?:ٔ|ی)?\s+[«\"']?([\w\u0600-\u06FF‌-]{1,40})[»\"']?",
        re.I,
    )
    _GENERIC_LANGUAGE_TERMS = frozenset(
        {"مخفف", "معادل", "ترجمه", "معنی", "معنا", "کلمه", "واژه", "خب", "انگلیسی", "فارسی"}
    )
    _GENERIC_SUBJECT_WORDS = frozenset(
        {
            "کامل", "جامع", "دقیق", "موضوع", "اطلاعات", "اینترنت", "منبع",
            "complete", "full", "detailed", "topic", "information", "online",
        }
    )

    @classmethod
    def _strip_substantive_greeting(cls, text: str) -> str:
        match = cls._GREETING_PREFIX.match(text)
        if not match:
            return text
        remainder = match.group(1).strip()
        if cls._WELLBEING.fullmatch(normalize_text(remainder)):
            return text
        normalized = normalize_text(remainder)
        is_question = bool(
            re.search(
                r"(?:چیست|چیه|یعنی|چرا|چطور|چگونه|کدام|آیا|چه\s+|"
                r"\b(?:what|why|how|which|is|does|define|explain)\b)",
                normalized,
                re.I,
            )
            or remainder.endswith(("؟", "?"))
        )
        return remainder if is_question and len(tokenize(remainder)) >= 2 else text

    @classmethod
    def _language_term(cls, text: str) -> str:
        normalized = normalize_text(text)
        match = cls._LANGUAGE_TERM.search(normalized)
        candidate = match.group(1) if match else ""
        if not candidate:
            suffix = re.search(
                r"[«\"']?([\w\u0600-\u06FF‌-]{2,40})[»\"']?\s+(?:به\s+)?(?:انگلیسی|فارسی)\b",
                normalized,
                re.I,
            )
            candidate = suffix.group(1) if suffix else ""
        candidate = candidate.strip(" -—:،؛؟?!.,\"'«»")
        if normalize_text(candidate) in cls._GENERIC_LANGUAGE_TERMS:
            return ""
        return candidate

    @staticmethod
    def _language_mode(text: str, fallback: str = "") -> str:
        normalized = normalize_text(text)
        if re.search(r"(?:انگلیسی|english)", normalized, re.I):
            return "english_equivalent"
        if re.search(r"(?:فارسی|persian|farsi)", normalized, re.I):
            return "persian_equivalent"
        return fallback

    @staticmethod
    def _language_query(term: str, mode: str) -> str:
        if mode == "persian_equivalent":
            return f"معادل یا ترجمه فارسی کلمه {term} چیست؟"
        return f"معادل یا مخفف انگلیسی کلمه {term} چیست؟"

    @classmethod
    def extract_subject(cls, query: str) -> str:
        clean = normalize_text(
            str(query).replace("\u2066", "").replace("\u2067", "").replace("\u2069", "")
        )
        clean = cls._RESEARCH_WORDS.sub(" ", clean)
        clean = cls._ABOUT_PREFIX.sub("", clean.strip())
        clean = re.sub(r"^(?:برای|از|the\s+topic\s+of)\s+", "", clean, flags=re.I)
        clean = re.sub(r"\s+(?:رو|را)\s*$", "", clean)
        clean = clean.strip(" -—:،؛؟?!.,\"'«»")
        words = tokenize(clean)
        if not 1 <= len(words) <= 8:
            return ""
        useful = [
            word for word in words
            if normalize_text(word) not in cls._GENERIC_SUBJECT_WORDS
        ]
        if not useful:
            return ""
        return " ".join(useful[:6]).strip()

    @staticmethod
    def _subject_present(subject: str, question: str) -> bool:
        subject_terms = set(content_terms(subject))
        question_terms = set(content_terms(question))
        if not subject_terms:
            return False
        return len(subject_terms & question_terms) >= max(1, len(subject_terms) // 2)

    @classmethod
    def resolve(cls, text: str, context: dict[str, Any]) -> ContextualQuery:
        raw = str(text).strip()
        original = " ".join(raw.split())
        # A pasted multi-message/test batch is one isolated payload. Detect this
        # before code/format guards so words such as «رسمی‌تر» inside one quoted
        # sub-message cannot preserve stale multi-line context accidentally.
        if ChallengeReasoner.is_prompt_batch(original):
            return ContextualQuery(original, original)
        # Preserve code layout and syntax verbatim. Flattening newlines destroys
        # Python block structure and can turn x=3 into an algebra question.
        if re.search(r"```|\b(?:def|for|while|if|elif|else|print|return|range)\b|(?:^|\n)\s*[A-Za-z_]\w*\s*=", raw, re.I):
            return ContextualQuery(raw, raw)
        # Explicit writing/format constraints are not vocabulary follow-ups even
        # when they contain words such as «فارسی», «انگلیسی» or «کلمه».
        if re.search(
            r"(?:بنویس|بازنویسی|رسمی.?تر|روان.?تر|دقیقاً\s*\d+\s*جمله|حداکثر\s*\d+\s*(?:کلمه|واژه)|"
            r"فقط\s+(?:فارسی|انگلیسی)|شامل\s+(?:کلمه|واژه)|write|rewrite|exactly\s+\d+\s+sentences?|at\s+most\s+\d+\s+words?|only\s+(?:persian|english)|include\s+the\s+word)",
            raw, re.I,
        ):
            return ContextualQuery(raw, raw)
        clean = cls._strip_substantive_greeting(original)
        normalized = normalize_text(clean)
        previous_subject = str(context.get("last_information_subject") or "").strip()
        previous_mode = str(context.get("last_information_mode") or "").strip()
        current_turn = int(context.get("_current_turn", 0) or 0)
        previous_turn = int(context.get("last_information_turn", 0) or 0)
        # Direct unit/API callers without turn metadata retain the legacy
        # behavior.  The agent always supplies turn metadata and therefore
        # expires a pronoun/translation subject after one intervening turn.
        information_fresh = not current_turn or (
            previous_turn > 0 and 0 <= current_turn - previous_turn <= 1
        )
        if not information_fresh:
            previous_subject = ""
            previous_mode = ""
        # Full-sentence translation commands must stay intact.  Older versions
        # tried to extract a single "term" and could turn "به انگلیسی ترجمه کن: ..."
        # into a query about the preposition «به».
        direct_translation = bool(re.search(
            r"(?:به\s+(?:انگلیسی|فارسی).{0,24}(?:ترجمه(?:(?:‌|\s)*اش)?\s+کن|بگو)|"
            r"ترجمه(?:(?:‌|\s)*اش)?\s+کن.{0,24}به\s+(?:انگلیسی|فارسی)|"
            r"(?:این(?:\s+جمله|\s+متن)?\s+(?:را|رو)\s+)?(?:انگلیسی|فارسی)(?:ش|اش|‌اش)?\s*(?:کن|بگو)|"
            r"translate(?:\s+this|\s+it|\s+the\s+following)?\s+(?:to|into)\s+(?:english|persian|farsi)|"
            r"say\s+(?:this|it)\s+in\s+(?:english|persian|farsi)|(?:english|persian|farsi)\s+translation)",
            normalized, re.I,
        ))
        if direct_translation:
            return ContextualQuery(original, clean)

        explicit_language = bool(cls._LANGUAGE_QUESTION.search(normalized))
        follow_up_language = bool(
            previous_mode and cls._LANGUAGE_FOLLOW_UP.search(normalized)
        )
        if explicit_language or follow_up_language:
            explicit_term = cls._language_term(clean)
            term = explicit_term or previous_subject
            mode = cls._language_mode(clean, previous_mode or "english_equivalent")
            if term:
                used_context = not explicit_term or follow_up_language
                return ContextualQuery(
                    original,
                    cls._language_query(term, mode),
                    term,
                    used_context,
                    mode,
                )
        if cls._RESEARCH_WORDS.search(normalize_text(clean)):
            explicit = cls.extract_subject(clean)
            return ContextualQuery(original, clean, explicit, False)
        subject = previous_subject
        if not subject and information_fresh:
            subject = str(context.get("last_research_subject") or "").strip()
        if not subject:
            return ContextualQuery(original, clean)
        if cls._subject_present(subject, clean):
            return ContextualQuery(original, clean, subject, False)
        if len(tokenize(clean)) > 22:
            return ContextualQuery(original, clean)
        if cls._DESKTOP_ACTION.search(normalize_text(clean)):
            return ContextualQuery(original, clean)
        if not cls._INFORMATION_QUESTION.search(clean) or not cls._FOLLOW_UP.search(clean):
            return ContextualQuery(original, clean)
        return ContextualQuery(
            original,
            f"{subject} — {clean}",
            subject,
            True,
        )


__all__ = ["ContextualQuery", "DialogueSubjectResolver"]
