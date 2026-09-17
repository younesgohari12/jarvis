from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from jarvis.agent.challenge_reasoner import ChallengeReasoner
from jarvis.agent.followup_model import FollowupIntentModel
from jarvis.agent.fluency import PersianFluencyEngine
from jarvis.utils.text import normalize_text


@dataclass(frozen=True, slots=True)
class ChainResponse:
    text: str
    intent: str
    state: dict[str, Any] = field(default_factory=dict)
    fact_key: str = ""
    fact_value: str = ""


class ConversationChainReasoner:
    """High-confidence conversational continuations that need turn state.

    The normal knowledge and fluency engines remain responsible for open-ended
    language.  This component only handles explicit facts and narrow follow-ups
    where losing the previous subject would produce a clearly wrong response.
    """

    _LEXICON = {
        "پنجره": ("window", "windows"),
        "کتاب": ("book", "books"),
        "درخت": ("tree", "trees"),
        "خانه": ("house", "houses"),
        "ماشین": ("car", "cars"),
        "کودک": ("child", "children"),
        "مرد": ("man", "men"),
        "زن": ("woman", "women"),
    }
    _SENTENCES = {
        "window": (
            "The window overlooks the garden.",
            "The window provides a view of the garden.",
        ),
        "book": (
            "This book explains the subject clearly.",
            "This book provides a clear explanation of the subject.",
        ),
        "tree": (
            "The tree stands beside the building.",
            "The tree is situated next to the building.",
        ),
        "house": (
            "The house is near the station.",
            "The house is located near the station.",
        ),
        "car": (
            "The car is ready for the journey.",
            "The car is prepared for the journey.",
        ),
        "child": (
            "The child completed the task.",
            "The child successfully completed the assigned task.",
        ),
        "man": (
            "The man entered the office.",
            "The man proceeded into the office.",
        ),
        "woman": (
            "The woman presented the report.",
            "The woman formally presented the report.",
        ),
    }
    _FACT_STORE = re.compile(
        r"(?P<label>(?:رمز|کد|شناسه|نام|شماره|آدرس)\s+.{1,60}?)\s+من\s+"
        r"(?P<value>.{1,100}?)\s+(?:است|هست)(?:\b|[؛،,.!?؟])",
        re.I,
    )
    _FACT_STORE_COMMAND = re.compile(
        r"(?P<label>(?:رمز|کد|شناسه|نام|شماره|آدرس)\s+.{1,60}?)\s+من\s+"
        r"(?P<value>.{1,100}?)(?:\s+(?:است|هست))?\s*(?:را|رو)?\s*[؛،,.!?؟]*\s*"
        r"(?=(?:یادت\s+(?:نگه\s+دار|بمونه|بماند)|به\s+خاطر\s+بسپار|به\s+یاد\s+داشته\s+باش|"
        r"حفظ(?:ش)?\s+کن|ذخیره\s+کن|ثبت\s+کن|فراموش\s+نکن|remember))",
        re.I,
    )
    _FACT_RECALL = re.compile(
        r"(?P<label>(?:رمز|کد|شناسه|نام|شماره|آدرس)\s+.{1,70}?)"
        r"(?=\s+(?:چه|چی|را|رو|یادت|ذخیره|ثبت|بگو|یادآوری))",
        re.I,
    )
    _TERM = re.compile(
        r"(?:کلمه|واژه)(?:ٔ|ی)?\s+[«\"']?([\w\u0600-\u06FF‌-]{1,40})[»\"']?",
        re.I,
    )
    _DIRECT_TERM = re.compile(
        r"(?:معادل|ترجمه)\s+انگلیسی\s+(?:(?:کلمه|واژه)(?:ٔ|ی)?\s+)?"
        r"[«\"']?([\w\u0600-\u06FF‌-]{1,40})[»\"']?(?=\s+(?:چیست|چیه|چی\s*می.?شه|است|می.?شود)|[؟?]|$)",
        re.I,
    )

    def __init__(self, model_path: Path | None = None) -> None:
        self.followup_model = FollowupIntentModel(model_path) if model_path else None

    @staticmethod
    def _current_turn(context: dict[str, Any]) -> int:
        try:
            return int(context.get("_current_turn", 0) or 0)
        except (TypeError, ValueError):
            return 0

    @classmethod
    def _fresh(cls, context: dict[str, Any], key: str, maximum_gap: int = 1) -> bool:
        current = cls._current_turn(context)
        try:
            previous = int(context.get(key, -100) or -100)
        except (TypeError, ValueError):
            return False
        return current > 0 and 0 <= current - previous <= maximum_gap

    @staticmethod
    def _canonical_fact_label(value: str) -> str:
        label = normalize_text(value).strip(" -—:،؛؟?!.,\"'«»")
        label = re.sub(
            r"(?:ٔ|‌?ای|ی)?\s+که\s+(?:(?:اول|قبلاً|قبلا)\s+)?"
            r"(?:گفتم|گفته\s+بودم|سپردم).*$",
            "",
            label,
        )
        label = re.sub(r"\s+من\s*(?:را|رو)?$", "", label)
        label = re.sub(r"\s+(?:قبلی|ذخیره.?شده|ثبت.?شده)$", "", label)
        label = re.sub(r"(جدید|شخصی|کاری)(?:‌?ام|م)$", r"\1", label)
        label = label.replace("ٔ", "")
        return re.sub(r"\s+", " ", label).strip()

    @classmethod
    def _fact_key(cls, label: str) -> str:
        return "user_fact:" + cls._canonical_fact_label(label).replace(" ", "_")[:90]

    @classmethod
    def _recall_fact(cls, label: str, facts: dict[str, str]) -> str:
        canonical = cls._canonical_fact_label(label)
        exact = facts.get(cls._fact_key(canonical))
        if exact:
            return exact
        requested = set(canonical.split())
        ranked: list[tuple[float, str]] = []
        for key, value in facts.items():
            if not key.startswith("user_fact:"):
                continue
            candidate = set(key.removeprefix("user_fact:").replace("_", " ").split())
            if not candidate or not requested:
                continue
            score = len(candidate & requested) / max(len(candidate), len(requested))
            if next(iter(candidate), "") == next(iter(requested), ""):
                score += 0.2
            ranked.append((score, value))
        if ranked and max(ranked)[0] >= 0.6:
            return max(ranked)[1]
        if re.search(r"(?:کد|رمز)\s+پروژه", canonical):
            return facts.get("project_code", "")
        return ""

    def _followup_intent(self, text: str) -> tuple[str, float]:
        normalized = normalize_text(text)
        deterministic = (
            ("summary", r"(?:خلاصه|جمع.?بندی|یک.?جمله.?ای|در\s+یک\s+(?:جمله|خط)|مختصر|لب\s+مطلب)"),
            ("example", r"(?:مثال|نمونه|مورد\s+کاربردی|دنیای\s+واقعی|ملموس)"),
            ("plural", r"(?:جمع|plural|تعداد\s+بیشتر).{0,60}(?:آن|این|همان|کلمه|واژه|لغت|word)"),
            ("rewrite_previous", r"(?:همان|قبلی|بالا|پاسخ|جمله|عبارت).{0,60}(?:رسمی|حرفه.?ای|بازنویسی|محترمانه)"),
            ("sentence", r"(?:جمله|sentence).{0,50}(?:همان|آن|این|کلمه|واژه|لغت)|(?:با|حاوی).{0,40}(?:جمله|sentence)"),
        )
        for label, pattern in deterministic:
            if re.search(pattern, normalized, re.I):
                return label, 1.0
        # A complete language question with its own term is a new request, not
        # a continuation.  This hard negative prevents stale word context from
        # hijacking queries such as "معادل انگلیسی کودک چیست؟".
        if self._term(normalized) and re.search(
            r"(?:معادل|ترجمه|مخفف|معنی).{0,80}(?:انگلیسی|فارسی)|"
            r"(?:انگلیسی|فارسی).{0,80}(?:معادل|ترجمه|مخفف|معنی)",
            normalized,
            re.I,
        ):
            return "", 0.0
        # The learned classifier is intentionally bounded to utterances that
        # contain a continuation cue.  Without this gate, lexical similarity
        # alone can mistake independent commands (close an app, write a short
        # story, etc.) for a follow-up.
        if not re.search(
            r"(?:قبلی|بالا|همان|همین|همونو|همون|آن|اون|این\s+(?:توضیح|پاسخ|لغت|واژه|کلمه)|"
            r"گفتی|ساختی|ترجمه\s+کردی|توضیحش|پاسخش|جمعش|معنایش|نتیجه.?اش|"
            r"کلمه.?ای\s+که|واژه.?ای\s+که|لغتی\s+که|ادامه\s+بده|هم\s+بزن|"
            r"کوتاه.?تر|رسمی.?ترش|حرف\s+پیش|پاسخ\s+پیش)",
            normalized,
            re.I,
        ):
            return "", 0.0
        if self.followup_model and self.followup_model.ready:
            label, confidence, margin = self.followup_model.predict(text)
            if label != "standalone" and confidence >= 0.2 and margin >= 0.035:
                return label, confidence
        return "", 0.0

    @staticmethod
    def _one_sentence_summary(text: str, language: str) -> str:
        clean = re.sub(r"\s+", " ", str(text)).strip()
        if re.search(r"\bHDD\b", clean, re.I) and re.search(r"\bSSD\b", clean, re.I):
            return (
                "SSD سریع‌تر، کم‌تأخیرتر و بدون قطعات متحرک است، اما HDD معمولاً ظرفیت زیاد را با قیمت کمتری ارائه می‌دهد."
                if language == "fa"
                else "An SSD is faster and has no moving parts, while an HDD usually offers more capacity for less money."
            )
        if re.search(r"\bTCP\b", clean, re.I) and re.search(r"\bUDP\b", clean, re.I):
            return (
                "TCP تحویل مرتب و مطمئن را با سربار بیشتر فراهم می‌کند، اما UDP سبک‌تر و کم‌تأخیرتر است و تحویل یا ترتیب داده‌ها را تضمین نمی‌کند."
                if language == "fa"
                else "TCP provides reliable, ordered delivery with more overhead, whereas UDP is lighter and lower-latency but does not guarantee delivery or order."
            )
        sentences = [part.strip() for part in re.split(r"(?<=[.!؟])\s+|\n+", clean) if part.strip()]
        if not sentences:
            return clean[:280]
        summary = sentences[0]
        if len(summary) < 80 and len(sentences) > 1:
            summary = summary.rstrip(".!؟") + "؛ " + sentences[1]
        return summary[:360].rstrip("؛، ")

    @staticmethod
    def _contextual_example(previous: str, language: str) -> str:
        if re.search(r"\bHDD\b", previous, re.I) and re.search(r"\bSSD\b", previous, re.I):
            return (
                "مثلاً ویندوز و بازی‌ها روی SSD سریع‌تر اجرا و بارگذاری می‌شوند، ولی برای آرشیو چند ترابایت فیلم با هزینهٔ کمتر می‌توان از HDD استفاده کرد."
                if language == "fa"
                else "For example, Windows and games load faster from an SSD, while an HDD can store a large media archive more cheaply."
            )
        if re.search(r"\bTCP\b", previous, re.I) and re.search(r"\bUDP\b", previous, re.I):
            return (
                "مثلاً دانلود فایل به ترتیب و صحت داده نیاز دارد و معمولاً از TCP استفاده می‌کند، اما تماس زنده می‌تواند برای کاهش تأخیر از UDP بهره ببرد."
                if language == "fa"
                else "For example, file downloads need TCP's reliable ordering, while a live call may use UDP to reduce latency."
            )
        return ""

    @staticmethod
    def _is_remember_request(text: str) -> bool:
        return bool(
            re.search(
                r"(?:یادت\s+(?:نگه\s+دار|بمونه|بماند)|به\s+خاطر\s+بسپار|"
                r"به\s+یاد\s+داشته\s+باش|حفظ(?:ش)?\s+کن|ذخیره\s+کن|ثبت\s+کن|"
                r"فراموش\s+نکن|remember)",
                text,
                re.I,
            )
        )

    @classmethod
    def _term(cls, text: str) -> str:
        match = cls._TERM.search(text) or cls._DIRECT_TERM.search(text)
        if not match:
            return ""
        return normalize_text(match.group(1)).strip(" -—:،؛؟?!.,\"'«»")

    @staticmethod
    def _is_ram_storage(text: str) -> bool:
        return bool(
            re.search(r"\bRAM\b|\bرم\b", text, re.I)
            and re.search(r"(?:ذخیره|storage|SSD|دیسک)", text, re.I)
            and re.search(r"(?:تفاوت|فرق|difference|چیست|چیه)", text, re.I)
        )

    @staticmethod
    def _is_cpu_gpu(text: str) -> bool:
        return bool(
            re.search(r"\bCPU\b|پردازنده", text, re.I)
            and re.search(r"\bGPU\b|پردازنده\s+گرافیکی", text, re.I)
            and re.search(r"(?:تفاوت|فرق|difference|چیست|چیه)", text, re.I)
        )

    def answer(
        self,
        text: str,
        language: str,
        context: dict[str, Any],
        facts: dict[str, str] | Callable[[str], str | None],
    ) -> ChainResponse | None:
        cls = type(self)
        if ChallengeReasoner.is_prompt_batch(text):
            return None
        clean = normalize_text(text)
        fact_values = facts if isinstance(facts, dict) else {
            "project_code": facts("project_code") or ""
        }
        current_turn = cls._current_turn(context)

        stored_original = cls._FACT_STORE.search(text) or cls._FACT_STORE_COMMAND.search(text)
        stored = stored_original or cls._FACT_STORE.search(clean) or cls._FACT_STORE_COMMAND.search(clean)
        if stored and cls._is_remember_request(clean):
            label = cls._canonical_fact_label(stored.group("label"))
            # Prefer the original match to preserve Latin casing and combining
            # marks; normalized Persian remains the fallback for spelling variants.
            value = stored.group("value").strip(" -—:،؛؟?!.,\"'«»")
            if value:
                response = (
                    f"باشه؛ «{label}» را به خاطر سپردم."
                    if language == "fa"
                    else f"Okay; I saved {label}."
                )
                return ChainResponse(
                    response,
                    "fact_remembered",
                    {"last_conversation_topic": "remembered_fact", "last_fact_label": label},
                    cls._fact_key(label),
                    value,
                )

        recalled = cls._FACT_RECALL.search(clean)
        if recalled:
            label = cls._canonical_fact_label(recalled.group("label"))
            value = cls._recall_fact(label, fact_values)
            if value:
                response = (
                    f"{label} که گفتی «{value}» بود."
                    if language == "fa"
                    else f"The saved value for {label} was “{value}”."
                )
                return ChainResponse(response, "fact_recalled", {"last_conversation_topic": "remembered_fact"})
            return ChainResponse(
                f"مقداری برای «{label}» در حافظه پیدا نکردم."
                if language == "fa" else f"I could not find a saved value for “{label}”.",
                "fact_not_found",
            )

        if cls._is_ram_storage(clean):
            if language == "fa":
                response = (
                    "RAM حافظهٔ کاریِ سریع و موقتی است؛ برنامه‌ها هنگام اجرا داده‌های فعال را آنجا "
                    "نگه می‌دارند و با خاموش‌شدن سیستم پاک می‌شود. حافظهٔ ذخیره‌سازی مثل SSD، فایل‌ها "
                    "را پایدار و بلندمدت نگه می‌دارد، اما معمولاً از RAM کندتر است."
                )
            else:
                response = (
                    "RAM is fast, temporary working memory for active programs and is cleared at shutdown; "
                    "storage such as an SSD keeps files persistently and is usually slower than RAM."
                )
            return ChainResponse(response, "knowledge_answer", {"last_explanation_topic": "ram_storage"})

        if cls._is_cpu_gpu(clean):
            if language == "fa":
                response = (
                    "CPU چند هستهٔ عمومی و قدرتمند برای کنترل سیستم و کارهای ترتیبی دارد؛ GPU تعداد "
                    "زیادی هستهٔ ساده‌تر برای انجام محاسبات مشابه به‌صورت موازی دارد. GPU همیشه سریع‌تر "
                    "نیست و برتری آن به موازی‌بودن مسئله بستگی دارد."
                )
            else:
                response = (
                    "A CPU has a few powerful general-purpose cores for control and sequential work; a GPU "
                    "has many simpler cores for highly parallel calculations. A GPU is not always faster."
                )
            return ChainResponse(response, "knowledge_answer", {"last_explanation_topic": "cpu_gpu"})

        if context.get("last_explanation_topic") == "ram_storage":
            if re.search(r"(?:مثال|نمونه|example)", clean, re.I):
                response = (
                    "مثلاً هنگام اجرای یک بازی، فایل‌های خود بازی روی SSD باقی می‌مانند، اما مرحله و "
                    "بافت‌هایی که همان لحظه استفاده می‌شوند موقتاً در RAM قرار می‌گیرند."
                    if language == "fa"
                    else "For example, a game's files remain on the SSD, while the current level and active textures are temporarily loaded into RAM."
                )
                return ChainResponse(response, "contextual_example", {"last_explanation_topic": "ram_storage"})
            if re.search(r"(?:خلاصه|در\s+یک\s+جمله|summari[sz]e|in\s+one\s+sentence)", clean, re.I):
                response = (
                    "RAM فضای کاریِ سریع و موقتی است، اما SSD یا دیسک فایل‌ها را به‌صورت پایدار نگه می‌دارد."
                    if language == "fa"
                    else "RAM is fast temporary workspace, while an SSD or disk stores files persistently."
                )
                return ChainResponse(response, "contextual_summary", {"last_explanation_topic": "ram_storage"})

        if context.get("last_explanation_topic") == "cpu_gpu":
            if re.search(r"(?:مثال|نمونه|example)", clean, re.I):
                response = (
                    "مثلاً CPU منطق بازی و تصمیم‌های مرحله‌به‌مرحله را مدیریت می‌کند، اما GPU هزاران "
                    "پیکسل تصویر را هم‌زمان محاسبه می‌کند."
                    if language == "fa"
                    else "For example, the CPU handles game logic step by step, while the GPU calculates thousands of pixels in parallel."
                )
                return ChainResponse(response, "contextual_example", {"last_explanation_topic": "cpu_gpu"})
            if re.search(r"(?:خلاصه|در\s+یک\s+جمله|summari[sz]e|in\s+one\s+sentence)", clean, re.I):
                response = (
                    "CPU برای کنترل و کارهای عمومی ساخته شده، درحالی‌که GPU در محاسبات موازی گسترده قوی‌تر است."
                    if language == "fa"
                    else "The CPU handles control and general work, while the GPU excels at massively parallel computation."
                )
                return ChainResponse(response, "contextual_summary", {"last_explanation_topic": "cpu_gpu"})

        explicit_term = cls._term(clean)
        asks_translation = bool(
            re.search(r"(?:معادل|ترجمه).{0,80}(?:انگلیسی)|(?:انگلیسی).{0,80}(?:معادل|ترجمه)", clean, re.I)
        )
        if asks_translation and explicit_term in cls._LEXICON:
            singular, plural = cls._LEXICON[explicit_term]
            response = (
                f"معادل انگلیسی «{explicit_term}» کلمهٔ {singular} است."
                if language == "fa"
                else f"The English equivalent of “{explicit_term}” is {singular}."
            )
            return ChainResponse(
                response,
                "knowledge_answer",
                {
                    "last_information_subject": explicit_term,
                    "last_information_mode": "english_equivalent",
                    "last_language_word": singular,
                    "last_language_plural": plural,
                    "last_language_turn": current_turn,
                },
            )

        word = str(context.get("last_language_word") or "").strip()
        plural = str(context.get("last_language_plural") or "").strip()
        followup_intent, followup_confidence = self._followup_intent(clean)
        if followup_intent in {"summary", "rewrite_previous"} and re.search(
            r"(?:متن\s+(?:بالا|قبلی)|پیام\s+قبلی|چیزی\s+که\s+گفتم|حرف\s+قبلی)",
            clean,
            re.I,
        ):
            # The grounded-text path can retrieve the previous *user* passage;
            # assistant-answer context must not pre-empt that evidence.
            return None
        language_fresh = cls._fresh(context, "last_language_turn")
        answer_fresh = cls._fresh(context, "last_answer_turn")
        if followup_intent == "plural" and word and plural and language_fresh:
            response = (
                f"جمع {word} در انگلیسی {plural} است."
                if language == "fa"
                else f"The plural of {word} is {plural}."
            )
            return ChainResponse(response, "language_plural", {
                "last_information_mode": "english_plural", "last_language_turn": current_turn,
            })

        if followup_intent == "sentence" and word and language_fresh:
            sentence, formal = cls._SENTENCES.get(word, (f"I used the word {word} in this sentence.", ""))
            return ChainResponse(
                sentence,
                "language_example",
                {
                    "last_language_sentence": sentence, "last_language_formal_sentence": formal,
                    "last_language_turn": current_turn,
                },
            )

        previous_sentence = str(context.get("last_language_sentence") or "").strip()
        formal_sentence = str(context.get("last_language_formal_sentence") or "").strip()
        if followup_intent == "rewrite_previous" and previous_sentence and language_fresh:
            response = formal_sentence or previous_sentence
            return ChainResponse(
                response,
                "language_formal_rewrite",
                {"last_language_sentence": response, "last_language_turn": current_turn},
            )

        previous_answer = str(context.get("last_assistant_text") or "").strip()
        if followup_intent == "summary":
            if answer_fresh and previous_answer:
                return ChainResponse(
                    cls._one_sentence_summary(previous_answer, language),
                    "contextual_summary",
                    {"followup_model_confidence": followup_confidence},
                )
            return ChainResponse(
                "مشخص نیست کدام توضیح را باید خلاصه کنم؛ متن یا موضوع را دوباره بفرست."
                if language == "fa" else "I am not sure which explanation to summarize; send the text or topic again.",
                "context_clarification",
            )
        if followup_intent == "example":
            if answer_fresh and previous_answer:
                example = cls._contextual_example(previous_answer, language)
                if example:
                    return ChainResponse(
                        example, "contextual_example",
                        {"followup_model_confidence": followup_confidence},
                    )
                return ChainResponse(
                    "موضوع قبلی را دارم، اما برای نساختن مثال نادرست بگو از کدام بخشش مثال می‌خواهی."
                    if language == "fa" else "I have the previous topic, but specify which part needs an example so I do not invent one.",
                    "context_clarification",
                )
            return ChainResponse(
                "برای کدام موضوع مثال می‌خواهی؟"
                if language == "fa" else "Which topic would you like an example for?",
                "context_clarification",
            )
        if followup_intent == "rewrite_previous":
            if answer_fresh and previous_answer and language == "fa":
                if re.search(r"(?:کوتاه|مختصر|خلاصه|shorter|concise)", clean, re.I):
                    rewritten = cls._one_sentence_summary(previous_answer, language)
                else:
                    rewritten = PersianFluencyEngine._rewrite(f": {previous_answer}")
                return ChainResponse(rewritten, "contextual_rewrite")
            shorter = bool(re.search(r"(?:کوتاه|مختصر|خلاصه|shorter|concise)", clean, re.I))
            return ChainResponse(
                ("مشخص نیست کدام متن را باید کوتاه‌تر کنم؛ متن یا موضوع را دوباره بفرست."
                 if shorter else "جملهٔ موردنظر را دوباره بفرست تا دقیقاً همان را رسمی‌تر کنم.")
                if language == "fa" else "Send the sentence again so I can rewrite that exact text more formally.",
                "context_clarification",
            )
        if followup_intent in {"plural", "sentence"}:
            return ChainResponse(
                "مرجع واژه مشخص نیست؛ اول خود کلمه یا ترجمه‌اش را بگو."
                if language == "fa" else "The referenced word is unclear; provide the word or its translation first.",
                "context_clarification",
            )
        return None


__all__ = ["ChainResponse", "ConversationChainReasoner"]
