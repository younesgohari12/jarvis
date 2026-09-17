from __future__ import annotations

import re
from pathlib import Path

from jarvis.agent.fluency import FluentResponse, PersianFluencyEngine
from jarvis.utils.text import normalize_text


class BilingualFluencyEngine(PersianFluencyEngine):
    """Project-trained Persian/English writing selector and bounded composer."""

    FORMAT = "jarvis-bilingual-fluency-v1"

    @staticmethod
    def _english_topic(text: str) -> str:
        value = normalize_text(text)
        match = re.search(r"(?:about|on|regarding|for)\s+(.+?)(?:\s+(?:please\s+)?(?:write|create|draft|make))?[.!?]*$", value, re.I)
        if match:
            topic = match.group(1)
        else:
            topic = re.sub(
                r"^(?:please\s+)?(?:write|create|draft|make|tell)\s+(?:me\s+)?"
                r"(?:a|an|some)?\s*(?:short\s+)?(?:text|message|caption|story|email|paragraph)\s*",
                "", value, flags=re.I,
            )
        topic = re.sub(r"\s*(?:please|for me)$", "", topic, flags=re.I).strip(" .,!?:;\"'")
        return topic[:120] or "the topic"

    @staticmethod
    def _english_kind(text: str, predicted: str) -> str:
        value = normalize_text(text)
        if re.search(r"(?:rewrite|rephrase|edit|make\s+it\s+(?:formal|clearer|shorter))", value, re.I):
            return "rewrite"
        if re.search(r"(?:formal|professional|polite).*(?:message|email|letter)|(?:message|email|letter).*(?:formal|professional|polite)", value, re.I):
            return "formal_message"
        if re.search(r"(?:caption|social\s+(?:post|media))", value, re.I):
            return "caption"
        if re.search(r"(?:story|tale)", value, re.I):
            return "story"
        return predicted if predicted in {
            "text", "formal_message", "caption", "story", "advice", "support", "rewrite",
        } else "text"

    @staticmethod
    def _english_output(label: str, text: str, topic: str) -> str:
        value = normalize_text(text)
        if label == "formal_message":
            if "order" in value and "tomorrow" in value:
                return (
                    "Hello,\n\nYour order has been prepared and will be shipped tomorrow. "
                    "We will send the tracking information as soon as the carrier receives the parcel.\n\n"
                    "Thank you for your patience and continued support."
                )
            return (
                f"Hello,\n\nI am writing regarding {topic}. We are reviewing the matter carefully and "
                "will share a clear update as soon as the relevant details are confirmed.\n\n"
                "Thank you for your time and cooperation."
            )
        if label == "caption":
            return (
                f"{topic.capitalize()} started as a small idea and grew through careful work, testing, "
                "and persistence. This is only the beginning, and we are excited to share the next steps. ✨"
            )
        if label == "story":
            return (
                f"At dusk, Maya noticed a small sign of {topic} where she least expected it. She stopped, "
                "looked again, and chose not to walk past. That brief pause changed the rest of her day; "
                "sometimes a quiet decision is enough to begin a new story."
            )
        if label == "advice":
            return (
                "Start by defining the smallest visible result you can produce. Work on it for twenty-five "
                "minutes without notifications, then review what changed. A small testable step usually "
                "creates more clarity than a long plan built around assumptions."
            )
        if label == "support":
            return (
                "I’m sorry this feels heavy right now. You do not have to solve everything at once or explain "
                "it perfectly. We can slow it down together: what is weighing on you most—exhaustion, pressure, "
                "or something specific that happened today?"
            )
        if label == "rewrite":
            match = re.search(
                r"(?:rewrite|rephrase|edit|polish|make\s+(?:this|it)\s+(?:formal|clearer|shorter|professional|friendlier))"
                r"\s*(?:this\s*)?[:：-]?\s*[\"']?(.+?)[\"']?$", text, re.I | re.S
            )
            if not match:
                match = re.search(r"[:：]\s*(.+)$", text, re.I | re.S)
            source = (match.group(1) if match else "").strip(" \"'")
            if not source:
                return "Put the source text in the same message, for example: “Make this more formal: ...”."
            source = re.sub(r"\s+", " ", source).strip()
            source = re.sub(r"\bi want to tell you that\b", "I’d like to let you know that", source, flags=re.I)
            source = re.sub(r"\ba lot of\b", "many", source, flags=re.I)
            source = re.sub(r"\bcan you please\b", "Could you please", source, flags=re.I)
            return "Rewritten version:\n\n" + source[:1].upper() + source[1:].rstrip(".!?") + "."
        return (
            f"{topic.capitalize()} becomes meaningful when it moves from an attractive idea to consistent "
            "action. Lasting progress is usually built through small decisions, honest feedback, and the "
            "patience to continue before the final result is visible."
        )

    @staticmethod
    def _quality_for_language(text: str, language: str, minimum_length: int = 45) -> tuple[bool, tuple[str, ...]]:
        value = text.strip()
        if not minimum_length <= len(value) <= 1800:
            return False, ("invalid_length",)
        if re.search(r"</?(?:tool|assistant|user|system)", value, re.I):
            return False, ("control_token",)
        letters = re.findall(r"[A-Za-z\u0600-\u06ff]", value)
        persian = re.findall(r"[\u0600-\u06ff]", value)
        ratio = len(persian) / max(1, len(letters))
        if language == "fa" and ratio < 0.72:
            return False, ("language_mismatch",)
        if language == "en" and ratio > 0.2:
            return False, ("language_mismatch",)
        sentences = [normalize_text(part) for part in re.split(r"[.!؟\n]+", value) if part.strip()]
        if len(sentences) != len(set(sentences)):
            return False, ("sentence_repetition",)
        return True, ("bounded_length", "no_control_tokens", f"{language}_consistency", "no_sentence_repetition")

    def compose(self, text: str, routed_intent: str, language: str) -> FluentResponse | None:
        if language not in {"fa", "en"} or not self.ready:
            return None
        predicted_joint, predicted_confidence = self.predict(text)
        prefix = f"{language}:"
        predicted = predicted_joint.removeprefix(prefix) if predicted_joint.startswith(prefix) else ""
        forced = self._FORCED_LABELS.get(routed_intent, "")
        if not forced and routed_intent not in {"unknown", "smalltalk"}:
            return None
        if not forced and predicted_confidence < 0.25:
            return None
        if language == "fa":
            label = self._writing_kind(text, forced or predicted)
            topic = self._topic(text)
            if label == "formal_message":
                output = self._formal_message(text, topic)
            elif label == "caption":
                output = self._caption(topic)
            elif label == "story":
                output = self._story(topic)
            elif label == "advice":
                output = self._advice(text)
            elif label == "support":
                output = self._support()
            elif label == "rewrite":
                output = self._rewrite(text)
            else:
                output = self._motivational_text(topic)
        else:
            label = self._english_kind(text, forced or predicted)
            topic = self._english_topic(text)
            output = self._english_output(label, text, topic)
        minimum_length = 20 if label == "rewrite" else 45
        accepted, checks = self._quality_for_language(output, language, minimum_length)
        if not accepted:
            fallback, similarity = self._nearest_response(text, f"{language}:{label}")
            accepted, checks = self._quality_for_language(fallback, language, minimum_length) if fallback else (False, ())
            if not accepted:
                return None
            output = fallback
            predicted_confidence = max(predicted_confidence, similarity)
        confidence = max(0.72 if forced else 0.64, min(0.96, predicted_confidence + 0.18))
        return FluentResponse(output, label, round(confidence, 4), "trained_bilingual_fluency_v11", checks)

    @staticmethod
    def polish_grounded(text: str, query: str, language: str) -> str:
        if language == "fa":
            return PersianFluencyEngine.polish_grounded(text, query, language)
        if language == "en" and re.search(r"(?:simple|plain\s+english|easy\s+to\s+understand)", query, re.I):
            value = text.strip()
            return value if value.lower().startswith("in simple terms") else f"In simple terms, {value[:1].lower() + value[1:]}"
        return text


__all__ = ["BilingualFluencyEngine"]
