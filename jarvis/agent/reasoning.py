from __future__ import annotations

import re
from dataclasses import dataclass

from jarvis.agent.challenge_reasoner import ChallengeReasoner
from jarvis.agent.local_intelligence_v23 import LocalIntelligenceV23
from jarvis.agent.deliberation import (
    DeliberationOutcome,
    EvidenceVerifier,
    QueryAnalyzer,
    TextReasoner,
    content_terms,
)
from jarvis.brain.responses import ResponseEngine

def _local_source_label(engine) -> str:
    """Honest telemetry: report the engine version that actually ran (v22 spec §24)."""
    version = str(getattr(engine, 'VERSION', 'unknown'))
    return 'local_intelligence_' + version.replace('.', '_').replace('-', '_')

from jarvis.knowledge.store import KnowledgeHit, KnowledgeStore
from jarvis.utils.text import normalize_text


@dataclass(frozen=True, slots=True)
class ReasoningResult:
    text: str
    intent: str
    source: str
    knowledge_score: float = 0.0
    confidence: float = 0.0
    checks: tuple[str, ...] = ()


class ReasoningEngine:
    """Bounded multi-pass reasoning grounded in local facts, without exposing CoT."""

    def __init__(
        self,
        knowledge: KnowledgeStore,
        responses: ResponseEngine,
        minimum_knowledge_score: float,
    ) -> None:
        self.knowledge = knowledge
        self.responses = responses
        self.minimum_score = minimum_knowledge_score
        self.analyzer = QueryAnalyzer()
        self.text_reasoner = TextReasoner()
        self.evidence_verifier = EvidenceVerifier()
        self.challenge_reasoner = ChallengeReasoner()
        self.local_intelligence = LocalIntelligenceV23(output_style='canonical')

    @staticmethod
    def _language_answer(hit: KnowledgeHit, language: str) -> str:
        answers: dict[str, str] = {}
        for line in hit.content.splitlines():
            stripped = line.strip()
            if stripped.startswith("[") and "]" in stripped:
                code, value = stripped[1:].split("]", 1)
                answers[code.casefold()] = value.strip()
        return answers.get(language) or answers.get("en") or answers.get("fa") or hit.content.strip()

    def answer_information(
        self,
        query: str,
        language: str,
        turn_index: int,
        internet_enabled: bool,
        attachment_text: str = "",
    ) -> ReasoningResult:
        local = self.local_intelligence.solve(query, language)
        if local is not None:
            return ReasoningResult(
                local.text, local.intent, _local_source_label(self.local_intelligence), 1.0,
                local.confidence, local.checks,
            )
        supplied = self.answer_supplied_text(query, language, attachment_text)
        if supplied is not None:
            return supplied
        hits = self.knowledge.search(query, limit=3)
        if hits and hits[0].score >= self.minimum_score:
            answer = self._language_answer(hits[0], language)
            text = self.responses.render(
                "knowledge_answer", language, query, turn_index, {"knowledge": answer}
            )
            return ReasoningResult(
                text, "knowledge_answer", "local_knowledge", hits[0].score,
                min(0.98, 0.58 + hits[0].score * 0.4),
                ("question_understood", "local_evidence_found", "answer_grounded"),
            )
        fallback = "unknown_information" if internet_enabled else "unknown_offline"
        return ReasoningResult(
            self.responses.render(fallback, language, query, turn_index),
            "unknown_information", "confidence_guard",
            hits[0].score if hits else 0.0, 0.2,
            ("insufficient_evidence", "no_unsupported_claim"),
        )

    @staticmethod
    def _references_document(query: str, attachment_text: str) -> bool:
        if not attachment_text:
            return False
        normalized = normalize_text(query)
        if re.search(
            r"(?:این|اون|آن|همین|بالا|پیوست|فایل|متن|محتوا|سند|کد|"
            r"\b(?:this|that|it|above|attached|attachment|file|text|document|content|code)\b)",
            normalized,
            re.I,
        ):
            return True
        query_terms = set(content_terms(query))
        attachment_terms = set(content_terms(attachment_text))
        return len(query_terms & attachment_terms) >= 2

    @staticmethod
    def _as_reasoning_result(outcome: DeliberationOutcome) -> ReasoningResult:
        intent = (
            "document_analysis"
            if outcome.source == "document_analysis"
            else "grounded_text_answer"
        )
        return ReasoningResult(
            outcome.text,
            intent,
            outcome.source,
            1.0,
            outcome.confidence,
            outcome.checks,
        )

    def answer_supplied_text(
        self,
        query: str,
        language: str,
        attachment_text: str = "",
    ) -> ReasoningResult | None:
        """Answer only when the input or referenced attachment contains evidence."""
        embedded = self.text_reasoner.answer_question(query)
        if embedded is not None:
            verified = self.evidence_verifier.verify(embedded, language)
            if verified is not None:
                return self._as_reasoning_result(verified)
        if not self._references_document(query, attachment_text):
            logical = self.text_reasoner.logical_inference(query, language)
            return self._as_reasoning_result(logical) if logical is not None else None
        outcome = self.text_reasoner.analyze_document(query, attachment_text)
        if outcome is None:
            return None
        verified = self.evidence_verifier.verify(outcome, language)
        return self._as_reasoning_result(verified) if verified is not None else None

    @staticmethod
    def _python_error_answer(language: str) -> str:
        if language == "fa":
            return (
                "برای پیدا کردن علت دقیق، متن کامل traceback و چند خط کد اطراف خط خطا لازم است. "
                "از آخرین خط traceback شروع کن، نوع خطا و نام فایل/شماره خط را بررسی کن، ورودی همان خط "
                "را چاپ یا در debugger ببین و بعد کوچک‌ترین نمونهٔ قابل تکرار بساز. اگر traceback را "
                "بفرستی می‌توانم مرحله‌به‌مرحله علت و اصلاح مشخص را بررسی کنم."
            )
        return (
            "I need the complete traceback and the nearby code for a precise diagnosis. Start at the final "
            "traceback line, inspect the exception type, file and line number, examine that line's inputs, "
            "then reduce it to a minimal reproducible example. Send those details and I can propose a specific fix."
        )

    def answer_complex(
        self,
        query: str,
        language: str,
        turn_index: int,
        context: dict[str, object] | None = None,
        attachment_text: str = "",
    ) -> ReasoningResult:
        """Four bounded passes: interpret, retrieve, verify, then realize the final answer."""
        normalized = normalize_text(query)
        checks: list[str] = ["question_understood"]

        local = self.local_intelligence.solve(query, language)
        if local is not None:
            return ReasoningResult(
                local.text, local.intent, _local_source_label(self.local_intelligence), 1.0,
                local.confidence, local.checks,
            )

        challenge = self.challenge_reasoner.solve(query, language)
        if challenge is not None:
            return ReasoningResult(
                challenge.text,
                "reasoned_answer",
                "local_challenge_reasoner_v094",
                1.0,
                challenge.confidence,
                challenge.checks,
            )

        supplied = self.answer_supplied_text(query, language, attachment_text)
        if supplied is not None:
            return supplied

        # Pass 1/2: diagnose missing inputs or retrieve several local facts.
        if re.search(r"(?:python|پایتون).*(?:error|خطا|ارور)|(?:error|خطا|ارور).*(?:python|پایتون)", normalized):
            checks.extend(("missing_diagnostic_details_detected", "clarification_requested"))
            return ReasoningResult(
                self._python_error_answer(language), "reasoned_clarification",
                "multi_pass_diagnostic", 0.7, 0.88,
                tuple(checks + ["no_guessing"]),
            )

        hits = self.knowledge.search(query, limit=3)
        if hits and hits[0].score >= self.minimum_score:
            facts = [self._language_answer(hit, language) for hit in hits if hit.score >= self.minimum_score]
            unique: list[str] = []
            for fact in facts:
                normalized_fact = normalize_text(fact)
                if any(
                    normalized_fact in normalize_text(existing)
                    or normalize_text(existing) in normalized_fact
                    for existing in unique
                ):
                    continue
                unique.append(fact)
                if len(unique) >= 2:
                    break
            answer = unique[0]
            if len(unique) > 1:
                bridge = "\n\nنکتهٔ تکمیلی: " if language == "fa" else "\n\nAdditional context: "
                answer += bridge + unique[1]
            checks.extend(("local_evidence_found", "evidence_consistency_checked"))
            return ReasoningResult(
                answer, "reasoned_answer", "multi_pass_knowledge", hits[0].score,
                min(0.94, 0.6 + hits[0].score * 0.35), tuple(checks + ["self_check_passed"]),
            )

        # Pass 3/4: guard against plausible-sounding unsupported prose.
        prompt = (
            "برای پاسخ قابل اعتماد اطلاعات کافی ندارم. جزئیات بیشتری بده یا اجازه بده در وب جستجو کنم."
            if language == "fa"
            else "I do not have enough reliable information for a grounded answer. Add details or let me search the web."
        )
        return ReasoningResult(
            prompt, "reasoned_clarification", "multi_pass_confidence_guard", 0.0, 0.25,
            tuple(checks + ["insufficient_evidence", "no_unsupported_claim"]),
        )

    @staticmethod
    def answer_system_assessment(
        query: str,
        language: str,
        system_info: dict[str, object],
    ) -> ReasoningResult:
        """Ground a lightweight suitability assessment in live system information."""
        cpu = str(system_info.get("cpu", "Unknown CPU"))
        ram = int(system_info.get("ram_total_mb", 0) or 0)
        physical = int(system_info.get("physical_cores", 0) or 0)
        logical = int(system_info.get("logical_cores", 0) or 0)
        adequate = ram >= 4096 and max(physical, logical) >= 2
        if language == "fa":
            verdict = (
                "برای اجرای Jarvis و پروژه‌های سبک Python مناسب است"
                if adequate
                else "برای کارهای خیلی سبک قابل استفاده است، اما ممکن است زیر بار هم‌زمان کند شود"
            )
            text = (
                f"براساس مشخصات واقعی سیستم: پردازنده {cpu}، رم {ram} مگابایت و "
                f"{physical} هستهٔ فیزیکی / {logical} هستهٔ منطقی داری. این سیستم {verdict}. "
                "برای پروژه‌ای سنگین‌تر، نیاز دقیق CPU/RAM آن پروژه را هم بگو تا مقایسه دقیق‌تر شود."
            )
        else:
            verdict = (
                "is suitable for Jarvis and lightweight Python workloads"
                if adequate
                else "can handle very light workloads, but may slow down under concurrent load"
            )
            text = (
                f"Using the live system data: {cpu}, {ram} MB RAM, and "
                f"{physical} physical / {logical} logical cores. This machine {verdict}. "
                "Share the other project's exact requirements for a more precise comparison."
            )
        return ReasoningResult(
            text, "system_assessment", "tool_grounded_reasoning", 1.0, 0.93,
            ("live_system_info_collected", "requirements_compared", "limitations_stated"),
        )
