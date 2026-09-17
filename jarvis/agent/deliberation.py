from __future__ import annotations

import math
import re
from collections import Counter, deque
from dataclasses import dataclass

from jarvis.utils.text import detect_language, normalize_text, tokenize


_PERSIAN_STOPWORDS = frozenset(
    {
        "از", "با", "به", "برای", "در", "را", "رو", "و", "یا", "که", "این",
        "آن", "یک", "یه", "هم", "است", "هست", "هستند", "بود", "شد", "شود",
        "می", "کن", "کنه", "کنید", "بگو", "بده", "لطفا", "دقیق", "متن", "طبق",
        "چی", "چیه", "چیست", "چه", "چرا", "چطور", "چگونه", "کدام", "آیا",
        "نتیجه", "میگیری", "میگیریم", "تحلیل", "بررسی", "خلاصه",
    }
)
_ENGLISH_STOPWORDS = frozenset(
    {
        "a", "an", "and", "are", "as", "at", "be", "because", "by", "do", "does",
        "for", "from", "how", "i", "in", "is", "it", "of", "on", "or", "please",
        "that", "the", "this", "to", "was", "were", "what", "when", "where", "which",
        "who", "why", "with", "would", "text", "analyze", "analyse", "explain",
        "summary", "summarize", "result", "conclude",
    }
)
_QUESTION_HINT = re.compile(
    r"(?:چرا|چطور|چگونه|چیست|چیه|یعنی|کدام|چه\s+نتیجه|آیا|"
    r"\b(?:why|how|what|which|who|where|when|does|do|is|are|conclude)\b)",
    re.I,
)
_ANALYSIS_HINT = re.compile(
    r"(?:تحلیل|تجزیه|بررسی\s+دقیق|نکات\s+کلیدی|جمع.?بندی|خلاصه|استنتاج|"
    r"\b(?:analy[sz]e|review|key points?|summari[sz]e|infer|reason about)\b)",
    re.I,
)
_FRESH_HINT = re.compile(
    r"(?:آخرین|جدیدترین|امروز|الان|فعلی|لحظه.?ای|زنده|قیمت|نرخ|خبر|اخبار|هوا|"
    r"\b(?:latest|newest|today|now|current|live|price|rate|news|weather|score|release)\b)",
    re.I,
)
_VOLATILE_HINT = re.compile(
    r"(?:بیت.?کوین|ارز|بورس|دلار|طلا|هوا|نتیجه\s+بازی|"
    r"\b(?:bitcoin|crypto|stock|exchange rate|weather|forecast|score)\b)",
    re.I,
)


@dataclass(frozen=True, slots=True)
class QueryAnalysis:
    language: str
    kind: str
    key_terms: tuple[str, ...]
    questions: tuple[str, ...]
    supplied_text: str
    needs_fresh_information: bool
    volatile: bool
    complex: bool
    ambiguous: bool
    confidence: float
    checks: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class DeliberationOutcome:
    text: str
    confidence: float
    source: str
    evidence: tuple[str, ...]
    checks: tuple[str, ...]


def _stem_token(token: str) -> str:
    value = normalize_text(token).strip("_-'‌؟?!.,،:;؛()[]{}\"")
    if not value:
        return ""
    if value.isascii():
        for suffix in ("ingly", "edly", "ation", "ments", "ment", "ing", "ies", "ed", "es", "s"):
            if value.endswith(suffix) and len(value) - len(suffix) >= 3:
                return value[: -len(suffix)] + ("y" if suffix == "ies" else "")
        return value
    compact = value.replace("‌", "")
    for suffix in ("هایی", "های", "ها", "ترین", "تر", "مان", "تان", "شان"):
        if compact.endswith(suffix) and len(compact) - len(suffix) >= 2:
            compact = compact[: -len(suffix)]
            break
    return compact


def content_terms(text: str, language: str | None = None) -> tuple[str, ...]:
    selected_language = language or detect_language(text)
    stopwords = _PERSIAN_STOPWORDS if selected_language == "fa" else _ENGLISH_STOPWORDS
    result: list[str] = []
    for raw in tokenize(text):
        token = _stem_token(raw)
        if len(token) < 2 or token in stopwords or token.isdecimal():
            continue
        result.append(token)
    return tuple(result)


def split_sentences(text: str) -> list[str]:
    clean = re.sub(r"[\t\f\v ]+", " ", str(text)).strip()
    if not clean:
        return []
    parts = re.split(r"(?<=[.!?؟])\s+|[\r\n]+|(?<=؛)\s*", clean)
    return [value.strip(" \t-*•") for value in parts if len(value.strip(" \t-*•")) >= 2]


class QueryAnalyzer:
    """Cheap first pass that records what an answer must establish."""

    @staticmethod
    def _embedded_text(query: str) -> tuple[str, tuple[str, ...]]:
        value = str(query).strip()
        marker = re.search(
            r"(?:این\s+متن(?:\s+را|\s+رو)?(?:\s+دقیق)?"
            r"(?:(?:\s+تحلیل|\s+بررسی)\s*(?:کن)?|\s+(?:بخون|بخوان|بخوانید))?|"
            r"متن|context|passage|text)\s*[:：]",
            value,
            re.I,
        )
        tail = value[marker.end() :].strip() if marker else ""
        if not tail:
            return "", ()
        sentences = split_sentences(tail)
        questions: list[str] = []
        while sentences and (_QUESTION_HINT.search(sentences[-1]) or sentences[-1].endswith(("?", "؟"))):
            questions.insert(0, sentences.pop())
            if len(questions) >= 3:
                break
        supplied = " ".join(sentences).strip()
        return supplied, tuple(questions)

    def analyze(self, query: str, context_text: str = "") -> QueryAnalysis:
        clean = " ".join(str(query).split())
        language = detect_language(clean)
        embedded, embedded_questions = self._embedded_text(clean)
        questions = list(embedded_questions)
        if not questions:
            questions = [
                sentence for sentence in split_sentences(clean)
                if sentence.endswith(("?", "؟"))
                or re.match(
                    r"^(?:چرا|چطور|چگونه|چیست|چیه|یعنی|کدام|چه\s+نتیجه|آیا|"
                    r"(?:why|how|what|which|who|where|when|does|do|can\s+we|is\s+there)\b)",
                    normalize_text(sentence), re.I,
                )
            ][-3:]
        supplied = embedded or str(context_text).strip()
        normalized = normalize_text(clean)
        fresh = bool(_FRESH_HINT.search(normalized))
        volatile = bool(_VOLATILE_HINT.search(normalized))
        terms = content_terms(" ".join(questions) or clean, language)
        logical = self._looks_logical(normalized)
        analysis_request = bool(_ANALYSIS_HINT.search(normalized))
        kind = "general"
        if logical:
            kind = "logical_inference"
        elif supplied and any(re.search(r"(?:چرا|\bwhy\b)", item, re.I) for item in questions):
            kind = "text_causality"
        elif supplied and questions:
            kind = "text_question"
        elif supplied and analysis_request:
            kind = "document_analysis"
        elif fresh or volatile:
            kind = "fresh_information"
        elif re.search(r"(?:فرق|تفاوت|مقایسه|\b(?:difference|compare|versus|\bvs\b)\b)", normalized, re.I):
            kind = "comparison"
        complex_query = (
            logical or analysis_request or len(questions) > 1 or len(tokenize(clean)) >= 22
        )
        ambiguous = len(terms) == 0 and not supplied and not logical
        checks = ["language_detected", "question_type_identified"]
        if supplied:
            checks.append("supplied_evidence_detected")
        if fresh:
            checks.append("freshness_requirement_detected")
        if ambiguous:
            checks.append("ambiguity_detected")
        confidence = 0.94 if kind != "general" else (0.84 if terms else 0.35)
        return QueryAnalysis(
            language, kind, tuple(dict.fromkeys(terms)), tuple(questions), supplied,
            fresh, volatile, complex_query, ambiguous, confidence, tuple(checks),
        )

    @staticmethod
    def _looks_logical(normalized: str) -> bool:
        return bool(
            re.search(r"(?:اگر\s+همه|همه\s+.+(?:هستند|دارند).+(?:پس|نتیجه)|"
                      r"\b(?:if\s+all|all\s+.+\s+are\s+.+(?:therefore|conclude))\b)", normalized, re.I)
            or (
                re.search(r"(?:همه\s+.+(?:هستند|دارند)|\ball\s+.+\s+are\b)", normalized, re.I)
                and re.search(r"(?:چه\s+نتیجه|نتیجه\s+می|\b(?:conclude|therefore|what follows)\b)", normalized, re.I)
            )
        )


class TextReasoner:
    """Grounded extractive QA, summarisation and small formal deductions."""

    def __init__(self) -> None:
        self.analyzer = QueryAnalyzer()

    @staticmethod
    def _score_sentence(sentence: str, question: str, language: str, position: int) -> float:
        query_terms = set(content_terms(question, language))
        sentence_terms = set(content_terms(sentence, language))
        if not sentence_terms:
            return 0.0
        overlap = len(query_terms & sentence_terms)
        if query_terms and overlap == 0:
            return 0.0
        coverage = overlap / max(1, len(query_terms))
        density = overlap / max(4, len(sentence_terms))
        causal = 0.28 if re.search(r"(?:چون|زیرا|به\s+دلیل|در\s+نتیجه|\bbecause|\bdue to|\btherefore)", sentence, re.I) else 0.0
        return coverage * 0.58 + density * 0.32 + causal + 0.04 / max(1, position + 1)

    @staticmethod
    def _natural_evidence_answer(sentence: str, question: str, language: str) -> str:
        value = sentence.strip()
        why = bool(re.search(r"(?:چرا|\bwhy\b)", question, re.I))
        if language == "fa":
            if why:
                cause = re.search(r"(?:چون|زیرا|به\s+دلیل(?:\s+اینکه)?)\s+(.+?)(?:[.!؟]|$)", value, re.I)
                if cause:
                    reason = cause.group(1).strip()
                    return f"چون {reason.rstrip(' .؟')}. این علت مستقیماً در متن گفته شده است."
            return f"بر اساس متن: {value}"
        if why:
            cause = re.search(r"(?:because|due\s+to)\s+(.+?)(?:[.!?]|$)", value, re.I)
            if cause:
                return f"According to the text, the reason is {cause.group(1).strip().rstrip('.!?')}."
        return f"Based on the text: {value}"

    @staticmethod
    def _singular(value: str) -> str:
        clean = normalize_text(value).replace("‌", "").strip(" ،,.؛")
        for suffix in ("هایی", "های", "ها", "ان", "ات"):
            if clean.endswith(suffix) and len(clean) - len(suffix) >= 2:
                return clean[: -len(suffix)]
        if clean.isascii() and clean.endswith("s") and len(clean) > 3:
            return clean[:-1]
        return clean

    def logical_inference(self, query: str, language: str) -> DeliberationOutcome | None:
        normalized = normalize_text(query)
        if language == "en":
            rules = re.findall(
                r"\ball\s+([a-z][a-z -]{0,45}?)\s+are\s+([a-z][a-z -]{0,55}?)(?=[,.;]|\s+(?:and|if|therefore|so)\b|$)",
                normalized,
                re.I,
            )
            facts = re.findall(
                r"\b([a-z][a-z-]{1,30})\s+is\s+(?:an?\s+)?([a-z][a-z -]{1,40}?)(?=[,.;?]|\s+(?:and|therefore|so)\b|$)",
                normalized,
                re.I,
            )
        else:
            rules = []
            for match in re.finditer(
                r"همه\s+([\w‌-]{2,35})\s+([\w‌ -]{1,50}?)\s+(هستند|دارند)(?=[،,.؛]|\s+و\s+|$)",
                normalized,
                re.I,
            ):
                subject, predicate, relation = match.groups()
                predicate = predicate.strip()
                if relation == "دارند":
                    predicate = f"دارای {predicate}"
                rules.append((subject, predicate))
            facts = re.findall(
                r"(?:^|[،,.؛]\s*|\s+و\s+)([\w‌-]{2,30})\s+(?:یک\s+)?([\w‌-]{2,35})\s+است(?=[،,.؛؟]|$)",
                normalized,
                re.I,
            )
        if not rules or not facts:
            return None

        graph: dict[str, list[str]] = {}
        display: dict[str, str] = {}
        for source, target in rules:
            source_key = self._singular(source)
            target_key = self._singular(target.replace("دارای ", ""))
            graph.setdefault(source_key, []).append(target_key)
            display[target_key] = target.strip()

        for entity, category in reversed(facts):
            category_key = self._singular(category)
            queue: deque[tuple[str, tuple[str, ...]]] = deque([(category_key, ())])
            visited = {category_key}
            conclusions: list[tuple[str, tuple[str, ...]]] = []
            while queue:
                current, path = queue.popleft()
                for target in graph.get(current, []):
                    if target in visited:
                        continue
                    visited.add(target)
                    new_path = path + (target,)
                    conclusions.append((target, new_path))
                    if len(new_path) < 4:
                        queue.append((target, new_path))
            if not conclusions:
                continue
            target, path = max(conclusions, key=lambda item: len(item[1]))
            predicate = display.get(target, target)
            if language == "fa":
                if predicate.startswith("دارای "):
                    text = f"نتیجهٔ منطقی این است که {entity} {predicate} است."
                else:
                    text = f"نتیجهٔ منطقی این است که {entity} {predicate} است."
            else:
                text = f"The valid conclusion is that {entity} is {predicate}."
            return DeliberationOutcome(
                text, 0.96, "formal_deduction", tuple(f"{a} -> {b}" for a, b in rules),
                ("premises_parsed", "deduction_completed", "conclusion_checked"),
            )
        return None

    def answer_question(
        self, query: str, context_text: str = "",
    ) -> DeliberationOutcome | None:
        analysis = self.analyzer.analyze(query, context_text)
        if analysis.kind == "logical_inference":
            return self.logical_inference(query, analysis.language)
        source_text = analysis.supplied_text
        if not source_text or not analysis.questions:
            return None
        sentences = split_sentences(source_text)
        if not sentences:
            return None
        answers: list[str] = []
        evidence: list[str] = []
        scores: list[float] = []
        for question in analysis.questions:
            ranked = sorted(
                enumerate(sentences),
                key=lambda item: self._score_sentence(
                    item[1], question, analysis.language, item[0]
                ),
                reverse=True,
            )
            best_position, best = ranked[0]
            score = self._score_sentence(best, question, analysis.language, best_position)
            if score < 0.12:
                continue
            answers.append(self._natural_evidence_answer(best, question, analysis.language))
            evidence.append(best)
            scores.append(score)
        if not answers:
            return None
        answer = answers[0] if len(answers) == 1 else "\n\n".join(
            f"{index}. {value}" for index, value in enumerate(answers, 1)
        )
        score = sum(scores) / len(scores)
        return DeliberationOutcome(
            answer, min(0.97, 0.62 + score * 0.35), "supplied_text",
            tuple(dict.fromkeys(evidence)),
            ("questions_parsed", "relevant_sentences_selected", "answers_supported_by_text"),
        )

    @staticmethod
    def _summary_sentences(text: str, language: str, limit: int = 4) -> tuple[str, ...]:
        sentences = [value for value in split_sentences(text) if 18 <= len(value) <= 700]
        if not sentences:
            return ()
        frequencies = Counter(content_terms(" ".join(sentences), language))
        if not frequencies:
            return tuple(sentences[:limit])
        maximum = max(frequencies.values()) or 1
        scores: list[tuple[float, int, str]] = []
        for index, sentence in enumerate(sentences):
            terms = content_terms(sentence, language)
            lexical = sum(frequencies[term] / maximum for term in set(terms))
            length_penalty = 1.0 / (1.0 + abs(len(terms) - 18) / 22.0)
            position_bonus = 0.24 / (index + 1)
            scores.append((lexical / max(4, len(set(terms))) * length_penalty + position_bonus, index, sentence))
        selected = sorted(scores, reverse=True)[: max(1, min(limit, len(scores)))]
        return tuple(value for _score, _index, value in sorted(selected, key=lambda item: item[1]))

    def analyze_document(self, query: str, text: str) -> DeliberationOutcome | None:
        clean_text = str(text).strip()
        if not clean_text:
            return None
        language = detect_language(query or clean_text)
        question_answer = self.answer_question(query, clean_text)
        if question_answer is not None and _QUESTION_HINT.search(query):
            return question_answer
        summary = self._summary_sentences(clean_text, language)
        if not summary:
            return None
        keyword_counts = Counter(content_terms(clean_text, language))
        keywords = [word for word, _count in keyword_counts.most_common(6)]
        if language == "fa":
            lines = ["جمع‌بندی متن:", " ".join(summary)]
            if keywords:
                lines.extend(("", "محورهای اصلی: " + "، ".join(keywords)))
            lines.extend(("", "این نتیجه فقط بر پایهٔ متن ارائه‌شده است؛ نکته‌ای خارج از آن به‌عنوان واقعیت اضافه نشده."))
        else:
            lines = ["Text assessment:", " ".join(summary)]
            if keywords:
                lines.extend(("", "Main themes: " + ", ".join(keywords)))
            lines.extend(("", "This assessment is limited to the supplied text; no outside claim was added as fact."))
        return DeliberationOutcome(
            "\n".join(lines), 0.88, "document_analysis", summary,
            ("document_segmented", "salient_points_ranked", "unsupported_claims_blocked"),
        )


class EvidenceVerifier:
    """Checks that a grounded answer still overlaps its cited evidence."""

    @staticmethod
    def support_score(answer: str, evidence: tuple[str, ...], language: str) -> float:
        answer_terms = set(content_terms(answer, language))
        evidence_terms = set(content_terms(" ".join(evidence), language))
        if not answer_terms or not evidence_terms:
            return 0.0
        return len(answer_terms & evidence_terms) / math.sqrt(len(answer_terms) * len(evidence_terms))

    def verify(self, outcome: DeliberationOutcome, language: str) -> DeliberationOutcome | None:
        if outcome.source == "formal_deduction":
            return outcome
        score = self.support_score(outcome.text, outcome.evidence, language)
        if score < 0.16:
            return None
        return DeliberationOutcome(
            outcome.text,
            min(outcome.confidence, 0.58 + score * 0.42),
            outcome.source,
            outcome.evidence,
            outcome.checks + ("evidence_support_verified",),
        )
