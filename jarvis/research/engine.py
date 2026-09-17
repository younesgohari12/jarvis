from __future__ import annotations

import html
import json
import re
import time
import urllib.parse
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

from jarvis.agent.deliberation import QueryAnalyzer, content_terms
from jarvis.search.engine import EvidencePassage, SearchEngine, SearchEvidence, SearchReport
from jarvis.utils.text import detect_language, normalize_text


@dataclass(frozen=True, slots=True)
class ResearchReport:
    query: str
    rewrites: tuple[str, ...]
    summary: str
    sources: tuple[SearchEvidence, ...]
    latency_ms: float
    attempts: int
    confidence: float = 0.0
    status: str = "insufficient"
    providers: tuple[str, ...] = ()
    errors: tuple[str, ...] = ()
    subject: str = ""


class ResearchEngine:
    """Bounded multi-query research with subject locking and cited synthesis."""

    _STATUS_RANK = {
        "verified": 5,
        "supported": 4,
        "limited": 3,
        "insufficient": 2,
        "no_results": 1,
        "unavailable": 0,
    }
    _PROMOTIONAL = re.compile(
        r"(?:عکس\s+کودکی|بیوگرافی.*(?:عکس|همسر|ثروت)|جدول\s+کامل|"
        r"برای\s+خرید|فروشگاه|دانلود\s+کنید|کلیک\s+کنید|"
        r"photo gallery|buy now|download|click here)",
        re.I,
    )
    _NARROW_QUESTION = re.compile(
        r"(?:[؟?]|چقدر|چند|کی|کجا|چه\s+تعداد|تا\s+الان|"
        r"\b(?:how many|how much|when|where|current|latest|now)\b)",
        re.I,
    )

    def __init__(
        self, search: SearchEngine, log_path: Path,
        persistence_enabled: Callable[[], bool] | None = None,
    ) -> None:
        self.search = search
        self.log_path = log_path
        self.analyzer = QueryAnalyzer()
        self.persistence_enabled = persistence_enabled or (lambda: True)

    def rewrite_queries(self, query: str) -> tuple[str, ...]:
        """Create generic, explainable rewrites without topic-specific tables."""
        clean = " ".join(str(query).split())[:260]
        if not clean:
            return ()
        analysis = self.analyzer.analyze(clean)
        language = analysis.language
        terms = list(dict.fromkeys(content_terms(clean, language)))[:12]
        compact = " ".join(terms) or clean
        values = [clean]
        if analysis.needs_fresh_information or analysis.volatile:
            values.append(f"{compact} {datetime.now().year}")
        normalized = normalize_text(clean)
        if re.search(r"(?:پزشک|سلامت|بیماری|درمان|دارو|\b(?:medical|health|disease|treatment)\b)", normalized, re.I):
            suffix = "راهنمای پزشکی معتبر یا مرور نظام‌مند" if language == "fa" else "authoritative medical guidance systematic review"
        elif re.search(r"(?:API|پایتون|کد|شبکه|پروتکل|نرم.?افزار|\b(?:python|software|network|protocol|programming|documentation)\b)", clean, re.I):
            suffix = "مستندات رسمی مشخصات فنی" if language == "fa" else "official documentation technical specification"
        elif re.search(r"(?:آمار|اقتصاد|جمعیت|قانون|\b(?:statistics|economy|population|law)\b)", normalized, re.I):
            suffix = "داده یا گزارش رسمی" if language == "fa" else "official data report"
        else:
            suffix = "منبع رسمی دانشگاهی" if language == "fa" else "official academic source"
        values.append(f"{compact} {suffix}")
        return tuple(dict.fromkeys(value.strip() for value in values if value.strip()))[:3]

    @staticmethod
    def _source_quality(evidence: SearchEvidence) -> float:
        host = (urllib.parse.urlparse(evidence.url).hostname or "").casefold()
        diversity_bonus = 0.04 if host else 0.0
        return (
            evidence.score * 0.46
            + evidence.source_quality * 0.42
            + evidence.corroboration * 0.12
            + diversity_bonus
        )

    @classmethod
    def _best_report(cls, reports: list[SearchReport]) -> SearchReport | None:
        if not reports:
            return None
        return max(
            reports,
            key=lambda report: (
                cls._STATUS_RANK.get(report.status, -1),
                report.confidence,
                len(report.passages),
                len(report.evidence),
            ),
        )

    @classmethod
    def _combine_reports(cls, reports: list[SearchReport]) -> SearchReport | None:
        """Merge evidence across rewrites instead of discarding all but one query."""
        best = cls._best_report(reports)
        if best is None:
            return None
        evidence_by_key: dict[str, SearchEvidence] = {}
        passages_by_signature: dict[str, EvidencePassage] = {}
        for report in reports:
            for source in report.evidence:
                key = cls._url_key(source.url)
                existing = evidence_by_key.get(key)
                if existing is None or cls._source_quality(source) > cls._source_quality(existing):
                    evidence_by_key[key] = source
            for passage in report.passages:
                signature = f"{cls._url_key(passage.url)}|{normalize_text(passage.text)[:220]}"
                existing_passage = passages_by_signature.get(signature)
                if existing_passage is None or passage.score > existing_passage.score:
                    passages_by_signature[signature] = passage
        evidence = sorted(evidence_by_key.values(), key=cls._source_quality, reverse=True)[:12]
        source_indexes = {cls._url_key(source.url): index for index, source in enumerate(evidence, 1)}
        passages = [
            EvidencePassage(
                passage.text, passage.url, passage.title, passage.score,
                source_indexes[cls._url_key(passage.url)], passage.corroboration,
            )
            for passage in sorted(
                passages_by_signature.values(),
                key=lambda item: (item.score, item.corroboration),
                reverse=True,
            )
            if cls._url_key(passage.url) in source_indexes
        ][:8]
        # Mirrored text on different hosts is not independent corroboration.
        signatures=set(); domains=set()
        for item in passages:
            signature=normalize_text(item.text).casefold()
            if signature in signatures: continue
            signatures.add(signature)
            domain=cls._domain(item.url)
            if domain: domains.add(domain)
        confidence = min(0.97, best.confidence + min(0.08, max(0, len(domains) - 1) * 0.025))
        status = best.status
        if len(domains) >= 2 and status in {"limited", "supported"} and confidence >= 0.58:
            status = "supported"
        fallback_summary = next(
            (report.summary for report in reports if report.summary.strip()),
            best.summary,
        )
        return SearchReport(
            best.query, fallback_summary, tuple(evidence), sum(report.fetched_pages for report in reports),
            confidence, status, tuple(passages),
            tuple(dict.fromkeys(provider for report in reports for provider in report.providers)),
            tuple(dict.fromkeys(error for report in reports for error in report.errors))[:8],
            any(report.fresh for report in reports),
        )

    @staticmethod
    def _url_key(url: str) -> str:
        parsed = urllib.parse.urlsplit(str(url).strip())
        host = (parsed.hostname or "").casefold().removeprefix("www.")
        path = urllib.parse.unquote(parsed.path).rstrip("/") or "/"
        kept_query = [
            (key, value)
            for key, value in urllib.parse.parse_qsl(parsed.query, keep_blank_values=True)
            if not key.casefold().startswith(("utm_", "ref", "source"))
        ]
        query = urllib.parse.urlencode(sorted(kept_query))
        return f"{host}{normalize_text(path)}" + (f"?{query}" if query else "")

    @staticmethod
    def _domain(url: str) -> str:
        return (urllib.parse.urlsplit(url).hostname or "").casefold().removeprefix("www.")

    @classmethod
    def _clean_finding(cls, text: str) -> str:
        value = html.unescape(" ".join(str(text).split()))
        value = re.sub(r"\s*\[\d+]\s*$", "", value).strip(" -*•")
        if cls._PROMOTIONAL.search(value):
            return ""
        while value.count("(") > value.count(")") and "(" in value:
            value = value.rsplit("(", 1)[0].rstrip()
        while value.count("[") > value.count("]") and "[" in value:
            value = value.rsplit("[", 1)[0].rstrip()
        if not 30 <= len(value) <= 620:
            return ""
        if value.endswith((":", "؛", "،", ";")):
            return ""
        if not value.endswith((".", "!", "?", "؟")):
            value += "."
        return value

    @staticmethod
    def _near_duplicate(text: str, selected: list[str]) -> bool:
        terms = set(content_terms(text))
        for existing in selected:
            other = set(content_terms(existing))
            union = terms | other
            if normalize_text(text)[:180] == normalize_text(existing)[:180]:
                return True
            if union and len(terms & other) / len(union) >= 0.78:
                return True
        return False

    @staticmethod
    def _supports_subject(
        finding: str,
        passage: EvidencePassage,
        subject_terms: set[str],
    ) -> bool:
        if not subject_terms:
            return True
        available = set(content_terms(f"{passage.title} {finding}"))
        return bool(subject_terms & available)

    @classmethod
    def _synthesize(
        cls,
        query: str,
        subject: str,
        report: SearchReport,
    ) -> tuple[str, tuple[SearchEvidence, ...], float, str]:
        language = detect_language(query)
        subject_terms = set(content_terms(subject, language))
        evidence_by_key = {cls._url_key(item.url): item for item in report.evidence}
        findings: list[tuple[str, EvidencePassage, SearchEvidence]] = []
        selected_text: list[str] = []
        domain_counts: dict[str, int] = {}
        for passage in report.passages:
            clean = cls._clean_finding(passage.text)
            if not clean or not cls._supports_subject(clean, passage, subject_terms):
                continue
            if cls._near_duplicate(clean, selected_text):
                continue
            source = evidence_by_key.get(cls._url_key(passage.url))
            if source is None:
                continue
            domain = cls._domain(source.url)
            if domain_counts.get(domain, 0) >= 2:
                continue
            selected_text.append(clean)
            findings.append((clean, passage, source))
            domain_counts[domain] = domain_counts.get(domain, 0) + 1
            if len(findings) >= 4:
                break

        if not findings:
            # Compatibility for custom SearchEngine implementations that only
            # return an already-cited summary.
            sources: list[SearchEvidence] = []
            seen: set[str] = set()
            old_to_new: dict[int, int] = {}
            for old_index, source in enumerate(report.evidence, 1):
                key = cls._url_key(source.url)
                if key in seen:
                    existing = next(
                        index for index, item in enumerate(sources, 1)
                        if cls._url_key(item.url) == key
                    )
                    old_to_new[old_index] = existing
                    continue
                if len(sources) >= 5:
                    continue
                seen.add(key)
                sources.append(source)
                old_to_new[old_index] = len(sources)

            def remap(match: re.Match[str]) -> str:
                new_index = old_to_new.get(int(match.group(1)))
                return f"[{new_index}]" if new_index else ""

            summary = re.sub(r"\[(\d+)]", remap, report.summary).strip()
            return summary, tuple(sources), report.confidence, report.status

        sources: list[SearchEvidence] = []
        source_indexes: dict[str, int] = {}
        for _finding, _passage, source in findings:
            key = cls._url_key(source.url)
            if key not in source_indexes:
                source_indexes[key] = len(sources) + 1
                sources.append(source)

        cited = [
            f"{finding} [{source_indexes[cls._url_key(source.url)]}]"
            for finding, _passage, source in findings
        ]
        narrow = bool(cls._NARROW_QUESTION.search(query))
        if language == "fa":
            if narrow:
                lines = ["پاسخ بر پایهٔ منابع بررسی‌شده:", cited[0]]
                if len(cited) > 1:
                    lines.extend(("", "شواهد تکمیلی:", *(f"• {item}" for item in cited[1:])))
            else:
                lines = ["جمع‌بندی تحقیق:", cited[0]]
                if len(cited) > 1:
                    lines.extend(("", "نکات مهم:", *(f"• {item}" for item in cited[1:])))
        else:
            heading = "Evidence-based answer:" if narrow else "Research summary:"
            lines = [heading, cited[0]]
            if len(cited) > 1:
                lines.extend(("", "Additional findings:", *(f"• {item}" for item in cited[1:])))

        domains = {cls._domain(source.url) for source in sources if cls._domain(source.url)}
        confidence = report.confidence
        status = report.status
        if len(domains) <= 1:
            confidence = min(confidence, 0.72)
            if status == "verified":
                status = "supported"
        elif len(domains) == 2:
            confidence = min(confidence, 0.9)
        return "\n".join(lines), tuple(sources), confidence, status

    def _search_once(self, query: str, subject: str) -> SearchReport:
        if subject:
            try:
                return self.search.search(query, subject=subject)
            except TypeError as exc:
                if "subject" not in str(exc):
                    raise
        return self.search.search(query)

    def research(self, query: str, subject: str = "") -> ResearchReport:
        clean_subject = " ".join(str(subject).split())[:120]
        rewrites = self.rewrite_queries(query)
        started = time.perf_counter()
        reports: list[SearchReport] = []
        executed: list[str] = []
        errors: list[str] = []
        for rewrite_index, rewritten in enumerate(rewrites):
            executed.append(rewritten)
            try:
                report = self._search_once(rewritten, clean_subject)
            except Exception as exc:
                errors.append(f"search: {type(exc).__name__}: {exc}")
                continue
            reports.append(report)
            errors.extend(report.errors)
            if rewrite_index >= 1 and report.status == "verified" and report.confidence >= 0.72:
                break

        best = self._combine_reports(reports)
        if best is None:
            summary = ""
            sources: tuple[SearchEvidence, ...] = ()
            confidence = 0.0
            status = "unavailable" if errors else "no_results"
        else:
            summary, sources, confidence, status = self._synthesize(
                str(query), clean_subject, best
            )
            if not summary and best.status not in {"unavailable", "no_results"}:
                status = "insufficient"

        providers = tuple(
            dict.fromkeys(provider for report in reports for provider in report.providers)
        )
        elapsed = (time.perf_counter() - started) * 1000.0
        result = ResearchReport(
            query=" ".join(str(query).split()),
            rewrites=tuple(executed),
            summary=summary,
            sources=sources,
            latency_ms=round(elapsed, 3),
            attempts=len(executed),
            confidence=round(confidence, 4),
            status=status,
            providers=providers,
            errors=tuple(dict.fromkeys(errors))[:8],
            subject=clean_subject,
        )
        if not self.persistence_enabled():
            return result
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        event = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "query": result.query,
            "subject": result.subject,
            "rewrites": list(result.rewrites),
            "status": result.status,
            "confidence": result.confidence,
            "results": len(result.sources),
            "selected_sources": [source.url for source in result.sources],
            "providers": list(result.providers),
            "latency_ms": result.latency_ms,
            "errors": list(result.errors),
        }
        try:
            with self.log_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(event, ensure_ascii=False) + "\n")
        except OSError:
            pass
        return result
