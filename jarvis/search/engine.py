from __future__ import annotations

import html
import re
import threading
import time
import urllib.parse
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, replace
from html.parser import HTMLParser

from jarvis.agent.deliberation import QueryAnalyzer, content_terms, split_sentences
from jarvis.tools.internet import InternetTool, SearchDiagnostics, SearchResult
from jarvis.utils.text import detect_language, normalize_text


class _ReadableHTML(HTMLParser):
    SKIP = {
        "script", "style", "noscript", "svg", "canvas", "nav", "footer",
        "header", "form", "button", "iframe", "template",
    }
    BLOCKS = {
        "article", "main", "section", "p", "div", "li", "h1", "h2", "h3",
        "h4", "blockquote", "pre", "td",
    }

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self._skip_depth = 0
        self.parts: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        lowered = tag.casefold()
        if lowered in self.SKIP:
            self._skip_depth += 1
        elif not self._skip_depth and lowered in self.BLOCKS:
            self.parts.append("\n")

    def handle_endtag(self, tag: str) -> None:
        lowered = tag.casefold()
        if lowered in self.SKIP and self._skip_depth:
            self._skip_depth -= 1
        elif not self._skip_depth and lowered in self.BLOCKS:
            self.parts.append("\n")

    def handle_data(self, data: str) -> None:
        if not self._skip_depth:
            clean = " ".join(data.split())
            if len(clean) >= 2:
                self.parts.append(clean)


@dataclass(frozen=True, slots=True)
class SearchEvidence:
    title: str
    url: str
    snippet: str
    score: float
    provider: str = ""
    domain: str = ""
    source_quality: float = 0.0
    corroboration: float = 0.0
    fetched: bool = False


@dataclass(frozen=True, slots=True)
class EvidencePassage:
    text: str
    url: str
    title: str
    score: float
    source_index: int
    corroboration: float = 0.0


@dataclass(frozen=True, slots=True)
class SearchReport:
    query: str
    summary: str
    evidence: tuple[SearchEvidence, ...]
    fetched_pages: int
    confidence: float = 0.0
    status: str = "insufficient"
    passages: tuple[EvidencePassage, ...] = ()
    providers: tuple[str, ...] = ()
    errors: tuple[str, ...] = ()
    fresh: bool = False


@dataclass(frozen=True, slots=True)
class _Candidate:
    text: str
    result_index: int
    score: float
    url: str
    title: str
    domain: str
    source_quality: float


class SearchEngine:
    """Multi-provider retrieval, page reading, claim ranking and source checks."""

    _BOILERPLATE = re.compile(
        r"(?:accept all cookies|cookie policy|privacy policy|sign in|log in|subscribe|"
        r"enable javascript|all rights reserved|skip to content|تبلیغات|حریم خصوصی|"
        r"ورود به حساب|عضویت|کوکی)",
        re.I,
    )
    _OFFICIAL_HOST_HINTS = (
        "docs.", "developer.", "support.", "learn.microsoft.com", "python.org",
        "who.int", "nasa.gov", "nih.gov", "medlineplus.gov", "nhs.uk",
        "mayoclinic.org", "clevelandclinic.org", "msdmanuals.com", "europa.eu",
        "w3.org", "ietf.org", "rfc-editor.org", "iso.org", "oecd.org",
        "worldbank.org", "imf.org", "un.org", "ecdc.europa.eu",
    )
    _SOCIAL_HOSTS = (
        "x.com", "twitter.com", "instagram.com", "facebook.com", "tiktok.com",
        "pinterest.com", "threads.net", "linkedin.com",
    )
    _COMMUNITY_HOSTS = (
        "reddit.com", "quora.com", "stackexchange.com", "stackoverflow.com",
    )
    _SOCIAL_QUERY = re.compile(
        r"(?:توییت|پست\s+(?:اینستاگرام|توییتر|شبکه)|شبکه\s+اجتماعی|"
        r"\b(?:tweet|social\s+media|instagram\s+post|reddit\s+thread)\b)",
        re.I,
    )
    _LOW_VALUE_RESULT = re.compile(
        r"(?:sign\s+in|log\s+in|صفحه\s+اصلی|home\s+page|دانلود\s+فایل\s+راهنما|"
        r"اشتراک\s+الکترونیک|نتایج\s+جستجو|search\s+results|همه\s+حقوق\s+محفوظ)",
        re.I,
    )
    _MEDICAL_QUERY = re.compile(
        r"(?:پزشک|پزشکی|سلامت|بیماری|درمان|دارو|علائم|درد|خون.?ریزی|ترشح|"
        r"بیضه|آلت\s+مردانه|کیر|واژن|مهبل|رحم|فرج|"
        r"\b(?:medical|health|disease|treatment|medicine|symptom|pain|"
        r"testicle|penis|vagina|uterus|vulva)\b)",
        re.I,
    )
    _LOW_QUALITY_MEDICAL = re.compile(
        r"(?:تشخیص\s+اندازه|اندازه.{0,25}(?:از\s+چهره|با\s+عکس)|"
        r"مشخصات\s+(?:یک\s+)?(?:واژن|آلت|کیر).{0,18}(?:خوب|ایده.?آل)|"
        r"(?:عکس|گالری).{0,30}(?:واژن|آلت|کیر)|"
        r"(?:بهترین|تضمینی|معجزه.?آسا).{0,30}(?:درمان|دارو)|"
        r"بلاگ\s+بیماری.{0,30}(?:نشیمنگاه|مقعد))",
        re.I,
    )

    def __init__(self, internet: InternetTool, fetch_top_pages: int = 3) -> None:
        self.internet = internet
        self.fetch_top_pages = max(0, min(4, int(fetch_top_pages)))
        self.analyzer = QueryAnalyzer()
        self._cache: OrderedDict[str, tuple[float, SearchReport]] = OrderedDict()
        self._cache_lock = threading.RLock()

    @staticmethod
    def _domain(url: str) -> str:
        return (urllib.parse.urlparse(url).hostname or "").casefold().removeprefix("www.")

    @staticmethod
    def _canonical_url(url: str) -> str:
        parsed = urllib.parse.urlsplit(str(url).strip())
        host = (parsed.hostname or "").casefold().removeprefix("www.")
        path = urllib.parse.unquote(parsed.path or "/").rstrip("/") or "/"
        query = urllib.parse.urlencode(sorted(urllib.parse.parse_qsl(parsed.query)))
        return urllib.parse.urlunsplit((parsed.scheme.casefold() or "https", host, path, query, ""))

    @classmethod
    def _deduplicate_results(cls, results: tuple[SearchResult, ...]) -> tuple[SearchResult, ...]:
        unique: dict[str, SearchResult] = {}
        for result in results:
            key = cls._canonical_url(result.url)
            existing = unique.get(key)
            if existing is None or (
                not existing.content and bool(result.content)
            ):
                unique[key] = result
        return tuple(unique.values())

    @classmethod
    def _unsafe_medical_result(cls, result: SearchResult, medical_query: bool) -> bool:
        if not medical_query:
            return False
        return bool(cls._LOW_QUALITY_MEDICAL.search(f"{result.title} {result.snippet}"))

    @classmethod
    def _unsafe_general_result(cls, result: SearchResult, social_query: bool) -> bool:
        host = cls._domain(result.url)
        if any(host == value or host.endswith("." + value) for value in cls._SOCIAL_HOSTS):
            return not social_query
        return bool(cls._LOW_VALUE_RESULT.search(f"{result.title} {result.snippet}"))

    @classmethod
    def _source_quality(cls, result: SearchResult, query_terms: set[str]) -> float:
        host = cls._domain(result.url)
        quality = 0.46
        if host.endswith((".gov", ".gov.uk", ".edu", ".ac.uk")):
            quality = 0.94
        elif host.endswith(("wikipedia.org", "wikimedia.org")):
            quality = 0.76
        elif any(hint in host for hint in cls._OFFICIAL_HOST_HINTS):
            quality = 0.88
        host_terms = set(re.split(r"[.\-_]", host))
        if query_terms & host_terms:
            quality += 0.12
        if any(host == value or host.endswith("." + value) for value in cls._SOCIAL_HOSTS):
            quality = min(quality, 0.16)
        elif any(host == value or host.endswith("." + value) for value in cls._COMMUNITY_HOSTS):
            quality = min(quality, 0.55)
        return max(0.1, min(1.0, quality))

    @classmethod
    def _result_score(
        cls,
        query_terms: set[str],
        result: SearchResult,
        rank: int,
        subject_terms: set[str] | None = None,
    ) -> float:
        title_terms = set(content_terms(result.title))
        text_terms = title_terms | set(content_terms(result.snippet))
        required_subject = subject_terms or set()
        subject_overlap = len(required_subject & text_terms)
        if required_subject and subject_overlap == 0:
            return 0.0
        overlap_count = len(query_terms & text_terms)
        if len(query_terms) >= 4 and overlap_count < 2:
            return 0.0
        overlap = overlap_count / max(1, len(query_terms))
        title_overlap = len(query_terms & title_terms) / max(1, len(query_terms))
        subject_coverage = subject_overlap / max(1, len(required_subject))
        source_quality = cls._source_quality(result, query_terms)
        return min(
            1.0,
            overlap * 0.48 + title_overlap * 0.22 + source_quality * 0.2
            + 0.1 / max(1, rank) + subject_coverage * 0.12,
        )

    @staticmethod
    def _extract_text(raw: str, content_type: str) -> str:
        if "html" not in content_type:
            return " ".join(str(raw).split())[:120_000]
        parser = _ReadableHTML()
        try:
            parser.feed(raw)
            parser.close()
        except Exception:
            return ""
        value = " ".join(part.strip() for part in parser.parts if part.strip())
        return html.unescape(value)[:120_000]

    @classmethod
    def _sentence_score(
        cls,
        sentence: str,
        query_terms: set[str],
        title_terms: set[str],
        position: int,
        result_score: float,
        source_quality: float,
        fresh: bool,
        subject_terms: set[str] | None = None,
    ) -> float:
        if cls._BOILERPLATE.search(sentence):
            return 0.0
        terms = set(content_terms(sentence))
        if not terms:
            return 0.0
        required_subject = subject_terms or set()
        if required_subject and not required_subject.intersection(terms):
            return 0.0
        overlap = len(query_terms & terms)
        if query_terms and overlap == 0:
            return 0.0
        coverage = overlap / max(1, len(query_terms))
        density = overlap / max(5, min(35, len(terms)))
        title_support = len(title_terms & terms) / max(1, len(title_terms))
        date_bonus = 0.1 if fresh and re.search(r"\b20\d{2}\b|امروز|today", sentence, re.I) else 0.0
        return min(
            1.0,
            coverage * 0.39 + density * 0.25 + title_support * 0.11
            + result_score * 0.12 + source_quality * 0.09
            + 0.04 / max(1, position + 1) + date_bonus,
        )

    @staticmethod
    def _candidate_duplicate(candidate: _Candidate, selected: list[_Candidate]) -> bool:
        signature = normalize_text(candidate.text)[:180]
        candidate_terms = set(content_terms(candidate.text))
        for existing in selected:
            if signature == normalize_text(existing.text)[:180]:
                return True
            existing_terms = set(content_terms(existing.text))
            union = candidate_terms | existing_terms
            if union and len(candidate_terms & existing_terms) / len(union) >= 0.84:
                return True
        return False

    @staticmethod
    def _corroboration(
        candidate: _Candidate, candidates: list[_Candidate], query_terms: set[str],
    ) -> float:
        claims = set(content_terms(candidate.text)) - query_terms
        if len(claims) < 2:
            return 0.0
        best = 0.0
        for other in candidates:
            if other.domain == candidate.domain:
                continue
            other_claims = set(content_terms(other.text)) - query_terms
            union = claims | other_claims
            if len(other_claims) < 2 or not union:
                continue
            best = max(best, len(claims & other_claims) / len(union))
        return min(1.0, best * 1.7)

    def _cached(self, key: str, fresh: bool) -> SearchReport | None:
        ttl = 45.0 if fresh else 300.0
        with self._cache_lock:
            item = self._cache.get(key)
            if item is None or time.monotonic() - item[0] > ttl:
                self._cache.pop(key, None)
                return None
            self._cache.move_to_end(key)
            return item[1]

    def _store_cache(self, key: str, report: SearchReport) -> None:
        with self._cache_lock:
            self._cache[key] = (time.monotonic(), report)
            self._cache.move_to_end(key)
            while len(self._cache) > 32:
                self._cache.popitem(last=False)

    def _raw_search(self, query: str) -> SearchDiagnostics:
        detailed = getattr(self.internet, "search_detailed", None)
        if callable(detailed):
            try:
                result = detailed(query)
                if isinstance(result, SearchDiagnostics):
                    return result
            except Exception as exc:
                return SearchDiagnostics((), (), (str(exc),))
        try:
            results = tuple(self.internet.search(query))
        except Exception as exc:
            return SearchDiagnostics((), (), (str(exc),))
        providers = tuple(dict.fromkeys(item.provider for item in results if item.provider))
        return SearchDiagnostics(results, providers, ())

    @staticmethod
    def _select_fetch_results(
        ranked: list[tuple[int, SearchResult, float]], limit: int,
    ) -> list[tuple[int, SearchResult, float]]:
        selected: list[tuple[int, SearchResult, float]] = []
        domains: set[str] = set()
        for item in ranked:
            _rank, result, _score = item
            if result.content:
                continue
            domain = SearchEngine._domain(result.url)
            if not domain or domain in domains:
                continue
            domains.add(domain)
            selected.append(item)
            if len(selected) >= limit:
                break
        return selected

    def search(self, query: str, subject: str = "") -> SearchReport:
        clean_query = " ".join(str(query).split())[:300]
        if not clean_query:
            raise ValueError("Search query is empty")
        analysis = self.analyzer.analyze(clean_query)
        fresh = analysis.needs_fresh_information or analysis.volatile
        clean_subject = " ".join(str(subject).split())[:120]
        subject_terms = set(content_terms(clean_subject, analysis.language))
        cache_key = f"{normalize_text(clean_query)}|{normalize_text(clean_subject)}"
        cached = self._cached(cache_key, fresh)
        if cached is not None:
            return cached

        diagnostics = self._raw_search(clean_query)
        query_terms = set(analysis.key_terms or content_terms(clean_query))
        medical_query = bool(self._MEDICAL_QUERY.search(normalize_text(clean_query)))
        social_query = bool(self._SOCIAL_QUERY.search(normalize_text(clean_query)))
        results = self._deduplicate_results(diagnostics.results)
        results = tuple(
            result for result in results
            if not self._unsafe_medical_result(result, medical_query)
            and not self._unsafe_general_result(result, social_query)
        )
        if not results:
            status = "unavailable" if diagnostics.errors else "no_results"
            report = SearchReport(
                clean_query, "", (), 0, 0.0, status, (),
                diagnostics.providers, diagnostics.errors, fresh,
            )
            self._store_cache(cache_key, report)
            return report

        scored = [
            (
                rank,
                result,
                self._result_score(query_terms, result, rank, subject_terms),
            )
            for rank, result in enumerate(results, 1)
        ]
        scored = [item for item in scored if item[2] >= 0.12]
        if not scored:
            report = SearchReport(
                clean_query, "", (), 0, 0.0, "insufficient", (),
                diagnostics.providers, diagnostics.errors, fresh,
            )
            self._store_cache(cache_key, report)
            return report
        ranked = sorted(scored, key=lambda item: item[2], reverse=True)
        evidence = [
            SearchEvidence(
                result.title, result.url, result.snippet, score,
                result.provider, self._domain(result.url),
                self._source_quality(result, query_terms), 0.0, bool(result.content),
            )
            for _rank, result, score in ranked
        ]
        index_by_url = {item.url: index for index, item in enumerate(evidence)}
        candidates: list[_Candidate] = []

        def add_text(result: SearchResult, raw_text: str, result_score: float) -> None:
            source_index = index_by_url[result.url]
            title_terms = set(content_terms(result.title))
            source_quality = evidence[source_index].source_quality
            sentences = split_sentences(raw_text)
            if not sentences and raw_text.strip():
                sentences = [" ".join(raw_text.split())]
            for position, sentence in enumerate(sentences[:350]):
                clean_sentence = " ".join(sentence.split()).strip()
                if not 24 <= len(clean_sentence) <= 620:
                    continue
                if medical_query and self._LOW_QUALITY_MEDICAL.search(clean_sentence):
                    continue
                score = self._sentence_score(
                    clean_sentence, query_terms, title_terms, position,
                    result_score, source_quality, fresh,
                    subject_terms,
                )
                if score >= 0.16:
                    candidates.append(
                        _Candidate(
                            clean_sentence, source_index, score, result.url, result.title,
                            self._domain(result.url), source_quality,
                        )
                    )

        for _rank, result, result_score in ranked:
            source_text = result.content or result.snippet
            if source_text:
                add_text(result, source_text, result_score)

        fetched = 0
        fetch_targets = self._select_fetch_results(ranked, self.fetch_top_pages)
        fetched_values: list[tuple[SearchResult, float, object]] = []
        if fetch_targets:
            with ThreadPoolExecutor(
                max_workers=min(4, len(fetch_targets)), thread_name_prefix="jarvis-page"
            ) as pool:
                futures = {
                    pool.submit(self.internet.fetch_text, result.url): (result, result_score)
                    for _rank, result, result_score in fetch_targets
                }
                for future in as_completed(futures):
                    result, result_score = futures[future]
                    try:
                        page = future.result()
                    except Exception:
                        continue
                    fetched_values.append((result, result_score, page))
        for result, result_score, page in fetched_values:
            fetched += 1
            readable = self._extract_text(page.text, page.content_type)
            if readable:
                add_text(result, readable, result_score)
                source_index = index_by_url[result.url]
                evidence[source_index] = replace(evidence[source_index], fetched=True)

        candidates.sort(key=lambda item: item.score, reverse=True)
        selected: list[_Candidate] = []
        domain_counts: dict[str, int] = {}
        for candidate in candidates:
            if self._candidate_duplicate(candidate, selected):
                continue
            if domain_counts.get(candidate.domain, 0) >= 2:
                continue
            selected.append(candidate)
            domain_counts[candidate.domain] = domain_counts.get(candidate.domain, 0) + 1
            if len(selected) >= 4:
                break

        passages: list[EvidencePassage] = []
        for candidate in selected:
            corroboration = self._corroboration(candidate, candidates, query_terms)
            evidence_index = candidate.result_index
            evidence[evidence_index] = replace(
                evidence[evidence_index],
                corroboration=max(evidence[evidence_index].corroboration, corroboration),
            )
            passages.append(
                EvidencePassage(
                    candidate.text, candidate.url, candidate.title, candidate.score,
                    evidence_index + 1, corroboration,
                )
            )

        domains = {passage.url.split("/", 3)[2].casefold() for passage in passages}
        max_quality = max((item.source_quality for item in evidence), default=0.0)
        max_corroboration = max((item.corroboration for item in passages), default=0.0)
        mean_score = (
            sum(item.score for item in passages[:3]) / min(3, len(passages))
            if passages else 0.0
        )
        confidence = min(
            0.97,
            mean_score * 0.55 + max_quality * 0.23
            + min(0.12, max(0, len(domains) - 1) * 0.06)
            + max_corroboration * 0.1,
        )
        if not passages:
            status = "insufficient"
        elif len(domains) >= 2 and (max_corroboration >= 0.22 or max_quality >= 0.86):
            status = "verified"
        elif max_quality >= 0.72 or confidence >= 0.55:
            status = "supported"
        else:
            status = "limited"
        if fresh and status == "limited":
            confidence = min(confidence, 0.49)

        language = detect_language(clean_query)
        summary_lines = [f"{item.text} [{item.source_index}]" for item in passages]
        if summary_lines and status == "limited":
            caution = (
                "تأیید مستقل کافی پیدا نشد؛ نتیجه را با احتیاط بخوان."
                if language == "fa" else
                "Independent confirmation was limited; treat this result cautiously."
            )
            summary_lines.append(caution)
        report = SearchReport(
            clean_query,
            "\n\n".join(summary_lines),
            tuple(evidence[:8]),
            fetched,
            round(confidence, 4),
            status,
            tuple(passages),
            diagnostics.providers,
            diagnostics.errors,
            fresh,
        )
        self._store_cache(cache_key, report)
        return report
