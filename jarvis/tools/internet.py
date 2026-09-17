from __future__ import annotations

import html
import ipaddress
import json
import re
import socket
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
import webbrowser
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from html.parser import HTMLParser
from xml.etree import ElementTree


class InternetToolError(RuntimeError):
    def __init__(self, message: str, code: str = "internet_error") -> None:
        self.code = code
        super().__init__(message)


@dataclass(frozen=True, slots=True)
class InternetResult:
    url: str
    status: int
    content_type: str
    text: str
    truncated: bool


@dataclass(frozen=True, slots=True)
class SearchResult:
    title: str
    url: str
    snippet: str
    provider: str = "duckduckgo"
    content: str = ""
    published: str = ""


@dataclass(frozen=True, slots=True)
class SearchDiagnostics:
    results: tuple[SearchResult, ...]
    providers: tuple[str, ...]
    errors: tuple[str, ...]


class _DuckDuckGoParser(HTMLParser):
    """Accept both DuckDuckGo HTML and Lite result layouts."""

    _TITLE_CLASSES = {"result__a", "result-link"}
    _SNIPPET_CLASSES = {"result__snippet", "result-snippet"}

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.results: list[SearchResult] = []
        self._capture = ""
        self._title = ""
        self._url = ""
        self._snippet = ""

    @staticmethod
    def _classes(attrs: dict[str, str | None]) -> set[str]:
        return set((attrs.get("class", "") or "").split())

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        values = dict(attrs)
        classes = self._classes(values)
        href = values.get("href", "") or ""
        is_redirect_result = tag == "a" and "uddg=" in href
        if tag == "a" and (classes & self._TITLE_CLASSES or is_redirect_result):
            self._flush()
            self._capture = "title"
            self._url = href
        elif tag in {"a", "div", "span", "td"} and classes & self._SNIPPET_CLASSES:
            self._capture = "snippet"

    def handle_endtag(self, tag: str) -> None:
        if tag in {"a", "div", "span", "td"}:
            self._capture = ""

    def handle_data(self, data: str) -> None:
        if self._capture == "title":
            self._title += data
        elif self._capture == "snippet":
            self._snippet += data

    @staticmethod
    def _target_url(value: str) -> str:
        url = html.unescape(value).strip()
        if url.startswith("//"):
            url = "https:" + url
        parsed = urllib.parse.urlparse(url)
        if parsed.netloc.casefold().endswith("duckduckgo.com"):
            target = urllib.parse.parse_qs(parsed.query).get("uddg", [""])[0]
            if target:
                url = urllib.parse.unquote(target)
        return url

    def _flush(self) -> None:
        title = " ".join(html.unescape(self._title).split())
        url = self._target_url(self._url)
        snippet = " ".join(html.unescape(self._snippet).split())
        if title and url.startswith(("http://", "https://")):
            self.results.append(SearchResult(title, url, snippet, "duckduckgo"))
        self._title = ""
        self._url = ""
        self._snippet = ""

    def close(self) -> None:
        self._flush()
        super().close()


class _SafeRedirectHandler(urllib.request.HTTPRedirectHandler):
    def __init__(self, validator: object, maximum_redirects: int) -> None:
        super().__init__()
        self._validator = validator
        self._maximum_redirects = maximum_redirects
        self._redirects = 0

    def redirect_request(
        self,
        request: urllib.request.Request,
        response: object,
        code: int,
        message: str,
        headers: object,
        new_url: str,
    ) -> urllib.request.Request | None:
        self._redirects += 1
        if self._redirects > self._maximum_redirects:
            raise InternetToolError("Redirect limit exceeded", "redirect_limit")
        validator = self._validator
        if callable(validator):
            validator(new_url)
        return super().redirect_request(request, response, code, message, headers, new_url)


class InternetTool:
    """Bounded standard-library networking with independent search fallbacks."""

    USER_AGENT = "Jarvis/0.11.0 (bounded desktop research client)"

    def __init__(
        self,
        timeout_seconds: float = 5.0,
        max_response_bytes: int = 524_288,
        max_search_results: int = 8,
        max_redirects: int = 3,
        cancel_event: threading.Event | None = None,
    ) -> None:
        self.timeout_seconds = max(0.5, float(timeout_seconds))
        self.max_response_bytes = max(1024, int(max_response_bytes))
        self.max_search_results = max(1, min(12, int(max_search_results)))
        self.max_redirects = max(0, min(8, int(max_redirects)))
        self.cancel_event = cancel_event

    def _cancelled(self) -> bool:
        return self.cancel_event is not None and self.cancel_event.is_set()

    @staticmethod
    def _parsed_url(url: str) -> urllib.parse.ParseResult:
        parsed = urllib.parse.urlparse(str(url).strip())
        if parsed.scheme not in {"http", "https"} or not parsed.hostname:
            raise InternetToolError("Only valid HTTP/HTTPS URLs are allowed", "invalid_url")
        if parsed.username or parsed.password:
            raise InternetToolError("Credentials in URLs are not allowed", "invalid_url")
        return parsed

    @staticmethod
    def _is_public_host(hostname: str) -> bool:
        if hostname.casefold() in {"localhost", "localhost.localdomain"}:
            return False
        try:
            addresses = socket.getaddrinfo(hostname, None, type=socket.SOCK_STREAM)
        except socket.gaierror as exc:
            raise InternetToolError(f"DNS lookup failed: {exc}", "dns_failed") from exc
        if not addresses:
            return False
        for address in addresses:
            try:
                if not ipaddress.ip_address(address[4][0]).is_global:
                    return False
            except ValueError:
                return False
        return True

    def _validate_public_url(self, url: str) -> urllib.parse.ParseResult:
        parsed = self._parsed_url(url)
        if not self._is_public_host(parsed.hostname or ""):
            raise InternetToolError(
                "Private, local, or reserved network addresses are blocked", "private_address"
            )
        return parsed

    def _opener(self) -> urllib.request.OpenerDirector:
        return urllib.request.build_opener(
            _SafeRedirectHandler(self._validate_public_url, self.max_redirects)
        )

    def _headers(self, language: str = "") -> dict[str, str]:
        accept_language = "fa-IR,fa;q=0.9,en;q=0.7" if language == "fa" else "en-US,en;q=0.9,fa;q=0.5"
        return {
            "User-Agent": self.USER_AGENT,
            "Accept": "text/html,application/json,application/xml,text/plain;q=0.9,*/*;q=0.2",
            "Accept-Language": accept_language,
            "Cache-Control": "no-cache",
        }

    def check_connection(self) -> bool:
        for url in ("https://www.wikipedia.org/", "https://duckduckgo.com/"):
            try:
                parsed = self._validate_public_url(url)
                request = urllib.request.Request(
                    parsed.geturl(), headers={**self._headers(), "Range": "bytes=0-0"}
                )
                with self._opener().open(request, timeout=min(2.5, self.timeout_seconds)) as response:
                    response.read(1)
                    if 200 <= int(getattr(response, "status", 200)) < 500:
                        return True
            except (InternetToolError, OSError, urllib.error.URLError):
                continue
        return False

    def fetch_text(self, url: str, *, language: str = "") -> InternetResult:
        parsed = self._validate_public_url(url)
        request = urllib.request.Request(parsed.geturl(), headers=self._headers(language))
        try:
            with self._opener().open(request, timeout=self.timeout_seconds) as response:
                self._validate_public_url(response.geturl())
                content_type = response.headers.get_content_type()
                is_text = (
                    content_type.startswith("text/")
                    or content_type in {"application/json", "application/xml", "application/xhtml+xml"}
                    or content_type.endswith(("+json", "+xml"))
                )
                if not is_text:
                    raise InternetToolError(
                        f"URL did not return text content ({content_type})", "unsupported_content"
                    )
                content_length = response.headers.get("Content-Length", "")
                if content_length.isdigit() and int(content_length) > self.max_response_bytes * 4:
                    raise InternetToolError(
                        "URL content is larger than the configured safety limit", "response_too_large"
                    )
                data = response.read(self.max_response_bytes + 1)
                charset = response.headers.get_content_charset() or "utf-8"
                try:
                    decoded = data[: self.max_response_bytes].decode(charset, errors="replace")
                except LookupError:
                    decoded = data[: self.max_response_bytes].decode("utf-8", errors="replace")
                return InternetResult(
                    response.geturl(), int(getattr(response, "status", 200)), content_type,
                    decoded, len(data) > self.max_response_bytes,
                )
        except InternetToolError:
            raise
        except urllib.error.HTTPError as exc:
            raise InternetToolError(f"HTTP {exc.code} returned by the server", "http_error") from exc
        except (OSError, urllib.error.URLError) as exc:
            raise InternetToolError(f"Cannot fetch URL: {exc}", "connection_failed") from exc

    def _fetch_with_retry(self, url: str, *, language: str = "") -> InternetResult:
        last_error: InternetToolError | None = None
        for attempt in range(2):
            if self._cancelled():
                raise InternetToolError("Search was cancelled", "cancelled")
            try:
                return self.fetch_text(url, language=language)
            except InternetToolError as exc:
                last_error = exc
                if exc.code not in {"connection_failed", "dns_failed", "http_error"} or attempt:
                    break
                time.sleep(0.12)
        assert last_error is not None
        raise last_error

    def _search_duckduckgo(self, query: str, language: str) -> list[SearchResult]:
        region = "ir-fa" if language == "fa" else "wt-wt"
        params = urllib.parse.urlencode({"q": query, "kl": region, "kp": "-1"})
        errors: list[str] = []
        for endpoint in (
            "https://html.duckduckgo.com/html/?" + params,
            "https://lite.duckduckgo.com/lite/?" + params,
        ):
            try:
                result = self._fetch_with_retry(endpoint, language=language)
                parser = _DuckDuckGoParser()
                parser.feed(result.text)
                parser.close()
            except Exception as exc:
                errors.append(str(exc))
                continue
            cleaned: list[SearchResult] = []
            seen: set[str] = set()
            for item in parser.results:
                normalized_url = item.url.split("#", 1)[0]
                if normalized_url in seen:
                    continue
                seen.add(normalized_url)
                cleaned.append(item)
                if len(cleaned) >= self.max_search_results:
                    break
            if cleaned:
                return cleaned
        raise InternetToolError(
            "DuckDuckGo returned no parseable results" + (f": {errors[-1]}" if errors else ""),
            "provider_unavailable",
        )

    def _search_wikipedia(self, query: str, language: str) -> list[SearchResult]:
        wiki_language = "fa" if language == "fa" else "en"
        endpoint = f"https://{wiki_language}.wikipedia.org/w/api.php"
        params = {
            "action": "query",
            "generator": "search",
            "gsrsearch": query,
            "gsrnamespace": "0",
            "gsrlimit": str(min(5, self.max_search_results)),
            "prop": "extracts|info",
            "exintro": "1",
            "explaintext": "1",
            "exsentences": "4",
            "inprop": "url",
            "redirects": "1",
            "format": "json",
            "formatversion": "2",
        }
        url = endpoint + "?" + urllib.parse.urlencode(params)
        response = self._fetch_with_retry(url, language=language)
        try:
            payload = json.loads(response.text)
        except json.JSONDecodeError as exc:
            raise InternetToolError("Wikipedia returned invalid JSON", "provider_invalid_response") from exc
        pages = payload.get("query", {}).get("pages", []) if isinstance(payload, dict) else []
        if isinstance(pages, dict):
            pages = list(pages.values())
        results: list[SearchResult] = []
        for page in pages if isinstance(pages, list) else []:
            if not isinstance(page, dict) or page.get("missing"):
                continue
            title = " ".join(str(page.get("title", "")).split())
            extract = " ".join(str(page.get("extract", "")).split())
            page_url = str(page.get("fullurl", ""))
            if not page_url and title:
                page_url = f"https://{wiki_language}.wikipedia.org/wiki/" + urllib.parse.quote(title.replace(" ", "_"))
            if title and page_url:
                results.append(
                    SearchResult(
                        title, page_url, extract[:900], f"wikipedia_{wiki_language}",
                        extract[:12_000], "",
                    )
                )
        return results

    def _search_bing_rss(self, query: str, language: str) -> list[SearchResult]:
        """Use Bing's lightweight RSS output as an independent HTML-free fallback."""
        params = urllib.parse.urlencode(
            {"q": query, "format": "rss", "setlang": "fa" if language == "fa" else "en"}
        )
        response = self._fetch_with_retry(
            "https://www.bing.com/search?" + params, language=language
        )
        try:
            root = ElementTree.fromstring(response.text)
        except ElementTree.ParseError as exc:
            raise InternetToolError(
                "Bing returned invalid RSS", "provider_invalid_response"
            ) from exc
        results: list[SearchResult] = []
        for item in root.findall(".//item")[: self.max_search_results]:
            title = " ".join((item.findtext("title") or "").split())
            url = (item.findtext("link") or "").strip()
            description = item.findtext("description") or ""
            description = html.unescape(re.sub(r"<[^>]+>", " ", description))
            snippet = " ".join(description.split())[:900]
            published = " ".join((item.findtext("pubDate") or "").split())[:100]
            if title and url.startswith(("http://", "https://")):
                results.append(SearchResult(title, url, snippet, "bing_rss", "", published))
        if not results:
            raise InternetToolError("Bing returned no parseable RSS results", "provider_unavailable")
        return results

    @staticmethod
    def _canonical_url(value: str) -> str:
        parsed = urllib.parse.urlsplit(value)
        kept = [
            (key, item)
            for key, item in urllib.parse.parse_qsl(parsed.query, keep_blank_values=True)
            if not key.casefold().startswith(("utm_", "ref", "source", "fbclid", "gclid"))
        ]
        return urllib.parse.urlunsplit(
            (
                parsed.scheme.casefold(), parsed.netloc.casefold(),
                parsed.path.rstrip("/") or "/", urllib.parse.urlencode(sorted(kept)), "",
            )
        )

    @staticmethod
    def _deduplicate(results: list[SearchResult], limit: int) -> tuple[SearchResult, ...]:
        output: list[SearchResult] = []
        seen_urls: set[str] = set()
        seen_titles: set[str] = set()
        for item in results:
            url_key = InternetTool._canonical_url(item.url)
            title_key = re.sub(r"\W+", " ", item.title.casefold()).strip()
            if not url_key or url_key in seen_urls or (title_key and title_key in seen_titles):
                continue
            seen_urls.add(url_key)
            if title_key:
                seen_titles.add(title_key)
            output.append(item)
            if len(output) >= limit:
                break
        return tuple(output)

    def search_detailed(self, query: str) -> SearchDiagnostics:
        clean_query = " ".join(str(query).split())[:300]
        if not clean_query:
            raise InternetToolError("Search query is empty", "empty_query")
        language = "fa" if re.search(r"[\u0600-\u06ff]", clean_query) else "en"
        providers: list[str] = []
        errors: list[str] = []
        gathered: dict[str, list[SearchResult]] = {}
        jobs = {
            "duckduckgo": lambda: self._search_duckduckgo(clean_query, language),
            "bing_rss": lambda: self._search_bing_rss(clean_query, language),
            f"wikipedia_{language}": lambda: self._search_wikipedia(clean_query, language),
        }
        with ThreadPoolExecutor(max_workers=3, thread_name_prefix="jarvis-search") as pool:
            futures = {pool.submit(job): name for name, job in jobs.items()}
            for future in as_completed(futures):
                name = futures[future]
                if self._cancelled():
                    errors.append("search: cancelled")
                    break
                try:
                    values = future.result()
                except Exception as exc:
                    errors.append(f"{name}: {exc}")
                    continue
                if values:
                    providers.append(name)
                    gathered[name] = values

        merged: list[SearchResult] = []
        longest = max((len(values) for values in gathered.values()), default=0)
        ordered_names = (
            (f"wikipedia_{language}", "bing_rss", "duckduckgo")
            if language == "fa" else ("duckduckgo", "bing_rss", f"wikipedia_{language}")
        )
        for index in range(longest):
            for name in ordered_names:
                values = gathered.get(name, [])
                if index < len(values):
                    merged.append(values[index])
        return SearchDiagnostics(
            self._deduplicate(merged, self.max_search_results),
            tuple(providers), tuple(errors),
        )

    def search(self, query: str) -> list[SearchResult]:
        return list(self.search_detailed(query).results)

    def open_in_browser(self, url: str) -> bool:
        parsed = self._parsed_url(url)
        return bool(webbrowser.open(parsed.geturl(), new=2))
