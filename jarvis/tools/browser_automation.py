from __future__ import annotations

import re
import threading
import urllib.parse
from dataclasses import dataclass, field
from typing import Any

from jarvis.tools.browser import BrowserActionResult, BrowserManager


@dataclass(frozen=True, slots=True)
class BrowserDOMResult:
    success: bool
    action: str
    verified: bool
    code: str
    url: str = ""
    detail: str = ""
    data: dict[str, Any] = field(default_factory=dict)


class BrowserAutomation:
    """Optional Playwright DOM control with a standard-browser fallback.

    JARVIS never downloads a browser or embeds Chromium at runtime. If the
    optional automation environment is installed, this class launches the
    user's selected installed browser through Playwright. Ordinary URL opening
    continues to work without Playwright.
    """

    SENSITIVE = re.compile(
        r"(?:buy|purchase|pay|checkout|send|submit|publish|post|email|transfer|"
        r"خرید|پرداخت|ارسال|انتشار|ثبت نهایی|واریز)", re.I
    )

    def __init__(self, browser: BrowserManager) -> None:
        self.browser = browser
        self._lock = threading.RLock()
        self._playwright: Any | None = None
        self._context: Any | None = None
        self._pages: list[Any] = []
        self._active = 0
        self._last_url = ""
        self._backend_error = ""

    @staticmethod
    def click_risk(arguments: dict[str, object]) -> str:
        label = str(arguments.get("label") or arguments.get("selector") or "")
        sensitive = bool(arguments.get("sensitive")) or bool(BrowserAutomation.SENSITIVE.search(label))
        return "privileged" if sensitive else "browser_interaction"

    @property
    def page(self) -> Any | None:
        if not self._pages:
            return None
        self._active = max(0, min(self._active, len(self._pages) - 1))
        return self._pages[self._active]

    def _ensure(self, browser: str = "") -> BrowserDOMResult | None:
        if self.page is not None:
            return None
        try:
            from playwright.sync_api import sync_playwright  # type: ignore[import-not-found]
        except ImportError:
            return BrowserDOMResult(
                False, "start_automation", True, "automation_dependency_missing",
                detail="Install requirements-automation.txt and run playwright install only if DOM automation is needed.",
            )
        try:
            self._playwright = sync_playwright().start()
            browser_name = str(browser).casefold().strip()
            if not browser_name:
                browser_name = str(self.browser.processes.default_browser_id()).casefold().strip()
            channel = "msedge" if browser_name in {"edge", "msedge"} else "chrome" if browser_name in {"chrome", "google_chrome"} else ""
            if not channel:
                self.close()
                return BrowserDOMResult(
                    False, "start_automation", True, "unsupported_automation_browser",
                    detail="DOM automation needs an installed Microsoft Edge or Google Chrome browser.",
                )
            launch_options: dict[str, Any] = {"headless": False, "channel": channel}
            executable = self._playwright.chromium.launch(**launch_options)
            self._context = executable.new_context(accept_downloads=False)
            self._pages = [self._context.new_page()]
            self._active = 0
            return None
        except Exception as exc:
            self._backend_error = str(exc)[:500]
            self.close()
            return BrowserDOMResult(
                False, "start_automation", True, "automation_start_failed",
                detail=self._backend_error,
            )

    def open_browser(self, browser: str = "", *, automation: bool = False) -> BrowserDOMResult:
        with self._lock:
            if automation:
                error = self._ensure(browser)
                if error:
                    return error
                return BrowserDOMResult(True, "open_browser", True, "automation_ready", self.current_url_value())
            result = self.browser.open_default() if not browser else self.browser.processes.open_app(browser)
            success = bool(getattr(result, "success", result))
            return BrowserDOMResult(success, "open_browser", bool(getattr(result, "verified", False)), getattr(result, "code", "opened" if success else "failed"))

    def open_url(self, url: str, browser: str = "", *, automation: bool = False) -> BrowserDOMResult:
        normalized = BrowserManager._validated_url(url)
        with self._lock:
            if automation or self.page is not None:
                error = self._ensure(browser)
                if error:
                    return error
                page = self.page
                assert page is not None
                try:
                    page.goto(normalized, wait_until="domcontentloaded", timeout=15_000)
                    self._last_url = str(page.url)
                    verified = self._last_url.startswith(("http://", "https://"))
                    return BrowserDOMResult(verified, "open_url", True, "navigated" if verified else "navigation_failed", self._last_url)
                except Exception as exc:
                    return BrowserDOMResult(False, "open_url", True, "navigation_failed", detail=str(exc)[:500])
            result: BrowserActionResult = self.browser.open_url(normalized, browser)
            if result.success:
                self._last_url = normalized
            return BrowserDOMResult(result.success, "open_url", result.verified, result.code, normalized)

    def search_google(self, query: str, browser: str = "", *, automation: bool = False) -> BrowserDOMResult:
        clean = " ".join(str(query).split())[:500]
        if not clean:
            return BrowserDOMResult(False, "search_google", True, "empty_query")
        url = "https://www.google.com/search?q=" + urllib.parse.quote_plus(clean)
        result = self.open_url(url, browser, automation=automation)
        return BrowserDOMResult(result.success, "search_google", result.verified, result.code, result.url, result.detail, {"query": clean})

    def search_youtube(self, query: str, browser: str = "", *, automation: bool = False) -> BrowserDOMResult:
        clean = " ".join(str(query).split())[:500]
        if not clean:
            return BrowserDOMResult(False, "search_youtube", True, "empty_query")
        url = "https://www.youtube.com/results?search_query=" + urllib.parse.quote_plus(clean)
        result = self.open_url(url, browser, automation=automation)
        return BrowserDOMResult(result.success, "search_youtube", result.verified, result.code, result.url, result.detail, {"query": clean})

    def current_url_value(self) -> str:
        page = self.page
        if page is not None:
            try:
                return str(page.url)
            except Exception:
                pass
        return self._last_url

    def current_url(self) -> BrowserDOMResult:
        value = self.current_url_value()
        return BrowserDOMResult(bool(value), "current_url", True, "url_available" if value else "url_unknown", value)

    def _navigation(self, action: str) -> BrowserDOMResult:
        page = self.page
        if page is None:
            return BrowserDOMResult(False, action, True, "automation_session_required")
        try:
            if action == "back":
                page.go_back(wait_until="domcontentloaded", timeout=10_000)
            elif action == "forward":
                page.go_forward(wait_until="domcontentloaded", timeout=10_000)
            else:
                page.reload(wait_until="domcontentloaded", timeout=10_000)
            self._last_url = str(page.url)
            return BrowserDOMResult(True, action, True, "navigated", self._last_url)
        except Exception as exc:
            return BrowserDOMResult(False, action, True, "navigation_failed", detail=str(exc)[:500])

    def navigate_back(self) -> BrowserDOMResult:
        return self._navigation("back")

    def navigate_forward(self) -> BrowserDOMResult:
        return self._navigation("forward")

    def refresh(self) -> BrowserDOMResult:
        return self._navigation("refresh")

    def new_tab(self, url: str = "") -> BrowserDOMResult:
        with self._lock:
            error = self._ensure()
            if error:
                return error
            assert self._context is not None
            page = self._context.new_page()
            self._pages.append(page)
            self._active = len(self._pages) - 1
            if url:
                return self.open_url(url, automation=True)
            return BrowserDOMResult(True, "new_tab", True, "tab_created", "about:blank", data={"tab": self._active})

    def close_tab(self, index: int | None = None) -> BrowserDOMResult:
        with self._lock:
            if not self._pages:
                return BrowserDOMResult(False, "close_tab", True, "no_tabs")
            target = self._active if index is None else max(0, min(len(self._pages) - 1, int(index)))
            try:
                self._pages[target].close()
            except Exception as exc:
                return BrowserDOMResult(False, "close_tab", True, "close_tab_failed", detail=str(exc)[:500])
            self._pages.pop(target)
            self._active = max(0, min(target - 1, len(self._pages) - 1))
            return BrowserDOMResult(True, "close_tab", True, "tab_closed", self.current_url_value(), data={"tabs": len(self._pages)})

    def switch_tab(self, index: int) -> BrowserDOMResult:
        target = int(index)
        if target < 0 or target >= len(self._pages):
            return BrowserDOMResult(False, "switch_tab", True, "invalid_tab")
        self._active = target
        page = self.page
        assert page is not None
        try:
            page.bring_to_front()
        except Exception as exc:
            return BrowserDOMResult(False, "switch_tab", True, "switch_failed", detail=str(exc)[:500])
        return BrowserDOMResult(True, "switch_tab", True, "tab_active", self.current_url_value(), data={"tab": target})

    def read_page(self, maximum_characters: int = 60_000) -> BrowserDOMResult:
        page = self.page
        if page is None:
            return BrowserDOMResult(False, "read_page", True, "automation_session_required")
        limit = max(1000, min(100_000, int(maximum_characters)))
        try:
            text = " ".join(str(page.locator("body").inner_text(timeout=8_000)).split())[:limit]
            return BrowserDOMResult(bool(text), "read_page", True, "page_read" if text else "empty_page", str(page.url), data={"text": text, "truncated": len(text) >= limit})
        except Exception as exc:
            return BrowserDOMResult(False, "read_page", True, "read_failed", detail=str(exc)[:500])

    def links(self, limit: int = 50) -> BrowserDOMResult:
        page = self.page
        if page is None:
            return BrowserDOMResult(False, "links", True, "automation_session_required")
        maximum = max(1, min(100, int(limit)))
        try:
            values = page.locator("a[href]").evaluate_all(
                "els => els.map(e => ({text:(e.innerText||e.textContent||'').trim(), url:e.href}))"
            )
            links: list[dict[str, str]] = []
            seen: set[str] = set()
            for item in values:
                url = str(item.get("url", ""))
                if not url.startswith(("http://", "https://")) or url in seen:
                    continue
                seen.add(url)
                links.append({"text": str(item.get("text", ""))[:200], "url": url})
                if len(links) >= maximum:
                    break
            return BrowserDOMResult(True, "links", True, "links_collected", str(page.url), data={"links": links})
        except Exception as exc:
            return BrowserDOMResult(False, "links", True, "links_failed", detail=str(exc)[:500])

    def click(self, label: str = "", selector: str = "", *, sensitive: bool = False) -> BrowserDOMResult:
        page = self.page
        if page is None:
            return BrowserDOMResult(False, "click", True, "automation_session_required")
        target = str(selector).strip() or str(label).strip()
        if not target:
            return BrowserDOMResult(False, "click", True, "missing_target")
        if self.SENSITIVE.search(target) and not sensitive:
            return BrowserDOMResult(False, "click", True, "sensitive_confirmation_required")
        try:
            locator = page.locator(selector).first if selector else page.get_by_text(label, exact=False).first
            locator.click(timeout=8_000)
            page.wait_for_timeout(250)
            self._last_url = str(page.url)
            return BrowserDOMResult(True, "click", False, "clicked", self._last_url, data={"target": target})
        except Exception as exc:
            return BrowserDOMResult(False, "click", True, "element_not_found", detail=str(exc)[:500])

    def fill(self, field: str, text: str, *, selector: str = "") -> BrowserDOMResult:
        page = self.page
        if page is None:
            return BrowserDOMResult(False, "fill", True, "automation_session_required")
        if not str(field).strip() or len(str(text)) > 10_000:
            return BrowserDOMResult(False, "fill", True, "invalid_arguments")
        try:
            if selector:
                locator = page.locator(selector).first
            else:
                locator = page.get_by_label(field, exact=False).first
                if locator.count() == 0:
                    locator = page.get_by_placeholder(field, exact=False).first
            locator.fill(str(text), timeout=8_000)
            return BrowserDOMResult(True, "fill", False, "field_filled", str(page.url), data={"field": field, "characters": len(str(text))})
        except Exception as exc:
            return BrowserDOMResult(False, "fill", True, "field_not_found", detail=str(exc)[:500])

    def close(self) -> None:
        with self._lock:
            for page in self._pages:
                try:
                    page.close()
                except Exception:
                    pass
            self._pages.clear()
            if self._context is not None:
                try:
                    browser = self._context.browser
                    self._context.close()
                    if browser is not None:
                        browser.close()
                except Exception:
                    pass
            self._context = None
            if self._playwright is not None:
                try:
                    self._playwright.stop()
                except Exception:
                    pass
            self._playwright = None
