from __future__ import annotations

import urllib.parse
import webbrowser
from dataclasses import dataclass

from jarvis.tools.processes import ProcessManager


@dataclass(frozen=True, slots=True)
class BrowserActionResult:
    success: bool
    action: str
    url: str
    browser: str = "default"
    verified: bool = False
    code: str = ""
    label: str = "Browser"


class BrowserManager:
    def __init__(self, processes: ProcessManager) -> None:
        self.processes = processes

    @staticmethod
    def _validated_url(raw_url: str) -> str:
        value = str(raw_url).strip()
        parsed = urllib.parse.urlsplit(value)
        if parsed.scheme.casefold() not in {"http", "https"} or not parsed.hostname:
            raise ValueError("Only absolute HTTP/HTTPS URLs can be opened")
        return value

    def open_url(self, raw_url: str, browser: str = "") -> BrowserActionResult:
        url = self._validated_url(raw_url)
        target = str(browser).strip().casefold()
        if target and target not in {"default", "default_browser"}:
            result = self.processes.open_app(target, (url,))
            return BrowserActionResult(
                result.success, "open_url", url, result.app_id, result.verified,
                result.code, result.label,
            )
        success = bool(webbrowser.open(url, new=2))
        return BrowserActionResult(success, "open_url", url, "default", False, "opened" if success else "open_failed")

    def open_default(self) -> BrowserActionResult:
        success = bool(webbrowser.open("about:blank", new=2))
        return BrowserActionResult(
            success, "open_browser", "about:blank", "default", False,
            "opened" if success else "open_failed", "Default Browser",
        )

    def search(self, query: str, browser: str = "", search_url: str = "") -> BrowserActionResult:
        clean = str(query).strip()
        if not clean:
            raise ValueError("Search query is required")
        url = search_url or f"https://www.google.com/search?q={urllib.parse.quote_plus(clean)}"
        result = self.open_url(url, browser)
        return BrowserActionResult(
            result.success, "search", result.url, result.browser,
            result.verified, result.code, result.label,
        )
