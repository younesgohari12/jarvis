from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from jarvis.tools.browser import BrowserActionResult
from jarvis.tools.processes import ProcessResult


@dataclass(frozen=True, slots=True)
class VerificationResult:
    success: bool
    verified: bool
    code: str = ""
    detail: str = ""


class ActionVerifier:
    """Interprets typed tool results instead of assuming that no exception means success."""

    @staticmethod
    def verify(tool: str, result: Any) -> VerificationResult:
        if isinstance(result, ProcessResult):
            return VerificationResult(result.success, result.verified, result.code, result.detail)
        if isinstance(result, BrowserActionResult):
            return VerificationResult(result.success, result.verified, result.code)
        if isinstance(result, bool):
            if tool == "is_app_running":
                return VerificationResult(True, True, "status_checked")
            return VerificationResult(result, False, "ok" if result else "operation_failed")
        if isinstance(result, tuple):
            if not result:
                return VerificationResult(True, True, "empty_collection")
            child_results = [ActionVerifier.verify(tool, item) for item in result]
            success = all(item.success for item in child_results)
            verified = all(item.verified for item in child_results)
            failed_codes = [item.code for item in child_results if not item.success]
            return VerificationResult(
                success, verified, "collection_verified" if success else "collection_partial_failure",
                ",".join(failed_codes),
            )
        generic_success = getattr(result, "success", None)
        if generic_success is not None:
            return VerificationResult(
                bool(generic_success), bool(getattr(result, "verified", False)),
                str(getattr(result, "code", "ok" if generic_success else "operation_failed")),
                str(getattr(result, "detail", "")),
            )
        if isinstance(result, dict) and "status" in result:
            status = str(result.get("status", ""))
            success = status not in {
                "failed", "error", "not_found", "blocked", "undo_failed",
            }
            return VerificationResult(success, True, status or "status_checked")
        if result is None:
            return VerificationResult(False, False, "empty_result")
        return VerificationResult(True, False, "ok")
