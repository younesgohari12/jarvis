from __future__ import annotations

from dataclasses import dataclass
from typing import Any, TYPE_CHECKING

from jarvis.agent.verifier import ActionVerifier, VerificationResult
from jarvis.tools.registry import ArgumentValidationError, ToolError, ToolRegistry

if TYPE_CHECKING:
    from jarvis.agent.planner import ToolCall


@dataclass(frozen=True, slots=True)
class ExecutionOutcome:
    success: bool
    result: Any = None
    verification: VerificationResult = VerificationResult(False, False, "not_run")
    code: str = ""
    missing: tuple[str, ...] = ()
    debug_detail: str = ""
    attempts: int = 1
    recovered: bool = False


class ToolExecutor:
    def __init__(self, registry: ToolRegistry, verifier: ActionVerifier | None = None) -> None:
        self.registry = registry
        self.verifier = verifier or ActionVerifier()

    def execute(self, call: ToolCall, *, confirmed: bool = False) -> ExecutionOutcome:
        try:
            missing = self.registry.validate(call.tool, call.arguments)
            if missing:
                return ExecutionOutcome(False, code="missing_argument", missing=missing)
            result = self.registry.invoke(call.tool, call.arguments, confirmed=confirmed)
            verification = self.verifier.verify(call.tool, result)
            if (
                not verification.success
                and verification.code in {"not_found", "open_failed", "launch_failed"}
                and call.tool in {"open_app", "restart_app"}
                and self.registry.spec("refresh_app_index") is not None
            ):
                self.registry.invoke("refresh_app_index", {}, confirmed=False)
                retried = self.registry.invoke(call.tool, call.arguments, confirmed=confirmed)
                retry_verification = self.verifier.verify(call.tool, retried)
                return ExecutionOutcome(
                    retry_verification.success, retried, retry_verification,
                    retry_verification.code, debug_detail=retry_verification.detail,
                    attempts=2, recovered=retry_verification.success,
                )
            return ExecutionOutcome(
                verification.success, result, verification, verification.code,
                debug_detail=verification.detail,
            )
        except ArgumentValidationError as exc:
            return ExecutionOutcome(
                False, code="missing_argument", missing=exc.missing,
                debug_detail=str(exc),
            )
        except ToolError as exc:
            return ExecutionOutcome(False, code="tool_failed", debug_detail=str(exc))
