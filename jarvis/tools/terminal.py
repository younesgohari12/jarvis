from __future__ import annotations

import os
import platform
import re
import shlex
import subprocess
import threading
import time
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class CommandAssessment:
    level: str
    code: str
    reason: str


@dataclass(frozen=True, slots=True)
class CommandResult:
    success: bool
    command: str
    return_code: int | None
    stdout: str
    stderr: str
    duration_ms: float
    verified: bool
    code: str
    policy_level: str


class CommandPolicy:
    """Deny-by-default command policy; it never delegates policy to a shell."""

    SAFE_EXECUTABLES = frozenset(
        {
            "whoami", "hostname", "ipconfig", "ping", "tracert", "pathping",
            "nslookup", "netstat", "arp", "route", "tasklist", "systeminfo",
            "where", "where.exe", "ver", "dir", "type", "wmic",
        }
    )
    SAFE_PYTHON_ARGS = frozenset({"--version", "-V", "-VV"})
    # BUG-001 (v22.4.2): explicit Route command grammar. `-4`/`-6` are global
    # address-family selectors, NOT operations; they can precede a mutating
    # subcommand (`route -4 add ...`). Classification must therefore skip
    # global flags first and then inspect the real operation, failing closed
    # on missing or unknown operations.
    ROUTE_GLOBAL_FLAGS = frozenset({"-4", "-6"})
    ROUTE_READONLY_OPERATIONS = frozenset({"print"})
    ROUTE_MUTATING_OPERATIONS = frozenset({"add", "delete", "change"})
    CAUTION_EXECUTABLES = frozenset({"git", "git.exe", "python", "python.exe", "py", "py.exe"})
    DANGEROUS_EXECUTABLES = frozenset(
        {"shutdown", "shutdown.exe", "taskkill", "taskkill.exe", "reg", "reg.exe", "sc", "sc.exe"}
    )
    BLOCKED_EXECUTABLES = frozenset(
        {
            "format", "format.com", "diskpart", "cipher", "bcdedit", "bootrec",
            "del", "erase", "rd", "rmdir", "rm", "dd", "mkfs", "fdisk",
        }
    )
    META = re.compile(r"(?:&&|\|\||[|;&><`]|\$\(|\r|\n)")
    SECRET = re.compile(
        r"(?i)(?:token|password|passwd|secret|api[_-]?key|authorization)\s*[=:]\s*\S+"
    )

    @classmethod
    def split(cls, command: str) -> tuple[str, ...]:
        clean = str(command).strip()
        if not clean or len(clean) > 2000 or cls.META.search(clean):
            return ()
        try:
            values = shlex.split(clean, posix=platform.system() != "Windows")
        except ValueError:
            return ()
        return tuple(value.strip('"') for value in values if value)

    @classmethod
    def assess(cls, command: str) -> CommandAssessment:
        clean = str(command).strip()
        if not clean:
            return CommandAssessment("blocked", "empty_command", "A command is required")
        if cls.META.search(clean):
            return CommandAssessment(
                "blocked", "shell_composition_blocked",
                "Pipes, redirection, command chaining and substitutions are blocked",
            )
        values = cls.split(clean)
        if not values:
            return CommandAssessment("blocked", "invalid_command", "Command parsing failed")
        executable = Path(values[0]).name.casefold()
        lowered = clean.casefold()
        if executable in cls.BLOCKED_EXECUTABLES:
            return CommandAssessment(
                "blocked", "destructive_command_blocked",
                "Destructive disk or bulk-delete commands are never exposed",
            )
        if any(
            marker in lowered
            for marker in (
                "remove-item -recurse", "invoke-expression", "iex ", "encodedcommand",
                "downloadstring", "frombase64string", "currentversion\\run",
            )
        ):
            return CommandAssessment(
                "blocked", "unsafe_script_blocked", "Unsafe script execution pattern"
            )
        if executable in cls.DANGEROUS_EXECUTABLES:
            return CommandAssessment(
                "dangerous", "explicit_confirmation_required",
                "The command can change processes, services, registry or power state",
            )
        if executable in cls.SAFE_EXECUTABLES:
            if executable == "route":
                return cls._assess_route(values)
            if executable == "ipconfig" and any(
                value.casefold() in {"/release", "/renew", "/registerdns"} for value in values[1:]
            ):
                return CommandAssessment("dangerous", "network_change_confirmation_required", "Network changes require confirmation")
            if executable == "wmic" and any(
                value.casefold() in {"call", "delete", "set", "create"} for value in values[1:]
            ):
                return CommandAssessment("dangerous", "wmic_change_confirmation_required", "WMIC changes require confirmation")
            return CommandAssessment("safe", "read_only", "Read-only system command")
        if executable in {"git", "git.exe"}:
            git_values = tuple(value.casefold() for value in values[1:])
            if any(
                value == "-c"
                or value.startswith("-c")
                or value.startswith("--config-env")
                or value.startswith("--exec-path")
                or "alias." in value
                for value in git_values
            ):
                return CommandAssessment(
                    "blocked", "git_execution_override_blocked",
                    "Git aliases and executable/configuration overrides are blocked",
                )
            if git_values == ("--version",):
                return CommandAssessment("safe", "read_only", "Git version query")
            return CommandAssessment(
                "dangerous", "git_confirmation_required",
                "Git can modify files, contact remotes, or invoke helpers and requires confirmation",
            )
        if executable in {"python", "python.exe", "py", "py.exe"}:
            if len(values) == 2 and values[1] in cls.SAFE_PYTHON_ARGS:
                return CommandAssessment("safe", "read_only", "Python version query")
            return CommandAssessment(
                "dangerous", "code_execution_confirmation_required",
                "Running Python code requires explicit confirmation",
            )
        return CommandAssessment(
            "blocked", "executable_not_allowlisted",
            f"Executable is not in the command allowlist: {executable}",
        )

    @classmethod
    def _assess_route(cls, values: tuple[str, ...]) -> CommandAssessment:
        """Classify a `route` command by its real operation, fail-closed.

        Grammar: route [-4|-6 ...] <operation> [args...]

        * global address-family selectors (-4/-6) are skipped and never
          treated as the operation (BUG-001);
        * `print` (with any selectors) stays read-only;
        * `add`/`delete`/`change` always require confirmation;
        * a missing or unknown operation is BLOCKED — it never defaults to
          safe (fail closed).
        """
        remainder = list(values[1:])
        while remainder and remainder[0].casefold() in cls.ROUTE_GLOBAL_FLAGS:
            remainder.pop(0)
        if not remainder:
            return CommandAssessment(
                "blocked", "route_operation_unverified",
                "Route command without a verifiable operation is not allowed",
            )
        operation = remainder[0].casefold()
        if operation in cls.ROUTE_READONLY_OPERATIONS:
            return CommandAssessment("safe", "read_only", "Route print is read-only")
        if operation in cls.ROUTE_MUTATING_OPERATIONS:
            return CommandAssessment(
                "dangerous", "network_change_confirmation_required",
                "Route changes require confirmation",
            )
        return CommandAssessment(
            "blocked", "route_unknown_operation_blocked",
            f"Unknown route operation is not allowed: {operation}",
        )

    @classmethod
    def redact(cls, value: str) -> str:
        return cls.SECRET.sub(lambda match: match.group(0).split("=", 1)[0] + "=[REDACTED]", value)


class CommandRunner:
    BUILTINS = frozenset({"dir", "type", "ver"})

    def __init__(
        self, maximum_output_bytes: int = 64 * 1024,
        cancel_event: threading.Event | None = None,
    ) -> None:
        self.maximum_output_bytes = max(4096, min(1024 * 1024, int(maximum_output_bytes)))
        self.cancel_event = cancel_event

    @staticmethod
    def risk_category(arguments: dict[str, object]) -> str:
        assessment = CommandPolicy.assess(str(arguments.get("command", "")))
        return {
            "safe": "read_only",
            "caution": "caution",
            "dangerous": "shell",
            "blocked": "denied",
        }[assessment.level]

    def run(
        self,
        command: str,
        *,
        cwd: str = "",
        timeout_seconds: float = 10.0,
    ) -> CommandResult:
        assessment = CommandPolicy.assess(command)
        if assessment.level == "blocked":
            return CommandResult(
                False, CommandPolicy.redact(command), None, "", assessment.reason, 0.0,
                True, assessment.code, assessment.level,
            )
        arguments = list(CommandPolicy.split(command))
        if not arguments:
            return CommandResult(
                False, "", None, "", "Command parsing failed", 0.0,
                True, "invalid_command", assessment.level,
            )
        working_directory: str | None = None
        if cwd:
            resolved = Path(cwd).expanduser().resolve()
            if not resolved.is_dir():
                return CommandResult(
                    False, CommandPolicy.redact(command), None, "", "Working folder does not exist",
                    0.0, True, "invalid_working_directory", assessment.level,
                )
            working_directory = str(resolved)
        if platform.system() == "Windows" and arguments[0].casefold() in self.BUILTINS:
            arguments = ["cmd.exe", "/d", "/s", "/c", *arguments]
        timeout = max(1.0, min(30.0, float(timeout_seconds)))
        started = time.perf_counter()
        try:
            process = subprocess.Popen(
                arguments,
                cwd=working_directory,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                encoding="utf-8",
                errors="replace",
                shell=False,
                env=os.environ.copy(),
                creationflags=(
                    getattr(subprocess, "CREATE_NO_WINDOW", 0)
                    if platform.system() == "Windows" else 0
                ),
            )
            deadline = time.monotonic() + timeout
            while True:
                remaining = deadline - time.monotonic()
                if self.cancel_event is not None and self.cancel_event.is_set():
                    process.terminate()
                    try:
                        stdout_value, stderr_value = process.communicate(timeout=0.5)
                    except subprocess.TimeoutExpired:
                        process.kill()
                        stdout_value, stderr_value = process.communicate()
                    return CommandResult(
                        False, CommandPolicy.redact(command), process.returncode,
                        CommandPolicy.redact(stdout_value), CommandPolicy.redact(stderr_value),
                        round((time.perf_counter() - started) * 1000.0, 3), True,
                        "cancelled", assessment.level,
                    )
                if remaining <= 0:
                    process.kill()
                    stdout_value, stderr_value = process.communicate()
                    return CommandResult(
                        False, CommandPolicy.redact(command), process.returncode,
                        CommandPolicy.redact(stdout_value),
                        CommandPolicy.redact(stderr_value or "Command timed out"),
                        round((time.perf_counter() - started) * 1000.0, 3), True,
                        "timeout", assessment.level,
                    )
                try:
                    stdout_value, stderr_value = process.communicate(timeout=min(0.15, remaining))
                    break
                except subprocess.TimeoutExpired:
                    continue
        except OSError as exc:
            return CommandResult(
                False, CommandPolicy.redact(command), None, "", str(exc)[:500],
                round((time.perf_counter() - started) * 1000.0, 3), True,
                "execution_failed", assessment.level,
            )
        limit = self.maximum_output_bytes
        stdout = CommandPolicy.redact(stdout_value.encode("utf-8")[:limit].decode("utf-8", "ignore"))
        stderr = CommandPolicy.redact(stderr_value.encode("utf-8")[:limit].decode("utf-8", "ignore"))
        success = process.returncode == 0
        return CommandResult(
            success,
            CommandPolicy.redact(command),
            process.returncode,
            stdout,
            stderr,
            round((time.perf_counter() - started) * 1000.0, 3),
            True,
            "completed" if success else "nonzero_exit",
            assessment.level,
        )
