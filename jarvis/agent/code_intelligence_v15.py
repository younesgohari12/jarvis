from __future__ import annotations

import ast
import math
import operator
import re
from dataclasses import dataclass
from typing import Any

from jarvis.utils.text import normalize_text


@dataclass(frozen=True, slots=True)
class CodeTraceAnswer:
    text: str
    confidence: float = 0.995
    checks: tuple[str, ...] = ("code_parsed", "safe_ast_interpreter", "trace_verified")


class _SafePythonEvaluator:
    """Tiny non-executing interpreter for educational Python trace questions."""

    MAX_STEPS = 5000
    MAX_RANGE = 1000

    def __init__(self) -> None:
        self.env: dict[str, Any] = {}
        self.output: list[str] = []
        self.steps = 0

    def _tick(self) -> None:
        self.steps += 1
        if self.steps > self.MAX_STEPS:
            raise ValueError("trace too large")

    def expr(self, node: ast.AST) -> Any:
        self._tick()
        if isinstance(node, ast.Constant):
            if isinstance(node.value, (str, int, float, bool, type(None))):
                return node.value
            raise ValueError("unsupported constant")
        if isinstance(node, ast.Name):
            if node.id in self.env:
                return self.env[node.id]
            raise ValueError("unknown name")
        if isinstance(node, ast.List):
            return [self.expr(item) for item in node.elts]
        if isinstance(node, ast.Tuple):
            return tuple(self.expr(item) for item in node.elts)
        if isinstance(node, ast.Dict):
            return {self.expr(k): self.expr(v) for k, v in zip(node.keys, node.values)}
        if isinstance(node, ast.UnaryOp):
            value = self.expr(node.operand)
            ops = {ast.USub: operator.neg, ast.UAdd: operator.pos, ast.Not: operator.not_, ast.Invert: operator.invert}
            fn = ops.get(type(node.op))
            if fn is None:
                raise ValueError("unsupported unary op")
            return fn(value)
        if isinstance(node, ast.BinOp):
            left, right = self.expr(node.left), self.expr(node.right)
            ops = {
                ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul,
                ast.Div: operator.truediv, ast.FloorDiv: operator.floordiv, ast.Mod: operator.mod,
                ast.Pow: operator.pow,
            }
            fn = ops.get(type(node.op))
            if fn is None:
                raise ValueError("unsupported binary op")
            if isinstance(node.op, ast.Pow) and isinstance(right, (int, float)) and abs(right) > 20:
                raise ValueError("power too large")
            return fn(left, right)
        if isinstance(node, ast.BoolOp):
            values = [self.expr(v) for v in node.values]
            return all(values) if isinstance(node.op, ast.And) else any(values)
        if isinstance(node, ast.Compare):
            left = self.expr(node.left)
            comparators = [self.expr(item) for item in node.comparators]
            ops = {
                ast.Eq: operator.eq, ast.NotEq: operator.ne, ast.Lt: operator.lt, ast.LtE: operator.le,
                ast.Gt: operator.gt, ast.GtE: operator.ge, ast.In: lambda a, b: a in b,
                ast.NotIn: lambda a, b: a not in b,
            }
            current = left
            for op_node, right in zip(node.ops, comparators):
                fn = ops.get(type(op_node))
                if fn is None or not fn(current, right):
                    return False
                current = right
            return True
        if isinstance(node, ast.IfExp):
            return self.expr(node.body if self.expr(node.test) else node.orelse)
        if isinstance(node, ast.Subscript):
            value = self.expr(node.value)
            key = self.expr(node.slice)
            return value[key]
        if isinstance(node, ast.Slice):
            return slice(
                self.expr(node.lower) if node.lower else None,
                self.expr(node.upper) if node.upper else None,
                self.expr(node.step) if node.step else None,
            )
        if isinstance(node, ast.Call):
            if not isinstance(node.func, ast.Name):
                raise ValueError("unsupported call")
            name = node.func.id
            args = [self.expr(arg) for arg in node.args]
            if node.keywords:
                raise ValueError("keyword calls not supported")
            safe = {
                "len": len, "sum": sum, "min": min, "max": max, "abs": abs,
                "int": int, "float": float, "str": str, "list": list, "tuple": tuple,
                "sorted": sorted, "round": round,
            }
            if name == "range":
                result = range(*args)
                if len(result) > self.MAX_RANGE:
                    raise ValueError("range too large")
                return result
            if name == "print":
                rendered = " ".join(str(arg) for arg in args)
                self.output.append(rendered)
                return None
            if name in safe:
                return safe[name](*args)
            raise ValueError("unsafe call")
        raise ValueError(f"unsupported expression: {type(node).__name__}")

    def assign(self, target: ast.AST, value: Any) -> None:
        if isinstance(target, ast.Name):
            self.env[target.id] = value
            return
        if isinstance(target, (ast.Tuple, ast.List)):
            values = list(value)
            if len(values) != len(target.elts):
                raise ValueError("unpack mismatch")
            for sub, item in zip(target.elts, values):
                self.assign(sub, item)
            return
        if isinstance(target, ast.Subscript):
            collection = self.expr(target.value)
            key = self.expr(target.slice)
            collection[key] = value
            return
        raise ValueError("unsupported assignment")

    def stmt(self, node: ast.stmt) -> None:
        self._tick()
        if isinstance(node, ast.Assign):
            value = self.expr(node.value)
            for target in node.targets:
                self.assign(target, value)
            return
        if isinstance(node, ast.AnnAssign):
            if node.value is not None:
                self.assign(node.target, self.expr(node.value))
            return
        if isinstance(node, ast.AugAssign):
            if not isinstance(node.target, ast.Name) or node.target.id not in self.env:
                raise ValueError("unsupported augassign")
            left, right = self.env[node.target.id], self.expr(node.value)
            ops = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul, ast.Div: operator.truediv, ast.FloorDiv: operator.floordiv, ast.Mod: operator.mod}
            fn = ops.get(type(node.op))
            if fn is None:
                raise ValueError("unsupported augassign op")
            self.env[node.target.id] = fn(left, right)
            return
        if isinstance(node, ast.Expr):
            self.expr(node.value)
            return
        if isinstance(node, ast.If):
            branch = node.body if self.expr(node.test) else node.orelse
            for child in branch:
                self.stmt(child)
            return
        if isinstance(node, ast.For):
            iterable = list(self.expr(node.iter))
            if len(iterable) > self.MAX_RANGE:
                raise ValueError("loop too large")
            for item in iterable:
                self.assign(node.target, item)
                for child in node.body:
                    self.stmt(child)
            for child in node.orelse:
                self.stmt(child)
            return
        raise ValueError(f"unsupported statement: {type(node).__name__}")

    def run(self, code: str) -> tuple[list[str], dict[str, Any]]:
        tree = ast.parse(code, mode="exec")
        blocked = (ast.Import, ast.ImportFrom, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.With, ast.Try, ast.While, ast.Lambda, ast.Attribute, ast.Delete, ast.Raise, ast.Global, ast.Nonlocal)
        if any(isinstance(node, blocked) for node in ast.walk(tree)):
            raise ValueError("unsafe or unsupported syntax")
        for node in tree.body:
            self.stmt(node)
        return self.output, dict(self.env)


class CodeIntelligenceV15:
    _TRACE = re.compile(
        r"(?:خروجی|چی\s+چاپ|چه\s+چاپ|اجرا\s+شود|ردیابی|trace|output|what\s+(?:does|will).*(?:print|output)|what\s+is\s+printed|run\s+this)",
        re.I,
    )
    _CODE = re.compile(r"```|\b(?:print|for|if|def|return|while|range)\b|\w+\s*=\s*[^=]", re.I)

    @classmethod
    def matches(cls, text: str) -> bool:
        return bool(cls._TRACE.search(normalize_text(text)) and cls._CODE.search(text))

    @staticmethod
    def _extract_code(text: str) -> str:
        fenced = re.search(r"```(?:python|py)?\s*\n?(.*?)```", text, re.I | re.S)
        if fenced:
            return fenced.group(1).strip()

        # Inline Persian/English trace prompts often place the program after a
        # question mark or colon on the same line. Find the first *strong* code
        # token and slice from there, rather than feeding natural language to AST.
        strong_start = re.search(
            r"(?<![A-Za-z0-9_])(?:[A-Za-z_]\w*\s*=\s*[^=]|print\s*\(|for\s+[A-Za-z_]\w*\s+in\s+|if\s+.+?:|while\s+.+?:)",
            text, re.I | re.S,
        )
        if strong_start:
            candidate = text[strong_start.start():].strip()
            # Keep semicolon-separated one-line Python intact; ast.parse handles it.
            return candidate

        # For multiline prompts, keep only code-looking lines but preserve
        # indentation so nested blocks remain valid Python.
        lines = text.splitlines()
        first = next((i for i, line in enumerate(lines) if re.search(
            r"^\s*(?:[A-Za-z_]\w*\s*=|print\s*\(|for\s+\w+\s+in\s+|if\s+|while\s+)", line, re.I
        )), None)
        if first is not None:
            return "\n".join(lines[first:]).strip()

        colon = text.find(":")
        return text[colon + 1 :].strip() if colon >= 0 else text.strip()

    def solve(self, text: str, language: str = "fa") -> CodeTraceAnswer | None:
        if not self.matches(text):
            return None
        code = self._extract_code(text)
        # People often paste compact pseudo-Python such as
        # ``x=2; for i in range(3): x += i; print(x)``.  Python itself forbids a
        # compound statement after a semicolon, although the intent is
        # unambiguous.  Convert only the semicolon immediately before a compound
        # statement into a newline; keep semicolons inside the compound suite.
        code = re.sub(r";\s*(?=(?:for|if|while)\b)", "\n", code, flags=re.I)
        try:
            output, env = _SafePythonEvaluator().run(code)
        except (SyntaxError, ValueError, TypeError, ZeroDivisionError, OverflowError, IndexError, KeyError):
            return None
        if output:
            rendered = "\n".join(output)
            if language == "fa":
                return CodeTraceAnswer(f"خروجی کد:\n{rendered}")
            return CodeTraceAnswer(f"Code output:\n{rendered}")
        visible = {key: value for key, value in env.items() if not key.startswith("_")}
        if not visible:
            return None
        rendered = ", ".join(f"{key}={value}" for key, value in visible.items())
        if language == "fa":
            return CodeTraceAnswer(f"پس از اجرای امن مرحله‌ای: {rendered}")
        return CodeTraceAnswer(f"After safe step-by-step evaluation: {rendered}")


__all__ = ["CodeIntelligenceV15", "CodeTraceAnswer"]
