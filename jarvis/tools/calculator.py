from __future__ import annotations

import ast
import math
import operator
import re
from dataclasses import dataclass
from typing import Callable


class CalculatorError(RuntimeError):
    pass


@dataclass(frozen=True, slots=True)
class CalculationResult:
    expression: str
    value: int | float


_DIGITS = str.maketrans("۰۱۲۳۴۵۶۷۸۹٠١٢٣٤٥٦٧٨٩", "01234567890123456789")
_OPERATORS: dict[type[ast.operator], Callable[[float, float], float]] = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}
_UNARY: dict[type[ast.unaryop], Callable[[float], float]] = {
    ast.UAdd: operator.pos,
    ast.USub: operator.neg,
}


class CalculatorTool:
    _MATHISH = re.compile(r"^[\d\s\.\+\-\*/%()^]+$")

    @classmethod
    def extract_expression(cls, text: str) -> str:
        value = str(text).translate(_DIGITS).replace("×", "*").replace("÷", "/")
        value = value.replace("^", "**")
        value = re.sub(
            r"(?:حساب\s*کن|چند(?:ه|میشه|می\s*شه)?|جواب|calculate|compute|what\s+is|equals?)",
            " ",
            value,
            flags=re.I,
        )
        candidates = re.findall(r"[\d\s\.\+\-\*/%()]+", value)
        candidates = [candidate.strip() for candidate in candidates if re.search(r"\d", candidate)]
        if not candidates:
            return ""
        expression = max(candidates, key=len).strip(" =؟?")
        return expression if cls._MATHISH.fullmatch(expression) else ""

    @classmethod
    def looks_like_calculation(cls, text: str) -> bool:
        expression = cls.extract_expression(text)
        if not expression or not re.search(r"[+\-*/%]", expression):
            return False
        normalized = str(text).translate(_DIGITS)
        explicit = bool(
            re.search(
                r"(?:حساب\s*کن|چند(?:ه|میشه|می\s*شه)?|"
                r"جواب\s+(?:این\s+)?(?:حساب|محاسبه|عبارت)|جواب\s*(?=\d)|"
                r"calculate|compute|"
                r"what\s+is|equals?)",
                normalized,
                re.I,
            )
        )
        if explicit:
            return True
        remainder = normalized.replace(expression, " ", 1)
        remainder = re.sub(r"[\s=؟?!.,،:;؛()]+", "", remainder)
        return not bool(re.search(r"[A-Za-z\u0600-\u06ff]", remainder))

    def calculate(self, expression: str) -> CalculationResult:
        clean = self.extract_expression(expression) or str(expression).translate(_DIGITS).strip()
        if not clean or len(clean) > 160:
            raise CalculatorError("Invalid or overly long expression")
        try:
            tree = ast.parse(clean, mode="eval")
            value = self._evaluate(tree.body, depth=0)
        except (SyntaxError, ValueError, ZeroDivisionError, OverflowError) as exc:
            raise CalculatorError(f"Cannot calculate expression: {exc}") from exc
        if not math.isfinite(float(value)) or abs(float(value)) > 1e100:
            raise CalculatorError("Result is outside the safe numeric range")
        final: int | float = int(value) if float(value).is_integer() else round(float(value), 12)
        return CalculationResult(clean, final)

    def _evaluate(self, node: ast.AST, depth: int) -> float:
        if depth > 20:
            raise CalculatorError("Expression nesting is too deep")
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            return float(node.value)
        if isinstance(node, ast.UnaryOp) and type(node.op) in _UNARY:
            return _UNARY[type(node.op)](self._evaluate(node.operand, depth + 1))
        if isinstance(node, ast.BinOp) and type(node.op) in _OPERATORS:
            left = self._evaluate(node.left, depth + 1)
            right = self._evaluate(node.right, depth + 1)
            if isinstance(node.op, ast.Pow) and (abs(right) > 12 or abs(left) > 1e12):
                raise CalculatorError("Exponent is outside the safe range")
            return float(_OPERATORS[type(node.op)](left, right))
        raise CalculatorError("Only arithmetic operators and numbers are supported")
