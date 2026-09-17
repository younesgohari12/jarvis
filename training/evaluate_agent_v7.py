from __future__ import annotations

import argparse
import json
import os
import re
import sys
import tempfile
import time
from collections import Counter, defaultdict
from datetime import UTC, datetime
from pathlib import Path, PureWindowsPath
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from jarvis.agent.permissions import PermissionLayer  # noqa: E402
from jarvis.runtime.bootstrap import build_runtime  # noqa: E402
from jarvis.utils.text import normalize_text  # noqa: E402


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, dict):
            raise ValueError(f"Expected object at {path.name}:{line_number}")
        rows.append(value)
    return rows


def _plan_tools(plan: Any) -> list[str]:
    if plan is None:
        return []
    return [step.tool_call.tool for step in plan.steps if step.tool_call is not None]


def _plan_risk(runtime: Any, tools: list[str]) -> str:
    maximum = 0
    for name in tools:
        spec = runtime.tools.spec(name)
        if spec is None:
            continue
        level = PermissionLayer.level_for(spec.effective_permission_category({}))
        if level is not None:
            maximum = max(maximum, int(level))
    return f"L{maximum}"


def _entity_name(route: Any) -> str:
    raw_path = str(route.arguments.get("path", ""))
    if not raw_path:
        return ""
    return PureWindowsPath(raw_path).name


def evaluate(dataset: Path, output: Path, *, limit: int = 0) -> dict[str, Any]:
    rows = _read_jsonl(dataset)
    if limit > 0:
        rows = rows[:limit]
    previous = os.environ.get("JARVIS_DATA_DIR")
    started = time.perf_counter()
    with tempfile.TemporaryDirectory(prefix="jarvis-v07-unseen-") as temporary:
        os.environ["JARVIS_DATA_DIR"] = str(Path(temporary) / "runtime")
        runtime = build_runtime(ROOT)
        try:
            totals: Counter[str] = Counter()
            passed: Counter[str] = Counter()
            dimension_totals: Counter[str] = Counter()
            dimension_passed: Counter[str] = Counter()
            errors: list[dict[str, Any]] = []
            response_sources: Counter[str] = Counter()
            latency: defaultdict[str, list[float]] = defaultdict(list)

            for row in rows:
                category = str(row.get("category", "other"))
                expected = dict(row.get("expected", {}))
                prompt = str(row.get("prompt", ""))
                case_started = time.perf_counter()
                route = runtime.agent.router.route(prompt)
                plan = runtime.agent.planner.plan(prompt, route)
                tools = _plan_tools(plan)
                observed: dict[str, Any] = {
                    "intent": route.intent,
                    "tools": tools,
                    "risk": _plan_risk(runtime, tools),
                    "name": _entity_name(route),
                }
                checks: dict[str, bool] = {}

                checks["intent"] = route.intent == str(expected.get("intent", ""))
                if "tool" in expected:
                    checks["tool"] = str(expected["tool"]) in tools
                if "steps" in expected:
                    checks["steps"] = tools == list(expected["steps"])
                if "name" in expected:
                    checks["name"] = observed["name"] == str(expected["name"])
                if "drive" in expected:
                    checks["drive"] = str(route.arguments.get("path", "")).upper().startswith(
                        f"{str(expected['drive']).upper()}:\\"
                    )
                if "risk" in expected:
                    checks["risk"] = observed["risk"] == str(expected["risk"])
                expected_arguments = expected.get("arguments", {})
                if isinstance(expected_arguments, dict):
                    for key, value in expected_arguments.items():
                        checks[f"argument:{key}"] = route.arguments.get(key) == value
                if expected.get("execute") is False:
                    checks["non_execution"] = plan is None or not plan.actionable

                if expected.get("must_include_any"):
                    response_started = time.perf_counter()
                    reply = runtime.agent.respond(prompt)
                    latency["response_ms"].append((time.perf_counter() - response_started) * 1000.0)
                    response_source = str(reply.data.get("reasoning_source", "agent"))
                    response_sources[response_source] += 1
                    lowered = reply.text.casefold()
                    checks["grounded_response"] = any(
                        str(token).casefold() in lowered
                        for token in expected["must_include_any"]
                    )
                    observed["response_source"] = response_source
                    observed["response_excerpt"] = reply.text[:180]

                totals[category] += 1
                for dimension, result in checks.items():
                    dimension_totals[dimension] += 1
                    dimension_passed[dimension] += int(result)
                case_passed = bool(checks) and all(checks.values())
                passed[category] += int(case_passed)
                latency["case_ms"].append((time.perf_counter() - case_started) * 1000.0)
                if not case_passed and len(errors) < 50:
                    errors.append({
                        "id": row.get("id"), "category": category, "prompt": prompt,
                        "failed": [name for name, result in checks.items() if not result],
                        "expected": expected, "observed": observed,
                    })

            category_accuracy = {
                name: round(passed[name] / max(1, total), 6)
                for name, total in sorted(totals.items())
            }
            dimension_accuracy = {
                name: round(dimension_passed[name] / max(1, total), 6)
                for name, total in sorted(dimension_totals.items())
            }
            total_passed = sum(passed.values())
            entity_total = dimension_totals["name"] + dimension_totals["drive"]
            entity_correct = dimension_passed["name"] + dimension_passed["drive"]
            entity_micro_f1 = entity_correct / entity_total if entity_total else None
            argument_dimensions = [
                name for name in dimension_totals if name.startswith("argument:")
            ]
            argument_total = sum(dimension_totals[name] for name in argument_dimensions)
            argument_correct = sum(dimension_passed[name] for name in argument_dimensions)
            argument_accuracy = argument_correct / argument_total if argument_total else None
            templates = {
                re.sub(
                    r"\b\d+\b|Atlas[_-]?\d+", "<value>",
                    normalize_text(str(row.get("prompt", ""))),
                )
                for row in rows
            }
            result: dict[str, Any] = {
                "format": "jarvis-agent-evaluation-v9",
                "dataset": str(dataset.relative_to(ROOT)),
                "dataset_cases": len(rows),
                "seen_in_training": False,
                "concept_split_policy": "benchmark-only concept groups excluded from training",
                "pretrained_source": None,
                "overall_exact_case_accuracy": round(total_passed / max(1, len(rows)), 6),
                "category_accuracy": category_accuracy,
                "dimension_accuracy": dimension_accuracy,
                "evaluation_center": {
                    "intent_accuracy": dimension_accuracy.get("intent"),
                    "entity_field_micro_f1": round(entity_micro_f1, 6) if entity_micro_f1 is not None else None,
                    "tool_selection_accuracy": dimension_accuracy.get("tool"),
                    "argument_field_accuracy": round(argument_accuracy, 6) if argument_accuracy is not None else None,
                    "plan_exact_match": dimension_accuracy.get("steps"),
                    "safety_non_execution_accuracy": dimension_accuracy.get("non_execution"),
                    "execution_success": None,
                    "recovery_success": None,
                    "conversation_quality": None,
                    "note": "Null metrics require interactive OS execution or human conversation ratings and were not fabricated.",
                },
                "category_counts": dict(sorted(totals.items())),
                "normalized_prompt_template_count": len(templates),
                "response_sources": dict(response_sources),
                "median_case_latency_ms": round(sorted(latency["case_ms"])[len(latency["case_ms"]) // 2], 3) if latency["case_ms"] else 0.0,
                "median_response_latency_ms": round(sorted(latency["response_ms"])[len(latency["response_ms"]) // 2], 3) if latency["response_ms"] else 0.0,
                "smart_brain_loaded_during_structured_evaluation": bool(runtime.neural and runtime.neural.loaded),
                "error_count": len(rows) - total_passed,
                "error_samples": errors,
                "elapsed_seconds": round(time.perf_counter() - started, 3),
                "evaluated_at_utc": datetime.now(UTC).isoformat(),
            }
        finally:
            runtime.close()
            if previous is None:
                os.environ.pop("JARVIS_DATA_DIR", None)
            else:
                os.environ["JARVIS_DATA_DIR"] = previous
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate JARVIS v0.9 on 1,000 held-out agent prompts")
    parser.add_argument("--dataset", type=Path, default=ROOT / "datasets" / "benchmarks" / "unseen_v003_1000.jsonl")
    parser.add_argument("--output", type=Path, default=ROOT / "models" / "agent_evaluation_v9.json")
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()
    result = evaluate(args.dataset, args.output, limit=max(0, args.limit))
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
