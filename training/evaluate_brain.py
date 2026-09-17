from __future__ import annotations

import json
import os
import sys
import tempfile
from pathlib import Path
from typing import Any, Iterator


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from jarvis.agent.task_context import TaskContext  # noqa: E402
from jarvis.runtime.bootstrap import build_runtime  # noqa: E402


def _read(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a JSON object in {path}")
    return payload


def _examples(path: Path) -> Iterator[tuple[str, str, str]]:
    for item in _read(path).get("intents", []):
        expected = str(item["tag"])
        route_type = str(item.get("route_type", "conversation"))
        for text in item.get("examples", []):
            yield str(text), expected, route_type


def _call_matches(actual: dict[str, Any], expected: dict[str, Any]) -> tuple[int, int]:
    if str(actual.get("tool", "")) != str(expected.get("tool", "")):
        return 0, max(1, len(expected.get("arguments", {})))
    correct = 0
    total = 0
    actual_arguments = actual.get("arguments", {})
    for key, value in expected.get("arguments", {}).items():
        total += 1
        if key == "path_nonempty":
            correct += int(bool(str(actual_arguments.get("path", "")).strip()))
        elif key == "query":
            correct += int(str(actual_arguments.get(key, "")).casefold() == str(value).casefold())
        else:
            correct += int(actual_arguments.get(key) == value)
    return correct, total


def _planned_calls(runtime: Any, text: str, route: Any, context: dict[str, Any]) -> tuple[list[dict[str, Any]], bool]:
    if route.intent != "multi_step_task":
        plan = runtime.agent.planner.plan(text, route)
        if not plan:
            return [], False
        calls = [step.tool_call.to_dict() for step in plan.steps if step.tool_call]
        return calls, bool(plan.needs_input)

    calls: list[dict[str, Any]] = []
    planning_context = dict(context)
    planning_context["_planning_chain"] = True
    for segment in route.arguments.get("segments", []):
        child = runtime.agent.router.route(str(segment), planning_context, allow_multi=False)
        plan = runtime.agent.planner.plan(str(segment), child)
        if not plan:
            continue
        if plan.needs_input:
            return calls, True
        for step in plan.steps:
            if step.tool_call:
                calls.append(step.tool_call.to_dict())
                planning_context = TaskContext.update(
                    planning_context, step.tool_call.tool, step.tool_call.arguments
                )
                planning_context["_planning_chain"] = True
    return calls, False


def evaluate() -> dict[str, Any]:
    previous = os.environ.get("JARVIS_DATA_DIR")
    with tempfile.TemporaryDirectory(prefix="jarvis-v04-eval-") as temporary:
        os.environ["JARVIS_DATA_DIR"] = temporary
        runtime = None
        try:
            runtime = build_runtime(ROOT)
            router = runtime.agent.router
            result: dict[str, Any] = {}
            for split in ("train", "validation", "test"):
                total = correct = type_correct = 0
                unknown_total = unknown_detected = predicted_unknown = 0
                confidence_total = 0.0
                errors: list[dict[str, str]] = []
                for text, expected, expected_type in _examples(
                    ROOT / "data" / "training" / f"{split}.json"
                ):
                    route = router.route(text)
                    total += 1
                    confidence_total += route.confidence
                    correct += int(route.intent == expected)
                    type_correct += int(route.route_type == expected_type)
                    if expected == "unknown":
                        unknown_total += 1
                        unknown_detected += int(route.intent == "unknown")
                    predicted_unknown += int(route.intent == "unknown")
                    if route.intent != expected and len(errors) < 20:
                        errors.append({
                            "text": text, "expected": expected,
                            "predicted": route.intent, "source": route.source,
                        })
                result[f"router_{split}_accuracy"] = round(correct / max(1, total), 6)
                result[f"router_{split}_route_type_accuracy"] = round(type_correct / max(1, total), 6)
                result[f"router_{split}_unknown_recall"] = round(unknown_detected / max(1, unknown_total), 6)
                result[f"router_{split}_unknown_precision"] = round(unknown_detected / max(1, predicted_unknown), 6)
                result[f"router_{split}_mean_confidence"] = round(confidence_total / max(1, total), 6)
                result[f"router_{split}_errors"] = errors

            counters = {
                "turns": 0, "intent": 0, "action_total": 0, "action": 0,
                "entity_total": 0, "entity": 0, "argument_total": 0, "argument": 0,
                "reference_total": 0, "reference": 0, "multi_total": 0, "multi": 0,
                "tool_total": 0, "tool": 0, "context_total": 0, "context": 0,
                "clarification_total": 0, "clarification": 0,
                "search": 0, "confirmation": 0,
            }
            scenarios = _read(runtime.config.paths.scenario_dataset).get("scenarios", [])
            for scenario in scenarios:
                if not isinstance(scenario, dict):
                    continue
                context: dict[str, Any] = {}
                for turn_index, item in enumerate(scenario.get("turns", [])):
                    text = str(item.get("text", ""))
                    expected = item.get("expected", {})
                    if not isinstance(expected, dict):
                        continue
                    route = router.route(text, context)
                    calls, clarification = _planned_calls(runtime, text, route, context)
                    expected_calls = expected.get("calls", [])
                    if not isinstance(expected_calls, list):
                        expected_calls = []
                    counters["turns"] += 1

                    expected_intent = (
                        "multi_step_task" if expected.get("multistep")
                        else str(expected.get("blocked_tool", "")) if expected.get("clarification")
                        else str(expected_calls[0].get("tool", "")) if expected_calls
                        else route.intent
                    )
                    counters["intent"] += int(route.intent == expected_intent)

                    expected_action = str(expected.get("action", ""))
                    if expected_action and expected_action != "multi":
                        counters["action_total"] += 1
                        counters["action"] += int(route.action == expected_action)
                    entity = str(expected.get("entity", ""))
                    if entity and not expected.get("multistep"):
                        counters["entity_total"] += 1
                        counters["entity"] += int(entity in route.entities)
                    if expected.get("reference"):
                        counters["reference_total"] += 1
                        counters["reference"] += int(route.referenced)
                    if expected.get("multistep"):
                        counters["multi_total"] += 1

                    tool_ok = len(calls) == len(expected_calls)
                    argument_ok = True
                    for actual, wanted in zip(calls, expected_calls):
                        correct, total = _call_matches(actual, wanted)
                        counters["argument"] += correct
                        counters["argument_total"] += total
                        argument_ok = argument_ok and correct == total
                        tool_ok = tool_ok and actual.get("tool") == wanted.get("tool")
                    counters["tool_total"] += 1
                    counters["tool"] += int(tool_ok and argument_ok)
                    if expected.get("multistep"):
                        counters["multi"] += int(tool_ok and argument_ok and len(calls) >= 2)
                    if turn_index > 0:
                        counters["context_total"] += 1
                        counters["context"] += int(tool_ok and argument_ok)
                    if expected.get("clarification"):
                        counters["clarification_total"] += 1
                        counters["clarification"] += int(clarification and not calls)
                    expected_search = any(call.get("tool") == "web_search" for call in expected_calls)
                    actual_search = any(call.get("tool") == "web_search" for call in calls)
                    counters["search"] += int(expected_search == actual_search)
                    expected_confirmation = bool(expected.get("confirmation")) or any(
                        str(call.get("tool", "")) in {
                            "delete_file", "delete_folder", "shutdown_system",
                            "restart_system", "sleep_system", "logoff_system",
                        }
                        for call in expected_calls
                    )
                    actual_confirmation = route.requires_confirmation or any(
                        bool(call.get("requires_confirmation")) for call in calls
                    )
                    counters["confirmation"] += int(
                        expected_confirmation == actual_confirmation
                    )

                    for call in calls:
                        context = TaskContext.update(
                            context, str(call["tool"]), dict(call.get("arguments", {}))
                        )

            turns = max(1, counters["turns"])
            result.update({
                "scenario_count": len(scenarios),
                "scenario_turn_count": counters["turns"],
                "intent_accuracy": round(counters["intent"] / turns, 6),
                "action_accuracy": round(counters["action"] / max(1, counters["action_total"]), 6),
                "entity_accuracy": round(counters["entity"] / max(1, counters["entity_total"]), 6),
                "argument_accuracy": round(counters["argument"] / max(1, counters["argument_total"]), 6),
                "reference_resolution_accuracy": round(counters["reference"] / max(1, counters["reference_total"]), 6),
                "multi_step_accuracy": round(counters["multi"] / max(1, counters["multi_total"]), 6),
                "tool_success_rate": round(counters["tool"] / max(1, counters["tool_total"]), 6),
                "context_accuracy": round(counters["context"] / max(1, counters["context_total"]), 6),
                "clarification_accuracy": round(counters["clarification"] / max(1, counters["clarification_total"]), 6),
                "search_decision_accuracy": round(counters["search"] / turns, 6),
                "confirmation_decision_accuracy": round(counters["confirmation"] / turns, 6),
                "unknown_precision": result.get("router_test_unknown_precision", 0.0),
                "unknown_recall": result.get("router_test_unknown_recall", 0.0),
            })

            metrics_path = ROOT / "models" / "training_metrics_v6.json"
            metrics = _read(metrics_path)
            metrics.update(result)
            metrics_path.write_text(
                json.dumps(metrics, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
            )
            return result
        finally:
            if runtime is not None:
                runtime.close()
            if previous is None:
                os.environ.pop("JARVIS_DATA_DIR", None)
            else:
                os.environ["JARVIS_DATA_DIR"] = previous


if __name__ == "__main__":
    print(json.dumps(evaluate(), ensure_ascii=False, indent=2))
