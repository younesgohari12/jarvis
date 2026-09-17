from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import tempfile
import time
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from jarvis.runtime.bootstrap import build_runtime  # noqa: E402


IGNORED_PARTS = {"__pycache__", "runtime_data", ".git", ".venv"}
IGNORED_FILES: set[str] = set()


def _project_size() -> int:
    total = 0
    for path in ROOT.rglob("*"):
        if not path.is_file() or any(part in IGNORED_PARTS for part in path.parts):
            continue
        if path.suffix in {".pyc", ".zip"} or path.name in IGNORED_FILES:
            continue
        total += path.stat().st_size
    return total


def _percentile(values: list[float], fraction: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, round((len(ordered) - 1) * fraction))
    return ordered[index]


def benchmark_core(rounds: int = 5) -> dict[str, Any]:
    previous = os.environ.get("JARVIS_DATA_DIR")
    startup_ms: list[float] = []
    result: dict[str, Any] = {}
    try:
        for _ in range(max(1, rounds)):
            with tempfile.TemporaryDirectory(prefix="jarvis-startup-bench-") as temporary:
                os.environ["JARVIS_DATA_DIR"] = temporary
                started = time.perf_counter()
                runtime = build_runtime(ROOT)
                startup_ms.append((time.perf_counter() - started) * 1000.0)
                runtime.close()

        with tempfile.TemporaryDirectory(prefix="jarvis-active-bench-") as temporary:
            os.environ["JARVIS_DATA_DIR"] = temporary
            runtime = build_runtime(ROOT)
            idle_ram = runtime.hardware.sample().app_ram_mb
            idle_wall = time.perf_counter()
            idle_cpu = time.process_time()
            time.sleep(1.5)
            idle_wall_elapsed = time.perf_counter() - idle_wall
            idle_cpu_elapsed = time.process_time() - idle_cpu

            prompts = (
                "سلام", "hello", "حالت چطوره؟", "what is your name?",
                "چیکار می تونی بکنی؟", "tell me about yourself",
                "چرا آسمون آبیه؟", "what is Python?",
            )
            latencies: list[float] = []
            active_wall = time.perf_counter()
            active_cpu = time.process_time()
            for index in range(160):
                started = time.perf_counter()
                runtime.agent.respond(prompts[index % len(prompts)])
                latencies.append((time.perf_counter() - started) * 1000.0)
            active_wall_elapsed = time.perf_counter() - active_wall
            active_cpu_elapsed = time.process_time() - active_cpu
            active_ram = runtime.hardware.sample().app_ram_mb
            result.update(
                {
                    "startup_median_ms": round(statistics.median(startup_ms), 3),
                    "startup_min_ms": round(min(startup_ms), 3),
                    "idle_core_ram_mb": round(idle_ram, 3),
                    "idle_core_process_cpu_percent": round(
                        idle_cpu_elapsed / max(idle_wall_elapsed, 1e-9) * 100.0, 4
                    ),
                    "active_ram_mb": round(active_ram, 3),
                    "active_process_cpu_percent": round(
                        active_cpu_elapsed / max(active_wall_elapsed, 1e-9) * 100.0, 3
                    ),
                    "response_median_ms": round(statistics.median(latencies), 3),
                    "response_p95_ms": round(_percentile(latencies, 0.95), 3),
                    "responses_per_second": round(160 / max(active_wall_elapsed, 1e-9), 2),
                }
            )
            runtime.close()
    finally:
        if previous is None:
            os.environ.pop("JARVIS_DATA_DIR", None)
        else:
            os.environ["JARVIS_DATA_DIR"] = previous

    metrics = json.loads(
        (ROOT / "models" / "training_metrics_v4.json").read_text(encoding="utf-8")
    )
    result.update(
        {
            "project_size_bytes": _project_size(),
            "model_size_bytes": (ROOT / "models" / "hybrid_brain_v4.jv.gz").stat().st_size,
            "model_parameters": runtime.brain.parameter_count if runtime.brain else 0,
            "neural_validation_accuracy": metrics["final_float_neural_validation_accuracy"],
            "neural_test_accuracy": metrics["final_float_neural_test_accuracy"],
            "router_validation_accuracy": metrics["router_validation_accuracy"],
            "router_test_accuracy": metrics["router_test_accuracy"],
            "router_unknown_recall": metrics["router_validation_unknown_recall"],
            "scenario_count": metrics["scenario_count"],
            "intent_accuracy": metrics["intent_accuracy"],
            "action_accuracy": metrics["action_accuracy"],
            "argument_accuracy": metrics["argument_accuracy"],
            "entity_accuracy": metrics["entity_accuracy"],
            "reference_resolution_accuracy": metrics["reference_resolution_accuracy"],
            "multi_step_accuracy": metrics["multi_step_accuracy"],
            "tool_success_rate": metrics["tool_success_rate"],
            "context_accuracy": metrics["context_accuracy"],
            "clarification_accuracy": metrics["clarification_accuracy"],
            "search_decision_accuracy": metrics["search_decision_accuracy"],
            "confirmation_decision_accuracy": metrics["confirmation_decision_accuracy"],
        }
    )
    return result


def benchmark_gui(seconds: float = 8.0) -> dict[str, Any]:
    previous = os.environ.get("JARVIS_DATA_DIR")
    runtime = None
    try:
        with tempfile.TemporaryDirectory(prefix="jarvis-gui-bench-") as temporary:
            os.environ["JARVIS_DATA_DIR"] = temporary
            runtime = build_runtime(ROOT)
            from jarvis.gui.app import JarvisGUI

            started_wall = time.perf_counter()
            started_cpu = time.process_time()
            gui = JarvisGUI(runtime, auto_close_ms=round(max(2.0, seconds) * 1000))
            gui.run()
            wall = time.perf_counter() - started_wall
            cpu = time.process_time() - started_cpu
            return {
                "gui_idle_seconds": round(wall, 3),
                "gui_idle_process_cpu_percent": round(cpu / max(wall, 1e-9) * 100.0, 3),
                "gui_profile": gui._active_performance,
            }
    finally:
        if runtime is not None and not getattr(runtime.memory, "_closed", True):
            runtime.close()
        if previous is None:
            os.environ.pop("JARVIS_DATA_DIR", None)
        else:
            os.environ["JARVIS_DATA_DIR"] = previous


def main() -> int:
    parser = argparse.ArgumentParser(description="Reproducible JARVIS v0.6 hybrid-runtime benchmark")
    parser.add_argument("--gui", action="store_true", help="also benchmark a display-backed GUI")
    parser.add_argument("--write", action="store_true", help="write models/benchmark_latest.json")
    args = parser.parse_args()
    result = benchmark_core()
    if args.gui:
        try:
            result.update(benchmark_gui())
        except Exception as exc:
            result["gui_benchmark_error"] = str(exc)
    if args.write:
        (ROOT / "models" / "benchmark_latest.json").write_text(
            json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
