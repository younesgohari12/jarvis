from __future__ import annotations

import importlib
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import main as entrypoint
from jarvis.brain.model import BrainLoadError
from jarvis.runtime.bootstrap import build_runtime
from jarvis.runtime.hardware import HardwareManager
from tests.helpers import TemporaryRuntime


ROOT = Path(__file__).resolve().parents[1]


class StartupTests(unittest.TestCase):
    def test_all_gui_modules_import_without_creating_a_window(self) -> None:
        expected = {
            "jarvis.gui.app": "JarvisGUI",
            "jarvis.gui.orb": "HolographicCore",
            "jarvis.gui.palette": "CommandPalette",
            "jarvis.gui.settings": "SettingsWindow",
            "jarvis.gui.monitor": "MonitorPanel",
            "jarvis.lab.app": "JarvisLab",
        }
        for module_name, symbol in expected.items():
            with self.subTest(module=module_name):
                module = importlib.import_module(module_name)
                self.assertTrue(hasattr(module, symbol))

    def test_main_constructs_and_runs_gui(self) -> None:
        with TemporaryRuntime() as runtime:
            with mock.patch.object(entrypoint, "build_runtime", return_value=runtime), mock.patch(
                "jarvis.gui.app.JarvisGUI"
            ) as gui_class:
                self.assertEqual(entrypoint.main([]), 0)
                gui_class.assert_called_once_with(runtime, auto_close_ms=None)
                gui_class.return_value.run.assert_called_once_with()

    def test_startup_order_produces_complete_runtime(self) -> None:
        with TemporaryRuntime() as runtime:
            self.assertEqual(runtime.config.version, "0.11.0")
            self.assertEqual(runtime.startup_warnings, ())
            self.assertIsNotNone(runtime.brain)
            self.assertIsNotNone(runtime.neural)
            self.assertEqual(runtime.neural.parameter_count, 23_077_376)
            self.assertGreater(runtime.knowledge.count(), 0)
            self.assertEqual(runtime.agent.current_personality, "Normal")
            self.assertTrue(runtime.memory.integrity_check())

    def test_brain_load_failure_does_not_crash_core(self) -> None:
        previous = os.environ.get("JARVIS_DATA_DIR")
        with tempfile.TemporaryDirectory(prefix="jarvis-no-brain-") as temporary:
            os.environ["JARVIS_DATA_DIR"] = temporary
            runtime = None
            try:
                with mock.patch(
                    "jarvis.runtime.bootstrap.HybridNeuralBrain.load",
                    side_effect=BrainLoadError("simulated damaged weights"),
                ):
                    runtime = build_runtime(ROOT)
                self.assertIsNone(runtime.brain)
                self.assertIn("simulated damaged weights", runtime.startup_warnings[0])
                self.assertTrue(runtime.agent.respond("سلام").text)
            finally:
                if runtime:
                    runtime.close()
                if previous is None:
                    os.environ.pop("JARVIS_DATA_DIR", None)
                else:
                    os.environ["JARVIS_DATA_DIR"] = previous

    def test_hardware_detection_always_has_cpu_fallback(self) -> None:
        hardware = HardwareManager()
        self.assertGreaterEqual(hardware.info.logical_cores, 1)
        self.assertTrue(hardware.info.cpu)
        self.assertTrue(hardware.info.platform)
        self.assertIn(hardware.recommended_performance(), {"LOW", "BALANCED", "ULTRA"})

    def test_all_performance_profiles_are_bounded(self) -> None:
        profiles = [HardwareManager.performance_profile(name) for name in ("LOW", "BALANCED", "ULTRA")]
        self.assertLess(profiles[0].active_fps, profiles[1].active_fps)
        self.assertLess(profiles[1].active_fps, profiles[2].active_fps)
        self.assertLess(profiles[0].particles, profiles[2].particles)
        self.assertTrue(all(profile.idle_fps <= profile.active_fps for profile in profiles))

    def test_runtime_dependencies_are_minimal_and_separate_from_training(self) -> None:
        requirements = (ROOT / "requirements.txt").read_text(encoding="utf-8")
        installed_lines = [
            line for line in requirements.splitlines()
            if line.strip() and not line.lstrip().startswith("#")
        ]
        self.assertEqual(installed_lines, ["-r requirements-runtime.txt"])
        runtime = (ROOT / "requirements-runtime.txt").read_text(encoding="utf-8")
        self.assertIn("numpy", runtime)
        self.assertNotIn("torch", runtime)
        training = (ROOT / "requirements-training.txt").read_text(encoding="utf-8")
        self.assertIn("torch", training)


if __name__ == "__main__":
    unittest.main()
