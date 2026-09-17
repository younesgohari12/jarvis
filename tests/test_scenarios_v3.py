from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path
from typing import Any

from jarvis.agent.intent_router import IntentRouter
from jarvis.brain.model import HybridNeuralBrain
from jarvis.config import load_config
from jarvis.entities.resolver import EntityResolver


ROOT = Path(__file__).resolve().parents[1]


class RouterScenarioTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls._temporary = tempfile.TemporaryDirectory(prefix="jarvis-v03-scenarios-")
        cls._previous = os.environ.get("JARVIS_DATA_DIR")
        os.environ["JARVIS_DATA_DIR"] = cls._temporary.name
        cls.config = load_config(ROOT)
        cls.entities = EntityResolver(cls.config.paths.websites, cls.config.paths.apps)
        cls.brain = HybridNeuralBrain.load(cls.config.paths.model, cls.config.brain)
        cls.router = IntentRouter(
            cls.brain, cls.config.paths.intent_signatures, cls.entities
        )

    @classmethod
    def tearDownClass(cls) -> None:
        if cls._previous is None:
            os.environ.pop("JARVIS_DATA_DIR", None)
        else:
            os.environ["JARVIS_DATA_DIR"] = cls._previous
        cls._temporary.cleanup()


def _load_scenarios() -> list[dict[str, Any]]:
    payload = json.loads(
        (ROOT / "data" / "training" / "scenarios_v3.json").read_text(encoding="utf-8")
    )
    return [value for value in payload["scenarios"] if isinstance(value, dict)]


def _scenario_test(scenario: dict[str, Any]):  # type: ignore[no-untyped-def]
    def test(self: RouterScenarioTests) -> None:
        text = str(scenario["text"])
        context = {"last_action": "open_url"} if any(
            token in text.casefold() for token in ("حالا", "now", "next")
        ) else {}
        route = self.router.route(text, context)
        self.assertEqual(route.intent, scenario["intent"], text)
        self.assertEqual(
            route.requires_confirmation, bool(scenario.get("confirmation", False)), text
        )
        for key, value in scenario.get("arguments", {}).items():
            self.assertEqual(route.arguments.get(key), value, f"{text}: argument {key}")
        entity = str(scenario.get("entity", ""))
        if entity:
            self.assertIn(entity, route.entities, text)
    return test


for _index, _scenario in enumerate(_load_scenarios(), 1):
    setattr(
        RouterScenarioTests,
        f"test_scenario_{_index:03d}",
        _scenario_test(_scenario),
    )


if __name__ == "__main__":
    unittest.main()
