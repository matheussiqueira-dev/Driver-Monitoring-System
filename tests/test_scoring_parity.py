from __future__ import annotations

import json
import unittest
from dataclasses import fields
from pathlib import Path

from dms.config import DMSConfig
from dms.scoring_scenarios import SCENARIOS, replay_scenario, replay_session

SHARED = Path(__file__).resolve().parents[1] / "shared"


class SharedScoringConfigTests(unittest.TestCase):
    def test_python_defaults_match_shared_config(self) -> None:
        shared = json.loads((SHARED / "scoring-config.json").read_text(encoding="utf-8"))
        defaults = DMSConfig()
        config_fields = {field.name for field in fields(DMSConfig)}

        for key, value in shared.items():
            with self.subTest(key=key):
                self.assertIn(key, config_fields)
                self.assertEqual(getattr(defaults, key), value)


class ScoringScenarioTests(unittest.TestCase):
    def test_committed_scenarios_match_reference_scorer(self) -> None:
        document = json.loads((SHARED / "scoring-scenarios.json").read_text(encoding="utf-8"))
        committed = {scenario["name"]: scenario for scenario in document["scenarios"]}

        self.assertEqual(set(committed), set(SCENARIOS))
        for name, frames in SCENARIOS.items():
            with self.subTest(scenario=name):
                self.assertEqual(committed[name]["frames"], frames)
                self.assertEqual(committed[name]["expected"], replay_scenario(frames))
                self.assertEqual(committed[name]["session"], replay_session(frames))

    def test_blink_is_not_drowsiness(self) -> None:
        outputs = replay_scenario(SCENARIOS["normal_blink"])

        self.assertFalse(any(output["drowsy"] for output in outputs))
        self.assertEqual(outputs[-1]["blink_rate"], 1.0)
        self.assertEqual(outputs[-1]["label"], "Atento")

    def test_brief_glance_does_not_trigger_offroad(self) -> None:
        outputs = replay_scenario(SCENARIOS["brief_glance_away"])

        self.assertFalse(any("Olhar fora" in output["events"] for output in outputs))

    def test_microsleep_follows_drowsiness(self) -> None:
        outputs = replay_scenario(SCENARIOS["drowsy_then_microsleep"])
        first_drowsy = next(i for i, output in enumerate(outputs) if output["drowsy"])
        first_microsleep = next(i for i, output in enumerate(outputs) if output["microsleep"])

        self.assertLess(first_drowsy, first_microsleep)
        self.assertIn("Microssono", outputs[-1]["events"])


if __name__ == "__main__":
    unittest.main()
