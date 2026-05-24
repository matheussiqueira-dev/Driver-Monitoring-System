from __future__ import annotations

import unittest
from types import SimpleNamespace

from dms.attention import AttentionScorer
from dms.config import DMSConfig
from dms.ear import EyeState


def eye_state(*, drowsy: bool = False, microsleep: bool = False) -> EyeState:
    return EyeState(
        ear=0.29,
        blink_rate=12.0,
        eyes_closed=drowsy or microsleep,
        drowsy=drowsy,
        microsleep=microsleep,
        closed_duration=2.2 if microsleep else 1.3 if drowsy else 0.0,
    )


class AttentionScorerTests(unittest.TestCase):
    def test_phone_far_from_face_uses_suspected_penalty(self) -> None:
        scorer = AttentionScorer(DMSConfig())

        state = scorer.update(
            eye_state=eye_state(),
            head_pose=SimpleNamespace(yaw=0.0, pitch=0.0),
            phone_present=True,
            phone_near_face=False,
            face_present=True,
            timestamp=10.0,
        )

        self.assertEqual(state.raw_score, 88.0)
        self.assertIn("Celular suspeito", state.events)

    def test_phone_near_face_uses_full_penalty(self) -> None:
        scorer = AttentionScorer(DMSConfig())

        state = scorer.update(
            eye_state=eye_state(),
            head_pose=SimpleNamespace(yaw=0.0, pitch=0.0),
            phone_present=True,
            phone_near_face=True,
            face_present=True,
            timestamp=10.0,
        )

        self.assertEqual(state.raw_score, 60.0)
        self.assertIn("Celular", state.events)

    def test_offroad_penalty_requires_duration(self) -> None:
        config = DMSConfig(offroad_time_s=1.0)
        scorer = AttentionScorer(config)
        pose = SimpleNamespace(yaw=26.0, pitch=0.0)

        first = scorer.update(
            eye_state=eye_state(),
            head_pose=pose,
            phone_present=False,
            phone_near_face=False,
            face_present=True,
            timestamp=10.0,
        )
        second = scorer.update(
            eye_state=eye_state(),
            head_pose=pose,
            phone_present=False,
            phone_near_face=False,
            face_present=True,
            timestamp=11.1,
        )

        self.assertNotIn("Olhar fora", first.events)
        self.assertIn("Olhar fora", second.events)
        self.assertEqual(second.raw_score, 80.0)


if __name__ == "__main__":
    unittest.main()
