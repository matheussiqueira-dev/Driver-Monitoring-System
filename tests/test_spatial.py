from __future__ import annotations

import json
import unittest
from pathlib import Path

from dms.spatial import is_phone_near_face
from dms.spatial_cases import build_cases

SPATIAL_CASES = Path(__file__).resolve().parents[1] / "shared" / "spatial-cases.json"


class PhoneFaceSpatialTests(unittest.TestCase):
    def test_phone_near_face_is_relevant(self) -> None:
        face = (300, 120, 520, 430)
        phone = (480, 350, 580, 510)

        self.assertTrue(is_phone_near_face(phone, face, (720, 1280, 3), pitch=7.0))

    def test_far_phone_with_downward_pitch_is_filtered(self) -> None:
        face = (300, 120, 520, 430)
        phone = (1040, 590, 1160, 700)

        self.assertFalse(is_phone_near_face(phone, face, (720, 1280, 3), pitch=18.0))

    def test_missing_face_never_triggers_proximity(self) -> None:
        phone = (480, 350, 580, 510)

        self.assertFalse(is_phone_near_face(phone, None, (720, 1280, 3), pitch=18.0))


class SpatialCasesTests(unittest.TestCase):
    def test_committed_cases_match_reference(self) -> None:
        committed = json.loads(SPATIAL_CASES.read_text(encoding="utf-8"))["cases"]
        self.assertEqual(committed, build_cases())
        self.assertTrue(any(case["near"] for case in committed))
        self.assertTrue(any(not case["near"] for case in committed))


if __name__ == "__main__":
    unittest.main()
