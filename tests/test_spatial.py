from __future__ import annotations

import unittest

from dms.spatial import is_phone_near_face


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


if __name__ == "__main__":
    unittest.main()
