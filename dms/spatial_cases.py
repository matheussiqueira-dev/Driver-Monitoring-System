"""Deterministic phone/face layouts used to check is_phone_near_face parity across runtimes."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from .spatial import is_phone_near_face

FRAME_SHAPE = (720, 1280, 3)
FACE_BBOX = (520, 120, 760, 430)
PHONE_W, PHONE_H = 100, 150
PITCHES: List[Optional[float]] = [None, 5.0, 18.0]


def build_cases() -> List[Dict[str, Any]]:
    cases: List[Dict[str, Any]] = []
    for cx in range(100, 1280, 180):
        for cy in range(80, 720, 120):
            phone = (cx - PHONE_W // 2, cy - PHONE_H // 2, cx + PHONE_W // 2, cy + PHONE_H // 2)
            for pitch in PITCHES:
                cases.append(
                    {
                        "phone": list(phone),
                        "face": list(FACE_BBOX),
                        "frame_height": FRAME_SHAPE[0],
                        "pitch": pitch,
                        "near": is_phone_near_face(phone, FACE_BBOX, FRAME_SHAPE, pitch),
                    }
                )
    cases.append(
        {"phone": [600, 300, 700, 450], "face": None, "frame_height": 720, "pitch": 18.0, "near": False}
    )
    return cases
