from __future__ import annotations

from typing import Optional, Tuple


BBox = Tuple[int, int, int, int]


def is_phone_near_face(phone_bbox: BBox, face_bbox: Optional[BBox], frame_shape, pitch: Optional[float]) -> bool:
    """Return whether a detected phone is spatially relevant to the driver.

    A generic "cell phone" detection anywhere in the frame should not be enough
    to trigger the highest attention penalty. The risk is stronger when the
    object is close to the face, low in the driver's field of view, and combined
    with a downward head pose.
    """

    if face_bbox is None:
        return False

    fx1, fy1, fx2, fy2 = face_bbox
    px1, py1, px2, py2 = phone_bbox
    face_cx = (fx1 + fx2) / 2.0
    face_cy = (fy1 + fy2) / 2.0
    phone_cx = (px1 + px2) / 2.0
    phone_cy = (py1 + py2) / 2.0
    face_w = max(1.0, fx2 - fx1)
    face_h = max(1.0, fy2 - fy1)

    dx = phone_cx - face_cx
    dy = phone_cy - face_cy
    distance = (dx**2 + dy**2) ** 0.5
    face_scale = max(face_w, face_h)

    near_face = distance < face_scale * 0.9
    below_face = phone_cy > (fy2 + 0.15 * face_h) or phone_cy > frame_shape[0] * 0.58
    plausible_interaction_zone = distance < face_scale * 1.45
    looking_down = pitch is not None and pitch > 10.0

    return near_face or (below_face and plausible_interaction_zone) or (
        below_face and looking_down and distance < face_scale * 1.8
    )
