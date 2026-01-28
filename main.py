from __future__ import annotations

import argparse
import time
from typing import Optional

import cv2

from dms.attention import AttentionScorer
from dms.config import DMSConfig
from dms.detection import HandDetector, YoloPhoneDetector
from dms.ear import EyeStateTracker, LEFT_EYE_IDX, RIGHT_EYE_IDX, compute_ear
from dms.face_mesh import FaceMeshDetector
from dms.head_pose import HeadPoseEstimator
from dms.utils import FPSCounter
from dms.visualization import draw_attention_bar, draw_detection_boxes, draw_metrics, draw_phone_boxes


def parse_source(value: str):
    if value.isdigit():
        return int(value)
    return value


def is_phone_near_face(phone_bbox, face_bbox, frame_shape, pitch: Optional[float]) -> bool:
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
    dist = (dx ** 2 + dy ** 2) ** 0.5
    near = dist < max(face_w, face_h) * 0.9

    frame_h = frame_shape[0]
    below_face = phone_cy > (fy2 + 0.2 * face_h) or phone_cy > frame_h * 0.6
    looking_down = pitch is not None and pitch > 10.0
    return near or below_face or looking_down


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Driver Monitoring System (DMS)")
    parser.add_argument("--source", default="0", help="Camera index or video path")
    parser.add_argument("--weights", default=None, help="YOLO weights path")
    parser.add_argument("--device", default=None, help="YOLO device (cpu or cuda)")
    parser.add_argument("--no-yolo", action="store_true", help="Disable YOLO phone detection")
    parser.add_argument("--no-hands", action="store_true", help="Disable MediaPipe hands fallback")
    parser.add_argument("--no-mesh", action="store_true", help="Disable face mesh overlay")
    parser.add_argument("--width", type=int, default=None, help="Capture width")
    parser.add_argument("--height", type=int, default=None, help="Capture height")
    parser.add_argument("--no-mirror", action="store_true", help="Disable mirroring the camera feed")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    config = DMSConfig()
    if args.width:
        config.frame_width = args.width
    if args.height:
        config.frame_height = args.height
    if args.weights:
        config.yolo_weights = args.weights
    if args.device:
        config.yolo_device = args.device
    if args.no_mirror:
        config.mirror = False
    if args.no_mesh:
        config.show_mesh = False

    cap = cv2.VideoCapture(parse_source(str(args.source)))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.frame_width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.frame_height)

    face_mesh = FaceMeshDetector(config)
    eye_tracker = EyeStateTracker(config)
    head_pose_estimator = HeadPoseEstimator(config)
    scorer = AttentionScorer(config)
    fps_counter = FPSCounter()

    yolo_detector = None
    if not args.no_yolo:
        yolo_detector = YoloPhoneDetector(config)

    hand_detector = None
    if not args.no_hands:
        hand_detector = HandDetector(config)

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        if config.mirror:
            frame = cv2.flip(frame, 1)

        timestamp = time.time()
        timestamp_ms = int(timestamp * 1000)
        face_result = face_mesh.process(frame, timestamp_ms)

        ear = None
        head_pose = None
        if face_result.face_present:
            left_ear = compute_ear(face_result.landmarks, LEFT_EYE_IDX)
            right_ear = compute_ear(face_result.landmarks, RIGHT_EYE_IDX)
            ear = (left_ear + right_ear) / 2.0
            head_pose = head_pose_estimator.estimate(face_result.landmarks, frame.shape[:2])

        eye_state = eye_tracker.update(ear, timestamp)

        detections = None
        phone_present = False
        phone_near_face = False
        if yolo_detector is not None:
            detections = yolo_detector.detect(frame)
            if detections.phones:
                phone_present = True
                pitch = head_pose.pitch if head_pose else None
                phone_near_face = any(
                    is_phone_near_face(det.bbox, face_result.bbox, frame.shape, pitch)
                    for det in detections.phones
                )

        hand_boxes = []
        if hand_detector is not None:
            hand_boxes = hand_detector.detect(frame, timestamp_ms)

        attention_state = scorer.update(
            eye_state=eye_state,
            head_pose=head_pose,
            phone_present=phone_present,
            phone_near_face=phone_near_face,
            face_present=face_result.face_present,
            timestamp=timestamp,
        )

        if config.show_mesh:
            face_mesh.draw(frame)
        if detections is not None:
            draw_detection_boxes(frame, detections.all)
        if hand_boxes:
            draw_phone_boxes(frame, hand_boxes, "hand")

        fps = fps_counter.update()
        draw_attention_bar(frame, attention_state.score)
        draw_metrics(frame, eye_state, head_pose, attention_state, fps)

        cv2.imshow("Driver Monitoring System", frame)
        key = cv2.waitKey(1) & 0xFF
        if key in (27, ord("q")):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
