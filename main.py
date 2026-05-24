from __future__ import annotations

import argparse
import sys
import time

import cv2

from dms.attention import AttentionScorer
from dms.camera import find_camera_index, list_camera_devices
from dms.config import DMSConfig
from dms.detection import HandDetector, YoloPhoneDetector
from dms.ear import EyeStateTracker, LEFT_EYE_IDX, RIGHT_EYE_IDX, compute_ear
from dms.face_mesh import FaceMeshDetector
from dms.head_pose import HeadPoseEstimator
from dms.overlay import FaceOverlayTracker
from dms.spatial import is_phone_near_face
from dms.utils import FPSCounter
from dms.visualization import draw_attention_bar, draw_detection_boxes, draw_metrics, draw_phone_boxes, draw_tracked_face_overlay


def parse_source(value: str):
    if value.isdigit():
        return int(value)
    return value


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Driver Monitoring System (DMS)")
    parser.add_argument("--source", default="0", help="Camera index or video path")
    parser.add_argument("--camera-name", default=None, help="Camera name (ex: Brio) for auto-selection")
    parser.add_argument("--list-cameras", action="store_true", help="List camera devices and exit")
    parser.add_argument("--weights", default=None, help="YOLO weights path")
    parser.add_argument("--device", default=None, help="YOLO device (cpu or cuda)")
    parser.add_argument("--no-yolo", action="store_true", help="Disable YOLO phone detection")
    parser.add_argument("--no-hands", action="store_true", help="Disable MediaPipe hands fallback")
    parser.add_argument("--no-mesh", action="store_true", help="Disable face mesh overlay")
    parser.add_argument("--no-face-overlay", action="store_true", help="Disable tracked face overlay")
    parser.add_argument("--no-debug-overlay", action="store_true", help="Hide overlay anchor/scale debug labels")
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
    if args.no_face_overlay:
        config.show_face_overlay = False
    if args.no_debug_overlay:
        config.show_debug = False

    if args.list_cameras:
        devices = list_camera_devices()
        if devices:
            print("Cameras disponiveis:")
            for idx, name in enumerate(devices):
                print(f"{idx}: {name}")
        else:
            print("Nao foi possivel listar cameras. Instale `pygrabber` no Windows para usar --list-cameras.")
        return

    source = parse_source(str(args.source))
    if isinstance(source, int) and args.camera_name:
        index = find_camera_index(args.camera_name)
        if index is None:
            print(f"Camera '{args.camera_name}' nao encontrada. Use --list-cameras para descobrir o indice.")
            return
        source = index
        print(f"Usando camera {index}: {args.camera_name}")

    if isinstance(source, int) and sys.platform.startswith("win"):
        cap = cv2.VideoCapture(source, cv2.CAP_DSHOW)
    else:
        cap = cv2.VideoCapture(source)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.frame_width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.frame_height)

    face_mesh = FaceMeshDetector(config)
    eye_tracker = EyeStateTracker(config)
    head_pose_estimator = HeadPoseEstimator(config)
    overlay_tracker = FaceOverlayTracker(config)
    scorer = AttentionScorer(config)
    fps_counter = FPSCounter()

    yolo_detector = None
    if not args.no_yolo:
        try:
            yolo_detector = YoloPhoneDetector(config)
        except Exception as exc:
            print(f"[DMS] YOLO desativado: {exc}")

    hand_detector = None
    if not args.no_hands:
        try:
            hand_detector = HandDetector(config)
        except Exception as exc:
            print(f"[DMS] Detector de maos desativado: {exc}")

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
        overlay_transform = overlay_tracker.update(face_result.landmarks, frame.shape, head_pose)

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
        if config.show_face_overlay:
            draw_tracked_face_overlay(frame, overlay_transform, config.show_debug)
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
