from __future__ import annotations

import argparse
import dataclasses
import sys
import time
from typing import Any, Dict, Optional, Union

import cv2
import numpy as np

from dms.attention import AttentionScorer
from dms.camera import find_camera_index, list_camera_devices
from dms.config import DMSConfig
from dms.ear import EyeStateTracker
from dms.errors import DMSError, SourceUnavailableError
from dms.face_mesh import FaceMeshDetector
from dms.head_pose import HeadPoseEstimator
from dms.overlay import FaceOverlayTracker
from dms.pipeline import DMSPipeline, FrameResult, HandSource, PhoneDetector
from dms.utils import FPSCounter
from dms.visualization import (
    draw_attention_bar,
    draw_detection_boxes,
    draw_metrics,
    draw_phone_boxes,
    draw_tracked_face_overlay,
)

Source = Union[int, str]
WINDOW_TITLE = "Driver Monitoring System"
EXIT_INTERRUPTED = 130


def parse_source(value: str) -> Source:
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
    parser.add_argument("--headless", action="store_true", help="Run without opening an OpenCV window")
    parser.add_argument(
        "--detect-every", type=int, default=None, help="Run YOLO/hand detection every N frames (default 3)"
    )
    parser.add_argument("--max-frames", type=int, default=None, help="Stop after processing this number of frames")
    return parser


def build_config(args: argparse.Namespace) -> DMSConfig:
    """Apply CLI overrides through dataclasses.replace so validation runs on the final values."""
    overrides: Dict[str, Any] = {}
    if args.width:
        overrides["frame_width"] = args.width
    if args.height:
        overrides["frame_height"] = args.height
    if args.weights:
        overrides["yolo_weights"] = args.weights
    if args.device:
        overrides["yolo_device"] = args.device
    if args.detect_every is not None:
        overrides["detector_interval_frames"] = args.detect_every
    if args.no_mirror:
        overrides["mirror"] = False
    if args.no_mesh:
        overrides["show_mesh"] = False
    if args.no_face_overlay:
        overrides["show_face_overlay"] = False
    if args.no_debug_overlay:
        overrides["show_debug"] = False
    return dataclasses.replace(DMSConfig(), **overrides)


def resolve_source(args: argparse.Namespace) -> Source:
    source = parse_source(str(args.source))
    if isinstance(source, int) and args.camera_name:
        index = find_camera_index(args.camera_name)
        if index is None:
            raise SourceUnavailableError(
                f"Camera '{args.camera_name}' nao encontrada. Use --list-cameras para descobrir o indice."
            )
        print(f"[DMS] Usando camera {index}: {args.camera_name}")
        return index
    return source


def open_capture(source: Source, config: DMSConfig) -> cv2.VideoCapture:
    if isinstance(source, int) and sys.platform.startswith("win"):
        cap = cv2.VideoCapture(source, cv2.CAP_DSHOW)
    else:
        cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise SourceUnavailableError(f"Nao foi possivel abrir a fonte de video: {source}")
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.frame_width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.frame_height)
    return cap


def build_pipeline(config: DMSConfig, args: argparse.Namespace) -> DMSPipeline:
    """Composition root: optional detectors degrade gracefully when unavailable."""
    phone_detector: Optional[PhoneDetector] = None
    if not args.no_yolo:
        try:
            from dms.detection import YoloPhoneDetector

            phone_detector = YoloPhoneDetector(config)
        except Exception as exc:  # torch/ultralytics may be missing or blocked by the OS
            print(f"[DMS] YOLO desativado: {exc}")

    hand_source: Optional[HandSource] = None
    if not args.no_hands:
        try:
            from dms.detection import HandDetector

            hand_source = HandDetector(config)
        except Exception as exc:
            print(f"[DMS] Detector de maos desativado: {exc}")

    return DMSPipeline(
        face_source=FaceMeshDetector(config),
        eye_tracker=EyeStateTracker(config),
        head_pose_source=HeadPoseEstimator(config),
        overlay_source=FaceOverlayTracker(config),
        scorer=AttentionScorer(config),
        phone_detector=phone_detector,
        hand_source=hand_source,
        detector_interval=config.detector_interval_frames,
    )


class FrameClock:
    """Monotonic timestamps: video time for files, wall clock for cameras.

    MediaPipe VIDEO mode requires strictly increasing timestamps, and using the
    video's own time keeps duration-based rules (drowsiness, off-road) correct
    when a file is processed faster or slower than real time. Some containers do
    not report CAP_PROP_POS_MSEC; then the frame index and the declared FPS are
    used instead.
    """

    FALLBACK_FPS = 30.0
    MIN_STEP_S = 0.001

    def __init__(self, cap: cv2.VideoCapture, is_file: bool) -> None:
        self.cap = cap
        self.is_file = is_file
        self.start = time.monotonic()
        self.last = -1.0
        self.frames = 0
        fps = cap.get(cv2.CAP_PROP_FPS) if is_file else 0.0
        self.fps = fps if fps and fps > 0 else self.FALLBACK_FPS

    def next(self) -> float:
        if self.is_file:
            position_s = self.cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
            timestamp = position_s if position_s > self.last else self.frames / self.fps
        else:
            timestamp = time.monotonic() - self.start
        self.frames += 1
        timestamp = max(timestamp, self.last + self.MIN_STEP_S)
        self.last = timestamp
        return timestamp


def draw_frame(frame: np.ndarray, result: FrameResult, config: DMSConfig, face_mesh: Any, fps: float) -> None:
    if config.show_mesh:
        face_mesh.draw(frame)
    if config.show_face_overlay:
        draw_tracked_face_overlay(frame, result.overlay, config.show_debug)
    if result.detections is not None:
        draw_detection_boxes(frame, result.detections.all)
    if result.hand_boxes:
        draw_phone_boxes(frame, result.hand_boxes, "hand")
    draw_attention_bar(frame, result.attention.score)
    draw_metrics(frame, result.eye_state, result.head_pose, result.attention, fps)


def print_cameras() -> None:
    devices = list_camera_devices()
    if not devices:
        print("Nao foi possivel listar cameras. Instale `pygrabber` no Windows para usar --list-cameras.")
        return
    print("Cameras disponiveis:")
    for idx, name in enumerate(devices):
        print(f"{idx}: {name}")


def run(args: argparse.Namespace) -> int:
    config = build_config(args)
    if args.list_cameras:
        print_cameras()
        return 0

    source = resolve_source(args)
    cap = open_capture(source, config)
    pipeline = build_pipeline(config, args)
    clock = FrameClock(cap, is_file=isinstance(source, str))
    fps_counter = FPSCounter()
    frames_processed = 0

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if config.mirror:
                frame = cv2.flip(frame, 1)

            result = pipeline.process(frame, clock.next())
            draw_frame(frame, result, config, pipeline.face_source, fps_counter.update())

            frames_processed += 1
            if args.max_frames is not None and frames_processed >= args.max_frames:
                break
            if args.headless:
                continue

            cv2.imshow(WINDOW_TITLE, frame)
            if cv2.waitKey(1) & 0xFF in (27, ord("q")):
                break
    finally:
        cap.release()
        if not args.headless:
            cv2.destroyAllWindows()
        print(f"[DMS] Frames processados: {frames_processed}")
    return 0


def main(argv: Optional[list[str]] = None) -> int:
    args = build_arg_parser().parse_args(argv)
    try:
        return run(args)
    except DMSError as exc:
        print(f"[DMS] {exc}", file=sys.stderr)
        return exc.exit_code
    except KeyboardInterrupt:
        return EXIT_INTERRUPTED


if __name__ == "__main__":
    sys.exit(main())
