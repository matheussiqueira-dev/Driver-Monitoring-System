import type { FaceLandmarker, FaceLandmarkerResult } from "@mediapipe/tasks-vision";
import type { Point } from "../domain/geometry";
import { AppError } from "../shared/errors";
import { loadVisionRuntime } from "./visionRuntime";

export { TASKS_VISION_VERSION } from "./visionRuntime";
const FACE_MODEL_URL =
  "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task";
const MIN_INFERENCE_INTERVAL_MS = 1000 / 30;

export type ModelStatus = "idle" | "loading" | "ready" | "error";

export interface Connection {
  readonly start: number;
  readonly end: number;
}

export interface LandmarkDetection {
  /** Normalized (0..1) landmarks of the first face, or null when no face. */
  readonly landmarks: readonly Point[] | null;
}

/**
 * Wraps MediaPipe Face Landmarker: lazy loading (the ~3 MB model only loads
 * when camera mode starts), throttled VIDEO-mode inference and status.
 */
export class FaceLandmarkerService {
  private landmarker: FaceLandmarker | null = null;
  private loading: Promise<void> | null = null;
  private lastInferenceMs = -Infinity;
  private statusValue: ModelStatus = "idle";
  private connectionList: readonly Connection[] = [];
  private delegateValue: "GPU" | "CPU" | null = null;
  frameId = 0;

  get status(): ModelStatus {
    return this.statusValue;
  }

  get delegate(): "GPU" | "CPU" | null {
    return this.delegateValue;
  }

  get connections(): readonly Connection[] {
    return this.connectionList;
  }

  ensureReady(): Promise<void> {
    if (this.landmarker) {
      return Promise.resolve();
    }
    this.loading ??= this.load().catch((error: unknown) => {
      this.statusValue = "error";
      this.loading = null;
      throw new AppError("MODEL_LOAD_FAILED", "Falha ao carregar o Face Landmarker", { cause: error });
    });
    return this.loading;
  }

  private async load(): Promise<void> {
    this.statusValue = "loading";
    const { vision, fileset } = await loadVisionRuntime();
    const FaceLandmarkerClass = vision.FaceLandmarker;
    const create = (delegate: "GPU" | "CPU"): Promise<FaceLandmarker> =>
      FaceLandmarkerClass.createFromOptions(fileset, {
        baseOptions: { modelAssetPath: FACE_MODEL_URL, delegate },
        runningMode: "VIDEO",
        numFaces: 1,
        minFaceDetectionConfidence: 0.5,
        minFacePresenceConfidence: 0.5,
        minTrackingConfidence: 0.5,
        outputFaceBlendshapes: false,
        outputFacialTransformationMatrixes: false,
      });
    try {
      this.landmarker = await create("GPU");
      this.delegateValue = "GPU";
    } catch {
      // WebGL unavailable or blocklisted (VMs, some drivers): the CPU delegate is slower but works.
      this.landmarker = await create("CPU");
      this.delegateValue = "CPU";
    }
    this.connectionList = [
      ...FaceLandmarkerClass.FACE_LANDMARKS_TESSELATION,
      ...FaceLandmarkerClass.FACE_LANDMARKS_CONTOURS,
      ...FaceLandmarkerClass.FACE_LANDMARKS_LEFT_EYE,
      ...FaceLandmarkerClass.FACE_LANDMARKS_RIGHT_EYE,
      ...FaceLandmarkerClass.FACE_LANDMARKS_LIPS,
    ];
    this.statusValue = "ready";
  }

  /** Returns null when inference was skipped (throttled, not ready or no new frame). */
  detect(video: HTMLVideoElement, timestampMs: number): LandmarkDetection | null {
    if (!this.landmarker || video.readyState < HTMLMediaElement.HAVE_CURRENT_DATA) {
      return null;
    }
    if (timestampMs - this.lastInferenceMs < MIN_INFERENCE_INTERVAL_MS) {
      return null;
    }
    const result: FaceLandmarkerResult = this.landmarker.detectForVideo(video, timestampMs);
    this.lastInferenceMs = timestampMs;
    this.frameId += 1;
    return { landmarks: result.faceLandmarks[0] ?? null };
  }

  close(): void {
    this.landmarker?.close();
    this.landmarker = null;
    this.loading = null;
    this.statusValue = "idle";
  }
}
