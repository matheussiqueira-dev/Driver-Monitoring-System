import type { ObjectDetector } from "@mediapipe/tasks-vision";
import { AppError } from "../shared/errors";
import type { ModelStatus } from "./faceLandmarker";
import { loadVisionRuntime } from "./visionRuntime";

/**
 * EfficientDet-Lite0 float16 (COCO, includes "cell phone"). On a small set of
 * public photos it found more phones than the int8 variant at the same speed,
 * and ran 2-3x faster than Lite2 (see MASTER_EVOLUTION_PLAN.md).
 */
const MODEL_URL =
  "https://storage.googleapis.com/mediapipe-models/object_detector/efficientdet_lite0/float16/1/efficientdet_lite0.tflite";
const PHONE_CATEGORY = "cell phone";
const SCORE_THRESHOLD = 0.3;
/**
 * Inference costs ~100+ ms on CPU, so it runs at most every 400 ms and the
 * result is reused in between (same idea as dms/pipeline.py). Phone use lasts
 * seconds, so 2.5 Hz is enough.
 */
const DEFAULT_INTERVAL_MS = 400;
/** Drop a cached result this old, e.g. after the tab was in the background. */
const MAX_RESULT_AGE_MS = 1000;

/** Phone box normalized to the source frame (0..1), before mirroring or object-fit mapping. */
export interface NormalizedBox {
  readonly x1: number;
  readonly y1: number;
  readonly x2: number;
  readonly y2: number;
  readonly score: number;
}

/**
 * Browser phone detection with MediaPipe Object Detector. Loaded lazily and
 * optional: if it fails, camera mode keeps working without phone signals.
 */
export class PhoneDetectorService {
  private detector: ObjectDetector | null = null;
  private loading: Promise<void> | null = null;
  private statusValue: ModelStatus = "idle";
  private lastBoxes: readonly NormalizedBox[] = [];
  private lastRunMs = -Infinity;
  private lastTimestampMs = -Infinity;

  constructor(private readonly intervalMs = DEFAULT_INTERVAL_MS) {}

  get status(): ModelStatus {
    return this.statusValue;
  }

  ensureReady(): Promise<void> {
    if (this.detector) {
      return Promise.resolve();
    }
    this.loading ??= this.load().catch((error: unknown) => {
      this.statusValue = "error";
      this.loading = null;
      throw new AppError("MODEL_LOAD_FAILED", "Falha ao carregar o detector de celular", { cause: error });
    });
    return this.loading;
  }

  private async load(): Promise<void> {
    this.statusValue = "loading";
    const { vision, fileset } = await loadVisionRuntime();
    const create = (delegate: "GPU" | "CPU"): Promise<ObjectDetector> =>
      vision.ObjectDetector.createFromOptions(fileset, {
        baseOptions: { modelAssetPath: MODEL_URL, delegate },
        runningMode: "VIDEO",
        categoryAllowlist: [PHONE_CATEGORY],
        scoreThreshold: SCORE_THRESHOLD,
        maxResults: 2,
      });
    try {
      this.detector = await create("GPU");
    } catch {
      this.detector = await create("CPU");
    }
    this.statusValue = "ready";
  }

  /** Phones in the current frame; runs inference at most every `intervalMs` and reuses the last result otherwise. */
  detect(video: HTMLVideoElement, timestampMs: number): readonly NormalizedBox[] {
    if (!this.detector || video.readyState < HTMLMediaElement.HAVE_CURRENT_DATA || video.videoWidth === 0) {
      return [];
    }
    // VIDEO mode requires strictly increasing timestamps per task.
    if (timestampMs - this.lastRunMs >= this.intervalMs && timestampMs > this.lastTimestampMs) {
      const result = this.detector.detectForVideo(video, timestampMs);
      this.lastTimestampMs = timestampMs;
      this.lastRunMs = timestampMs;
      const { videoWidth: width, videoHeight: height } = video;
      this.lastBoxes = result.detections.flatMap((detection) => {
        const box = detection.boundingBox;
        const score = detection.categories[0]?.score ?? 0;
        if (!box) {
          return [];
        }
        return [
          {
            x1: box.originX / width,
            y1: box.originY / height,
            x2: (box.originX + box.width) / width,
            y2: (box.originY + box.height) / height,
            score,
          },
        ];
      });
    }
    return timestampMs - this.lastRunMs > MAX_RESULT_AGE_MS ? [] : this.lastBoxes;
  }

  reset(): void {
    this.lastBoxes = [];
    this.lastRunMs = -Infinity;
  }

  close(): void {
    this.detector?.close();
    this.detector = null;
    this.loading = null;
    this.statusValue = "idle";
    this.reset();
  }
}
