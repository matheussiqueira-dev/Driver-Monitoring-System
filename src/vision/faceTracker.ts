import type { HeadPose } from "../domain/attention";
import { computeMeanEar } from "../domain/eyeState";
import { boundingBox, clamp, distance, type Box, type Point } from "../domain/geometry";
import { estimateHeadPose, faceAnchor } from "./headPose";

const DEFAULT_ALPHA = 0.58;
const FAST_ALPHA = 0.88;
const FAST_REANCHOR_DISTANCE_PX = 72;
const HOLD_FRAMES = 4;
const FADE_FRAMES = 10;
const FPS_WINDOW_MS = 1000;

export type TrackStatus = "tracking" | "holding" | "fading";

export interface FaceTrack {
  readonly landmarks: readonly Point[];
  readonly rawLandmarksCount: number;
  readonly box: Box;
  readonly anchor: Point;
  readonly ear: number | null;
  readonly pose: HeadPose;
  readonly opacity: number;
  readonly status: TrackStatus;
  readonly trackingFps: number;
}

/** EMA over landmark positions; reacts faster when the face jumps. */
export class LandmarkSmoother {
  private previous: Point[] | null = null;

  constructor(private readonly alpha = DEFAULT_ALPHA) {}

  reset(): void {
    this.previous = null;
  }

  update(points: readonly Point[]): readonly Point[] {
    const previous = this.previous;
    if (!previous || previous.length !== points.length) {
      this.previous = points.map((point) => ({ ...point }));
      return this.previous;
    }

    let maxMovement = 0;
    points.forEach((point, index) => {
      const prior = previous[index];
      if (prior) {
        maxMovement = Math.max(maxMovement, distance(point, prior));
      }
    });

    const alpha = maxMovement > FAST_REANCHOR_DISTANCE_PX ? FAST_ALPHA : this.alpha;
    this.previous = points.map((point, index) => {
      const prior = previous[index] ?? point;
      return {
        x: alpha * point.x + (1 - alpha) * prior.x,
        y: alpha * point.y + (1 - alpha) * prior.y,
        z: alpha * (point.z ?? 0) + (1 - alpha) * (prior.z ?? 0),
      };
    });
    return this.previous;
  }
}

/**
 * Turns per-frame landmark results into a stable face track: smooths points,
 * derives EAR/pose, and holds then fades the overlay when the face is lost.
 */
export class FaceTracker {
  private readonly smoother = new LandmarkSmoother();
  private track: FaceTrack | null = null;
  private missedFrames = 0;
  private resultTimes: number[] = [];

  get current(): FaceTrack | null {
    return this.track;
  }

  reset(): void {
    this.smoother.reset();
    this.track = null;
    this.missedFrames = 0;
    this.resultTimes = [];
  }

  /** @param canvasLandmarks landmarks already mapped to canvas pixels */
  update(canvasLandmarks: readonly Point[] | null, timestampMs: number): FaceTrack | null {
    if (!canvasLandmarks || canvasLandmarks.length === 0) {
      return this.markMissing();
    }

    const landmarks = this.smoother.update(canvasLandmarks);
    const box = boundingBox(landmarks);
    const anchor = faceAnchor(landmarks);
    const pose = estimateHeadPose(landmarks);
    if (!box || !anchor || !pose) {
      return this.markMissing();
    }

    this.resultTimes.push(timestampMs);
    while (this.resultTimes.length > 2 && timestampMs - (this.resultTimes[0] ?? timestampMs) > FPS_WINDOW_MS) {
      this.resultTimes.shift();
    }
    const first = this.resultTimes[0] ?? timestampMs;
    const span = timestampMs - first;
    const trackingFps = this.resultTimes.length > 1 && span > 0 ? ((this.resultTimes.length - 1) * 1000) / span : 0;

    this.missedFrames = 0;
    this.track = {
      landmarks,
      rawLandmarksCount: canvasLandmarks.length,
      box,
      anchor,
      ear: computeMeanEar(landmarks),
      pose,
      opacity: 1,
      status: "tracking",
      trackingFps,
    };
    return this.track;
  }

  markMissing(): FaceTrack | null {
    this.missedFrames += 1;
    if (!this.track) {
      return null;
    }
    if (this.missedFrames <= HOLD_FRAMES) {
      this.track = { ...this.track, opacity: 0.85, status: "holding" };
      return this.track;
    }
    if (this.missedFrames <= HOLD_FRAMES + FADE_FRAMES) {
      const fade = 1 - (this.missedFrames - HOLD_FRAMES) / FADE_FRAMES;
      this.track = { ...this.track, opacity: clamp(fade, 0, 1), status: "fading" };
      return this.track;
    }
    this.reset();
    return null;
  }
}
