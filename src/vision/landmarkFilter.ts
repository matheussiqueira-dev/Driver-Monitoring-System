import type { Point } from "../domain/geometry";

/** Base cutoff (Hz) when the face is still: low enough to remove MediaPipe jitter. */
const MIN_CUTOFF_HZ = 1.7;
/** Extra cutoff per (inter-ocular distances / s) of speed: fast motion is followed with little lag. */
const SPEED_BETA = 4;
/** Speeds below this (inter-ocular distances / s) are landmark jitter, not motion: they do not open the filter. */
const JITTER_SPEED = 0.35;
/** Cutoff (Hz) of the speed estimate itself, so jitter does not open the filter. */
const DERIVATIVE_CUTOFF_HZ = 1;
/** A jump larger than this (in inter-ocular distances, median over points) is a re-detection: snap. */
const SNAP_DISTANCE = 1;
const MAX_GAP_S = 0.5;
const FALLBACK_DT_S = 1 / 30;
const LEFT_EYE_OUTER = 33;
const RIGHT_EYE_OUTER = 263;

/** EMA weight of a new sample for a first-order low-pass at `cutoffHz` after `dtS` seconds. */
export function emaAlpha(cutoffHz: number, dtS: number): number {
  return 1 - Math.exp(-2 * Math.PI * cutoffHz * dtS);
}

/**
 * Speed-adaptive EMA over all face landmarks (One Euro filter family).
 *
 * Alpha comes from the real time between results, so smoothing is the same at
 * 15, 30 or 60 FPS. Each point's cutoff grows with its own smoothed speed: a
 * still face is heavily filtered (no jitter) while head turns and iris
 * saccades pass through almost unfiltered (no perceptible lag). Speed is
 * normalized by the inter-ocular distance, so the behaviour does not depend on
 * face size, canvas size or devicePixelRatio (points are in CSS pixels).
 * Mirrors dms/landmark_filter.py.
 */
export class LandmarkSmoother {
  private values: Float64Array | null = null;
  private speeds: Float64Array | null = null;
  private lastTimestampMs: number | null = null;

  reset(): void {
    this.values = null;
    this.speeds = null;
    this.lastTimestampMs = null;
  }

  update(points: readonly Point[], timestampMs: number): readonly Point[] {
    const count = points.length;
    const previous = this.values;
    const speeds = this.speeds;
    const lastMs = this.lastTimestampMs;
    const rawDtS = lastMs === null ? 0 : (timestampMs - lastMs) / 1000;
    this.lastTimestampMs = timestampMs;

    if (!previous || !speeds || previous.length !== count * 3 || rawDtS > MAX_GAP_S) {
      return this.snap(points);
    }
    const dtS = rawDtS > 0 ? rawDtS : FALLBACK_DT_S;
    const scale = interOcular(points);
    if (medianDisplacement(points, previous) > SNAP_DISTANCE * scale) {
      return this.snap(points);
    }

    const derivativeAlpha = emaAlpha(DERIVATIVE_CUTOFF_HZ, dtS);
    const output: Point[] = new Array<Point>(count);
    for (let index = 0; index < count; index += 1) {
      const point = points[index] as Point;
      const offset = index * 3;
      const px = previous[offset] as number;
      const py = previous[offset + 1] as number;
      const pz = previous[offset + 2] as number;
      const z = point.z ?? 0;

      const rawSpeed = Math.hypot(point.x - px, point.y - py) / dtS / scale;
      const speed = (speeds[index] as number) + derivativeAlpha * (rawSpeed - (speeds[index] as number));
      speeds[index] = speed;
      const alpha = emaAlpha(MIN_CUTOFF_HZ + SPEED_BETA * Math.max(0, speed - JITTER_SPEED), dtS);

      const x = px + alpha * (point.x - px);
      const y = py + alpha * (point.y - py);
      const nz = pz + alpha * (z - pz);
      previous[offset] = x;
      previous[offset + 1] = y;
      previous[offset + 2] = nz;
      output[index] = { x, y, z: nz };
    }
    return output;
  }

  private snap(points: readonly Point[]): readonly Point[] {
    const values = new Float64Array(points.length * 3);
    points.forEach((point, index) => {
      values[index * 3] = point.x;
      values[index * 3 + 1] = point.y;
      values[index * 3 + 2] = point.z ?? 0;
    });
    this.values = values;
    this.speeds = new Float64Array(points.length);
    return points.map((point) => ({ x: point.x, y: point.y, z: point.z ?? 0 }));
  }
}

function interOcular(points: readonly Point[]): number {
  const a = points[LEFT_EYE_OUTER];
  const b = points[RIGHT_EYE_OUTER];
  if (a && b) {
    const span = Math.hypot(a.x - b.x, a.y - b.y);
    if (span > 1) {
      return span;
    }
  }
  let minX = Infinity;
  let maxX = -Infinity;
  for (const point of points) {
    minX = Math.min(minX, point.x);
    maxX = Math.max(maxX, point.x);
  }
  return Math.max(1, (maxX - minX) * 0.45);
}

function medianDisplacement(points: readonly Point[], previous: Float64Array): number {
  const moves = points.map((point, index) =>
    Math.hypot(point.x - (previous[index * 3] as number), point.y - (previous[index * 3 + 1] as number)),
  );
  moves.sort((a, b) => a - b);
  return moves[Math.floor(moves.length / 2)] ?? 0;
}
