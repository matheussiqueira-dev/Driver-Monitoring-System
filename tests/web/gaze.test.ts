import { describe, expect, it } from "vitest";
import { combineGaze, EyeGazeCalibration, gazeDeviation } from "../../src/domain/gaze";
import type { Point } from "../../src/domain/geometry";
import { EYEBALL_RADIUS_RATIO, IRIS, estimateEyeGaze, eyeRotation } from "../../src/vision/irisGaze";
import { emaAlpha, LandmarkSmoother } from "../../src/vision/landmarkFilter";

const CORNERS = [
  [33, 133],
  [362, 263],
] as const;

/** Two level 40 px wide eyes with the iris shifted by a fraction of the eyeball radius. */
function eyesWithIris(shiftX = 0, shiftY = 0): Point[] {
  const points: Point[] = Array.from({ length: 478 }, () => ({ x: 0, y: 0, z: 0 }));
  const radius = 40 * EYEBALL_RADIUS_RATIO;
  [IRIS.first, IRIS.second].forEach((iris, eye) => {
    const left = 200 + eye * 100;
    const [outer, inner] = CORNERS[eye] ?? [0, 0];
    points[outer] = { x: left, y: 150, z: 0 };
    points[inner] = { x: left + 40, y: 150, z: 0 };
    const cx = left + 20 + shiftX * radius;
    const cy = 150 + shiftY * radius;
    const ring = [
      [0, 0],
      [6, 0],
      [0, -6],
      [-6, 0],
      [0, 6],
    ] as const;
    iris.forEach((index, i) => {
      const [dx, dy] = ring[i] ?? [0, 0];
      points[index] = { x: cx + dx, y: cy + dy, z: 0 };
    });
  });
  return points;
}

describe("iris gaze", () => {
  it("is zero with the iris centered between the canthi", () => {
    const gaze = estimateEyeGaze(eyesWithIris());
    expect(gaze?.yaw).toBeCloseTo(0, 6);
    expect(gaze?.pitch).toBeCloseTo(0, 6);
  });

  it("follows the spherical eye model (+yaw right, +pitch down on the canvas)", () => {
    const gaze = estimateEyeGaze(eyesWithIris(0.5, 0.25));
    expect(gaze?.yaw).toBeCloseTo(30, 3);
    expect(gaze?.pitch).toBeCloseTo((Math.asin(0.25) * 180) / Math.PI, 3);
  });

  it("does not turn head roll into eye yaw", () => {
    const a = { x: 0, y: 0 };
    const b = { x: 40 * Math.cos(0.4), y: 40 * Math.sin(0.4) };
    const rotation = eyeRotation({ x: b.x / 2, y: b.y / 2 }, a, b);
    expect(rotation?.yaw).toBeCloseTo(0, 6);
    expect(rotation?.pitch).toBeCloseTo(0, 6);
  });

  it("returns null without refined iris points or with an implausible iris", () => {
    expect(estimateEyeGaze(eyesWithIris().slice(0, 468))).toBeNull();
    expect(estimateEyeGaze(eyesWithIris(3, 0))).toBeNull();
  });

  it("adds the eye rotation to the head pose and ignores closed eyes", () => {
    const head = { yaw: 4, pitch: 2, roll: 0 };
    expect(combineGaze(head, { yaw: 25, pitch: 1 }, 0.3, 0.21)).toEqual({ yaw: 29, pitch: 3 });
    expect(combineGaze(head, { yaw: 25, pitch: 1 }, 0.1, 0.21)).toEqual({ yaw: 4, pitch: 2 });
    expect(combineGaze(null, { yaw: 25, pitch: 1 }, 0.3, 0.21)).toBeNull();
    expect(gazeDeviation({ yaw: 10, pitch: 15 }, 20, 15)).toBeCloseTo(1, 6);
  });
});

describe("LandmarkSmoother", () => {
  it("derives alpha from the real frame interval", () => {
    expect(emaAlpha(1, 1 / (2 * Math.PI))).toBeCloseTo(1 - Math.exp(-1), 10);
  });

  it("removes static jitter", () => {
    const base = eyesWithIris();
    const smoother = new LandmarkSmoother();
    let seed = 7;
    const noise = (): number => {
      seed = (seed * 16807) % 2147483647;
      return (seed / 2147483647 - 0.5) * 2;
    };
    const xs: number[] = [];
    for (let frame = 0; frame < 90; frame += 1) {
      const noisy = base.map((point) => ({ x: point.x + noise(), y: point.y + noise(), z: 0 }));
      const out = smoother.update(noisy, (frame * 1000) / 30);
      if (frame >= 30) {
        xs.push(out[33]?.x ?? 0);
      }
    }
    const mean = xs.reduce((sum, x) => sum + x, 0) / xs.length;
    const std = Math.sqrt(xs.reduce((sum, x) => sum + (x - mean) ** 2, 0) / xs.length);
    // Uniform noise in [-1, 1] has std 0.577.
    expect(std).toBeLessThan(0.25);
  });

  it("converges in the same wall-clock time at 30 and 60 FPS when the face is slow", () => {
    const base = eyesWithIris();
    const moved = base.map((point) => ({ ...point, x: point.x + 0.5 }));
    const after = (fps: number): number => {
      const smoother = new LandmarkSmoother();
      smoother.update(base, 0);
      let x = 0;
      for (let frame = 1; frame <= fps / 5; frame += 1) {
        x = smoother.update(moved, (frame * 1000) / fps)[33]?.x ?? 0;
      }
      return x;
    };
    expect(after(30)).toBeCloseTo(after(60), 1);
  });

  it("snaps on a re-detection jump and after a long gap", () => {
    const base = eyesWithIris();
    const smoother = new LandmarkSmoother();
    smoother.update(base, 0);
    const jumped = base.map((point) => ({ ...point, x: point.x + 400 }));
    expect(smoother.update(jumped, 33)[33]?.x).toBe(600);
    expect(smoother.update(base, 2000)[33]?.x).toBe(200);
  });
});

describe("EyeGazeCalibration", () => {
  const ahead = { yaw: 0, pitch: 0, roll: 0 };

  it("removes the driver's neutral iris offset", () => {
    const calibration = new EyeGazeCalibration();
    const rest = { yaw: 1.5, pitch: -12 };
    for (let t = 0; t < 5; t += 0.1) {
      calibration.apply(rest, ahead, true, t);
    }
    const out = calibration.apply(rest, ahead, true, 5);
    expect(out?.yaw).toBeCloseTo(0, 6);
    expect(out?.pitch).toBeCloseTo(0, 6);
  });

  it("is not pulled by sustained off-road gaze or closed eyes", () => {
    const calibration = new EyeGazeCalibration();
    calibration.apply({ yaw: 0, pitch: -10 }, ahead, true, 0);
    for (let t = 0.1; t < 20; t += 0.1) {
      calibration.apply({ yaw: 0, pitch: 18 }, ahead, true, t);
      calibration.apply({ yaw: 0, pitch: 0 }, ahead, false, t);
    }
    expect(calibration.apply({ yaw: 0, pitch: 18 }, ahead, true, 20)?.pitch).toBeCloseTo(28, 6);
  });
});
