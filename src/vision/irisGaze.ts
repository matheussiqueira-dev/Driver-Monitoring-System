import type { EyeGaze } from "../domain/gaze";
import type { Point } from "../domain/geometry";

/** Face Landmarker refined iris points: center first, then the 4 iris extremes. */
export const IRIS = {
  first: [468, 469, 470, 471, 472],
  second: [473, 474, 475, 476, 477],
} as const;

/** Eye corners (canthi) paired with each iris. */
const EYE_CORNERS = {
  first: [33, 133],
  second: [362, 263],
} as const;

/** Eyeball radius over palpebral fissure width (~12 mm / ~29 mm in adults). */
export const EYEBALL_RADIUS_RATIO = 0.42;

export interface IrisEye {
  readonly center: Point;
  readonly radius: number;
}

/** Iris center (mean of the 5 iris points, steadier than point 468/473 alone) and radius. */
export function irisEye(landmarks: readonly Point[], indices: readonly number[]): IrisEye | null {
  const points = indices.map((index) => landmarks[index]);
  if (points.some((point) => point === undefined)) {
    return null;
  }
  const valid = points as Point[];
  const center = {
    x: valid.reduce((sum, point) => sum + point.x, 0) / valid.length,
    y: valid.reduce((sum, point) => sum + point.y, 0) / valid.length,
  };
  const ring = valid.slice(1);
  const radius = ring.reduce((sum, point) => sum + Math.hypot(point.x - center.x, point.y - center.y), 0) / ring.length;
  return { center, radius };
}

/**
 * Eye-in-head rotation from one eye: the iris offset from the canthi midpoint,
 * measured along the eye axis (so head roll does not leak into yaw), divided
 * by the eyeball radius gives sin(angle) of a spherical eye model.
 * Returns null when the iris lands outside the eyeball (bad landmarks).
 */
export function eyeRotation(iris: Point, cornerA: Point, cornerB: Point): EyeGaze | null {
  const [left, right] = cornerA.x <= cornerB.x ? [cornerA, cornerB] : [cornerB, cornerA];
  const width = Math.hypot(right.x - left.x, right.y - left.y);
  if (width < 1) {
    return null;
  }
  const ux = (right.x - left.x) / width;
  const uy = (right.y - left.y) / width;
  const dx = iris.x - (left.x + right.x) / 2;
  const dy = iris.y - (left.y + right.y) / 2;
  const radius = width * EYEBALL_RADIUS_RATIO;
  const horizontal = (dx * ux + dy * uy) / radius;
  // Perpendicular to the eye axis, pointing down on screen when the head is level.
  const vertical = (-dx * uy + dy * ux) / radius;
  if (Math.abs(horizontal) > 1 || Math.abs(vertical) > 1) {
    return null;
  }
  return { yaw: (Math.asin(horizontal) * 180) / Math.PI, pitch: (Math.asin(vertical) * 180) / Math.PI };
}

/** Mean eye-in-head rotation over the eyes with usable iris landmarks; null without iris points. */
export function estimateEyeGaze(landmarks: readonly Point[]): EyeGaze | null {
  const rotations: EyeGaze[] = [];
  for (const side of ["first", "second"] as const) {
    const iris = irisEye(landmarks, IRIS[side]);
    const [a, b] = EYE_CORNERS[side];
    const cornerA = landmarks[a];
    const cornerB = landmarks[b];
    if (iris && cornerA && cornerB) {
      const rotation = eyeRotation(iris.center, cornerA, cornerB);
      if (rotation) {
        rotations.push(rotation);
      }
    }
  }
  if (rotations.length === 0) {
    return null;
  }
  return {
    yaw: rotations.reduce((sum, rotation) => sum + rotation.yaw, 0) / rotations.length,
    pitch: rotations.reduce((sum, rotation) => sum + rotation.pitch, 0) / rotations.length,
  };
}
