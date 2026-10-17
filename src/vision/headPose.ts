import type { HeadPose } from "../domain/attention";
import { distance, midpoint, type Point } from "../domain/geometry";

export const LANDMARK = {
  noseTip: 1,
  chin: 152,
  forehead: 10,
  leftEyeOuter: 33,
  rightEyeOuter: 263,
} as const;

/**
 * Lightweight head pose from 2D landmarks (degrees, approximate).
 * The Python engine uses solvePnP; on the web this geometric estimate keeps
 * inference cheap and is accurate enough for threshold-based attention rules.
 */
export function estimateHeadPose(landmarks: readonly Point[]): HeadPose | null {
  const nose = landmarks[LANDMARK.noseTip];
  const chin = landmarks[LANDMARK.chin];
  const eyeA = landmarks[LANDMARK.leftEyeOuter];
  const eyeB = landmarks[LANDMARK.rightEyeOuter];
  if (!nose || !chin || !eyeA || !eyeB) {
    return null;
  }

  const [leftEye, rightEye] = eyeA.x <= eyeB.x ? [eyeA, eyeB] : [eyeB, eyeA];
  const eyeMid = midpoint(leftEye, rightEye);
  const faceWidth = Math.max(1, distance(leftEye, rightEye));
  const faceHeight = Math.max(1, distance(landmarks[LANDMARK.forehead] ?? eyeMid, chin));

  return {
    yaw: ((nose.x - eyeMid.x) / faceWidth) * 62,
    pitch: ((nose.y - eyeMid.y) / faceHeight - 0.24) * 85,
    roll: (Math.atan2(rightEye.y - leftEye.y, rightEye.x - leftEye.x) * 180) / Math.PI,
  };
}

/** Stable point between nose, eyes and chin used to anchor the pose vector. */
export function faceAnchor(landmarks: readonly Point[]): Point | null {
  const nose = landmarks[LANDMARK.noseTip];
  const chin = landmarks[LANDMARK.chin];
  const eyeA = landmarks[LANDMARK.leftEyeOuter];
  const eyeB = landmarks[LANDMARK.rightEyeOuter];
  if (!nose || !chin || !eyeA || !eyeB) {
    return null;
  }
  const eyes = midpoint(eyeA, eyeB);
  const faceMid = midpoint(eyes, chin);
  return {
    x: nose.x * 0.52 + faceMid.x * 0.28 + eyes.x * 0.2,
    y: nose.y * 0.52 + faceMid.y * 0.28 + eyes.y * 0.2,
  };
}
