/** Axis-aligned box as corners, in the same pixel space for face and phone. */
export interface Corners {
  readonly x1: number;
  readonly y1: number;
  readonly x2: number;
  readonly y2: number;
}

/**
 * Whether a detected phone is spatially relevant to the driver. Mirrors
 * dms/spatial.py is_phone_near_face; parity is checked against
 * shared/spatial-cases.json. A phone anywhere in the frame is only "suspected";
 * it counts as in use when close to the face, or low in the field of view
 * within reach, or low while the head is pitched down.
 */
export function isPhoneNearFace(
  phone: Corners,
  face: Corners | null,
  frameHeight: number,
  pitch: number | null,
): boolean {
  if (face === null) {
    return false;
  }
  const faceCx = (face.x1 + face.x2) / 2;
  const faceCy = (face.y1 + face.y2) / 2;
  const phoneCx = (phone.x1 + phone.x2) / 2;
  const phoneCy = (phone.y1 + phone.y2) / 2;
  const faceW = Math.max(1, face.x2 - face.x1);
  const faceH = Math.max(1, face.y2 - face.y1);

  const dx = phoneCx - faceCx;
  const dy = phoneCy - faceCy;
  const distance = (dx ** 2 + dy ** 2) ** 0.5;
  const faceScale = Math.max(faceW, faceH);

  const nearFace = distance < faceScale * 0.9;
  const belowFace = phoneCy > face.y2 + 0.15 * faceH || phoneCy > frameHeight * 0.58;
  const plausibleInteractionZone = distance < faceScale * 1.45;
  const lookingDown = pitch !== null && pitch > 10;

  return (
    nearFace || (belowFace && plausibleInteractionZone) || (belowFace && lookingDown && distance < faceScale * 1.8)
  );
}
