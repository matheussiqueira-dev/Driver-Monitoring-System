import { describe, expect, it } from "vitest";
import spatialDocument from "../../shared/spatial-cases.json";
import { isPhoneNearFace, type Corners } from "../../src/domain/spatial";

type Box = readonly number[];

interface SpatialCase {
  phone: Box;
  face: Box | null;
  frame_height: number;
  pitch: number | null;
  near: boolean;
}

const cases = (spatialDocument as { cases: SpatialCase[] }).cases;
function corners(box: Box): Corners {
  const [x1, y1, x2, y2] = box;
  if (box.length !== 4 || x1 === undefined || y1 === undefined || x2 === undefined || y2 === undefined) {
    throw new Error(`Invalid box: ${JSON.stringify(box)}`);
  }
  return { x1, y1, x2, y2 };
}

describe("isPhoneNearFace parity with dms/spatial.py", () => {
  it("covers both outcomes", () => {
    expect(cases.some((testCase) => testCase.near)).toBe(true);
    expect(cases.some((testCase) => !testCase.near)).toBe(true);
  });

  it("matches every committed case", () => {
    const mismatches = cases.filter(
      (testCase) =>
        isPhoneNearFace(
          corners(testCase.phone),
          testCase.face ? corners(testCase.face) : null,
          testCase.frame_height,
          testCase.pitch,
        ) !== testCase.near,
    );
    expect(mismatches).toEqual([]);
  });
});
