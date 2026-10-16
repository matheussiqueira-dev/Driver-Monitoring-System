import { readFileSync } from "node:fs";
import { resolve } from "node:path";
import { describe, expect, it } from "vitest";
import { TASKS_VISION_VERSION } from "../../src/vision/faceLandmarker";

interface PackageManifest {
  readonly dependencies?: Readonly<Record<string, string>>;
}

describe("FaceLandmarkerService", () => {
  it("loads WASM from the same tasks-vision version that is bundled", () => {
    const manifest = JSON.parse(
      readFileSync(resolve(import.meta.dirname, "../../package.json"), "utf8"),
    ) as PackageManifest;

    // A JS/WASM version mismatch fails at runtime only, so pin them together here.
    expect(manifest.dependencies?.["@mediapipe/tasks-vision"]).toBe(TASKS_VISION_VERSION);
  });
});
