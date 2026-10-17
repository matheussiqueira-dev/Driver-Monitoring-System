/** Pinned to the npm package version so JS, WASM and models stay in sync. */
export const TASKS_VISION_VERSION = "1.1.1";
const WASM_URL = `https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@${TASKS_VISION_VERSION}/wasm`;

export type VisionModule = typeof import("@mediapipe/tasks-vision");
export type VisionFileset = Awaited<ReturnType<VisionModule["FilesetResolver"]["forVisionTasks"]>>;

export interface VisionRuntime {
  readonly vision: VisionModule;
  readonly fileset: VisionFileset;
}

let runtime: Promise<VisionRuntime> | null = null;

/**
 * Loads the tasks-vision module and its WASM fileset once, shared by every
 * task (face landmarks, phone detection). Dynamic import keeps MediaPipe out
 * of the initial bundle; a failed load is retried on the next call.
 */
export function loadVisionRuntime(): Promise<VisionRuntime> {
  runtime ??= (async () => {
    const vision = await import("@mediapipe/tasks-vision");
    const fileset = await vision.FilesetResolver.forVisionTasks(WASM_URL);
    return { vision, fileset };
  })().catch((error: unknown) => {
    runtime = null;
    throw error;
  });
  return runtime;
}
