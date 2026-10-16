import { afterEach, describe, expect, it, vi } from "vitest";
import { CameraSource } from "../../src/infra/camera";
import { AppError } from "../../src/shared/errors";

class FakeTrack {
  stopped = false;
  stop(): void {
    this.stopped = true;
  }
}

function fakeStream(): { stream: MediaStream; track: FakeTrack } {
  const track = new FakeTrack();
  return { stream: { getTracks: () => [track] } as unknown as MediaStream, track };
}

function fakeVideo(): HTMLVideoElement {
  return {
    srcObject: null,
    pause: vi.fn(),
    play: vi.fn(() => Promise.resolve()),
  } as unknown as HTMLVideoElement;
}

/** getUserMedia whose calls resolve only when the test says so. */
function deferredMediaDevices() {
  const pending: Array<(stream: MediaStream) => void> = [];
  const getUserMedia = vi.fn(
    () =>
      new Promise<MediaStream>((resolve) => {
        pending.push(resolve);
      }),
  );
  vi.stubGlobal("navigator", { mediaDevices: { getUserMedia } });
  return pending;
}

afterEach(() => {
  vi.unstubAllGlobals();
});

describe("CameraSource", () => {
  it("starts and stops the stream", async () => {
    const pending = deferredMediaDevices();
    const camera = new CameraSource(fakeVideo());
    const started = camera.start();
    const { stream, track } = fakeStream();
    pending[0]?.(stream);

    await expect(started).resolves.toBe(true);
    expect(camera.isActive).toBe(true);
    camera.stop();
    expect(track.stopped).toBe(true);
    expect(camera.isActive).toBe(false);
  });

  it("releases a stream that arrives after stop() (no camera left on)", async () => {
    const pending = deferredMediaDevices();
    const camera = new CameraSource(fakeVideo());
    const started = camera.start();
    camera.stop();
    const { stream, track } = fakeStream();
    pending[0]?.(stream);

    await expect(started).resolves.toBe(false);
    expect(track.stopped).toBe(true);
    expect(camera.isActive).toBe(false);
  });

  it("keeps only the newest stream when start() is called twice", async () => {
    const pending = deferredMediaDevices();
    const camera = new CameraSource(fakeVideo());
    const first = camera.start();
    const second = camera.start();
    const older = fakeStream();
    const newer = fakeStream();
    pending[1]?.(newer.stream);
    pending[0]?.(older.stream);

    await expect(second).resolves.toBe(true);
    await expect(first).resolves.toBe(false);
    expect(older.track.stopped).toBe(true);
    expect(newer.track.stopped).toBe(false);
  });

  it("maps a missing mediaDevices to CAMERA_UNSUPPORTED", async () => {
    vi.stubGlobal("navigator", {});
    await expect(new CameraSource(fakeVideo()).start()).rejects.toMatchObject({ code: "CAMERA_UNSUPPORTED" });
  });

  it("maps a permission denial to CAMERA_PERMISSION_DENIED", async () => {
    const denied = new DOMException("denied", "NotAllowedError");
    vi.stubGlobal("navigator", { mediaDevices: { getUserMedia: () => Promise.reject(denied) } });
    const error: unknown = await new CameraSource(fakeVideo()).start().catch((reason: unknown) => reason);
    expect(error).toBeInstanceOf(AppError);
    expect((error as AppError).code).toBe("CAMERA_PERMISSION_DENIED");
  });
});
