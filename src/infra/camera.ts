import { AppError, toAppError } from "../shared/errors";

/** Owns the getUserMedia stream lifecycle for the preview element. */
export class CameraSource {
  private stream: MediaStream | null = null;
  /** Bumped by every start/stop, so a slow getUserMedia from an outdated request can be discarded. */
  private generation = 0;

  constructor(private readonly video: HTMLVideoElement) {}

  get isActive(): boolean {
    return this.stream !== null;
  }

  /** @returns false when a newer start/stop superseded this call (its stream is released). */
  async start(): Promise<boolean> {
    const generation = ++this.generation;
    this.releaseStream();
    // Undefined outside secure contexts (http://, some embedded browsers) despite the DOM typings.
    const mediaDevices = navigator.mediaDevices as MediaDevices | undefined;
    if (!mediaDevices) {
      throw new AppError("CAMERA_UNSUPPORTED", "getUserMedia indisponivel neste navegador");
    }

    let stream: MediaStream;
    try {
      stream = await mediaDevices.getUserMedia({
        video: { facingMode: "user", width: { ideal: 1280 }, height: { ideal: 720 } },
        audio: false,
      });
    } catch (error) {
      throw toAppError(error, "CAMERA_UNAVAILABLE");
    }

    if (generation !== this.generation) {
      stopTracks(stream);
      return false;
    }
    this.stream = stream;
    try {
      this.video.srcObject = stream;
      await this.video.play();
    } catch (error) {
      if (generation === this.generation) {
        this.stop();
      }
      throw toAppError(error, "CAMERA_UNAVAILABLE");
    }
    return generation === this.generation;
  }

  stop(): void {
    this.generation += 1;
    this.releaseStream();
  }

  private releaseStream(): void {
    if (this.stream) {
      stopTracks(this.stream);
    }
    this.stream = null;
    this.video.pause();
    this.video.srcObject = null;
  }
}

function stopTracks(stream: MediaStream): void {
  for (const track of stream.getTracks()) {
    track.stop();
  }
}
