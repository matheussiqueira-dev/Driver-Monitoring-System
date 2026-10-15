import { AppError, toAppError } from "../shared/errors";

/** Owns the getUserMedia stream lifecycle for the preview element. */
export class CameraSource {
  private stream: MediaStream | null = null;

  constructor(private readonly video: HTMLVideoElement) {}

  get isActive(): boolean {
    return this.stream !== null;
  }

  async start(): Promise<void> {
    // Undefined outside secure contexts (http://, some embedded browsers) despite the DOM typings.
    const mediaDevices = navigator.mediaDevices as MediaDevices | undefined;
    if (!mediaDevices) {
      throw new AppError("CAMERA_UNSUPPORTED", "getUserMedia indisponivel neste navegador");
    }
    try {
      this.stream = await mediaDevices.getUserMedia({
        video: { facingMode: "user", width: { ideal: 1280 }, height: { ideal: 720 } },
        audio: false,
      });
      this.video.srcObject = this.stream;
      await this.video.play();
    } catch (error) {
      this.stop();
      throw toAppError(error, "CAMERA_UNAVAILABLE");
    }
  }

  stop(): void {
    this.stream?.getTracks().forEach((track) => {
      track.stop();
    });
    this.stream = null;
    this.video.pause();
    this.video.srcObject = null;
  }
}
