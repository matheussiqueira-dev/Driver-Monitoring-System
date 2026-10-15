import type { AttentionState } from "../domain/attention";
import type { FrameSignals } from "../app/signals";
import type { CoordinateMapper } from "./coordinateMapper";
import type { Connection } from "./faceLandmarker";
import type { FaceTrack } from "./faceTracker";

export interface OverlayLayers {
  mesh: boolean;
  pose: boolean;
  objects: boolean;
}

const RISK_COLORS = { ok: "#56f08d", warn: "#f7c75b", danger: "#ff5e67" } as const;
const LABEL_FONT = "12px Inter, system-ui, sans-serif";

/** Draws the face mesh, pose vector and detection boxes on the overlay canvas. */
export class OverlayRenderer {
  constructor(
    private readonly canvas: HTMLCanvasElement,
    private readonly ctx: CanvasRenderingContext2D,
    private readonly mapper: CoordinateMapper,
  ) {}

  /** Resize the backing store to the CSS size x devicePixelRatio and clear. */
  beginFrame(): void {
    const { width, height, dpr } = this.mapper.size;
    const pixelWidth = Math.round(width * dpr);
    const pixelHeight = Math.round(height * dpr);
    if (this.canvas.width !== pixelWidth || this.canvas.height !== pixelHeight) {
      this.canvas.width = pixelWidth;
      this.canvas.height = pixelHeight;
      this.canvas.style.width = `${width}px`;
      this.canvas.style.height = `${height}px`;
    }
    this.ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    this.ctx.clearRect(0, 0, width, height);
  }

  drawTrack(
    track: FaceTrack | null,
    attention: AttentionState,
    connections: readonly Connection[],
    layers: OverlayLayers,
  ): void {
    if (!track || track.opacity <= 0) {
      return;
    }
    const ctx = this.ctx;
    ctx.save();
    ctx.globalAlpha = track.opacity;

    if (layers.mesh) {
      ctx.strokeStyle = "rgba(86, 240, 141, 0.72)";
      ctx.lineWidth = 0.8;
      ctx.beginPath();
      for (const { start, end } of connections) {
        const a = track.landmarks[start];
        const b = track.landmarks[end];
        if (a && b) {
          ctx.moveTo(a.x, a.y);
          ctx.lineTo(b.x, b.y);
        }
      }
      ctx.stroke();
      ctx.fillStyle = "rgba(86, 240, 141, 0.88)";
      for (const point of track.landmarks) {
        ctx.beginPath();
        ctx.arc(point.x, point.y, 1.35, 0, Math.PI * 2);
        ctx.fill();
      }
    }

    const { box } = track;
    ctx.strokeStyle = track.status === "tracking" ? "#3488ff" : "#f7c75b";
    ctx.lineWidth = 2;
    ctx.strokeRect(box.x, box.y, box.width, box.height);
    this.drawLabel(box.x, Math.max(18, box.y - 8), `face ${track.status}`, "#3bd8ff");

    if (layers.pose) {
      this.drawPoseVector(
        track.anchor.x,
        track.anchor.y,
        track.pose.yaw,
        track.pose.pitch,
        RISK_COLORS[attention.level],
      );
    }
    ctx.restore();
  }

  /** Demo overlay: boxes in relative coordinates over the demo media, driven by the demo signals. */
  drawDemo(signals: FrameSignals, attention: AttentionState, layers: OverlayLayers): void {
    const { width, height } = this.mapper.size;
    const box = (x: number, y: number, w: number, h: number, label: string, color: string): void => {
      this.ctx.strokeStyle = color;
      this.ctx.lineWidth = 2;
      this.ctx.strokeRect(x * width, y * height, w * width, h * height);
      this.drawLabel(x * width, Math.max(18, y * height - 6), label, color);
    };

    if (signals.facePresent) {
      box(0.29, 0.13, 0.42, 0.76, "face", RISK_COLORS[attention.level]);
      if (layers.pose && signals.headPose) {
        this.drawPoseVector(
          0.5 * width,
          0.45 * height,
          signals.headPose.yaw,
          signals.headPose.pitch,
          RISK_COLORS[attention.level],
        );
      }
    }
    if (layers.objects && signals.phonePresent) {
      const near = signals.phoneNearFace;
      box(
        near ? 0.6 : 0.73,
        near ? 0.36 : 0.52,
        0.14,
        0.18,
        near ? "cell phone (perto do rosto)" : "cell phone",
        near ? RISK_COLORS.danger : RISK_COLORS.warn,
      );
    }
  }

  private drawPoseVector(x: number, y: number, yaw: number, pitch: number, color: string): void {
    const ctx = this.ctx;
    ctx.strokeStyle = color;
    ctx.lineWidth = 3;
    ctx.beginPath();
    ctx.moveTo(x, y);
    ctx.lineTo(x + yaw * 2, y + pitch * 1.6);
    ctx.stroke();
    ctx.fillStyle = color;
    ctx.beginPath();
    ctx.arc(x, y, 4, 0, Math.PI * 2);
    ctx.fill();
  }

  private drawLabel(x: number, y: number, text: string, color: string): void {
    const ctx = this.ctx;
    ctx.font = LABEL_FONT;
    const width = Math.max(48, ctx.measureText(text).width + 14);
    ctx.fillStyle = "rgba(2, 8, 6, 0.7)";
    ctx.fillRect(x, y - 16, width, 20);
    ctx.fillStyle = color;
    ctx.fillText(text, x + 7, y - 2);
  }
}
