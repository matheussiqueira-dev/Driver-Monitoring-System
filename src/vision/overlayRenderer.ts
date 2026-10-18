import type { AttentionState, RiskLevel } from "../domain/attention";
import type { FrameSignals } from "../app/signals";
import type { GazeDirection } from "../domain/gaze";
import { clamp, type Point } from "../domain/geometry";
import type { Corners } from "../domain/spatial";
import type { CoordinateMapper } from "./coordinateMapper";
import type { Connection, FaceConnections } from "./faceLandmarker";
import type { FaceTrack } from "./faceTracker";
import { IRIS, irisEye } from "./irisGaze";

export interface OverlayLayers {
  mesh: boolean;
  pose: boolean;
  objects: boolean;
}

/** A detected phone in canvas pixels, already classified against the face. */
export interface PhoneOverlay {
  readonly box: Corners;
  readonly score: number;
  readonly near: boolean;
}

type Rgb = readonly [number, number, number];

/**
 * Monochrome dark palette with two neon accents (kept in sync with
 * dms/visualization.py): teal for the tracked critical zones, acid pink for
 * alerts. Everything else is white at low alpha.
 */
export const PALETTE = {
  neon: [0, 255, 204] as Rgb,
  acid: [255, 46, 99] as Rgb,
  mesh: "rgba(255, 255, 255, 0.08)",
  frame: "rgba(255, 255, 255, 0.42)",
  text: "rgba(236, 240, 243, 0.9)",
  plate: "rgba(6, 8, 10, 0.72)",
} as const;

/** Alert mix (0 = neon, 1 = acid) targeted by each risk level. */
const LEVEL_MIX: Record<RiskLevel, number> = { ok: 0, warn: 0.5, danger: 1 };
/** Time constant of the color transition between risk levels. */
const MIX_TAU_MS = 280;
const MESH_LINE_WIDTH = 0.5;
const ZONE_LINE_WIDTH = 1;
const IRIS_GLOW_BLUR = 8;
const ZONE_GLOW_BLUR = 6;
const NOSE_TIP = 1;
/** Pose arrow length as a fraction of the face width. */
const ARROW_LENGTH = 0.85;
const LABEL_FONT = "500 10.5px ui-monospace, 'SFMono-Regular', 'Cascadia Mono', Menlo, Consolas, monospace";

function toHsl([r, g, b]: Rgb): [number, number, number] {
  const [rn, gn, bn] = [r / 255, g / 255, b / 255];
  const max = Math.max(rn, gn, bn);
  const min = Math.min(rn, gn, bn);
  const l = (max + min) / 2;
  const d = max - min;
  if (d === 0) {
    return [0, 0, l];
  }
  const s = d / (1 - Math.abs(2 * l - 1));
  const h = max === rn ? ((gn - bn) / d) % 6 : max === gn ? (bn - rn) / d + 2 : (rn - gn) / d + 4;
  return [(h * 60 + 360) % 360, s, l];
}

function fromHsl(h: number, s: number, l: number): Rgb {
  const c = (1 - Math.abs(2 * l - 1)) * s;
  const x = c * (1 - Math.abs(((h / 60) % 2) - 1));
  const m = l - c / 2;
  const [r, g, b] =
    h < 60
      ? [c, x, 0]
      : h < 120
        ? [x, c, 0]
        : h < 180
          ? [0, c, x]
          : h < 240
            ? [0, x, c]
            : h < 300
              ? [x, 0, c]
              : [c, 0, x];
  return [(r + m) * 255, (g + m) * 255, (b + m) * 255];
}

/**
 * Interpolates hue along the shortest arc (plus saturation/lightness) instead
 * of RGB: neon teal to acid pink passes through electric violet, not gray.
 */
export function mixRgb(a: Rgb, b: Rgb, t: number): Rgb {
  const k = clamp(t, 0, 1);
  const [ha, sa, la] = toHsl(a);
  const [hb, sb, lb] = toHsl(b);
  const dh = ((hb - ha + 540) % 360) - 180;
  return fromHsl((ha + dh * k + 360) % 360, sa + (sb - sa) * k, la + (lb - la) * k);
}

export function rgba(color: Rgb, alpha: number): string {
  const [r, g, b] = color.map((channel) => Math.round(channel));
  return `rgba(${r}, ${g}, ${b}, ${clamp(alpha, 0, 1).toFixed(3)})`;
}

/** Hermite smoothstep: 0 below `edge0`, 1 above `edge1`. */
function smoothstep(edge0: number, edge1: number, value: number): number {
  const t = clamp((value - edge0) / (edge1 - edge0), 0, 1);
  return t * t * (3 - 2 * t);
}

/**
 * Draws the face overlay: a faint graphite mesh, neon EAR/MAR/iris zones that
 * shift to acid pink as the risk grows, and a gradient gaze arrow whose
 * opacity follows how far the gaze is from the road.
 */
export class OverlayRenderer {
  private alertMix = 0;
  private lastMixMs: number | null = null;

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
    gaze: GazeDirection | null,
    connections: FaceConnections,
    layers: OverlayLayers,
    phones: readonly PhoneOverlay[] = [],
    nowMs = 0,
  ): void {
    const accent = this.accentColor(attention.level, nowMs);
    if (layers.objects) {
      this.drawPhones(phones);
    }
    if (!track || track.opacity <= 0) {
      return;
    }
    const ctx = this.ctx;
    ctx.save();
    ctx.globalAlpha = track.opacity;
    const points = track.landmarks;

    if (layers.mesh) {
      ctx.lineWidth = MESH_LINE_WIDTH;
      ctx.strokeStyle = PALETTE.mesh;
      this.strokeConnections(points, connections.mesh);
      this.glowStroke(points, connections.eyes, accent, ZONE_LINE_WIDTH, ZONE_GLOW_BLUR);
      this.glowStroke(points, connections.lips, accent, ZONE_LINE_WIDTH, ZONE_GLOW_BLUR);
      this.drawIrises(points, connections.irises, accent);
    }

    const { box } = track;
    this.drawBrackets(box.x, box.y, box.width, box.height);
    // Below the face: the space above it is where phone labels usually land.
    this.drawLabel(box.x, box.y + box.height + 20, `FACE · ${track.status.toUpperCase()}`, PALETTE.text);

    if (layers.pose) {
      const origin = points[NOSE_TIP] ?? track.anchor;
      const direction = gaze ?? track.pose;
      this.drawGazeArrow(origin, direction, box.width, attention.gazeDeviation, accent);
    }
    ctx.restore();
  }

  /** Demo overlay: boxes in relative coordinates over the demo media, driven by the demo signals. */
  drawDemo(
    signals: FrameSignals,
    attention: AttentionState,
    gaze: GazeDirection | null,
    layers: OverlayLayers,
    nowMs = 0,
  ): void {
    const { width, height } = this.mapper.size;
    const accent = this.accentColor(attention.level, nowMs);

    if (signals.facePresent) {
      const faceWidth = 0.42 * width;
      this.drawBrackets(0.29 * width, 0.13 * height, faceWidth, 0.76 * height);
      this.drawLabel(0.29 * width, Math.max(18, 0.13 * height - 6), "FACE · DEMO", PALETTE.text);
      if (layers.pose && gaze) {
        this.drawGazeArrow({ x: 0.5 * width, y: 0.45 * height }, gaze, faceWidth, attention.gazeDeviation, accent);
      }
    }
    if (layers.objects && signals.phonePresent) {
      const near = signals.phoneNearFace;
      this.drawPhones([
        {
          box: {
            x1: (near ? 0.6 : 0.73) * width,
            y1: (near ? 0.36 : 0.52) * height,
            x2: (near ? 0.74 : 0.87) * width,
            y2: (near ? 0.54 : 0.7) * height,
          },
          score: Number.NaN,
          near,
        },
      ]);
    }
  }

  /** Eases the accent between neon and acid with a dt-based exponential, so level changes never snap. */
  private accentColor(level: RiskLevel, nowMs: number): Rgb {
    const target = LEVEL_MIX[level];
    const dtMs = this.lastMixMs === null ? Infinity : Math.max(0, nowMs - this.lastMixMs);
    this.lastMixMs = nowMs;
    this.alertMix += (target - this.alertMix) * (1 - Math.exp(-dtMs / MIX_TAU_MS));
    return mixRgb(PALETTE.neon, PALETTE.acid, this.alertMix);
  }

  private strokeConnections(points: readonly Point[], connections: readonly Connection[]): void {
    const ctx = this.ctx;
    ctx.beginPath();
    for (const { start, end } of connections) {
      const a = points[start];
      const b = points[end];
      if (a && b) {
        ctx.moveTo(a.x, a.y);
        ctx.lineTo(b.x, b.y);
      }
    }
    ctx.stroke();
  }

  private glowStroke(
    points: readonly Point[],
    connections: readonly Connection[],
    color: Rgb,
    lineWidth: number,
    blur: number,
  ): void {
    const ctx = this.ctx;
    ctx.save();
    ctx.lineWidth = lineWidth;
    ctx.lineJoin = "round";
    ctx.strokeStyle = rgba(color, 0.95);
    ctx.shadowColor = rgba(color, 1);
    ctx.shadowBlur = blur;
    this.strokeConnections(points, connections);
    ctx.restore();
  }

  private drawIrises(points: readonly Point[], connections: readonly Connection[], color: Rgb): void {
    this.glowStroke(points, connections, color, ZONE_LINE_WIDTH, IRIS_GLOW_BLUR);
    const ctx = this.ctx;
    ctx.save();
    ctx.fillStyle = rgba(color, 1);
    ctx.shadowColor = rgba(color, 1);
    ctx.shadowBlur = IRIS_GLOW_BLUR;
    for (const indices of [IRIS.first, IRIS.second]) {
      const eye = irisEye(points, indices);
      if (eye) {
        ctx.beginPath();
        ctx.arc(eye.center.x, eye.center.y, Math.max(1, eye.radius * 0.22), 0, Math.PI * 2);
        ctx.fill();
      }
    }
    ctx.restore();
  }

  /**
   * Gaze vector projected out of the nose: a unit 3D direction (yaw, pitch)
   * seen from the camera, so a driver looking at the road gives a short arrow
   * and a sideways glance a long one. Opacity and color follow the deviation
   * from the road thresholds (VATS); past the limit it turns acid.
   */
  private drawGazeArrow(
    origin: Point,
    gaze: GazeDirection,
    faceWidth: number,
    deviation: number | null,
    accent: Rgb,
  ): void {
    const ctx = this.ctx;
    const yaw = (gaze.yaw * Math.PI) / 180;
    const pitch = (gaze.pitch * Math.PI) / 180;
    const length = Math.max(24, faceWidth * ARROW_LENGTH);
    const tip = {
      x: origin.x + Math.sin(yaw) * Math.cos(pitch) * length,
      y: origin.y + Math.sin(pitch) * length,
    };
    const level = deviation ?? 0;
    const opacity = 0.3 + 0.7 * smoothstep(0, 1, level);
    const color = mixRgb(accent, PALETTE.acid, smoothstep(0.65, 1, level));
    const shaft = Math.hypot(tip.x - origin.x, tip.y - origin.y);

    ctx.save();
    ctx.shadowColor = rgba(color, 0.9);
    ctx.shadowBlur = IRIS_GLOW_BLUR;
    ctx.fillStyle = rgba(color, opacity);
    ctx.beginPath();
    ctx.arc(origin.x, origin.y, 2.5, 0, Math.PI * 2);
    ctx.fill();

    if (shaft > 6) {
      const gradient = ctx.createLinearGradient(origin.x, origin.y, tip.x, tip.y);
      gradient.addColorStop(0, rgba(color, 0));
      gradient.addColorStop(1, rgba(color, opacity));
      ctx.strokeStyle = gradient;
      ctx.lineWidth = 2;
      ctx.lineCap = "round";
      ctx.beginPath();
      ctx.moveTo(origin.x, origin.y);
      ctx.lineTo(tip.x, tip.y);
      ctx.stroke();

      const ux = (tip.x - origin.x) / shaft;
      const uy = (tip.y - origin.y) / shaft;
      const head = Math.min(10, shaft * 0.4);
      ctx.beginPath();
      ctx.moveTo(tip.x + ux * 2, tip.y + uy * 2);
      ctx.lineTo(tip.x - ux * head - uy * head * 0.5, tip.y - uy * head + ux * head * 0.5);
      ctx.lineTo(tip.x - ux * head + uy * head * 0.5, tip.y - uy * head - ux * head * 0.5);
      ctx.closePath();
      ctx.fill();
    }
    ctx.restore();
  }

  /** Corner brackets instead of a full box: frames the face without covering it. */
  private drawBrackets(x: number, y: number, width: number, height: number): void {
    const ctx = this.ctx;
    const arm = Math.min(18, width * 0.18, height * 0.18);
    ctx.save();
    ctx.strokeStyle = PALETTE.frame;
    ctx.lineWidth = 1;
    ctx.beginPath();
    for (const [cx, cy, sx, sy] of [
      [x, y, 1, 1],
      [x + width, y, -1, 1],
      [x + width, y + height, -1, -1],
      [x, y + height, 1, -1],
    ] as const) {
      ctx.moveTo(cx + sx * arm, cy);
      ctx.lineTo(cx, cy);
      ctx.lineTo(cx, cy + sy * arm);
    }
    ctx.stroke();
    ctx.restore();
  }

  private drawPhones(phones: readonly PhoneOverlay[]): void {
    const ctx = this.ctx;
    for (const { box, score, near } of phones) {
      ctx.save();
      ctx.lineWidth = near ? 1.5 : 1;
      ctx.strokeStyle = near ? rgba(PALETTE.acid, 1) : PALETTE.frame;
      if (near) {
        ctx.shadowColor = rgba(PALETTE.acid, 1);
        ctx.shadowBlur = ZONE_GLOW_BLUR;
      }
      ctx.strokeRect(box.x1, box.y1, box.x2 - box.x1, box.y2 - box.y1);
      ctx.restore();
      const confidence = Number.isFinite(score) ? ` ${Math.round(score * 100)}%` : "";
      const label = `PHONE${confidence}${near ? " · PERTO DO ROSTO" : ""}`;
      this.drawLabel(box.x1, Math.max(18, box.y1 - 6), label, near ? rgba(PALETTE.acid, 1) : PALETTE.text);
    }
  }

  private drawLabel(x: number, y: number, text: string, color: string): void {
    const ctx = this.ctx;
    ctx.save();
    ctx.font = LABEL_FONT;
    const width = ctx.measureText(text).width + 12;
    ctx.fillStyle = PALETTE.plate;
    ctx.fillRect(x, y - 15, width, 18);
    ctx.fillStyle = color;
    ctx.fillText(text, x + 6, y - 2);
    ctx.restore();
  }
}
