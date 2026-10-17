import type { ScoringConfig } from "../domain/config";
import { AlertPolicy, type AlertTone } from "../domain/alertPolicy";
import { buildSessionReport, type SessionReport } from "../domain/report";
import { applySensitivity } from "../domain/sensitivity";
import { isPhoneNearFace, type Corners } from "../domain/spatial";
import type { TelemetryEvent, EventPayload } from "../infra/analytics";
import type { CameraSource } from "../infra/camera";
import { toAppError, userMessageFor, type AppError } from "../shared/errors";
import type { MonitorView, SourceMode } from "../ui/monitorView";
import type { CoordinateMapper, MediaSize, Viewport } from "../vision/coordinateMapper";
import type { FaceLandmarkerService } from "../vision/faceLandmarker";
import type { FaceTrack, FaceTracker } from "../vision/faceTracker";
import type { OverlayLayers, OverlayRenderer, PhoneOverlay } from "../vision/overlayRenderer";
import type { NormalizedBox, PhoneDetectorService } from "../vision/phoneDetector";
import { MonitoringPipeline, type PipelineResult } from "./pipeline";
import { demoSignals, type FrameSignals } from "./signals";

const DEMO_STEP_S = 1 / 30;
const MAX_FRAME_GAP_S = 0.25;
const NOTICE_DURATION_MS = 6000;
const CAMERA_LOADING_MESSAGE = "Iniciando camera e Face Mesh...";
const NO_FACE: FrameSignals = {
  facePresent: false,
  ear: null,
  mar: null,
  headPose: null,
  phonePresent: false,
  phoneNearFace: false,
};

export interface MonitorDependencies {
  readonly baseConfig: ScoringConfig;
  readonly view: MonitorView;
  readonly renderer: OverlayRenderer;
  readonly mapper: CoordinateMapper;
  readonly landmarker: FaceLandmarkerService;
  /** Optional in practice: camera mode keeps working if this model fails to load. */
  readonly phoneDetector: PhoneDetectorService;
  readonly tracker: FaceTracker;
  readonly camera: CameraSource;
  readonly media: { readonly demo: HTMLImageElement; readonly video: HTMLVideoElement };
  readonly track: (name: TelemetryEvent, payload?: EventPayload) => void;
  /** Audible alert sink; decides itself whether sound is enabled. */
  readonly alert: { play(tone: AlertTone): void };
  /** Called when the controller changes mode on its own (e.g. camera failure falls back to demo). */
  readonly onModeChange: (mode: SourceMode) => void;
}

/** Counts processed frames over a sliding one-second window. */
class RateMeter {
  private times: number[] = [];

  tick(nowMs: number): void {
    this.times.push(nowMs);
    while (this.times.length > 0 && nowMs - (this.times[0] ?? nowMs) > 1000) {
      this.times.shift();
    }
  }

  get rate(): number {
    return this.times.length;
  }

  reset(): void {
    this.times = [];
  }
}

/**
 * Application service for the monitor: owns mode, render loop and the wiring
 * between sources (demo signals or camera + Face Landmarker), the scoring
 * pipeline and the view. All collaborators are injected.
 */
export class MonitorController {
  private mode: SourceMode = "demo";
  private sensitivity: number;
  private readonly pipeline: MonitoringPipeline;
  private readonly layers: OverlayLayers = { mesh: true, pose: true, objects: true };
  private readonly rate = new RateMeter();
  private readonly alertPolicy = new AlertPolicy();
  private viewport: Viewport = { width: 1, height: 1, dpr: 1 };
  private lastResult: PipelineResult | null = null;
  private phones: readonly PhoneOverlay[] = [];
  private lastFrameMs: number | null = null;
  private simTimeS = 0;
  private simAccumulatorS = 0;
  private rafId: number | null = null;
  private modeRequest = 0;
  /** Message that must stay visible for a while (e.g. why the camera could not start). */
  private notice: { readonly text: string; readonly untilMs: number } | null = null;

  constructor(
    private readonly deps: MonitorDependencies,
    initialSensitivity: number,
  ) {
    this.sensitivity = initialSensitivity;
    this.pipeline = new MonitoringPipeline(applySensitivity(deps.baseConfig, initialSensitivity));
  }

  get currentMode(): SourceMode {
    return this.mode;
  }

  start(): void {
    this.rafId ??= requestAnimationFrame(this.frame);
  }

  stop(): void {
    if (this.rafId !== null) {
      cancelAnimationFrame(this.rafId);
      this.rafId = null;
    }
    this.deps.camera.stop();
  }

  setViewport(viewport: Viewport): void {
    this.viewport = viewport;
  }

  setLayer(layer: keyof OverlayLayers, enabled: boolean): void {
    this.layers[layer] = enabled;
  }

  setSensitivity(value: number): void {
    this.sensitivity = value;
    this.pipeline.reconfigure(applySensitivity(this.deps.baseConfig, value));
  }

  /** Snapshot of the current session for export. */
  buildReport(now: Date = new Date()): SessionReport {
    const { session } = this.pipeline;
    return buildSessionReport({
      generatedAt: now,
      mode: this.mode,
      sensitivity: this.sensitivity,
      config: applySensitivity(this.deps.baseConfig, this.sensitivity),
      summary: session.summary(),
      onsets: session.onsets,
      scoreHistory: session.history,
    });
  }

  /** Switch source. Resolves to the mode actually active (camera may fall back to demo). */
  async setMode(mode: SourceMode): Promise<SourceMode> {
    const request = ++this.modeRequest;
    this.mode = mode;
    this.notice = null;
    this.resetSession();

    if (mode !== "camera") {
      this.stopCamera();
      return mode;
    }

    try {
      const [started] = await Promise.all([this.deps.camera.start(), this.deps.landmarker.ensureReady()]);
      if (request !== this.modeRequest || !started) {
        // The user switched again while loading; the newer request owns the camera.
        return this.mode;
      }
      // Drop demo samples collected while the camera and model were loading.
      this.resetSession();
      this.showCamera(true);
      this.startPhoneDetector();
      this.deps.track("camera_ready", {
        width: this.deps.media.video.videoWidth,
        height: this.deps.media.video.videoHeight,
      });
      this.deps.track("face_mesh_ready", { runtime: "web", delegate: this.deps.landmarker.delegate });
      return "camera";
    } catch (error) {
      const appError = toAppError(error);
      this.deps.track(appError.code === "MODEL_LOAD_FAILED" ? "face_mesh_error" : "camera_start_failed", {
        code: appError.code,
        message: appError.message,
      });
      if (request === this.modeRequest) {
        this.fallBackToDemo(appError);
      }
      return this.mode;
    }
  }

  /** Loads in the background; a failure only disables phone signals. */
  private startPhoneDetector(): void {
    this.deps.phoneDetector.ensureReady().then(
      () => {
        this.deps.track("phone_detector_ready", { runtime: "web" });
      },
      (error: unknown) => {
        const appError = toAppError(error);
        this.deps.track("phone_detector_error", { code: appError.code, message: appError.message });
      },
    );
  }

  private fallBackToDemo(error: AppError): void {
    this.modeRequest += 1;
    this.stopCamera();
    this.mode = "demo";
    this.resetSession();
    this.notice = {
      text: `${userMessageFor(error)} Demo mantida ativa.`,
      untilMs: performance.now() + NOTICE_DURATION_MS,
    };
    this.deps.onModeChange("demo");
  }

  private activeNotice(nowMs: number): string | null {
    if (this.notice && nowMs < this.notice.untilMs) {
      return this.notice.text;
    }
    this.notice = null;
    return null;
  }

  private resetSession(): void {
    this.pipeline.reset();
    this.deps.tracker.reset();
    this.deps.phoneDetector.reset();
    this.alertPolicy.reset();
    this.phones = [];
    this.rate.reset();
    this.lastResult = null;
    this.simTimeS = 0;
    this.simAccumulatorS = 0;
  }

  private stopCamera(): void {
    this.deps.camera.stop();
    this.showCamera(false);
  }

  private showCamera(visible: boolean): void {
    const { demo, video } = this.deps.media;
    video.hidden = !visible;
    video.classList.toggle("is-mirrored", visible);
    demo.hidden = visible;
  }

  private readonly frame = (nowMs: number): void => {
    this.rafId = requestAnimationFrame(this.frame);
    const gapS = this.lastFrameMs === null ? 0 : Math.min(MAX_FRAME_GAP_S, (nowMs - this.lastFrameMs) / 1000);
    this.lastFrameMs = nowMs;

    const isCamera = this.mode === "camera" && this.deps.camera.isActive;
    const media = this.deps.media;
    const source: MediaSize = isCamera
      ? { width: media.video.videoWidth, height: media.video.videoHeight }
      : { width: media.demo.naturalWidth, height: media.demo.naturalHeight };
    this.deps.mapper.update(this.viewport, source, isCamera);
    this.deps.renderer.beginFrame();

    try {
      if (isCamera) {
        this.cameraFrame(nowMs);
      } else {
        this.demoFrame(nowMs, gapS);
      }
    } catch (error) {
      const appError = toAppError(error);
      this.deps.track("client_error", { code: appError.code, message: appError.message, source: "frame" });
      if (isCamera) {
        this.fallBackToDemo(appError);
      }
    }
  };

  private demoFrame(nowMs: number, gapS: number): void {
    const profile = this.mode === "review" ? "review" : "demo";
    this.simAccumulatorS += gapS;
    // Fixed 30 Hz simulation: blink/drowsiness rules depend on frame cadence.
    while (this.simAccumulatorS >= DEMO_STEP_S || this.lastResult === null) {
      this.simAccumulatorS = Math.max(0, this.simAccumulatorS - DEMO_STEP_S);
      this.simTimeS += DEMO_STEP_S;
      this.lastResult = this.process(demoSignals(profile, this.simTimeS), this.simTimeS);
      this.rate.tick(nowMs);
    }
    this.deps.renderer.drawDemo(this.lastResult.signals, this.lastResult.attention, this.layers);
    const status = this.mode === "camera" ? CAMERA_LOADING_MESSAGE : this.activeNotice(nowMs);
    this.render(this.lastResult, nowMs, status, null);
  }

  private cameraFrame(nowMs: number): void {
    const detection = this.deps.landmarker.detect(this.deps.media.video, nowMs);
    let track: FaceTrack | null = this.deps.tracker.current;
    if (detection) {
      const mapped = detection.landmarks?.map((landmark) => this.deps.mapper.toCanvasPoint(landmark)) ?? null;
      track = this.deps.tracker.update(mapped, nowMs);
      this.phones = this.classifyPhones(this.deps.phoneDetector.detect(this.deps.media.video, nowMs), track);
      this.lastResult = this.process(this.signalsFromTrack(track, this.phones), nowMs / 1000);
      this.rate.tick(nowMs);
    }
    const result = this.lastResult ?? this.process(NO_FACE, nowMs / 1000);
    this.lastResult = result;
    this.deps.renderer.drawTrack(track, result.attention, this.deps.landmarker.connections, this.layers, this.phones);
    const status = this.deps.landmarker.status === "ready" ? null : `Face Mesh: ${this.deps.landmarker.status}`;
    this.render(result, nowMs, status, this.debugLines(track));
  }

  /** Maps detector boxes to canvas pixels (same space as the face track) and applies the proximity rule. */
  private classifyPhones(boxes: readonly NormalizedBox[], track: FaceTrack | null): readonly PhoneOverlay[] {
    const { mapper } = this.deps;
    const face: Corners | null =
      track?.status === "tracking"
        ? { x1: track.box.x, y1: track.box.y, x2: track.box.x + track.box.width, y2: track.box.y + track.box.height }
        : null;
    return boxes.map((normalized) => {
      // Mirroring swaps the horizontal corners, so take min/max after mapping.
      const a = mapper.toCanvasPoint({ x: normalized.x1, y: normalized.y1 });
      const b = mapper.toCanvasPoint({ x: normalized.x2, y: normalized.y2 });
      const box = { x1: Math.min(a.x, b.x), y1: Math.min(a.y, b.y), x2: Math.max(a.x, b.x), y2: Math.max(a.y, b.y) };
      const near = isPhoneNearFace(box, face, mapper.size.height, track?.pose.pitch ?? null);
      return { box, score: normalized.score, near };
    });
  }

  private signalsFromTrack(track: FaceTrack | null, phones: readonly PhoneOverlay[]): FrameSignals {
    const phonePresent = phones.length > 0;
    if (!track || track.status !== "tracking") {
      return { ...NO_FACE, phonePresent };
    }
    return {
      facePresent: true,
      ear: track.ear,
      mar: track.mar,
      headPose: track.pose,
      phonePresent,
      phoneNearFace: phones.some((phone) => phone.near),
    };
  }

  private process(signals: FrameSignals, timestampS: number): PipelineResult {
    const result = this.pipeline.process(signals, timestampS);
    for (const event of result.startedEvents) {
      this.deps.track("attention_event", { event, mode: this.mode, sensitivity: this.sensitivity });
    }
    const tone = this.alertPolicy.next(result.startedEvents, performance.now());
    if (tone) {
      this.deps.alert.play(tone);
    }
    return result;
  }

  private render(
    result: PipelineResult,
    nowMs: number,
    statusMessage: string | null,
    debugLines: readonly string[] | null,
  ): void {
    this.deps.view.render(
      {
        result,
        mode: this.mode,
        fps: this.rate.rate,
        statusMessage,
        history: this.pipeline.session.history,
        onsets: this.pipeline.session.onsets,
        summary: this.pipeline.session.summary(),
        debugLines,
      },
      nowMs,
    );
  }

  private debugLines(track: FaceTrack | null): readonly string[] {
    const { video } = this.deps.media;
    const { width, height, dpr } = this.deps.mapper.size;
    return [
      `frameId: ${this.deps.landmarker.frameId}`,
      `landmarks: ${track?.rawLandmarksCount ?? 0}`,
      `video: ${video.videoWidth}x${video.videoHeight}`,
      `canvas: ${Math.round(width)}x${Math.round(height)} @${dpr.toFixed(2)}x`,
      `faceMesh: ${this.deps.landmarker.status} (${this.deps.landmarker.delegate ?? "-"})`,
      `phoneDetector: ${this.deps.phoneDetector.status} (${this.phones.length})`,
      `mirrored: ${String(this.deps.mapper.isMirrored)}`,
      `trackingFps: ${(track?.trackingFps ?? 0).toFixed(1)}`,
      `track: ${track?.status ?? "none"}`,
    ];
  }
}
