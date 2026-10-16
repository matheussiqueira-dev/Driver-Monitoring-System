import "./styles.css";
import { MonitorController } from "./app/monitorController";
import { DEFAULT_SCORING_CONFIG } from "./domain/config";
import { initAnalytics, trackEvent } from "./infra/analytics";
import { CameraSource } from "./infra/camera";
import { registerServiceWorker } from "./infra/pwa";
import { toAppError, userMessageFor } from "./shared/errors";
import { requireAll, requireElement } from "./ui/dom";
import { MonitorView, type SourceMode } from "./ui/monitorView";
import { CoordinateMapper } from "./vision/coordinateMapper";
import { FaceLandmarkerService } from "./vision/faceLandmarker";
import { FaceTracker } from "./vision/faceTracker";
import { OverlayRenderer, type OverlayLayers } from "./vision/overlayRenderer";

const SOURCE_MODES: readonly SourceMode[] = ["demo", "camera", "review"];
const LAYERS: readonly (keyof OverlayLayers)[] = ["mesh", "pose", "objects"];

function isSourceMode(value: string | undefined): value is SourceMode {
  return SOURCE_MODES.includes(value as SourceMode);
}

function isLayer(value: string | undefined): value is keyof OverlayLayers {
  return LAYERS.includes(value as keyof OverlayLayers);
}

/** Composition root: builds the object graph and binds DOM controls. */
function bootstrap(): void {
  initAnalytics();
  registerServiceWorker();

  const shell = requireElement(".video-shell", HTMLElement);
  const canvas = requireElement("#overlayCanvas", HTMLCanvasElement);
  const video = requireElement("#cameraPreview", HTMLVideoElement);
  const demo = requireElement(".video-shell img", HTMLImageElement);
  const sensitivity = requireElement("#sensitivityRange", HTMLInputElement);
  const context = canvas.getContext("2d");
  if (!context) {
    throw new Error("Canvas 2D indisponivel");
  }

  const modeButtons = requireAll(".source-button", HTMLButtonElement);
  const markActive = (mode: SourceMode): void => {
    for (const button of modeButtons) {
      const active = button.dataset.mode === mode;
      button.classList.toggle("is-active", active);
      button.setAttribute("aria-pressed", String(active));
    }
  };

  const mapper = new CoordinateMapper();
  const view = new MonitorView();
  const controller = new MonitorController(
    {
      baseConfig: DEFAULT_SCORING_CONFIG,
      view,
      renderer: new OverlayRenderer(canvas, context, mapper),
      mapper,
      landmarker: new FaceLandmarkerService(),
      tracker: new FaceTracker(),
      camera: new CameraSource(video),
      media: { demo, video },
      track: trackEvent,
      onModeChange: markActive,
    },
    Number(sensitivity.value),
  );

  const syncViewport = (): void => {
    const rect = shell.getBoundingClientRect();
    controller.setViewport({ width: rect.width, height: rect.height, dpr: window.devicePixelRatio || 1 });
  };
  new ResizeObserver(syncViewport).observe(shell);
  syncViewport();

  for (const button of modeButtons) {
    button.addEventListener("click", () => {
      const mode = button.dataset.mode;
      if (!isSourceMode(mode) || mode === controller.currentMode) {
        return;
      }
      markActive(mode);
      trackEvent("source_mode_changed", { mode });
      void controller.setMode(mode).then(markActive);
    });
  }

  for (const checkbox of requireAll("[data-layer]", HTMLInputElement)) {
    checkbox.addEventListener("change", () => {
      const layer = checkbox.dataset.layer;
      if (isLayer(layer)) {
        controller.setLayer(layer, checkbox.checked);
        trackEvent("overlay_layer_toggled", { layer, enabled: checkbox.checked });
      }
    });
  }

  sensitivity.addEventListener("change", () => {
    controller.setSensitivity(Number(sensitivity.value));
    trackEvent("sensitivity_changed", { value: Number(sensitivity.value) });
  });

  window.addEventListener("pagehide", () => {
    controller.stop();
  });
  // Restored from the back/forward cache: the loop and the camera were stopped on pagehide.
  window.addEventListener("pageshow", (event) => {
    if (!event.persisted) {
      return;
    }
    if (controller.currentMode === "camera") {
      void controller.setMode("demo").then(markActive);
    }
    controller.start();
  });
  controller.start();
}

try {
  bootstrap();
} catch (error) {
  const appError = toAppError(error);
  trackEvent("client_error", { code: appError.code, message: appError.message, source: "bootstrap" });
  const strip = document.querySelector("#alertStrip");
  if (strip) {
    strip.textContent = userMessageFor(appError);
  }
  console.error(appError);
}
