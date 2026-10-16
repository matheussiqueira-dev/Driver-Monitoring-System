type Primitive = string | number | boolean | null;
export type EventPayload = Readonly<Record<string, Primitive | undefined>>;

/** Every custom event the app may emit; a typo becomes a compile error. */
export type TelemetryEvent =
  | "source_mode_changed"
  | "overlay_layer_toggled"
  | "sensitivity_changed"
  | "camera_ready"
  | "camera_start_failed"
  | "face_mesh_ready"
  | "face_mesh_error"
  | "attention_event"
  | "report_exported"
  | "web_vital"
  | "client_error"
  | "pwa_service_worker_ready"
  | "pwa_service_worker_error";

type VercelAnalytics = (...args: unknown[]) => void;

declare global {
  interface Window {
    va?: VercelAnalytics;
    vaq?: unknown[][];
  }
}

const LOCAL_HOSTS = new Set(["localhost", "127.0.0.1", "::1", ""]);
const MAX_KEY_LENGTH = 80;
const MAX_VALUE_LENGTH = 180;

function isProductionLikeHost(): boolean {
  return window.location.protocol === "https:" && !LOCAL_HOSTS.has(window.location.hostname);
}

/** Truncate keys/values so free-form data (e.g. error messages) cannot bloat events. */
export function sanitizePayload(payload: EventPayload): Record<string, Primitive> {
  const result: Record<string, Primitive> = {};
  for (const [key, value] of Object.entries(payload)) {
    if (value === undefined) {
      continue;
    }
    result[key.slice(0, MAX_KEY_LENGTH)] = typeof value === "string" ? value.slice(0, MAX_VALUE_LENGTH) : value;
  }
  return result;
}

function loadScript(src: string): void {
  if (document.querySelector(`script[src="${CSS.escape(src)}"]`)) {
    return;
  }
  const script = document.createElement("script");
  script.defer = true;
  script.src = src;
  document.head.appendChild(script);
}

function metaContent(name: string): string {
  return document.querySelector(`meta[name="${name}"]`)?.getAttribute("content") ?? "";
}

export function trackEvent(name: TelemetryEvent, payload: EventPayload = {}): void {
  window.va?.("event", { name, data: sanitizePayload(payload) });
}

function observePerformance(): void {
  if (!("PerformanceObserver" in window)) {
    return;
  }
  const observe = (type: string, onEntry: (entry: PerformanceEntry) => void): void => {
    try {
      new PerformanceObserver((list) => {
        list.getEntries().forEach(onEntry);
      }).observe({ type, buffered: true });
    } catch {
      // Metric not supported by this browser.
    }
  };
  observe("paint", (entry) => {
    if (entry.name === "first-contentful-paint") {
      trackEvent("web_vital", { metric: "FCP", value: Math.round(entry.startTime) });
    }
  });
  observe("largest-contentful-paint", (entry) => {
    trackEvent("web_vital", { metric: "LCP", value: Math.round(entry.startTime) });
  });
}

function observeErrors(): void {
  window.addEventListener("error", (event) => {
    trackEvent("client_error", {
      message: event.message || "unknown",
      source: event.filename ? new URL(event.filename, window.location.href).pathname : "inline",
    });
  });
  window.addEventListener("unhandledrejection", (event: PromiseRejectionEvent) => {
    const reason: unknown = event.reason;
    trackEvent("client_error", {
      message: reason instanceof Error ? reason.message : String(reason),
      source: "promise",
    });
  });
}

/** Queue-based Vercel Analytics bootstrap; scripts only load on production hosts. */
export function initAnalytics(): void {
  window.va ??= (...args: unknown[]) => {
    (window.vaq ??= []).push(args);
  };
  // Drop URL fragments (#monitor, #dashboard) so page views are not split per section.
  window.va("beforeSend", (event: { url: string }) => {
    try {
      const url = new URL(event.url);
      url.hash = "";
      return { ...event, url: url.toString() };
    } catch {
      return event;
    }
  });
  if (isProductionLikeHost()) {
    for (const meta of ["vercel-insights-src", "vercel-speed-insights-src"]) {
      const src = metaContent(meta);
      if (src) {
        loadScript(src);
      }
    }
  }
  observePerformance();
  observeErrors();
}
