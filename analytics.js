const LOCAL_HOSTS = new Set(["localhost", "127.0.0.1", "::1", ""]);
const MAX_VALUE_LENGTH = 180;

function isProductionLikeHost() {
  return window.location.protocol === "https:" && !LOCAL_HOSTS.has(window.location.hostname);
}

function sanitizeValue(value) {
  if (value === null || typeof value === "number" || typeof value === "boolean") {
    return value;
  }
  return String(value).slice(0, MAX_VALUE_LENGTH);
}

function sanitizePayload(payload = {}) {
  return Object.fromEntries(
    Object.entries(payload).map(([key, value]) => [key.slice(0, 80), sanitizeValue(value)]),
  );
}

function loadScript(src) {
  if (document.querySelector(`script[src="${src}"]`)) {
    return;
  }
  const script = document.createElement("script");
  script.defer = true;
  script.src = src;
  script.dataset.observability = "vercel";
  document.head.appendChild(script);
}

function flushPerformanceMetric(name, value, rating = "observed") {
  trackEvent("web_vital", {
    metric: name,
    value: Math.round(value),
    rating,
  });
}

function observePerformance() {
  if (!("PerformanceObserver" in window)) {
    return;
  }

  try {
    const paintObserver = new PerformanceObserver((list) => {
      for (const entry of list.getEntries()) {
        if (entry.name === "first-contentful-paint") {
          flushPerformanceMetric("FCP", entry.startTime);
        }
      }
    });
    paintObserver.observe({ type: "paint", buffered: true });
  } catch {
    // Browser does not support this metric.
  }

  try {
    const lcpObserver = new PerformanceObserver((list) => {
      const entries = list.getEntries();
      const last = entries[entries.length - 1];
      if (last) {
        flushPerformanceMetric("LCP", last.startTime);
      }
    });
    lcpObserver.observe({ type: "largest-contentful-paint", buffered: true });
  } catch {
    // Browser does not support this metric.
  }
}

function observeErrors() {
  window.addEventListener("error", (event) => {
    trackEvent("client_error", {
      message: event.message || "unknown",
      source: event.filename ? new URL(event.filename, window.location.href).pathname : "inline",
    });
  });

  window.addEventListener("unhandledrejection", (event) => {
    trackEvent("client_error", {
      message: event.reason?.message || event.reason || "unhandledrejection",
      source: "promise",
    });
  });
}

export function initAnalytics() {
  window.va =
    window.va ||
    function vaQueue() {
      (window.vaq = window.vaq || []).push(arguments);
    };

  window.va("beforeSend", (event) => {
    try {
      const url = new URL(event.url);
      url.hash = "";
      event.url = url.toString();
    } catch {
      return event;
    }
    return event;
  });

  if (isProductionLikeHost()) {
    loadScript("/_vercel/insights/script.js");
  }

  observePerformance();
  observeErrors();
}

export function trackEvent(name, payload = {}) {
  const safeName = String(name).slice(0, 80);
  const safePayload = sanitizePayload(payload);

  if (typeof window.va === "function") {
    window.va("event", safeName, safePayload);
  }
  if (typeof window.gtag === "function") {
    window.gtag("event", safeName, safePayload);
  }
}
