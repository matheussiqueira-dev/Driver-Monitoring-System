export function registerPwa(trackEvent = () => {}) {
  if (!("serviceWorker" in navigator) || window.location.protocol === "file:") {
    return;
  }

  window.addEventListener("load", () => {
    navigator.serviceWorker
      .register("/service-worker.js", { scope: "/" })
      .then((registration) => {
        trackEvent("pwa_service_worker_ready", {
          scope: registration.scope,
        });
      })
      .catch((error) => {
        trackEvent("pwa_service_worker_error", {
          message: error.message || "registration failed",
        });
      });
  });
}
