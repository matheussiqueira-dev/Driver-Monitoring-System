import { trackEvent } from "./analytics";

export function registerServiceWorker(): void {
  if (!("serviceWorker" in navigator) || !import.meta.env.PROD) {
    return;
  }
  window.addEventListener("load", () => {
    navigator.serviceWorker
      .register("/service-worker.js", { scope: "/" })
      .then((registration) => {
        trackEvent("pwa_service_worker_ready", { scope: registration.scope });
      })
      .catch((error: unknown) => {
        trackEvent("pwa_service_worker_error", { message: error instanceof Error ? error.message : String(error) });
      });
  });
}
