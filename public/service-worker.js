// __BUILD_ID__ is replaced at build time (vite.config.ts), so every deploy gets its own cache.
const CACHE_NAME = "dms-__BUILD_ID__";
const APP_SHELL = ["/", "/manifest.webmanifest", "/favicon.svg", "/assets/icon-192.png", "/assets/icon-512.png"];

self.addEventListener("install", (event) => {
  event.waitUntil(
    caches
      .open(CACHE_NAME)
      .then((cache) => cache.addAll(APP_SHELL))
      .then(() => self.skipWaiting()),
  );
});

self.addEventListener("activate", (event) => {
  event.waitUntil(
    caches
      .keys()
      .then((keys) => Promise.all(keys.filter((key) => key !== CACHE_NAME).map((key) => caches.delete(key))))
      .then(() => self.clients.claim()),
  );
});

function putInCache(request, response) {
  if (response.ok && response.type === "basic") {
    const copy = response.clone();
    caches.open(CACHE_NAME).then((cache) => cache.put(request, copy));
  }
  return response;
}

self.addEventListener("fetch", (event) => {
  const { request } = event;
  const url = new URL(request.url);
  if (request.method !== "GET" || url.origin !== self.location.origin) {
    return;
  }

  // HTML: network first, so a new deploy is picked up immediately; cache is the offline fallback.
  if (request.mode === "navigate") {
    event.respondWith(
      fetch(request)
        .then((response) => putInCache("/", response))
        .catch(() => caches.match("/")),
    );
    return;
  }

  // Static files: stale-while-revalidate. Built JS/CSS have content hashes, so a cached copy is never stale.
  event.respondWith(
    caches.match(request).then((cached) => {
      const network = fetch(request)
        .then((response) => putInCache(request, response))
        .catch(() => cached);
      return cached || network;
    }),
  );
});
