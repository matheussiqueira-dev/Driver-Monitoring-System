import { readFileSync, writeFileSync } from "node:fs";
import { resolve } from "node:path";
import type { Plugin } from "vite";
import { defineConfig } from "vitest/config";

/**
 * Stamps the service worker cache name with a per-build id, so each deploy
 * gets a fresh cache and stale app shells are evicted automatically.
 */
function serviceWorkerBuildId(): Plugin {
  let outDir = "dist";
  return {
    name: "dms-service-worker-build-id",
    apply: "build",
    configResolved(config) {
      outDir = config.build.outDir;
    },
    closeBundle() {
      const file = resolve(outDir, "service-worker.js");
      const buildId = Date.now().toString(36);
      writeFileSync(file, readFileSync(file, "utf8").replaceAll("__BUILD_ID__", buildId));
    },
  };
}

interface VercelHeaderRule {
  readonly source: string;
  readonly headers: readonly { readonly key: string; readonly value: string }[];
}

/**
 * The catch-all security headers from vercel.json (CSP, frame, referrer...),
 * applied to `vite preview` so local runs and E2E tests behave like production.
 */
function productionHeaders(): Record<string, string> {
  const config = JSON.parse(readFileSync(resolve(import.meta.dirname, "vercel.json"), "utf8")) as {
    headers?: readonly VercelHeaderRule[];
  };
  const rule = config.headers?.find((entry) => entry.source === "/(.*)");
  // HSTS on localhost would pin the browser to HTTPS for every localhost port.
  const headers = (rule?.headers ?? []).filter((header) => header.key !== "Strict-Transport-Security");
  return Object.fromEntries(headers.map((header) => [header.key, header.value]));
}

export default defineConfig({
  plugins: [serviceWorkerBuildId()],
  build: {
    outDir: "dist",
    assetsDir: "assets",
    target: "es2022",
    sourcemap: true,
  },
  server: {
    port: 3000,
  },
  preview: {
    headers: productionHeaders(),
  },
  test: {
    include: ["tests/web/**/*.test.ts"],
    environment: "node",
  },
});
