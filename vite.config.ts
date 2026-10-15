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
  test: {
    include: ["tests/web/**/*.test.ts"],
    environment: "node",
  },
});
