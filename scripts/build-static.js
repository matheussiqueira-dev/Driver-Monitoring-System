const fs = require("node:fs");
const path = require("node:path");

const root = process.cwd();
const dist = path.join(root, "dist");
const staticFiles = [
  "index.html",
  "styles.css",
  "app.js",
  "analytics.js",
  "pwa.js",
  "favicon.svg",
  "manifest.webmanifest",
  "service-worker.js",
  "robots.txt",
  "sitemap.xml",
];
const requiredFiles = [
  ...staticFiles,
  path.join("assets", "demo.gif"),
  path.join("assets", "icon-192.png"),
  path.join("assets", "icon-512.png"),
  path.join("assets", "og-image.png"),
];

for (const file of requiredFiles) {
  if (!fs.existsSync(path.join(root, file))) {
    throw new Error(`Missing required static asset: ${file}`);
  }
}

fs.rmSync(dist, { recursive: true, force: true });
fs.mkdirSync(path.join(dist, "assets"), { recursive: true });

for (const file of staticFiles) {
  fs.copyFileSync(path.join(root, file), path.join(dist, file));
}

for (const file of fs.readdirSync(path.join(root, "assets"))) {
  fs.copyFileSync(path.join(root, "assets", file), path.join(dist, "assets", file));
}

console.log("Static build ready for Vercel: dist/");
