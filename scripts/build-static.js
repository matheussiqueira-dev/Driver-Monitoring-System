const fs = require("node:fs");
const path = require("node:path");

const root = process.cwd();
const dist = path.join(root, "dist");
const requiredFiles = ["index.html", "styles.css", "app.js", "favicon.svg", path.join("assets", "demo.gif")];

for (const file of requiredFiles) {
  if (!fs.existsSync(path.join(root, file))) {
    throw new Error(`Missing required static asset: ${file}`);
  }
}

fs.rmSync(dist, { recursive: true, force: true });
fs.mkdirSync(path.join(dist, "assets"), { recursive: true });

for (const file of ["index.html", "styles.css", "app.js", "favicon.svg"]) {
  fs.copyFileSync(path.join(root, file), path.join(dist, file));
}

fs.copyFileSync(path.join(root, "assets", "demo.gif"), path.join(dist, "assets", "demo.gif"));

console.log("Static build ready for Vercel: dist/");
