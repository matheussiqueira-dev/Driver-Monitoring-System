const canvas = document.querySelector("#overlayCanvas");
const ctx = canvas.getContext("2d");
const videoShell = document.querySelector(".video-shell");
const demoImage = document.querySelector(".video-shell img");
const cameraPreview = document.querySelector("#cameraPreview");
const scoreValue = document.querySelector("#scoreValue");
const scoreLabel = document.querySelector("#scoreLabel");
const earValue = document.querySelector("#earValue");
const poseValue = document.querySelector("#poseValue");
const fpsValue = document.querySelector("#fpsValue");
const attentionFill = document.querySelector("#attentionFill");
const alertStrip = document.querySelector("#alertStrip");
const sensitivityRange = document.querySelector("#sensitivityRange");
const scoreChart = document.querySelector("#scoreChart");
const eventList = document.querySelector("#eventList");
const insightText = document.querySelector("#insightText");
const riskState = document.querySelector("#riskState");
const kpiScore = document.querySelector("#kpiScore");
const kpiEvents = document.querySelector("#kpiEvents");
const kpiEar = document.querySelector("#kpiEar");
const kpiFps = document.querySelector("#kpiFps");

const state = {
  mode: "demo",
  mediaStream: null,
  layers: {
    mesh: true,
    pose: true,
    objects: true,
  },
  history: Array.from({ length: 24 }, (_, index) => 72 + Math.sin(index * 0.6) * 10 + index * 0.35),
  lastHistoryUpdate: 0,
};

const chartBars = state.history.map((value) => {
  const bar = document.createElement("span");
  bar.className = "score-bar";
  scoreChart.appendChild(bar);
  return bar;
});

function resizeCanvas() {
  const rect = videoShell.getBoundingClientRect();
  const ratio = window.devicePixelRatio || 1;
  canvas.width = Math.max(1, Math.round(rect.width * ratio));
  canvas.height = Math.max(1, Math.round(rect.height * ratio));
  canvas.style.width = `${rect.width}px`;
  canvas.style.height = `${rect.height}px`;
  ctx.setTransform(ratio, 0, 0, ratio, 0, 0);
}

function point(x, y, width, height) {
  return [x * width, y * height];
}

function ellipse(cx, cy, rx, ry, count, start = 0, end = Math.PI * 2) {
  return Array.from({ length: count }, (_, index) => {
    const t = start + ((end - start) * index) / Math.max(1, count - 1);
    return [cx + Math.cos(t) * rx, cy + Math.sin(t) * ry];
  });
}

const meshLines = [
  ellipse(0.5, 0.52, 0.17, 0.29, 34),
  ellipse(0.43, 0.45, 0.045, 0.024, 12),
  ellipse(0.57, 0.45, 0.045, 0.024, 12),
  ellipse(0.5, 0.64, 0.07, 0.026, 16),
  [
    [0.5, 0.43],
    [0.49, 0.5],
    [0.48, 0.56],
    [0.51, 0.58],
    [0.54, 0.56],
  ],
  [
    [0.39, 0.39],
    [0.46, 0.37],
    [0.54, 0.37],
    [0.61, 0.39],
  ],
];

function drawPath(points, width, height, color, close = false) {
  if (!points.length) {
    return;
  }

  ctx.beginPath();
  points.forEach(([x, y], index) => {
    const [px, py] = point(x, y, width, height);
    if (index === 0) {
      ctx.moveTo(px, py);
    } else {
      ctx.lineTo(px, py);
    }
  });
  if (close) {
    ctx.closePath();
  }
  ctx.strokeStyle = color;
  ctx.lineWidth = 1;
  ctx.stroke();
}

function drawBox(x, y, w, h, label, color, width, height) {
  const [px, py] = point(x, y, width, height);
  const bw = w * width;
  const bh = h * height;
  ctx.strokeStyle = color;
  ctx.lineWidth = 2;
  ctx.strokeRect(px, py, bw, bh);
  ctx.fillStyle = "rgba(2, 8, 6, 0.68)";
  ctx.fillRect(px, Math.max(0, py - 22), Math.max(58, label.length * 8), 20);
  ctx.fillStyle = color;
  ctx.font = "12px Inter, system-ui, sans-serif";
  ctx.fillText(label, px + 7, Math.max(14, py - 8));
}

function getMetrics(time) {
  const sensitivity = Number(sensitivityRange.value);
  const wave = Math.sin(time / 900);
  const eventWave = Math.sin(time / 2100);
  const modeOffset = state.mode === "review" ? -9 : state.mode === "camera" ? 2 : 0;
  const score = Math.max(
    18,
    Math.min(100, 78 + modeOffset + wave * 9 + (sensitivity - 60) * 0.23 - Math.max(0, eventWave) * 16),
  );
  const ear = Math.max(0.18, 0.286 + Math.sin(time / 720) * 0.022 - Math.max(0, eventWave) * 0.028);
  const yaw = -3.2 + Math.sin(time / 980) * 8.4;
  const pitch = 4.8 + Math.cos(time / 1180) * 7.8 + Math.max(0, eventWave) * 5.2;
  const roll = 0.7 + Math.sin(time / 1320) * 2.1;
  const fps = 29.4 + Math.sin(time / 1500) * 1.7;
  const phoneRisk = eventWave > 0.54;
  const offroad = Math.abs(yaw) > 9.8 || pitch > 13.6;

  return { score, ear, yaw, pitch, roll, fps, phoneRisk, offroad };
}

function scoreMeta(score) {
  if (score >= 70) {
    return { label: "Atento", color: "#56f08d", level: "ok" };
  }
  if (score >= 40) {
    return { label: "Alerta", color: "#f7c75b", level: "warn" };
  }
  return { label: "Distraido", color: "#ff5e67", level: "danger" };
}

function updateDom(metrics, meta, time) {
  scoreValue.textContent = metrics.score.toFixed(1);
  scoreLabel.textContent = `(${meta.label})`;
  earValue.textContent = metrics.ear.toFixed(3);
  poseValue.textContent = `${metrics.yaw.toFixed(1)} / ${metrics.pitch.toFixed(1)} / ${metrics.roll.toFixed(1)}`;
  fpsValue.textContent = metrics.fps.toFixed(1);
  attentionFill.style.height = `${Math.max(8, metrics.score)}%`;
  attentionFill.style.background = meta.color;

  alertStrip.classList.toggle("is-warning", meta.level === "warn");
  alertStrip.classList.toggle("is-danger", meta.level === "danger");
  if (metrics.phoneRisk) {
    alertStrip.textContent = "Alerta: celular proximo ao rosto com pitch descendente";
  } else if (metrics.offroad) {
    alertStrip.textContent = "Alerta: olhar fora da pista acima do limiar";
  } else {
    alertStrip.textContent = "Condicao estavel: motorista atento";
  }

  if (time - state.lastHistoryUpdate > 700) {
    state.history.shift();
    state.history.push(metrics.score);
    state.lastHistoryUpdate = time;
  }

  chartBars.forEach((bar, index) => {
    const value = state.history[index];
    const barMeta = scoreMeta(value);
    bar.style.height = `${Math.max(10, value)}%`;
    bar.style.background = barMeta.color;
  });

  const criticalEvents = state.history.filter((value) => value < 58).length;
  kpiScore.textContent = (state.history.reduce((sum, value) => sum + value, 0) / state.history.length).toFixed(1);
  kpiEvents.textContent = String(Math.min(criticalEvents, 9)).padStart(2, "0");
  kpiEar.textContent = metrics.ear.toFixed(3);
  kpiFps.textContent = metrics.fps.toFixed(1);
  riskState.textContent = meta.level === "ok" ? "baixo risco" : meta.level === "warn" ? "risco moderado" : "risco alto";

  const events = [
    ["ok", "Face detectada com tracking estavel"],
    [metrics.offroad ? "warn" : "ok", metrics.offroad ? "Head pose acima do limiar" : "Pose dentro do limiar configurado"],
    [metrics.phoneRisk ? "danger" : "warn", metrics.phoneRisk ? "Celular em zona de risco" : "Celular filtrado por proximidade facial"],
  ];
  eventList.innerHTML = events
    .map(([level, text]) => `<li><span class="event-dot ${level}"></span>${text}</li>`)
    .join("");

  insightText.textContent = metrics.phoneRisk
    ? "Priorize a penalidade de celular somente quando houver proximidade facial e pitch para baixo para reduzir falsos positivos."
    : metrics.offroad
      ? "Aumente a suavizacao temporal quando yaw e pitch oscilarem em trechos com iluminacao instavel."
      : "A sessao segue estavel. Mantenha o peso de head pose acima de EAR em cenarios com oculos escuros.";
}

function drawOverlay(metrics, meta) {
  const rect = videoShell.getBoundingClientRect();
  const width = rect.width;
  const height = rect.height;
  ctx.clearRect(0, 0, width, height);

  if (state.layers.mesh) {
    meshLines.forEach((line, index) => {
      drawPath(line, width, height, index === 0 ? "rgba(86, 240, 141, 0.9)" : "rgba(86, 240, 141, 0.66)", index < 4);
    });

    ctx.fillStyle = "rgba(86, 240, 141, 0.85)";
    meshLines.flat().forEach(([x, y]) => {
      const [px, py] = point(x, y, width, height);
      ctx.beginPath();
      ctx.arc(px, py, 1.7, 0, Math.PI * 2);
      ctx.fill();
    });
  }

  drawBox(0.29, 0.13, 0.42, 0.76, "face 0.98", "#3488ff", width, height);

  if (state.layers.pose) {
    const [cx, cy] = point(0.5, 0.53, width, height);
    ctx.strokeStyle = meta.color;
    ctx.lineWidth = 3;
    ctx.beginPath();
    ctx.moveTo(cx, cy);
    ctx.lineTo(cx + metrics.yaw * 3.2, cy + metrics.pitch * 2.2);
    ctx.stroke();
    ctx.fillStyle = meta.color;
    ctx.font = "12px Inter, system-ui, sans-serif";
    ctx.fillText("pose", cx + 12, cy - 12);
  }

  if (state.layers.objects) {
    drawBox(0.04, 0.42, 0.34, 0.44, "driver 0.91", "#3488ff", width, height);
    if (metrics.phoneRisk) {
      drawBox(0.66, 0.58, 0.16, 0.2, "cell phone 0.84", "#ff5e67", width, height);
    } else {
      drawBox(0.73, 0.42, 0.14, 0.18, "object 0.42", "#f7c75b", width, height);
    }
  }
}

function animate(time = 0) {
  const metrics = getMetrics(time);
  const meta = scoreMeta(metrics.score);
  updateDom(metrics, meta, time);
  drawOverlay(metrics, meta);
  requestAnimationFrame(animate);
}

document.querySelectorAll(".source-button").forEach((button) => {
  button.addEventListener("click", async () => {
    document.querySelectorAll(".source-button").forEach((item) => item.classList.remove("is-active"));
    button.classList.add("is-active");
    await setMode(button.dataset.mode);
  });
});

async function setMode(mode) {
  state.mode = mode;

  if (mode !== "camera" && state.mediaStream) {
    state.mediaStream.getTracks().forEach((track) => track.stop());
    state.mediaStream = null;
  }

  if (mode !== "camera") {
    cameraPreview.hidden = true;
    demoImage.hidden = false;
    return;
  }

  if (!navigator.mediaDevices?.getUserMedia) {
    alertStrip.textContent = "Camera indisponivel neste navegador";
    alertStrip.classList.add("is-warning");
    return;
  }

  try {
    state.mediaStream = await navigator.mediaDevices.getUserMedia({
      video: {
        facingMode: "user",
        width: { ideal: 1280 },
        height: { ideal: 720 },
      },
      audio: false,
    });
    cameraPreview.srcObject = state.mediaStream;
    cameraPreview.hidden = false;
    demoImage.hidden = true;
    await cameraPreview.play();
  } catch (error) {
    cameraPreview.hidden = true;
    demoImage.hidden = false;
    alertStrip.textContent = "Permissao de camera negada. Demo mantida ativa.";
    alertStrip.classList.add("is-warning");
  }
}

document.querySelectorAll("[data-layer]").forEach((checkbox) => {
  checkbox.addEventListener("change", () => {
    state.layers[checkbox.dataset.layer] = checkbox.checked;
  });
});

new ResizeObserver(resizeCanvas).observe(videoShell);
window.addEventListener("load", resizeCanvas);
resizeCanvas();
animate();
