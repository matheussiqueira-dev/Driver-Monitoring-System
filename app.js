import { FaceLandmarker, FilesetResolver } from "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.35";

const FACE_MODEL_URL =
  "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/latest/face_landmarker.task";
const WASM_URL = "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.35/wasm";

const LEFT_EYE = [362, 385, 387, 263, 373, 380];
const RIGHT_EYE = [33, 160, 158, 133, 153, 144];
const NOSE_TIP = 1;
const CHIN = 152;
const FOREHEAD = 10;
const LEFT_EYE_OUTER = 33;
const RIGHT_EYE_OUTER = 263;
const LEFT_CHEEK = 234;
const RIGHT_CHEEK = 454;

const OVERLAY_ALPHA = 0.58;
const FAST_REANCHOR_DISTANCE = 72;
const MIN_FACE_INFERENCE_INTERVAL_MS = 1000 / 30;
const HOLD_FRAMES = 4;
const FADE_FRAMES = 10;

const elements = {
  canvas: document.querySelector("#overlayCanvas"),
  videoShell: document.querySelector(".video-shell"),
  demoImage: document.querySelector(".video-shell img"),
  cameraPreview: document.querySelector("#cameraPreview"),
  scoreValue: document.querySelector("#scoreValue"),
  scoreLabel: document.querySelector("#scoreLabel"),
  earValue: document.querySelector("#earValue"),
  poseValue: document.querySelector("#poseValue"),
  fpsValue: document.querySelector("#fpsValue"),
  attentionFill: document.querySelector("#attentionFill"),
  alertStrip: document.querySelector("#alertStrip"),
  sensitivityRange: document.querySelector("#sensitivityRange"),
  scoreChart: document.querySelector("#scoreChart"),
  eventList: document.querySelector("#eventList"),
  insightText: document.querySelector("#insightText"),
  riskState: document.querySelector("#riskState"),
  kpiScore: document.querySelector("#kpiScore"),
  kpiEvents: document.querySelector("#kpiEvents"),
  kpiEar: document.querySelector("#kpiEar"),
  kpiFps: document.querySelector("#kpiFps"),
  trackingDebug: document.querySelector("#trackingDebug"),
};

const ctx = elements.canvas.getContext("2d");

const state = {
  mode: "demo",
  mirrored: true,
  mediaStream: null,
  rafId: null,
  cameraStatus: "idle",
  faceMeshStatus: "idle",
  layers: {
    mesh: true,
    pose: true,
    objects: true,
  },
  history: Array.from({ length: 24 }, (_, index) => 72 + Math.sin(index * 0.6) * 10 + index * 0.35),
  lastHistoryUpdate: 0,
};

const chartBars = state.history.map(() => {
  const bar = document.createElement("span");
  bar.className = "score-bar";
  elements.scoreChart.appendChild(bar);
  return bar;
});

function clamp(value, min, max) {
  return Math.max(min, Math.min(max, value));
}

function distance(a, b) {
  return Math.hypot(a.x - b.x, a.y - b.y);
}

function midpoint(a, b) {
  return {
    x: (a.x + b.x) * 0.5,
    y: (a.y + b.y) * 0.5,
    z: ((a.z || 0) + (b.z || 0)) * 0.5,
  };
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

function computeEar(landmarks, indices) {
  if (!landmarks || indices.some((index) => !landmarks[index])) {
    return null;
  }
  const [p1, p2, p3, p4, p5, p6] = indices.map((index) => landmarks[index]);
  const vertical1 = distance(p2, p6);
  const vertical2 = distance(p3, p5);
  const horizontal = distance(p1, p4);
  return horizontal > 0 ? (vertical1 + vertical2) / (2 * horizontal) : null;
}

function computeBlinkSafeEar(landmarks) {
  const left = computeEar(landmarks, LEFT_EYE);
  const right = computeEar(landmarks, RIGHT_EYE);
  if (left === null || right === null) {
    return null;
  }
  return (left + right) * 0.5;
}

function computePoseApprox(landmarks) {
  if (!landmarks?.[NOSE_TIP] || !landmarks?.[CHIN] || !landmarks?.[LEFT_EYE_OUTER] || !landmarks?.[RIGHT_EYE_OUTER]) {
    return { yaw: 0, pitch: 0, roll: 0 };
  }

  const eyeA = landmarks[LEFT_EYE_OUTER];
  const eyeB = landmarks[RIGHT_EYE_OUTER];
  const leftEye = eyeA.x <= eyeB.x ? eyeA : eyeB;
  const rightEye = eyeA.x <= eyeB.x ? eyeB : eyeA;
  const nose = landmarks[NOSE_TIP];
  const chin = landmarks[CHIN];
  const eyeMid = midpoint(leftEye, rightEye);
  const faceWidth = Math.max(1, distance(leftEye, rightEye));
  const faceHeight = Math.max(1, distance(landmarks[FOREHEAD] || eyeMid, chin));
  const roll = (Math.atan2(rightEye.y - leftEye.y, rightEye.x - leftEye.x) * 180) / Math.PI;
  const yaw = ((nose.x - eyeMid.x) / faceWidth) * 62;
  const pitch = ((nose.y - eyeMid.y) / faceHeight - 0.24) * 85;
  return { yaw, pitch, roll };
}

function computeFaceBox(landmarks) {
  const xs = landmarks.map((point) => point.x);
  const ys = landmarks.map((point) => point.y);
  const x1 = Math.min(...xs);
  const y1 = Math.min(...ys);
  const x2 = Math.max(...xs);
  const y2 = Math.max(...ys);
  return { x: x1, y: y1, width: x2 - x1, height: y2 - y1 };
}

function computeAnchor(landmarks) {
  const nose = landmarks[NOSE_TIP];
  const chin = landmarks[CHIN];
  const eyes = midpoint(landmarks[LEFT_EYE_OUTER], landmarks[RIGHT_EYE_OUTER]);
  const faceMid = midpoint(eyes, chin);
  return {
    x: nose.x * 0.52 + faceMid.x * 0.28 + eyes.x * 0.2,
    y: nose.y * 0.52 + faceMid.y * 0.28 + eyes.y * 0.2,
  };
}

class OverlayCoordinateMapper {
  constructor(shell, canvas, context) {
    this.shell = shell;
    this.canvas = canvas;
    this.ctx = context;
    this.width = 1;
    this.height = 1;
    this.dpr = 1;
    this.mapping = {
      sourceWidth: 1,
      sourceHeight: 1,
      displayWidth: 1,
      displayHeight: 1,
      offsetX: 0,
      offsetY: 0,
      scale: 1,
      mirrored: false,
    };
  }

  sync(source, mirrored) {
    const rect = this.shell.getBoundingClientRect();
    this.width = Math.max(1, rect.width);
    this.height = Math.max(1, rect.height);
    this.dpr = window.devicePixelRatio || 1;

    const pixelWidth = Math.round(this.width * this.dpr);
    const pixelHeight = Math.round(this.height * this.dpr);
    if (this.canvas.width !== pixelWidth || this.canvas.height !== pixelHeight) {
      this.canvas.width = pixelWidth;
      this.canvas.height = pixelHeight;
      this.canvas.style.width = `${this.width}px`;
      this.canvas.style.height = `${this.height}px`;
    }
    this.ctx.setTransform(this.dpr, 0, 0, this.dpr, 0, 0);

    const sourceWidth = source?.videoWidth || source?.naturalWidth || this.width;
    const sourceHeight = source?.videoHeight || source?.naturalHeight || this.height;
    const scale = Math.max(this.width / sourceWidth, this.height / sourceHeight);
    const displayWidth = sourceWidth * scale;
    const displayHeight = sourceHeight * scale;
    this.mapping = {
      sourceWidth,
      sourceHeight,
      displayWidth,
      displayHeight,
      offsetX: (this.width - displayWidth) * 0.5,
      offsetY: (this.height - displayHeight) * 0.5,
      scale,
      mirrored,
    };
  }

  clear() {
    this.ctx.clearRect(0, 0, this.width, this.height);
  }

  toCanvasPoint(landmark) {
    const nx = this.mapping.mirrored ? 1 - landmark.x : landmark.x;
    return {
      x: this.mapping.offsetX + nx * this.mapping.sourceWidth * this.mapping.scale,
      y: this.mapping.offsetY + landmark.y * this.mapping.sourceHeight * this.mapping.scale,
      z: (landmark.z || 0) * this.mapping.sourceWidth * this.mapping.scale,
    };
  }
}

class LandmarkSmoother {
  constructor(alpha = OVERLAY_ALPHA) {
    this.alpha = alpha;
    this.previous = null;
  }

  reset() {
    this.previous = null;
  }

  update(points) {
    if (!this.previous || this.previous.length !== points.length) {
      this.previous = points.map((point) => ({ ...point }));
      return this.previous;
    }

    let maxMovement = 0;
    for (let index = 0; index < points.length; index += 1) {
      maxMovement = Math.max(maxMovement, distance(points[index], this.previous[index]));
    }

    const alpha = maxMovement > FAST_REANCHOR_DISTANCE ? 0.88 : this.alpha;
    this.previous = points.map((point, index) => ({
      x: alpha * point.x + (1 - alpha) * this.previous[index].x,
      y: alpha * point.y + (1 - alpha) * this.previous[index].y,
      z: alpha * (point.z || 0) + (1 - alpha) * (this.previous[index].z || 0),
    }));
    return this.previous;
  }
}

class FaceMeshProcessor {
  constructor() {
    this.faceLandmarker = null;
    this.initializing = null;
    this.lastInferenceTimestamp = 0;
    this.frameId = 0;
    this.lastResultTimestamp = 0;
    this.connections = [];
  }

  async ensureReady() {
    if (this.faceLandmarker) {
      return this.faceLandmarker;
    }
    if (this.initializing) {
      return this.initializing;
    }

    state.faceMeshStatus = "loading";
    this.initializing = (async () => {
      const vision = await FilesetResolver.forVisionTasks(WASM_URL);
      this.faceLandmarker = await FaceLandmarker.createFromOptions(vision, {
        baseOptions: {
          modelAssetPath: FACE_MODEL_URL,
          delegate: "GPU",
        },
        runningMode: "VIDEO",
        numFaces: 1,
        minFaceDetectionConfidence: 0.5,
        minFacePresenceConfidence: 0.5,
        minTrackingConfidence: 0.5,
        outputFaceBlendshapes: false,
        outputFacialTransformationMatrixes: true,
      });
      this.connections = [
        ...(FaceLandmarker.FACE_LANDMARKS_TESSELATION || []),
        ...(FaceLandmarker.FACE_LANDMARKS_CONTOURS || []),
        ...(FaceLandmarker.FACE_LANDMARKS_LEFT_EYE || []),
        ...(FaceLandmarker.FACE_LANDMARKS_RIGHT_EYE || []),
        ...(FaceLandmarker.FACE_LANDMARKS_LIPS || []),
      ];
      state.faceMeshStatus = "ready";
      return this.faceLandmarker;
    })().catch((error) => {
      state.faceMeshStatus = "error";
      this.initializing = null;
      throw error;
    });

    return this.initializing;
  }

  detect(video, timestampMs) {
    if (!this.faceLandmarker || video.readyState < HTMLMediaElement.HAVE_CURRENT_DATA) {
      return { updated: false, result: null };
    }
    if (timestampMs - this.lastInferenceTimestamp < MIN_FACE_INFERENCE_INTERVAL_MS) {
      return { updated: false, result: null };
    }

    const result = this.faceLandmarker.detectForVideo(video, timestampMs);
    this.lastInferenceTimestamp = timestampMs;
    this.frameId += 1;
    this.lastResultTimestamp = timestampMs;
    return { updated: true, result };
  }
}

class FaceTracker {
  constructor() {
    this.smoother = new LandmarkSmoother();
    this.lastTrack = null;
    this.missedFrames = 0;
    this.faceFpsTimes = [];
  }

  reset() {
    this.smoother.reset();
    this.lastTrack = null;
    this.missedFrames = 0;
    this.faceFpsTimes = [];
  }

  update(result, mapper, timestampMs) {
    const rawLandmarks = result?.faceLandmarks?.[0] || null;
    if (!rawLandmarks) {
      return this.markMissing(timestampMs);
    }

    const mapped = rawLandmarks.map((landmark) => mapper.toCanvasPoint(landmark));
    const landmarks = this.smoother.update(mapped);
    const box = computeFaceBox(landmarks);
    const anchor = computeAnchor(landmarks);
    const ear = computeBlinkSafeEar(landmarks);
    const pose = computePoseApprox(landmarks);

    this.faceFpsTimes.push(timestampMs);
    while (this.faceFpsTimes.length > 2 && timestampMs - this.faceFpsTimes[0] > 1000) {
      this.faceFpsTimes.shift();
    }
    const faceFps =
      this.faceFpsTimes.length > 1
        ? ((this.faceFpsTimes.length - 1) * 1000) / (this.faceFpsTimes[this.faceFpsTimes.length - 1] - this.faceFpsTimes[0])
        : 0;

    this.missedFrames = 0;
    this.lastTrack = {
      landmarks,
      rawLandmarksCount: rawLandmarks.length,
      box,
      anchor,
      ear,
      pose,
      opacity: 1,
      status: "tracking",
      faceFps,
      timestampMs,
    };
    return this.lastTrack;
  }

  markMissing(timestampMs) {
    this.missedFrames += 1;
    if (!this.lastTrack) {
      return null;
    }
    if (this.missedFrames <= HOLD_FRAMES) {
      this.lastTrack = { ...this.lastTrack, opacity: 0.85, status: "holding", timestampMs };
      return this.lastTrack;
    }
    if (this.missedFrames <= HOLD_FRAMES + FADE_FRAMES) {
      const fade = 1 - (this.missedFrames - HOLD_FRAMES) / FADE_FRAMES;
      this.lastTrack = { ...this.lastTrack, opacity: clamp(fade, 0, 1), status: "fading", timestampMs };
      return this.lastTrack;
    }

    this.reset();
    return null;
  }
}

class LandmarkRenderer {
  constructor(context, mapper, processor) {
    this.ctx = context;
    this.mapper = mapper;
    this.processor = processor;
  }

  draw(track, metrics) {
    if (!track || track.opacity <= 0) {
      return;
    }

    this.ctx.save();
    this.ctx.globalAlpha = track.opacity;

    if (state.layers.mesh) {
      this.drawConnections(track.landmarks, this.processor.connections);
      this.drawPoints(track.landmarks);
    }

    this.drawFaceBox(track);

    if (state.layers.pose) {
      this.drawPose(track, metrics);
    }

    this.ctx.restore();
  }

  drawConnections(landmarks, connections) {
    this.ctx.strokeStyle = "rgba(86, 240, 141, 0.72)";
    this.ctx.lineWidth = 0.8;
    this.ctx.beginPath();

    for (const connection of connections) {
      const start = connection.start ?? connection[0];
      const end = connection.end ?? connection[1];
      const a = landmarks[start];
      const b = landmarks[end];
      if (!a || !b) {
        continue;
      }
      this.ctx.moveTo(a.x, a.y);
      this.ctx.lineTo(b.x, b.y);
    }
    this.ctx.stroke();
  }

  drawPoints(landmarks) {
    this.ctx.fillStyle = "rgba(86, 240, 141, 0.88)";
    for (const point of landmarks) {
      this.ctx.beginPath();
      this.ctx.arc(point.x, point.y, 1.35, 0, Math.PI * 2);
      this.ctx.fill();
    }
  }

  drawFaceBox(track) {
    const { box } = track;
    this.ctx.strokeStyle = track.status === "tracking" ? "#3488ff" : "#f7c75b";
    this.ctx.lineWidth = 2;
    this.ctx.strokeRect(box.x, box.y, box.width, box.height);
    this.drawLabel(box.x, Math.max(18, box.y - 8), `face ${track.status}`);
  }

  drawPose(track, metrics) {
    const anchor = track.anchor;
    const yawVector = metrics.yaw * 2.0;
    const pitchVector = metrics.pitch * 1.6;
    this.ctx.strokeStyle = metrics.meta.color;
    this.ctx.lineWidth = 3;
    this.ctx.beginPath();
    this.ctx.moveTo(anchor.x, anchor.y);
    this.ctx.lineTo(anchor.x + yawVector, anchor.y + pitchVector);
    this.ctx.stroke();

    this.ctx.fillStyle = "#56f08d";
    this.ctx.beginPath();
    this.ctx.arc(anchor.x, anchor.y, 4, 0, Math.PI * 2);
    this.ctx.fill();
    this.drawLabel(anchor.x + 10, anchor.y - 10, "anchor");
  }

  drawLabel(x, y, text) {
    this.ctx.font = "12px Inter, system-ui, sans-serif";
    const width = Math.max(64, this.ctx.measureText(text).width + 14);
    this.ctx.fillStyle = "rgba(2, 8, 6, 0.7)";
    this.ctx.fillRect(x, y - 16, width, 20);
    this.ctx.fillStyle = "#3bd8ff";
    this.ctx.fillText(text, x + 7, y - 2);
  }
}

const mapper = new OverlayCoordinateMapper(elements.videoShell, elements.canvas, ctx);
const faceProcessor = new FaceMeshProcessor();
const faceTracker = new FaceTracker();
const renderer = new LandmarkRenderer(ctx, mapper, faceProcessor);

function getDemoMetrics(time) {
  const sensitivity = Number(elements.sensitivityRange.value);
  const wave = Math.sin(time / 900);
  const eventWave = Math.sin(time / 2100);
  const modeOffset = state.mode === "review" ? -9 : 0;
  const score = clamp(78 + modeOffset + wave * 9 + (sensitivity - 60) * 0.23 - Math.max(0, eventWave) * 16, 18, 100);
  const ear = Math.max(0.18, 0.286 + Math.sin(time / 720) * 0.022 - Math.max(0, eventWave) * 0.028);
  const yaw = -3.2 + Math.sin(time / 980) * 8.4;
  const pitch = 4.8 + Math.cos(time / 1180) * 7.8 + Math.max(0, eventWave) * 5.2;
  const roll = 0.7 + Math.sin(time / 1320) * 2.1;
  const fps = 29.4 + Math.sin(time / 1500) * 1.7;
  const phoneRisk = eventWave > 0.54;
  const offroad = Math.abs(yaw) > 9.8 || pitch > 13.6;
  const meta = scoreMeta(score);
  return { score, ear, yaw, pitch, roll, fps, phoneRisk, offroad, meta, facePresent: true, events: [] };
}

function getCameraMetrics(track) {
  if (!track) {
    const meta = scoreMeta(28);
    return {
      score: 28,
      ear: null,
      yaw: 0,
      pitch: 0,
      roll: 0,
      fps: 0,
      phoneRisk: false,
      offroad: false,
      meta,
      facePresent: false,
      events: ["Sem rosto"],
    };
  }

  const yaw = track.pose.yaw;
  const pitch = track.pose.pitch;
  const roll = track.pose.roll;
  const ear = track.ear;
  const drowsy = ear !== null && ear < 0.2;
  const offroad = Math.abs(yaw) > 20 || Math.abs(pitch) > 18;
  const score = clamp(100 - (drowsy ? 30 : 0) - (offroad ? 20 : 0) - (track.status === "tracking" ? 0 : 18), 0, 100);
  const meta = scoreMeta(score);
  const events = [];
  if (drowsy) {
    events.push("EAR baixo");
  }
  if (offroad) {
    events.push("Pose fora");
  }
  if (track.status !== "tracking") {
    events.push(track.status === "holding" ? "Rosto em hold" : "Rosto em fade");
  }
  return {
    score,
    ear,
    yaw,
    pitch,
    roll,
    fps: track.faceFps,
    phoneRisk: false,
    offroad,
    meta,
    facePresent: true,
    events,
  };
}

function updateDom(metrics, time) {
  elements.scoreValue.textContent = metrics.score.toFixed(1);
  elements.scoreLabel.textContent = `(${metrics.meta.label})`;
  elements.earValue.textContent = metrics.ear === null ? "--" : metrics.ear.toFixed(3);
  elements.poseValue.textContent = `${metrics.yaw.toFixed(1)} / ${metrics.pitch.toFixed(1)} / ${metrics.roll.toFixed(1)}`;
  elements.fpsValue.textContent = metrics.fps.toFixed(1);
  elements.attentionFill.style.height = `${Math.max(8, metrics.score)}%`;
  elements.attentionFill.style.background = metrics.meta.color;

  elements.alertStrip.classList.toggle("is-warning", metrics.meta.level === "warn");
  elements.alertStrip.classList.toggle("is-danger", metrics.meta.level === "danger");
  if (state.mode === "camera" && state.faceMeshStatus !== "ready") {
    elements.alertStrip.textContent = `Face Mesh: ${state.faceMeshStatus}`;
  } else if (!metrics.facePresent) {
    elements.alertStrip.textContent = "Rosto nao detectado no frame atual";
  } else if (metrics.events.length) {
    elements.alertStrip.textContent = `Alerta: ${metrics.events.join(", ")}`;
  } else {
    elements.alertStrip.textContent = "Condicao estavel: motorista atento";
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
  elements.kpiScore.textContent = (state.history.reduce((sum, value) => sum + value, 0) / state.history.length).toFixed(1);
  elements.kpiEvents.textContent = String(Math.min(criticalEvents, 9)).padStart(2, "0");
  elements.kpiEar.textContent = metrics.ear === null ? "--" : metrics.ear.toFixed(3);
  elements.kpiFps.textContent = metrics.fps.toFixed(1);
  elements.riskState.textContent =
    metrics.meta.level === "ok" ? "baixo risco" : metrics.meta.level === "warn" ? "risco moderado" : "risco alto";

  const objectStatus = state.mode === "camera" ? "Objetos: aguardando modelo web" : "Celular filtrado por proximidade facial";
  const events = [
    [metrics.facePresent ? "ok" : "danger", metrics.facePresent ? "Face Mesh atualizado no frame" : "Sem landmarks no frame"],
    [metrics.offroad ? "warn" : "ok", metrics.offroad ? "Head pose acima do limiar" : "Pose dentro do limiar configurado"],
    [state.mode === "camera" ? "warn" : metrics.phoneRisk ? "danger" : "warn", objectStatus],
  ];
  elements.eventList.innerHTML = events
    .map(([level, text]) => `<li><span class="event-dot ${level}"></span>${text}</li>`)
    .join("");

  elements.insightText.textContent =
    state.mode === "camera"
      ? "O canvas usa landmarks reais do frame atual, mapeados pelo mesmo recorte do video e com espelhamento aplicado."
      : "A sessao demo usa o GIF como material visual. No modo Camera, os landmarks passam a vir do MediaPipe em tempo real.";
}

function drawDemoOverlay(metrics) {
  const width = mapper.width;
  const height = mapper.height;
  const point = (x, y) => [x * width, y * height];
  const drawBox = (x, y, w, h, label, color) => {
    const [px, py] = point(x, y);
    ctx.strokeStyle = color;
    ctx.lineWidth = 2;
    ctx.strokeRect(px, py, w * width, h * height);
    ctx.fillStyle = "rgba(2, 8, 6, 0.68)";
    ctx.fillRect(px, Math.max(0, py - 22), Math.max(58, label.length * 8), 20);
    ctx.fillStyle = color;
    ctx.font = "12px Inter, system-ui, sans-serif";
    ctx.fillText(label, px + 7, Math.max(14, py - 8));
  };

  drawBox(0.29, 0.13, 0.42, 0.76, "demo face", "#3488ff");
  if (state.layers.objects) {
    drawBox(0.04, 0.42, 0.34, 0.44, "driver 0.91", "#3488ff");
    drawBox(0.73, 0.42, 0.14, 0.18, metrics.phoneRisk ? "cell phone 0.84" : "object 0.42", metrics.phoneRisk ? "#ff5e67" : "#f7c75b");
  }
}

function updateDebug(track) {
  const mapping = mapper.mapping;
  const lines = [
    `frameId: ${faceProcessor.frameId}`,
    `lastResultMs: ${Math.round(faceProcessor.lastResultTimestamp || 0)}`,
    `landmarks: ${track?.rawLandmarksCount || 0}`,
    `video: ${elements.cameraPreview.videoWidth || 0}x${elements.cameraPreview.videoHeight || 0}`,
    `canvas: ${Math.round(mapper.width)}x${Math.round(mapper.height)} @${mapper.dpr.toFixed(2)}x`,
    `camera: ${state.cameraStatus}`,
    `faceMesh: ${state.faceMeshStatus}`,
    `mirrored: ${String(mapping.mirrored)}`,
    `trackingFps: ${(track?.faceFps || 0).toFixed(1)}`,
    `anchor: ${track ? `${track.anchor.x.toFixed(0)},${track.anchor.y.toFixed(0)}` : "--"}`,
    `bbox: ${track ? `${track.box.width.toFixed(0)}x${track.box.height.toFixed(0)}` : "--"}`,
  ];
  elements.trackingDebug.textContent = lines.join("\n");
  elements.trackingDebug.hidden = state.mode !== "camera";
}

function renderFrame(time = 0) {
  const source = state.mode === "camera" ? elements.cameraPreview : elements.demoImage;
  mapper.sync(source, state.mode === "camera" && state.mirrored);
  mapper.clear();

  if (state.mode === "camera") {
    let track = faceTracker.lastTrack;
    if (state.faceMeshStatus === "ready") {
      const now = performance.now();
      const { updated, result } = faceProcessor.detect(elements.cameraPreview, now);
      if (updated) {
        track = faceTracker.update(result, mapper, now);
      }
    } else if (state.cameraStatus === "ready") {
      track = faceTracker.markMissing(performance.now());
    }

    const metrics = getCameraMetrics(track);
    renderer.draw(track, metrics);
    updateDom(metrics, time);
    updateDebug(track);
  } else {
    const metrics = getDemoMetrics(time);
    drawDemoOverlay(metrics);
    updateDom(metrics, time);
    updateDebug(null);
  }

  state.rafId = requestAnimationFrame(renderFrame);
}

async function startCamera() {
  if (!navigator.mediaDevices?.getUserMedia) {
    state.cameraStatus = "unsupported";
    throw new Error("Camera indisponivel neste navegador");
  }

  state.cameraStatus = "requesting";
  state.mediaStream = await navigator.mediaDevices.getUserMedia({
    video: {
      facingMode: "user",
      width: { ideal: 1280 },
      height: { ideal: 720 },
    },
    audio: false,
  });

  elements.cameraPreview.srcObject = state.mediaStream;
  elements.cameraPreview.hidden = false;
  elements.demoImage.hidden = true;
  elements.cameraPreview.classList.toggle("is-mirrored", state.mirrored);
  await elements.cameraPreview.play();
  state.cameraStatus = "ready";
}

function stopCamera() {
  if (state.mediaStream) {
    state.mediaStream.getTracks().forEach((track) => track.stop());
    state.mediaStream = null;
  }
  elements.cameraPreview.pause();
  elements.cameraPreview.srcObject = null;
  elements.cameraPreview.hidden = true;
  elements.cameraPreview.classList.remove("is-mirrored");
  elements.demoImage.hidden = false;
  state.cameraStatus = "idle";
  faceTracker.reset();
}

async function setMode(mode) {
  state.mode = mode;
  elements.trackingDebug.hidden = mode !== "camera";

  if (mode !== "camera") {
    stopCamera();
    return;
  }

  try {
    await Promise.all([startCamera(), faceProcessor.ensureReady()]);
  } catch (error) {
    console.error(error);
    stopCamera();
    state.mode = "demo";
    document.querySelectorAll(".source-button").forEach((item) => item.classList.toggle("is-active", item.dataset.mode === "demo"));
    elements.alertStrip.textContent = "Nao foi possivel iniciar camera/Face Mesh. Demo mantida ativa.";
    elements.alertStrip.classList.add("is-warning");
  }
}

document.querySelectorAll(".source-button").forEach((button) => {
  button.addEventListener("click", async () => {
    document.querySelectorAll(".source-button").forEach((item) => item.classList.remove("is-active"));
    button.classList.add("is-active");
    await setMode(button.dataset.mode);
  });
});

document.querySelectorAll("[data-layer]").forEach((checkbox) => {
  checkbox.addEventListener("change", () => {
    state.layers[checkbox.dataset.layer] = checkbox.checked;
  });
});

new ResizeObserver(() => mapper.sync(state.mode === "camera" ? elements.cameraPreview : elements.demoImage, state.mode === "camera" && state.mirrored)).observe(
  elements.videoShell,
);

window.addEventListener("beforeunload", () => {
  if (state.rafId) {
    cancelAnimationFrame(state.rafId);
  }
  stopCamera();
});

mapper.sync(elements.demoImage, false);
state.rafId = requestAnimationFrame(renderFrame);
