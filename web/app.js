import {
  FaceLandmarker,
  HandLandmarker,
  FilesetResolver,
} from "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14/+esm";

const video = document.getElementById("video");
const canvas = document.getElementById("canvas");
const ctx = canvas.getContext("2d");
const statusEl = document.getElementById("status");
const filterNameEl = document.getElementById("filter-name");
const btnPrev = document.getElementById("btn-prev");
const btnNext = document.getElementById("btn-next");
const btnSnap = document.getElementById("btn-snap");
const btnRec = document.getElementById("btn-rec");

// ── Filter definitions ──────────────────────────────────────────────────────

const FILTER_DEFS = [
  { name: "bday_hat", src: "assets/bday_hat.png", placement: "hat" },
  { name: "pats_eyes", src: "assets/pats_eyes.png", placement: "cheeks" },
  { name: "bike", src: "assets/bike.png", placement: "glasses" },
  { name: "chef_hat", src: "assets/chef_hat.png", placement: "hat" },
];

function loadImage(src) {
  return new Promise((resolve) => {
    const img = new Image();
    img.onload = () => resolve(img);
    img.onerror = () => { console.warn("Failed to load", src); resolve(null); };
    img.src = src;
  });
}

// landmarks: array of {x, y, z} in normalized coords (0-1)
// mx(lm) mirrors x so the canvas (which is drawn flipped) aligns with overlay

function mx(lm, w) { return (1 - lm.x) * w; }
function my(lm, h) { return lm.y * h; }

function placeHat(landmarks, img, w, h) {
  if (!img) return;
  const leftEye = landmarks[33];
  const rightEye = landmarks[263];
  const forehead = landmarks[10];

  const eyeWidth = Math.abs(rightEye.x - leftEye.x) * w;
  const hatW = eyeWidth * 2.5;
  const hatH = hatW * 0.9;
  const fx = mx(forehead, w);
  const fy = my(forehead, h);

  ctx.drawImage(img, fx - hatW / 2, fy - hatH, hatW, hatH);
}

function placeCheeks(landmarks, img, w, h) {
  if (!img) return;
  const leftEye = landmarks[33];
  const rightEye = landmarks[263];
  const eyeDist = Math.abs(rightEye.x - leftEye.x) * w;

  const newW = eyeDist * 1.4;
  const newH = img.height * (newW / img.width);

  // Mirror: average of mirrored left+right x stays the same as unmirrored center
  const centerX = ((1 - (leftEye.x + rightEye.x) / 2)) * w;
  const centerY = ((leftEye.y + rightEye.y) / 2) * h + newH;

  ctx.drawImage(img, centerX - newW / 2, centerY - newH / 2, newW, newH);
}

function placeGlasses(landmarks, img, w, h) {
  if (!img) return;
  const left = landmarks[33];
  const right = landmarks[263];

  const x1 = mx(left, w); const y1 = my(left, h);
  const x2 = mx(right, w); const y2 = my(right, h);

  const eyeDist = Math.hypot(x2 - x1, y2 - y1);
  const newW = eyeDist * 1.6;
  const newH = img.height * (newW / img.width);
  const cx = (x1 + x2) / 2;
  const cy = (y1 + y2) / 2;

  ctx.drawImage(img, cx - newW / 2, cy - newH / 2, newW, newH);
}

const PLACE = { hat: placeHat, cheeks: placeCheeks, glasses: placeGlasses };

// state 

let filters = [];
let currentFilter = 0;
let faceLandmarker = null;
let handLandmarker = null;
let faceResult = null;
let handResult = null;

// Hand swipe
let prevWristX = null;
let lastSwipeTime = 0;
const SWIPE_COOLDOWN = 500; // ms

// Recording
let mediaRecorder = null;
let recordedChunks = [];
let isRecording = false;



// init 

async function init() {
  statusEl.textContent = "Loading models...";

  const vision = await FilesetResolver.forVisionTasks(
    "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14/wasm"
  );

  [faceLandmarker, handLandmarker] = await Promise.all([
    FaceLandmarker.createFromOptions(vision, {
      baseOptions: {
        modelAssetPath:
          "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task",
        delegate: "GPU",
      },
      outputFaceBlendshapes: false,
      runningMode: "VIDEO",
      numFaces: 2,
    }),
    HandLandmarker.createFromOptions(vision, {
      baseOptions: {
        modelAssetPath:
          "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task",
        delegate: "GPU",
      },
      runningMode: "VIDEO",
      numHands: 1,
    }),
  ]);

  statusEl.textContent = "Loading filters...";
  const images = await Promise.all(FILTER_DEFS.map((f) => loadImage(f.src)));
  filters = FILTER_DEFS.map((f, i) => ({ ...f, image: images[i] }));

  statusEl.textContent = "Starting camera...";
  const stream = await navigator.mediaDevices.getUserMedia({ video: true });
  video.srcObject = stream;
  await video.play();

  // Wait until real frame dimensions
  await new Promise((r) => {
    const check = () => (video.videoWidth > 0 ? r() : requestAnimationFrame(check));
    check();
  });

  canvas.width = video.videoWidth;
  canvas.height = video.videoHeight;

  statusEl.textContent = "Wave your hand to change filters";
  updateFilterLabel();
  requestAnimationFrame(loop);
}

// Main loop

function loop() {
  try {
    const now = performance.now();

    faceResult = faceLandmarker.detectForVideo(video, now);
    handResult = handLandmarker.detectForVideo(video, now);

    const w = canvas.width;
    const h = canvas.height;

    // Draw mirrored video
    ctx.save();
    ctx.translate(w, 0);
    ctx.scale(-1, 1);
    ctx.drawImage(video, 0, 0, w, h);
    ctx.restore();

    // Debug: show face count live
    //const faceCount = faceResult?.faceLandmarks?.length ?? 0;
    //statusEl.textContent = `faces: ${faceCount} | filter: ${filters[currentFilter]?.name} | img: ${filters[currentFilter]?.image ? "ok" : "null"}`;

    // Apply face filters
    if (faceCount > 0) {
      for (const faceLandmarks of faceResult.faceLandmarks) {
        const f = filters[currentFilter];
        const fn = PLACE[f.placement];
        if (fn) fn(faceLandmarks, f.image, w, h);
      }
    }

    // Hand swipe
    if (handResult?.landmarks?.length) {
      const wristX = handResult.landmarks[0][0].x;
      if (prevWristX !== null && now - lastSwipeTime > SWIPE_COOLDOWN) {
        const diff = wristX - prevWristX;
        if (diff > 0.1) {
          currentFilter = (currentFilter - 1 + filters.length) % filters.length;
          lastSwipeTime = now;
          updateFilterLabel();
        } else if (diff < -0.1) {
          currentFilter = (currentFilter + 1) % filters.length;
          lastSwipeTime = now;
          updateFilterLabel();
        }
      }
      prevWristX = wristX;
    }

    // REC indicator
    if (isRecording) {
      ctx.fillStyle = "red";
      ctx.beginPath();
      ctx.arc(w - 30, 30, 10, 0, Math.PI * 2);
      ctx.fill();
      ctx.fillStyle = "white";
      ctx.font = "bold 16px sans-serif";
      ctx.fillText("REC", w - 70, 36);
    }
  } catch (e) {
    statusEl.textContent = "Loop error: " + e.message;
    console.error("Loop error:", e);
  }

  requestAnimationFrame(loop);
}

// UI helpers

function updateFilterLabel() {
  filterNameEl.textContent = filters[currentFilter]?.name ?? "";
}

function takeSnapshot() {
  canvas.toBlob((blob) => {
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = `snapshot_${filters[currentFilter].name}.png`;
    a.click();
    URL.revokeObjectURL(url);
  });
}

function toggleRecording() {
  if (!isRecording) {
    recordedChunks = [];
    const stream = canvas.captureStream(30);
    mediaRecorder = new MediaRecorder(stream, { mimeType: "video/webm" });
    mediaRecorder.ondataavailable = (e) => {
      if (e.data.size > 0) recordedChunks.push(e.data);
    };
    mediaRecorder.onstop = () => {
      const blob = new Blob(recordedChunks, { type: "video/webm" });
      const url = URL.createObjectURL(blob);
      const a = document.createElement("a");
      a.href = url;
      a.download = `video_${filters[currentFilter].name}_${Date.now()}.webm`;
      a.click();
      URL.revokeObjectURL(url);
    };
    mediaRecorder.start();
    isRecording = true;
    btnRec.classList.add("active");
    btnRec.textContent = "Stop (V)";
  } else {
    mediaRecorder.stop();
    isRecording = false;
    btnRec.classList.remove("active");
    btnRec.textContent = "Record (V)";
  }
}

// Button & keyboard events 

btnPrev.addEventListener("click", () => {
  currentFilter = (currentFilter - 1 + filters.length) % filters.length;
  updateFilterLabel();
});
btnNext.addEventListener("click", () => {
  currentFilter = (currentFilter + 1) % filters.length;
  updateFilterLabel();
});
btnSnap.addEventListener("click", takeSnapshot);
btnRec.addEventListener("click", toggleRecording);

document.addEventListener("keydown", (e) => {
  if (e.code === "Space") { e.preventDefault(); takeSnapshot(); }
  if (e.key === "v") { toggleRecording(); }
  if (e.key === "Escape") { /* nothing to close in browser */ }
  if (e.key === "ArrowLeft") { currentFilter = (currentFilter - 1 + filters.length) % filters.length; updateFilterLabel(); }
  if (e.key === "ArrowRight") { currentFilter = (currentFilter + 1) % filters.length; updateFilterLabel(); }
});

// Start

init().catch((err) => {
  statusEl.textContent = "Error: " + err.message;
  console.error(err);
});
