const sequenceLength = 45;
const featureCount = 258;
const handFeatureCount = 126;
const confidenceThreshold = 0.35;
const immediateRecognitionThreshold = 0.8;
const warmupLength = 15;
const minimumSegmentFrames = 12;
const stableGapFrames = 8;
const maximumSegmentFrames = 90;
const motionThreshold = 0.012;
const apiBase = (window.ISL_API_BASE_URL || window.location.origin).replace(/\/$/, "");

const elements = {
  video: document.querySelector("#camera"),
  preview: document.querySelector("#preview"),
  placeholder: document.querySelector("#camera-placeholder"),
  start: document.querySelector("#start-camera"),
  capture: document.querySelector("#capture"),
  stop: document.querySelector("#stop-camera"),
  upload: document.querySelector("#video-upload"),
  referenceUpload: document.querySelector("#reference-video-upload"),
  referenceVideo: document.querySelector("#reference-video"),
  reset: document.querySelector("#reset"),
  retry: document.querySelector("#retry"),
  message: document.querySelector("#message"),
  apiStatus: document.querySelector("#api-status"),
  badge: document.querySelector("#capture-badge"),
  quality: document.querySelector("#quality-label"),
  mediapipe: document.querySelector("#mediapipe-state"),
  resultState: document.querySelector("#result-state"),
  pose: document.querySelector("#pose-state"),
  left: document.querySelector("#left-state"),
  right: document.querySelector("#right-state"),
  frameCount: document.querySelector("#frame-count"),
  diagnostics: document.querySelector("#frame-diagnostics"),
  expectedSign: document.querySelector("#expected-sign"),
  progress: document.querySelector("#progress-bar"),
  prediction: document.querySelector("#prediction"),
  confidence: document.querySelector("#confidence"),
  referenceMatch: document.querySelector("#reference-match"),
  alternatives: document.querySelector("#alternatives"),
  action: document.querySelector("#action-result"),
  faceToggle: document.querySelector("#face-toggle"),
  singleMode: document.querySelector("#single-mode"),
  continuousMode: document.querySelector("#continuous-mode"),
  threshold: document.querySelector("#confidence-threshold"),
  history: document.querySelector("#history"),
  historyCount: document.querySelector("#history-count"),
  clearHistory: document.querySelector("#clear-history"),
  classFilter: document.querySelector("#class-filter"),
  classList: document.querySelector("#class-list"),
  model: document.querySelector("#model-version"),
  latency: document.querySelector("#latency")
};

let stream;
let hands;
let animationFrame;
let processing = false;
let warmupFrames = 0;
let sequence = [];
let handSequence = [];
let capturing = false;
let predictionHistory = [];
let diagnosticFrames = { total: 0, pose: 0, left: 0, right: 0 };
let validationTrials = [];
let uploadMode = false;
let uploadVideo;
let uploadResolve;
let referenceVideoUrl;
let actionMapping = {};
let faceVisible = true;
let continuousMode = false;
let activeSegment = [];
let segmentGap = 0;
let previousMotionPoints;
let segmentBusy = false;
let recognitionHistory = [];
let classNames = [];

fetch("action-mapping.json").then((response) => response.ok ? response.json() : {}).then((mapping) => {
  actionMapping = mapping;
}).catch(() => {});
fetch("classes.json").then((response) => response.ok ? response.json() : {}).then((mapping) => {
  classNames = Object.keys(mapping).sort((a, b) => Number(a) - Number(b)).map((key) => mapping[key]);
  elements.expectedSign.replaceChildren(new Option("Optional validation sign", ""), ...classNames.map((name) => new Option(name, name)));
  renderClassList();
}).catch(() => {});

function setMessage(text, kind = "") {
  elements.message.textContent = text;
  elements.message.dataset.kind = kind;
}

function setSignal(element, present) {
  element.textContent = present ? "READY" : "MISSING";
  element.classList.toggle("good", present);
}

function setFrameCount(count) {
  elements.frameCount.textContent = count;
  elements.progress.style.width = `${Math.round((count / sequenceLength) * 100)}%`;
}

function setMode(continuous) {
  continuousMode = continuous;
  elements.singleMode.classList.toggle("active", !continuous);
  elements.continuousMode.classList.toggle("active", continuous);
  setMessage(continuous ? "Continuous mode: move to start a sign, pause to submit it." : "Single-sign mode: capture when ready.");
}

function updateDiagnostics(quality) {
  diagnosticFrames.total += 1;
  diagnosticFrames.pose += quality.pose ? 1 : 0;
  diagnosticFrames.left += quality.left ? 1 : 0;
  diagnosticFrames.right += quality.right ? 1 : 0;
  elements.diagnostics.textContent = `${diagnosticFrames.total} / ${diagnosticFrames.pose} / ${diagnosticFrames.left} / ${diagnosticFrames.right}`;
}

function drawPrivacyPreview(faceLandmarks = []) {
  const context = elements.preview.getContext("2d");
  if (!elements.video.videoWidth || !elements.video.videoHeight) return;
  elements.preview.width = elements.video.videoWidth;
  elements.preview.height = elements.video.videoHeight;
  context.save();
  context.scale(-1, 1);
  context.drawImage(elements.video, -elements.preview.width, 0, elements.preview.width, elements.preview.height);
  context.restore();

  if (faceVisible) return;
  const points = faceLandmarks || [];
  let bounds;
  if (points.length) {
    const xs = points.map((point) => (1 - point.x) * elements.preview.width);
    const ys = points.map((point) => point.y * elements.preview.height);
    bounds = {
      left: Math.max(0, Math.min(...xs) - 18),
      top: Math.max(0, Math.min(...ys) - 14),
      right: Math.min(elements.preview.width, Math.max(...xs) + 18),
      bottom: Math.min(elements.preview.height, Math.max(...ys) + 14)
    };
  } else {
    bounds = { left: elements.preview.width * 0.28, top: 0, right: elements.preview.width * 0.72, bottom: elements.preview.height * 0.38 };
  }
  context.fillStyle = "rgba(8, 35, 40, 0.96)";
  context.fillRect(bounds.left, bounds.top, bounds.right - bounds.left, bounds.bottom - bounds.top);
  context.fillStyle = "rgba(255, 255, 255, 0.85)";
  context.font = `${Math.max(12, elements.preview.width / 70)}px Arial`;
  context.fillText("FACE HIDDEN", bounds.left + 10, Math.min(bounds.bottom - 10, bounds.top + 24));
}

function renderClassList() {
  const filter = elements.classFilter.value.trim().toLowerCase();
  elements.classList.replaceChildren(...classNames.filter((name) => name.toLowerCase().includes(filter)).map((name) => {
    const item = document.createElement("div");
    item.textContent = name;
    return item;
  }));
}

function motionScore(results) {
  const points = [];
  const detected = getHandLandmarks(results);
  for (const landmarks of [detected.Left, detected.Right]) {
    if (landmarks?.[0]) points.push(landmarks[0]);
  }
  if (!points.length || !previousMotionPoints || previousMotionPoints.length !== points.length) {
    previousMotionPoints = points;
    return 0;
  }
  const score = points.reduce((total, point, index) => total + Math.hypot(point.x - previousMotionPoints[index].x, point.y - previousMotionPoints[index].y), 0) / points.length;
  previousMotionPoints = points;
  return score;
}

function sampleFrames(frames) {
  return Array.from({ length: sequenceLength }, (_, index) => frames[Math.floor(index * (frames.length - 1) / (sequenceLength - 1))]);
}

function segmentFrames(frames) {
  const segments = [];
  let current = [];
  let gap = 0;
  for (const frame of frames) {
    if (!current.length && frame.motion < motionThreshold) continue;
    if (frame.motion >= motionThreshold) gap = 0;
    else gap += 1;
    current.push(frame);
    if (current.length >= minimumSegmentFrames && (gap >= stableGapFrames || current.length >= maximumSegmentFrames)) {
      if (current.length >= sequenceLength) segments.push(sampleFrames(current));
      current = [];
      gap = 0;
    }
  }
  if (current.length >= sequenceLength) segments.push(sampleFrames(current));
  return segments.filter((segment) => segment.length === sequenceLength);
}

function submitSegment(segment) {
  if (segmentBusy || segment.length < sequenceLength) return;
  activeSegment = [];
  segmentGap = 0;
  segmentBusy = true;
  const sampled = sampleFrames(segment);
  sequence = sampled.map((frame) => frame.features);
  handSequence = sampled.map((frame) => frame.handFeatures);
  void sendSequence().finally(() => { segmentBusy = false; });
}

function extractLandmarks(results) {
  const pose = results.poseLandmarks || [];
  const left = results.leftHandLandmarks || [];
  const right = results.rightHandLandmarks || [];
  const values = [];

  for (let index = 0; index < 33; index += 1) {
    const landmark = pose[index];
    values.push(landmark ? landmark.x : 0, landmark ? landmark.y : 0, landmark ? landmark.z : 0, landmark ? (landmark.visibility ?? 0) : 0);
  }
  for (const landmarks of [left, right]) {
    for (let index = 0; index < 21; index += 1) {
      const landmark = landmarks[index];
      values.push(landmark ? landmark.x : 0, landmark ? landmark.y : 0, landmark ? landmark.z : 0);
    }
  }
  if (values.length !== featureCount || values.some((value) => !Number.isFinite(value))) {
    throw new Error("Invalid browser landmark vector");
  }
  return values;
}

function normalizeHandLandmarks(landmarks) {
  const points = landmarks.map((landmark) => [landmark.x, landmark.y, landmark.z]);
  const wrist = points[0];
  const centered = points.map((point) => point.map((value, index) => value - wrist[index]));
  const scale = Math.max(...centered.map((point) => Math.hypot(...point)));
  return centered.flatMap((point) => scale > 0 ? point.map((value) => value / scale) : point);
}

function getHandLandmarks(results) {
  const detected = results.multiHandLandmarks || [];
  const handedness = results.multiHandedness || [];
  const handsBySide = { Left: [], Right: [] };
  detected.forEach((landmarks, index) => {
    const label = handedness[index]?.label;
    if (label in handsBySide && handsBySide[label].length === 0) handsBySide[label] = landmarks;
  });
  return handsBySide;
}

function extractHandLandmarks(results) {
  const detected = getHandLandmarks(results);
  const left = detected.Left;
  const right = detected.Right;
  const values = [
    ...(left.length === 21 ? normalizeHandLandmarks(left) : Array(63).fill(0)),
    ...(right.length === 21 ? normalizeHandLandmarks(right) : Array(63).fill(0))
  ];
  if (values.length !== handFeatureCount || values.some((value) => !Number.isFinite(value))) {
    throw new Error("Invalid browser hand landmark vector");
  }
  return values;
}

function assessQuality(results) {
  const detected = getHandLandmarks(results);
  const pose = false;
  const left = detected.Left.length === 21;
  const right = detected.Right.length === 21;
  // The hand-only model accepts the same zero-fill convention as its dataset;
  // a usable capture requires at least one detected hand, not a pose landmark.
  return { pose, left, right, complete: left || right };
}

function clearResult() {
  predictionHistory = [];
  elements.resultState.textContent = "NO RESULT";
  elements.prediction.textContent = "Gesture unclear";
  elements.confidence.textContent = "--";
  elements.referenceMatch.textContent = "--";
  elements.alternatives.replaceChildren();
  elements.action.textContent = "None configured";
}

function showPrediction(result) {
  const top = result.top_predictions || [];
  const best = top[0];
  const threshold = Number(elements.threshold.value);
  const expected = elements.expectedSign.value.trim();
  const match = result.reference_match;
  elements.referenceMatch.textContent = match?.available ? `${match.match_percent.toFixed(1)}%` : "Unavailable";
  if (!best || best.confidence < threshold) {
    elements.resultState.textContent = expected ? "NOT MATCHING" : "NO SIGN DETECTED";
    elements.prediction.textContent = best ? best.label : "No sign detected";
    elements.confidence.textContent = best ? `${(best.confidence * 100).toFixed(1)}%` : "--";
    recordValidation(result, best, top);
    return;
  }

  predictionHistory.push(best.index);
  predictionHistory = predictionHistory.slice(-3);
  const stable = best.confidence >= immediateRecognitionThreshold || predictionHistory.filter((index) => index === best.index).length >= 2;
  if (expected) {
    elements.resultState.textContent = stable && best.label === expected && match?.available ? "MATCHING" : "NOT MATCHING";
    elements.prediction.textContent = best.label;
  } else if (!stable) {
    elements.resultState.textContent = "STABILIZING";
    elements.prediction.textContent = best.label;
  } else {
    elements.resultState.textContent = match?.available ? "MATCHING" : "REFERENCE UNAVAILABLE";
    elements.prediction.textContent = best.label;
  }
  elements.confidence.textContent = `${(best.confidence * 100).toFixed(1)}%`;
  elements.alternatives.replaceChildren(...top.slice(1).map((item) => {
    const row = document.createElement("div");
    row.className = "alternative";
    row.innerHTML = `<span>${item.label}</span><strong>${(item.confidence * 100).toFixed(1)}%</strong>`;
    return row;
  }));
  elements.model.textContent = result.model_version || "--";
  elements.action.textContent = actionMapping[best.label] || "None configured";
  addHistory(best, result);
  recordValidation(result, best, top);
}

function addHistory(best, result) {
  if (!best || best.confidence < Number(elements.threshold.value)) return;
  const item = { time: new Date(), label: best.label, confidence: best.confidence, action: actionMapping[best.label] || "None configured" };
  recognitionHistory.unshift(item);
  elements.historyCount.textContent = recognitionHistory.length;
  elements.history.replaceChildren(...recognitionHistory.map((entry) => {
    const row = document.createElement("div");
    row.className = "history-item";
    row.innerHTML = `<span>${entry.time.toLocaleTimeString()}</span><strong>${entry.label}</strong><span>${(entry.confidence * 100).toFixed(1)}%</span>`;
    return row;
  }));
}

function recordValidation(result, best, top) {
  const expected = elements.expectedSign.value.trim();
  if (expected && best) {
    validationTrials.push({
      expected,
      prediction: best.label,
      confidence: best.confidence,
      top3: top.map((item) => ({ label: item.label, confidence: item.confidence })),
      quality: result.quality || {},
      timestamp: new Date().toISOString(),
      correct: best.label === expected
    });
    const blob = new Blob([JSON.stringify(validationTrials, null, 2)], { type: "application/json" });
    const link = document.createElement("a");
    link.href = URL.createObjectURL(blob);
    link.download = "webcam-validation.json";
    link.click();
    URL.revokeObjectURL(link.href);
  }
}

async function checkApi() {
  try {
    const response = await fetch(`${apiBase}/health/hand`, { signal: AbortSignal.timeout(5000) });
    if (!response.ok) throw new Error("Hand-recognition API unavailable");
    elements.apiStatus.textContent = "Hand API online";
    elements.apiStatus.classList.add("online");
    return true;
  } catch (error) {
    elements.apiStatus.textContent = "API offline";
    elements.apiStatus.classList.remove("online");
    return false;
  }
}

async function sendSequence() {
  const started = performance.now();
  elements.resultState.textContent = "ANALYZING";
  elements.prediction.textContent = "Reading gesture...";
  try {
    const response = await fetch(`${apiBase}/predict/hand-sequence`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ hand_sequence: handSequence, expected_sign: elements.expectedSign.value.trim() || null }),
      signal: AbortSignal.timeout(20000)
    });
    if (!response.ok) throw new Error(`API returned ${response.status}`);
    const result = await response.json();
    showPrediction(result);
    elements.latency.textContent = `${Math.round(performance.now() - started)} ms`;
    setMessage("Sequence analyzed. Capture another sign when ready.");
    return result;
  } catch (error) {
    elements.resultState.textContent = "ERROR";
    elements.prediction.textContent = "Inference unavailable";
    elements.confidence.textContent = "--";
    setMessage(error.message, "error");
    await checkApi();
  }
}

function onResults(results) {
  drawPrivacyPreview();
  const quality = assessQuality(results);
    const motion = motionScore(results);
  updateDiagnostics(quality);
  setSignal(elements.pose, quality.pose);
  setSignal(elements.left, quality.left);
  setSignal(elements.right, quality.right);

  if (quality.complete) warmupFrames += 1;

  if (uploadMode) {
    if (uploadResolve) {
      const resolve = uploadResolve;
      uploadResolve = undefined;
      resolve({
        quality,
        features: quality.complete ? extractLandmarks(results) : undefined,
        handFeatures: quality.complete ? extractHandLandmarks(results) : undefined,
        motion
      });
    }
    return;
  }

  if (continuousMode) {
    if (quality.complete && (motion >= motionThreshold || activeSegment.length)) {
      activeSegment.push({ features: extractLandmarks(results), handFeatures: extractHandLandmarks(results), motion });
      segmentGap = motion >= motionThreshold ? 0 : segmentGap + 1;
      setFrameCount(Math.min(activeSegment.length, sequenceLength));
      elements.badge.textContent = `CAPTURING ${Math.min(activeSegment.length, sequenceLength)}/${sequenceLength}`;
      if (!segmentBusy && activeSegment.length >= sequenceLength && (segmentGap >= stableGapFrames || activeSegment.length >= maximumSegmentFrames)) {
        submitSegment(activeSegment);
      }
    } else if (!quality.complete && activeSegment.length >= sequenceLength) {
      segmentGap += 1;
      if (segmentGap >= stableGapFrames) submitSegment(activeSegment);
    }
    const warmed = warmupFrames >= warmupLength;
    elements.quality.textContent = warmed ? "WAITING FOR SIGN" : `WARMUP ${Math.min(warmupFrames, warmupLength)}/${warmupLength}`;
    elements.capture.disabled = true;
    return;
  }

  if (!quality.complete) {
    warmupFrames = 0;
    if (capturing) {
      sequence = [];
      handSequence = [];
      setFrameCount(0);
      setMessage("Keep at least one hand visible. Capture restarted.", "warning");
    }
  }
  const warmed = warmupFrames >= warmupLength;
  elements.quality.textContent = warmed ? "READY" : `WARMUP ${Math.min(warmupFrames, warmupLength)}/${warmupLength}`;
  elements.capture.disabled = !warmed || capturing;

  if (capturing && quality.complete) {
    sequence.push(extractLandmarks(results));
    handSequence.push(extractHandLandmarks(results));
    setFrameCount(sequence.length);
    elements.badge.textContent = `CAPTURING ${sequence.length}/${sequenceLength}`;
    if (sequence.length === sequenceLength) {
      capturing = false;
      elements.badge.textContent = "ANALYZING";
      elements.capture.disabled = true;
      void sendSequence();
    }
  }
}

async function processUploadedVideo(file) {
  await initializeHands();
  uploadMode = true;
  cancelAnimationFrame(animationFrame);
  elements.badge.textContent = "ANALYZING VIDEO";
  elements.resultState.textContent = "ANALYZING";
  elements.prediction.textContent = "Extracting temporal landmarks...";
  setMessage("Sampling 45 frames from the uploaded video...");
  uploadVideo = document.createElement("video");
  uploadVideo.muted = true;
  uploadVideo.playsInline = true;
  uploadVideo.src = URL.createObjectURL(file);
  try {
    await new Promise((resolve, reject) => {
      uploadVideo.onloadedmetadata = resolve;
      uploadVideo.onerror = () => reject(new Error("Could not read uploaded video"));
    });
    uploadVideo.pause();
    const validFrames = [];
    const sampleCount = Math.max(sequenceLength, Math.ceil(uploadVideo.duration * 30));
    const times = Array.from({ length: sampleCount }, (_, index) => uploadVideo.duration * index / (sampleCount - 1));
    for (const [index, time] of times.entries()) {
      if (index > 0) {
        await new Promise((resolve, reject) => {
          uploadVideo.onseeked = resolve;
          uploadVideo.onerror = () => reject(new Error("Could not decode uploaded video frame"));
          uploadVideo.currentTime = time;
        });
      }
      const result = await new Promise((resolve, reject) => {
        uploadResolve = resolve;
        hands.send({ image: uploadVideo }).catch(reject);
      });
      if (result.quality.complete) {
        validFrames.push({ features: result.features, handFeatures: result.handFeatures, motion: result.motion });
      }
    }
    if (validFrames.length < sequenceLength) {
      throw new Error(`Uploaded video has only ${validFrames.length}/${sequenceLength} valid landmark frames`);
    }
    const segments = continuousMode ? segmentFrames(validFrames) : [sampleFrames(validFrames)];
    if (!segments.length) throw new Error("No complete sign segments were detected");
    for (const segment of segments) {
      sequence = segment.map((frame) => frame.features);
      handSequence = segment.map((frame) => frame.handFeatures);
      setFrameCount(sequence.length);
      await sendSequence();
    }
  } catch (error) {
    elements.resultState.textContent = "ERROR";
    elements.prediction.textContent = "Video not clear";
    setMessage(error.message, "error");
  } finally {
    uploadMode = false;
    uploadResolve = undefined;
    const uploadUrl = uploadVideo.src;
    uploadVideo.removeAttribute("src");
    uploadVideo.load();
    URL.revokeObjectURL(uploadUrl);
    uploadVideo = undefined;
    elements.badge.textContent = stream ? "IDLE" : "VIDEO COMPLETE";
    if (stream) animationFrame = requestAnimationFrame(frameLoop);
  }
}

async function frameLoop() {
  if (!stream || uploadMode) return;
  if (!processing) {
    processing = true;
    try {
      await hands.send({ image: elements.video });
    } catch (error) {
      setMessage(`Landmark processing failed: ${error.message}`, "error");
    } finally {
      processing = false;
    }
  }
  animationFrame = requestAnimationFrame(frameLoop);
}

async function initializeHands() {
  if (hands) return;
  elements.mediapipe.textContent = "INITIALIZING";
  hands = new Hands({ locateFile: (file) => `vendor/mediapipe-hands/${file}` });
  hands.setOptions({ maxNumHands: 2, modelComplexity: 1, minDetectionConfidence: 0.5, minTrackingConfidence: 0.5 });
  hands.onResults(onResults);
  elements.mediapipe.textContent = "READY";
}

async function startCamera() {
  if (stream) return;
  try {
    stream = await navigator.mediaDevices.getUserMedia({ video: { width: { ideal: 960 }, height: { ideal: 720 }, facingMode: "user" }, audio: false });
    elements.video.srcObject = stream;
    await elements.video.play();
    await initializeHands();
    warmupFrames = 0;
    sequence = [];
    handSequence = [];
    diagnosticFrames = { total: 0, pose: 0, left: 0, right: 0 };
    elements.diagnostics.textContent = "0 / 0 / 0 / 0";
    setFrameCount(0);
    elements.placeholder.hidden = true;
    elements.start.disabled = true;
    elements.stop.disabled = false;
    setMessage("Camera ready. Hold both hands and your pose in frame.");
    animationFrame = requestAnimationFrame(frameLoop);
    await checkApi();
  } catch (error) {
    stopCamera();
    elements.mediapipe.textContent = "ERROR";
    setMessage(`Could not start camera: ${error.message}`, "error");
  }
}

function stopCamera() {
  cancelAnimationFrame(animationFrame);
  if (stream) stream.getTracks().forEach((track) => track.stop());
  stream = undefined;
  hands = undefined;
  capturing = false;
  warmupFrames = 0;
  sequence = [];
  handSequence = [];
  diagnosticFrames = { total: 0, pose: 0, left: 0, right: 0 };
  elements.video.srcObject = null;
  elements.placeholder.hidden = false;
  elements.start.disabled = false;
  elements.capture.disabled = true;
  elements.stop.disabled = true;
  elements.badge.textContent = "IDLE";
  elements.quality.textContent = "WAITING";
  elements.mediapipe.textContent = "NOT STARTED";
  setFrameCount(0);
  setSignal(elements.pose, false);
  setSignal(elements.left, false);
  setSignal(elements.right, false);
  clearResult();
  setMessage("Camera stopped.");
}

elements.start.addEventListener("click", startCamera);
elements.stop.addEventListener("click", stopCamera);
elements.faceToggle.addEventListener("click", () => {
  faceVisible = !faceVisible;
  elements.faceToggle.textContent = faceVisible ? "ON" : "OFF";
  elements.faceToggle.classList.toggle("off", !faceVisible);
  document.querySelector("#privacy-badge").textContent = faceVisible ? "FACE VISIBLE" : "PRIVACY MODE: ON";
  if (faceVisible) drawPrivacyPreview();
});
elements.singleMode.addEventListener("click", () => setMode(false));
elements.continuousMode.addEventListener("click", () => setMode(true));
elements.threshold.addEventListener("change", () => {
  const value = Number(elements.threshold.value);
  elements.threshold.value = Number.isFinite(value) ? Math.min(1, Math.max(0, value)) : 0.35;
});
elements.classFilter.addEventListener("input", renderClassList);
elements.clearHistory.addEventListener("click", () => {
  recognitionHistory = [];
  elements.historyCount.textContent = "0";
  elements.history.replaceChildren(Object.assign(document.createElement("span"), { className: "history-empty", textContent: "No signs recognized yet" }));
});
elements.upload.addEventListener("change", async (event) => {
  const [file] = event.target.files;
  if (!file) return;
  event.target.value = "";
  try {
    await processUploadedVideo(file);
  } catch (error) {
    setMessage(error.message, "error");
  }
});
elements.referenceUpload.addEventListener("change", (event) => {
  const [file] = event.target.files;
  event.target.value = "";
  if (!file) return;
  if (referenceVideoUrl) URL.revokeObjectURL(referenceVideoUrl);
  referenceVideoUrl = URL.createObjectURL(file);
  elements.referenceVideo.src = referenceVideoUrl;
  elements.referenceVideo.hidden = false;
  setMessage(`Reference video selected: ${file.name}`);
});
elements.reset.addEventListener("click", () => {
  sequence = [];
  handSequence = [];
  setFrameCount(0);
  clearResult();
  setMessage(stream ? "Result cleared. Capture a new sign when ready." : "Result cleared.");
});
elements.retry.addEventListener("click", async () => {
  const online = await checkApi();
  if (!online) setMessage("API is still unavailable. Check the service and retry.", "error");
});
elements.capture.addEventListener("click", () => {
  if (!stream || warmupFrames < warmupLength) return;
  capturing = true;
  sequence = [];
  handSequence = [];
  setFrameCount(0);
  elements.badge.textContent = "CAPTURING 0/45";
  elements.capture.disabled = true;
  elements.resultState.textContent = "RECORDING";
  elements.prediction.textContent = "Hold gesture steady";
  setMessage("Capturing 45 valid frames...");
});

document.querySelector("#privacy-badge").textContent = "FACE VISIBLE";
void checkApi();
