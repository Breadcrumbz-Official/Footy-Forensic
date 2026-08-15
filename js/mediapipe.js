/* mediapipe.js — MediaPipe Tasks Vision PoseLandmarker wrapper.
 *
 * Uses the current @mediapipe/tasks-vision web API (FilesetResolver +
 * PoseLandmarker.createFromOptions). Everything runs in the browser; no frame
 * is ever uploaded anywhere. The model and wasm are fetched from a CDN once and
 * then cached by the browser.
 *
 * This drives the LIVE camera overlay only. Analysis pose — the whole-clip
 * VIDEO-mode pass with the heavy model — runs on the server (server/pose.py),
 * which sees the full-resolution original. Still VIDEO mode rather than IMAGE
 * here, because the preview is a continuous stream and the temporal filter is
 * what stops the skeleton juddering between frames.
 *
 * Confirmed against the published API types (vision.d.ts) for
 * @mediapipe/tasks-vision@0.10.35: detectForVideo(videoFrame, timestamp, opts?).
 */

import { FilesetResolver, PoseLandmarker }
  from 'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.35/vision_bundle.mjs';

const WASM_PATH = 'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.35/wasm';
const MODEL_URL =
  'https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/1/pose_landmarker_full.task';

let landmarker = null;
let backend = null;
let initPromise = null;

// detectForVideo requires a strictly increasing timestamp for the lifetime of
// the landmarker instance. We keep our own virtual clock (ms) instead of using
// real video time, because the three phases are analyzed as separate clips
// that are not chronological relative to each other in this counter's terms
// (e.g. re-picking an earlier phase after a later one). Deltas between
// consecutive frames within one clip still track real elapsed time, since the
// filter's smoothing strength depends on realistic inter-frame timing.
let virtualClockMs = 0;

async function build(vision) {
  return PoseLandmarker.createFromOptions(vision, {
    baseOptions: { modelAssetPath: MODEL_URL, delegate: 'GPU' },
    runningMode: 'VIDEO',
    numPoses: 1,
    minPoseDetectionConfidence: 0.5,
    minPosePresenceConfidence: 0.5
  });
}

/** Idempotent init. Prefers the GPU delegate, falls back to CPU. */
export function initPose() {
  if (initPromise) return initPromise;
  initPromise = (async () => {
    const vision = await FilesetResolver.forVisionTasks(WASM_PATH);
    try {
      landmarker = await build(vision);
      backend = 'GPU';
    } catch (err) {
      console.warn('GPU delegate unavailable, falling back to CPU:', err);
      landmarker = await PoseLandmarker.createFromOptions(vision, {
        baseOptions: { modelAssetPath: MODEL_URL, delegate: 'CPU' },
        runningMode: 'VIDEO',
        numPoses: 1,
        minPoseDetectionConfidence: 0.5,
        minPosePresenceConfidence: 0.5
      });
      backend = 'CPU';
    }
    return { backend };
  })();
  return initPromise;
}

export const getBackend = () => backend;

// Live-preview detection is driven by a timer rather than a fixed frame
// source, so frame spacing is irregular — we measure real elapsed time between
// calls instead of assuming a dt.
let lastLiveAt = null;

/**
 * Run one live-preview detection, called on a throttled interval while the
 * camera preview is on (see app.js) to drive the optional skeleton / joint
 * angle overlay.
 */
export async function detectPoseLive(canvas) {
  if (!landmarker) await initPose();
  const now = performance.now();
  const deltaMs = lastLiveAt == null ? 33 : Math.max(1, Math.round(now - lastLiveAt));
  lastLiveAt = now;
  virtualClockMs += deltaMs;
  const res = landmarker.detectForVideo(canvas, virtualClockMs);
  return res?.landmarks?.length
    ? { landmarks: res.landmarks[0], worldLandmarks: res.worldLandmarks?.[0] ?? null }
    : null;
}

/** Call when live preview (re)starts so a gap while it was off (e.g. the
 *  overlay was toggled off, or the camera was closed) isn't read as one huge
 *  real elapsed frame. */
export function resetLiveClock() {
  lastLiveAt = null;
}

/** Release GPU/wasm resources. */
export function closePose() {
  landmarker?.close();
  landmarker = null;
  initPromise = null;
  virtualClockMs = 0;
  lastLiveAt = null;
}
