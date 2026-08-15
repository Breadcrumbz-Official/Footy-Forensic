/* mediapipe.js — MediaPipe Tasks Vision PoseLandmarker wrapper.
 *
 * Uses the current @mediapipe/tasks-vision web API (FilesetResolver +
 * PoseLandmarker.createFromOptions). Everything runs in the browser; no frame
 * is ever uploaded anywhere. The model and wasm are fetched from a CDN once and
 * then cached by the browser.
 *
 * Runs in VIDEO mode via detectForVideo(), fed short clips (not single stills)
 * around each picked moment — see js/video.js:captureWindow(). VIDEO mode gives
 * MediaPipe temporal context across those frames, which damps the landmark
 * jitter a single motion-blurred frame would otherwise produce; IMAGE mode has
 * no such context. Confirmed against the published API types (vision.d.ts) for
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

/**
 * Run pose detection across a short ordered clip of frames (VIDEO mode), so
 * MediaPipe can use temporal context instead of judging each frame in
 * isolation. `items` is `[{canvas, time}]` with `time` the real capture time
 * in seconds, ascending within the clip.
 * @returns {Array<{landmarks:Array, worldLandmarks:Array}|null>} one entry per
 *   input frame, null where no person was found in that frame.
 */
export async function detectPoseSequence(items) {
  if (!landmarker) await initPose();
  const out = [];
  let prevTime = null;
  for (const { canvas, time } of items) {
    const deltaMs = prevTime == null ? 33 : Math.max(1, Math.round((time - prevTime) * 1000));
    virtualClockMs += deltaMs;
    prevTime = time;
    const res = landmarker.detectForVideo(canvas, virtualClockMs);
    out.push(res?.landmarks?.length
      ? { landmarks: res.landmarks[0], worldLandmarks: res.worldLandmarks?.[0] ?? null }
      : null);
  }
  virtualClockMs += 200; // gap so the next clip never collides, whatever real time it comes from
  return out;
}

// Live-preview detection (during camera recording) is driven by a timer, not
// a fixed frame source, so frame spacing is irregular in a way the burst clip
// above isn't — we measure real elapsed time between calls instead of
// assuming a dt.
let lastLiveAt = null;

/**
 * Run one live-preview detection, meant to be called on a throttled interval
 * while the camera preview is on (see app.js) to drive an optional live
 * skeleton/joint-angle overlay. Shares the same landmarker and monotonic
 * clock as detectPoseSequence, so it stays valid to mix the two in one
 * session (live preview, then later phase-selection on the recorded clip).
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
