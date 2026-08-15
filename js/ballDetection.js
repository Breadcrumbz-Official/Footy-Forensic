/* ballDetection.js — finds the ball in the LIVE camera preview.
 *
 * This is a framing aid, not a measurement: it shows the player that the ball
 * is in shot and being seen before they commit to a take. Nothing here is
 * scored. The real ball tracking — the whole-clip Viterbi track, the size and
 * shape gates, the rescue crop — lives in server/ball_detection.py, which sees
 * the full-resolution original and has future frames to reason over. Keeping a
 * second copy of that here is how the two would drift apart.
 *
 * API confirmed against the published types for @mediapipe/tasks-vision@0.10.35:
 *   ObjectDetector.createFromOptions(fileset, options): Promise<ObjectDetector>
 *   detectForVideo(videoFrame, timestamp, opts?): DetectionResult
 *   DetectionResult.detections[].boundingBox = {originX, originY, width, height}
 *   DetectionResult.detections[].categories[] = {categoryName, score, ...}
 * Bounding boxes come back in PIXELS of the frame passed in, not normalized.
 */

import { FilesetResolver, ObjectDetector }
  from 'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.35/vision_bundle.mjs';

const WASM_PATH = 'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.35/wasm';
// Lite0 (4.5MB) was tried first for the smaller download, but it was trained
// almost entirely on COCO's small, distant "sports ball" examples: tested
// against a ball filling a real framing shot, its top guess was "frisbee" or
// "suitcase" — never "sports ball" above the score floor, so the live overlay
// never drew anything. Lite2 int8 (7.5MB, same architecture the server uses
// for the analysis that counts) gets both close-up and distant balls right
// with 0.6-0.8 confidence. Confirmed by direct test against real photos.
const MODEL_URL =
  'https://storage.googleapis.com/mediapipe-models/object_detector/efficientdet_lite2/int8/1/efficientdet_lite2.tflite';

const BALL_CATEGORY = 'sports ball';   // the COCO class a football comes back as
const SCORE_FLOOR = 0.2;

/* Plausibility gates — all scale-free, mirroring the server's. */
// A ball's bounding box is near-square from any angle. Long thin boxes are
// line markings, shins, or smeared grass. Loose, because a fast ball smears.
const MIN_ROUNDNESS = 0.5;
// A size-5 ball is ~22cm across; hip-to-shoulder is ~50cm, so a ball is ~0.45
// torso lengths wide. Generous, because the ball can sit nearer or further
// from the camera than the player.
const MIN_DIAM_TORSO = 0.18;
const MAX_DIAM_TORSO = 0.95;

// Cost of jumping: a struck ball leaves at ~25m/s, which at 30fps is ~1.7
// torso lengths in one frame.
const STEP_WEIGHT = 0.6;
const MAX_STEP_TORSO = 2.5;

// CPU, deliberately. The GPU delegate for ObjectDetector hung the renderer
// outright in testing — not a rejected promise but a synchronous wasm-level
// wedge, which no try/catch or Promise.race can recover from, so the usual
// "try GPU, fall back to CPU" pattern used in mediapipe.js is not safe here.
// The pose landmarker keeps GPU because it drives the overlay at 30fps, where
// the difference actually matters; this runs at ~5fps.
const DELEGATE = 'CPU';

let detector = null;
let backend = null;
let initPromise = null;

// detectForVideo needs strictly increasing timestamps for the life of the
// instance, so we drive it off our own counter rather than real video time.
let clockMs = 0;

function build(vision) {
  return ObjectDetector.createFromOptions(vision, {
    baseOptions: { modelAssetPath: MODEL_URL, delegate: DELEGATE },
    runningMode: 'VIDEO',
    // Ask the model for nothing else. Cheaper than filtering after the fact,
    // and a stray "person" or "tennis racket" can never reach us.
    categoryAllowlist: [BALL_CATEGORY],
    scoreThreshold: SCORE_FLOOR,
    maxResults: 8
  });
}

/** Idempotent init. See DELEGATE above for why this is CPU-only. */
export function initBall() {
  if (initPromise) return initPromise;
  initPromise = (async () => {
    const vision = await FilesetResolver.forVisionTasks(WASM_PATH);
    detector = await build(vision);
    backend = DELEGATE;
    return { backend };
  })();
  return initPromise;
}

export const getBallBackend = () => backend;

/** Detections from one frame, reduced to plausible ball candidates. */
function candidates(result, scale) {
  const out = [];
  for (const d of result?.detections ?? []) {
    const b = d.boundingBox;
    if (!b || b.width <= 0 || b.height <= 0) continue;
    if (Math.min(b.width, b.height) / Math.max(b.width, b.height) < MIN_ROUNDNESS) continue;

    const diam = (b.width + b.height) / 2;
    // Only size-gate when we actually know how big the player is; with no pose
    // we would be inventing a reference.
    if (scale > 0) {
      const rel = diam / scale;
      if (rel < MIN_DIAM_TORSO || rel > MAX_DIAM_TORSO) continue;
    }

    out.push({
      x: b.originX + b.width / 2,
      y: b.originY + b.height / 2,
      r: diam / 2,
      score: d.categories?.[0]?.score ?? 0
    });
  }
  return out;
}

// The live overlay cannot use the server's whole-clip track: that needs every
// frame before it can decide anything, and live has no future. Instead we carry
// the last accepted position forward and prefer candidates near it — the same
// "confidence minus travel cost" idea, resolved greedily one frame at a time.
// Weaker than the offline track, which is fine here: nothing is scored from it.
let liveLast = null;
let liveMisses = 0;
let lastLiveAt = null;

// How many consecutive empty frames before we accept the ball is genuinely gone
// rather than briefly blurred or hidden behind the player's own leg.
const LIVE_FORGET_AFTER = 6;

/**
 * One live ball detection for the camera preview.
 * @param {HTMLCanvasElement} canvas current preview frame
 * @param {number|null} scale torso length in px from the live pose, or null
 * @returns {{x,y,r,score}|null} best current estimate, or null
 */
export async function detectBallLive(canvas, scale) {
  if (!detector) await initBall();
  const now = performance.now();
  const deltaMs = lastLiveAt == null ? 33 : Math.max(1, Math.round(now - lastLiveAt));
  lastLiveAt = now;
  clockMs += deltaMs;

  const cands = candidates(detector.detectForVideo(canvas, clockMs), scale || 0);
  if (!cands.length) {
    if (++liveMisses > LIVE_FORGET_AFTER) liveLast = null;
    return liveLast;
  }

  let pick;
  if (liveLast && scale > 0) {
    const cost = v => (1 - v.score) + STEP_WEIGHT *
      Math.min(Math.hypot(v.x - liveLast.x, v.y - liveLast.y) / scale, MAX_STEP_TORSO);
    pick = cands.reduce((best, c) => (cost(c) < cost(best) ? c : best));
  } else {
    pick = cands.reduce((best, c) => (c.score > best.score ? c : best));
  }

  liveMisses = 0;
  liveLast = pick;
  return pick;
}

/** Call when live preview (re)starts, so a gap while it was off is not read as
 *  one huge elapsed frame and a stale position is not carried across. */
export function resetBallLive() {
  lastLiveAt = null;
  liveLast = null;
  liveMisses = 0;
}

/** Release wasm resources. */
export function closeBall() {
  detector?.close();
  detector = null;
  initPromise = null;
  clockMs = 0;
  resetBallLive();
}
