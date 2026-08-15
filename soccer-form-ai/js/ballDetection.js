/* ballDetection.js — finds the ball in a clip using MediaPipe's ObjectDetector.
 *
 * Why this is a module and not a single detect() call: one frame's raw output
 * is not trustworthy for this job. The COCO "sports ball" class also fires on
 * round-ish background clutter (in testing, on a stretched patch of grass that
 * scored 0.33), and the one frame we care about most — contact — is exactly
 * where the ball is fastest, most motion-blurred, and most likely to be missed
 * outright. Either failure would quietly corrupt a metric.
 *
 * So we read the WHOLE clip and pick the single most temporally coherent track
 * through it. A real ball moves a short, smooth distance frame to frame; a
 * false positive appears once somewhere unrelated and vanishes. Frames the
 * detector missed inside an otherwise good track are interpolated and flagged
 * as such; frames outside it stay null rather than being extrapolated.
 *
 * Every gate below is expressed relative to the player's own torso length,
 * measured by the pose model on the same frame — so, as everywhere else in
 * this app, there are no hard-coded pixel distances.
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
// EfficientDet-Lite0, uint8 quantized: 4.5MB, versus 13.8MB for the float32
// build of the same model. On a phone that download difference matters more
// than the accuracy difference does, because the temporal track below is what
// actually carries reliability here, not any single frame's confidence.
const MODEL_URL =
  'https://storage.googleapis.com/mediapipe-tasks/object_detector/efficientdet_lite0_uint8.tflite';

const BALL_CATEGORY = 'sports ball';   // the COCO class a football comes back as
const SCORE_FLOOR = 0.2;               // below this the track logic can't rescue it anyway

/* Plausibility gates — all scale-free. */
// A ball's bounding box is near-square from any angle. Long thin boxes are
// line markings, shins, or smeared grass. Kept loose because a ball moving
// fast does smear into an oval.
const MIN_ROUNDNESS = 0.5;
// A size-5 ball is ~22cm across; hip-to-shoulder is ~50cm, so a ball is
// ~0.45 torso lengths wide. The band is generous because the ball can sit
// nearer or further from the camera than the player.
const MIN_DIAM_TORSO = 0.18;
const MAX_DIAM_TORSO = 0.95;

/* Track-selection costs (see bestTrack). */
// Cost of declaring "no ball in this frame". Must sit above the emission cost
// of a plausible detection (1 - score, so ~0.3 at score 0.7) or the track
// would rather see nothing than accept a real ball.
const MISS_COST = 0.8;
const STEP_WEIGHT = 0.6;
// A struck ball leaves at ~25m/s; at 30fps that is ~0.8m, or ~1.7 torso
// lengths, in one frame. Displacement beyond this is capped rather than
// rejected, so a genuinely fast ball is penalised but still reachable.
const MAX_STEP_TORSO = 2.5;

let detector = null;
let backend = null;
let initPromise = null;

// detectForVideo needs strictly increasing timestamps for the life of the
// instance, same constraint as the pose landmarker — and for the same reason
// (phases are separate clips, picked in any order) we drive it off our own
// counter instead of real video time. This one is independent of the pose
// clock; they are different task instances and do not share state.
let clockMs = 0;

function build(vision, delegate) {
  return ObjectDetector.createFromOptions(vision, {
    baseOptions: { modelAssetPath: MODEL_URL, delegate },
    runningMode: 'VIDEO',
    // Ask the model for nothing else. Cheaper than filtering after the fact,
    // and it means a stray "person" or "tennis racket" can never reach us.
    categoryAllowlist: [BALL_CATEGORY],
    scoreThreshold: SCORE_FLOOR,
    maxResults: 8
  });
}

/** Idempotent init. Prefers the GPU delegate, falls back to CPU. */
export function initBall() {
  if (initPromise) return initPromise;
  initPromise = (async () => {
    const vision = await FilesetResolver.forVisionTasks(WASM_PATH);
    try {
      detector = await build(vision, 'GPU');
      backend = 'GPU';
    } catch (err) {
      console.warn('Ball detector: GPU delegate unavailable, falling back to CPU:', err);
      detector = await build(vision, 'CPU');
      backend = 'CPU';
    }
    return { backend };
  })();
  return initPromise;
}

export const getBallBackend = () => backend;

/**
 * Find the ball across one ordered clip.
 *
 * @param {Array<{canvas:HTMLCanvasElement, time:number}>} items clip frames, ascending in time
 * @param {Array<number|null>} scales torso length in px for each frame (null where
 *        no pose was found); used to size- and motion-gate candidates
 * @returns {Array<{x,y,r,score,interpolated}|null>} one entry per input frame,
 *          in that frame's pixel space. null = no trustworthy ball there.
 */
export async function detectBallSequence(items, scales = []) {
  if (!detector) await initBall();

  const filled = fillScales(scales, items.length);
  const perFrame = [];
  let prevTime = null;
  for (let i = 0; i < items.length; i++) {
    const { canvas, time } = items[i];
    const deltaMs = prevTime == null ? 33 : Math.max(1, Math.round((time - prevTime) * 1000));
    clockMs += deltaMs;
    prevTime = time;
    perFrame.push(candidates(detector.detectForVideo(canvas, clockMs), filled[i]));
  }
  clockMs += 200; // gap so the next clip never collides with this one

  return fillGaps(bestTrack(perFrame, filled));
}

/**
 * Torso length is the yardstick for every gate here, but the pose model may
 * have failed on some frames of the clip. Borrow the median of the frames it
 * did read — the player's size does not meaningfully change within ~0.3s.
 */
function fillScales(scales, n) {
  const known = [];
  for (let i = 0; i < n; i++) {
    const s = scales[i];
    if (typeof s === 'number' && isFinite(s) && s > 0) known.push(s);
  }
  const median = known.length
    ? known.slice().sort((a, b) => a - b)[Math.floor(known.length / 2)]
    : 0;
  return Array.from({ length: n }, (_, i) => {
    const s = scales[i];
    return (typeof s === 'number' && isFinite(s) && s > 0) ? s : median;
  });
}

/** Detections from one frame, reduced to plausible ball candidates. */
function candidates(result, scale) {
  const out = [];
  for (const d of result?.detections ?? []) {
    const b = d.boundingBox;
    if (!b || b.width <= 0 || b.height <= 0) continue;
    if (Math.min(b.width, b.height) / Math.max(b.width, b.height) < MIN_ROUNDNESS) continue;

    const diam = (b.width + b.height) / 2;
    // Only size-gate when we actually know how big the player is; with no
    // pose on any frame of the clip we would be inventing a reference.
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

/**
 * Pick the most coherent path through the per-frame candidates (Viterbi).
 *
 * Each frame's states are its candidates plus a "not visible here" state, so
 * the track can survive the ball being missed mid-clip without being forced
 * onto whatever junk the detector did return. Cost of a state is its
 * unconfidence (1 - score); cost of moving between two real candidates is how
 * far the ball would have had to travel, in torso lengths. A false positive is
 * therefore only chosen if nothing more coherent exists — and a lone one, far
 * from everything else, loses to the miss state.
 */
function bestTrack(perFrame, scales) {
  const n = perFrame.length;
  if (!n) return [];

  const MISS = -1;
  const statesAt = i => [...perFrame[i].map((_, k) => k), MISS];
  const emission = (i, k) => k === MISS ? MISS_COST : 1 - perFrame[i][k].score;
  const transition = (i, from, to) => {
    // Nothing to compare against across a gap — moving in or out of the miss
    // state is free, and paying for it twice would just bias against gaps.
    if (from === MISS || to === MISS) return 0;
    const a = perFrame[i - 1][from], b = perFrame[i][to];
    const s = scales[i] || scales[i - 1] || 0;
    if (!s) return 0;
    return STEP_WEIGHT * Math.min(Math.hypot(a.x - b.x, a.y - b.y) / s, MAX_STEP_TORSO);
  };

  let states = statesAt(0);
  let costs = states.map(k => emission(0, k));
  let paths = states.map(k => [k]);

  for (let i = 1; i < n; i++) {
    const next = statesAt(i);
    const nextCosts = [], nextPaths = [];
    for (const to of next) {
      let best = Infinity, bestJ = 0;
      states.forEach((from, j) => {
        const c = costs[j] + transition(i, from, to);
        if (c < best) { best = c; bestJ = j; }
      });
      nextCosts.push(best + emission(i, to));
      nextPaths.push([...paths[bestJ], to]);
    }
    states = next; costs = nextCosts; paths = nextPaths;
  }

  const winner = costs.indexOf(Math.min(...costs));
  return paths[winner].map((k, i) => k === MISS ? null : perFrame[i][k]);
}

/**
 * Fill frames the detector missed *between* two good ones. Over the ~0.3s of a
 * clip the ball's path is close enough to straight for this to beat having no
 * reading at all. We deliberately do not extrapolate past the ends of the
 * track — that would be a guess, not an interpolation — and everything filled
 * in is flagged so the UI can say where the number came from.
 */
function fillGaps(track) {
  const known = [];
  track.forEach((b, i) => { if (b) known.push(i); });
  if (!known.length) return track.map(() => null);

  return track.map((b, i) => {
    if (b) return { ...b, interpolated: false };
    let before = null, after = null;
    for (const k of known) {
      if (k < i) before = k;
      else if (after === null) after = k;
    }
    if (before === null || after === null) return null;
    const t = (i - before) / (after - before);
    const a = track[before], c = track[after];
    return {
      x: a.x + (c.x - a.x) * t,
      y: a.y + (c.y - a.y) * t,
      r: a.r + (c.r - a.r) * t,
      score: Math.min(a.score, c.score),
      interpolated: true
    };
  });
}

/** Release wasm resources. */
export function closeBall() {
  detector?.close();
  detector = null;
  initPromise = null;
  clockMs = 0;
}
