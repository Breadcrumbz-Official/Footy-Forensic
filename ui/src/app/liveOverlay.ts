/* liveOverlay.ts — the live camera overlay: skeleton, joint angles, ball ring.
 *
 * Ported from the previous client (js/mediapipe.js, js/ballDetection.js,
 * js/biomechanics.js, js/overlay.js). This runs entirely in the browser and
 * never uploads a frame — it has to be local, because a round trip per frame
 * would never keep up with a preview.
 *
 * Nothing here is scored. Analysis pose is the server's whole-clip pass with
 * the heavy model at full resolution (server/pose.py); this exists so the
 * player can see that the camera is actually picking them up BEFORE they commit
 * to a take. The joint angles are read off the same three landmark coordinates
 * the server scores with, so a live "142°" at the knee is what that joint would
 * score if you picked that instant.
 */

import { FilesetResolver, PoseLandmarker, ObjectDetector } from '@mediapipe/tasks-vision';
import type { NormalizedLandmark } from '@mediapipe/tasks-vision';

// Keep this in step with the @mediapipe/tasks-vision version in package.json:
// the wasm binaries and the JS API are versioned together and must match.
const MP_VERSION = '0.10.35';
const WASM_PATH = `https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@${MP_VERSION}/wasm`;
const POSE_MODEL =
  'https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/1/pose_landmarker_full.task';
// EfficientDet-Lite0 uint8: 4.5MB against 13.8MB for float32. On a phone that
// download matters more than the accuracy does for a preview. The server runs
// Lite2 for the detection that actually counts.
const BALL_MODEL =
  'https://storage.googleapis.com/mediapipe-tasks/object_detector/efficientdet_lite0_uint8.tflite';

/* ── Geometry ────────────────────────────────────────────────────────────── */

export interface Pt { x: number; y: number; v: number }
export interface Ball { x: number; y: number; r: number; score: number }

/** MediaPipe Pose landmark indices (33-point model). */
export const LM = {
  NOSE: 0,
  L_SH: 11, R_SH: 12, L_EL: 13, R_EL: 14, L_WR: 15, R_WR: 16,
  L_HIP: 23, R_HIP: 24, L_KN: 25, R_KN: 26, L_AN: 27, R_AN: 28,
  L_HEEL: 29, R_HEEL: 30, L_FOOT: 31, R_FOOT: 32,
} as const;

const CONNECTIONS: [number, number][] = [
  [LM.L_SH, LM.R_SH], [LM.L_HIP, LM.R_HIP],
  [LM.L_SH, LM.L_HIP], [LM.R_SH, LM.R_HIP],
  [LM.L_SH, LM.L_EL], [LM.L_EL, LM.L_WR],
  [LM.R_SH, LM.R_EL], [LM.R_EL, LM.R_WR],
  [LM.L_HIP, LM.L_KN], [LM.L_KN, LM.L_AN],
  [LM.L_AN, LM.L_HEEL], [LM.L_HEEL, LM.L_FOOT], [LM.L_AN, LM.L_FOOT],
  [LM.R_HIP, LM.R_KN], [LM.R_KN, LM.R_AN],
  [LM.R_AN, LM.R_HEEL], [LM.R_HEEL, LM.R_FOOT], [LM.R_AN, LM.R_FOOT],
];

const clamp = (v: number, lo: number, hi: number) => Math.min(hi, Math.max(lo, v));
const dist = (a: Pt, b: Pt) => Math.hypot(a.x - b.x, a.y - b.y);
const mid = (a: Pt, b: Pt): Pt => ({ x: (a.x + b.x) / 2, y: (a.y + b.y) / 2, v: Math.min(a.v, b.v) });

/**
 * Convert MediaPipe's normalized landmarks to pixel space.
 *
 * MediaPipe normalizes x by width and y by height independently, so on a
 * non-square frame that space is anisotropically stretched and any angle
 * measured in it is simply wrong. Everything below works in pixels.
 */
export function toPixels(landmarks: NormalizedLandmark[], w: number, h: number): Pt[] {
  return landmarks.map(l => ({ x: l.x * w, y: l.y * h, v: l.visibility ?? 1 }));
}

/** Interior angle at vertex `b`, in degrees, formed by a-b-c. 180 = straight. */
export function angleDeg(a: Pt, b: Pt, c: Pt): number {
  const v1x = a.x - b.x, v1y = a.y - b.y;
  const v2x = c.x - b.x, v2y = c.y - b.y;
  const n1 = Math.hypot(v1x, v1y), n2 = Math.hypot(v2x, v2y);
  if (n1 < 1e-6 || n2 < 1e-6) return NaN;
  return Math.acos(clamp((v1x * v2x + v1y * v2y) / (n1 * n2), -1, 1)) * 180 / Math.PI;
}

const minVis = (p: Pt[], idxs: number[]) => idxs.reduce((m, i) => Math.min(m, p[i]?.v ?? 0), 1);

/** Torso length in pixels — the unit for every normalized distance in this app,
 *  on both sides of the wire. Falls back to hip width when foreshortened. */
export function torsoScale(p: Pt[]): number {
  const t = dist(mid(p[LM.L_HIP], p[LM.R_HIP]), mid(p[LM.L_SH], p[LM.R_SH]));
  const hipW = dist(p[LM.L_HIP], p[LM.R_HIP]);
  return Math.max(t, hipW * 2.2, 1e-3);
}

/* ── Pose ────────────────────────────────────────────────────────────────── */

let landmarker: PoseLandmarker | null = null;
let poseBackend: string | null = null;
let posePromise: Promise<{ backend: string }> | null = null;

// detectForVideo demands strictly increasing timestamps for the lifetime of the
// instance, so we keep our own clock rather than using wall time.
let poseClockMs = 0;
let poseLastAt: number | null = null;

/** Idempotent init. Prefers the GPU delegate, falls back to CPU. */
export function initPose(): Promise<{ backend: string }> {
  if (posePromise) return posePromise;
  posePromise = (async () => {
    const vision = await FilesetResolver.forVisionTasks(WASM_PATH);
    const options = {
      runningMode: 'VIDEO' as const,
      numPoses: 1,
      minPoseDetectionConfidence: 0.5,
      minPosePresenceConfidence: 0.5,
    };
    try {
      landmarker = await PoseLandmarker.createFromOptions(vision, {
        baseOptions: { modelAssetPath: POSE_MODEL, delegate: 'GPU' }, ...options,
      });
      poseBackend = 'GPU';
    } catch {
      landmarker = await PoseLandmarker.createFromOptions(vision, {
        baseOptions: { modelAssetPath: POSE_MODEL, delegate: 'CPU' }, ...options,
      });
      poseBackend = 'CPU';
    }
    return { backend: poseBackend };
  })();
  return posePromise;
}

export const getPoseBackend = () => poseBackend;

/** One live pose detection against the current preview frame. */
export function detectPoseLive(source: HTMLVideoElement | HTMLCanvasElement): Pt[] | null {
  if (!landmarker) return null;
  const w = source instanceof HTMLVideoElement ? source.videoWidth : source.width;
  const h = source instanceof HTMLVideoElement ? source.videoHeight : source.height;
  if (!w || !h) return null;

  const now = performance.now();
  poseClockMs += poseLastAt === null ? 33 : Math.max(1, Math.round(now - poseLastAt));
  poseLastAt = now;

  const res = landmarker.detectForVideo(source, poseClockMs);
  return res?.landmarks?.length ? toPixels(res.landmarks[0], w, h) : null;
}

/* ── Ball ────────────────────────────────────────────────────────────────── */

const BALL_CATEGORY = 'sports ball';   // the COCO class a football comes back as
const SCORE_FLOOR = 0.2;
// Plausibility gates, all scale-free, mirroring the server's.
const MIN_ROUNDNESS = 0.5;
const MIN_DIAM_TORSO = 0.18;
const MAX_DIAM_TORSO = 0.95;
const STEP_WEIGHT = 0.6;
const MAX_STEP_TORSO = 2.5;
// How many consecutive empty frames before the ball is accepted as gone rather
// than briefly blurred or hidden behind the player's own leg.
const LIVE_FORGET_AFTER = 6;

let detector: ObjectDetector | null = null;
let ballPromise: Promise<{ backend: string }> | null = null;
let ballClockMs = 0;
let ballLastAt: number | null = null;
let ballLast: Ball | null = null;
let ballMisses = 0;

/**
 * CPU, deliberately. The GPU delegate for ObjectDetector wedged the renderer at
 * the wasm level in testing on the previous client — not a rejected promise but
 * a synchronous hang, which no try/catch can recover from, so the "try GPU,
 * fall back to CPU" pattern used for pose is not safe here. This runs at ~5fps
 * where pose runs at 30, so the cost is not felt.
 */
export function initBall(): Promise<{ backend: string }> {
  if (ballPromise) return ballPromise;
  ballPromise = (async () => {
    const vision = await FilesetResolver.forVisionTasks(WASM_PATH);
    detector = await ObjectDetector.createFromOptions(vision, {
      baseOptions: { modelAssetPath: BALL_MODEL, delegate: 'CPU' },
      runningMode: 'VIDEO',
      // Ask the model for nothing else: cheaper than filtering afterwards, and
      // a stray "person" can never reach us.
      categoryAllowlist: [BALL_CATEGORY],
      scoreThreshold: SCORE_FLOOR,
      maxResults: 8,
    });
    return { backend: 'CPU' };
  })();
  return ballPromise;
}

/**
 * One live ball detection. `scale` is torso length in px from the live pose, or
 * 0 when there is no pose — with no player we have no size reference, so the
 * size gate is skipped rather than invented.
 */
export function detectBallLive(source: HTMLVideoElement | HTMLCanvasElement, scale: number): Ball | null {
  if (!detector) return null;
  const now = performance.now();
  ballClockMs += ballLastAt === null ? 33 : Math.max(1, Math.round(now - ballLastAt));
  ballLastAt = now;

  const result = detector.detectForVideo(source, ballClockMs);
  const cands: Ball[] = [];
  for (const d of result?.detections ?? []) {
    const b = d.boundingBox;
    if (!b || b.width <= 0 || b.height <= 0) continue;
    if (Math.min(b.width, b.height) / Math.max(b.width, b.height) < MIN_ROUNDNESS) continue;
    const diam = (b.width + b.height) / 2;
    if (scale > 0) {
      const rel = diam / scale;
      if (rel < MIN_DIAM_TORSO || rel > MAX_DIAM_TORSO) continue;
    }
    cands.push({
      x: b.originX + b.width / 2,
      y: b.originY + b.height / 2,
      r: diam / 2,
      score: d.categories?.[0]?.score ?? 0,
    });
  }

  if (!cands.length) {
    if (++ballMisses > LIVE_FORGET_AFTER) ballLast = null;
    return ballLast;
  }

  // Live has no future to look at, so the server's whole-clip track is not
  // available: carry the last accepted position forward and prefer candidates
  // near it — the same "confidence minus travel cost" idea, resolved greedily.
  let pick: Ball;
  if (ballLast && scale > 0) {
    const cost = (v: Ball) => (1 - v.score) + STEP_WEIGHT *
      Math.min(Math.hypot(v.x - ballLast!.x, v.y - ballLast!.y) / scale, MAX_STEP_TORSO);
    pick = cands.reduce((best, c) => (cost(c) < cost(best) ? c : best));
  } else {
    pick = cands.reduce((best, c) => (c.score > best.score ? c : best));
  }
  ballMisses = 0;
  ballLast = pick;
  return pick;
}

/** Call when the preview (re)starts, so a gap while it was off is not read as
 *  one huge elapsed frame and a stale ball is not carried across. */
export function resetLiveClocks() {
  poseLastAt = null;
  ballLastAt = null;
  ballLast = null;
  ballMisses = 0;
}

/** Release GPU/wasm resources. */
export function closeLive() {
  landmarker?.close();
  detector?.close();
  landmarker = null;
  detector = null;
  posePromise = null;
  ballPromise = null;
  poseBackend = null;
  poseClockMs = 0;
  ballClockMs = 0;
  resetLiveClocks();
}

/* ── Drawing ─────────────────────────────────────────────────────────────── */

const CYAN = '#00e5ff';
const YELLOW = '#ffdd00';
const PINK = '#ff3ba7';

// [label, pointA, vertex, pointC]. The label is drawn at `vertex`, where the
// interior angle is measured — the same angleDeg(a, vertex, c) the server uses
// for backswingKneeFlex / kickLegExtension / ankleLock.
const JOINT_ANGLES: [string, number, number, number][] = [
  ['L knee', LM.L_HIP, LM.L_KN, LM.L_AN],
  ['R knee', LM.R_HIP, LM.R_KN, LM.R_AN],
  ['L hip', LM.L_SH, LM.L_HIP, LM.L_KN],
  ['R hip', LM.R_SH, LM.R_HIP, LM.R_KN],
  ['L ankle', LM.L_KN, LM.L_AN, LM.L_FOOT],
  ['R ankle', LM.R_KN, LM.R_AN, LM.R_FOOT],
];

/**
 * Draw the overlay on a transparent canvas sitting over the preview. Never
 * draws the frame itself — the <video> shows that underneath.
 */
export function drawOverlay(
  canvas: HTMLCanvasElement,
  pts: Pt[] | null,
  { skeleton = true, angles = false, ball = null as Ball | null, mirrored = false } = {},
) {
  const g = canvas.getContext('2d');
  if (!g) return;
  g.clearRect(0, 0, canvas.width, canvas.height);
  g.save();
  // The front camera preview is mirrored for the user's benefit, but detection
  // ran on the unmirrored frame, so the overlay has to be flipped to match what
  // they are looking at.
  if (mirrored) {
    g.translate(canvas.width, 0);
    g.scale(-1, 1);
  }

  const unit = Math.max(2, Math.round(Math.min(canvas.width, canvas.height) / 160));

  if (ball) {
    g.strokeStyle = PINK;
    g.lineWidth = unit * 1.4;
    g.beginPath();
    g.arc(ball.x, ball.y, Math.max(ball.r, unit * 2), 0, Math.PI * 2);
    g.stroke();
    g.fillStyle = PINK;
    g.beginPath();
    g.arc(ball.x, ball.y, unit * 0.9, 0, Math.PI * 2);
    g.fill();
  }

  if (pts && skeleton) {
    g.strokeStyle = CYAN;
    g.lineWidth = unit;
    for (const [a, b] of CONNECTIONS) {
      const pa = pts[a], pb = pts[b];
      if (!pa || !pb || pa.v < 0.3 || pb.v < 0.3) continue;
      g.beginPath();
      g.moveTo(pa.x, pa.y);
      g.lineTo(pb.x, pb.y);
      g.stroke();
    }
    g.fillStyle = YELLOW;
    for (let i = 0; i < pts.length; i++) {
      if (i > 0 && i < 11) continue;           // skip the dense face points
      const p = pts[i];
      if (!p || p.v < 0.3) continue;
      g.beginPath();
      g.arc(p.x, p.y, unit * 1.3, 0, Math.PI * 2);
      g.fill();
    }
  }

  if (pts && angles) {
    const fontPx = Math.max(12, unit * 6);
    g.font = `bold ${fontPx}px system-ui, sans-serif`;
    g.textBaseline = 'middle';
    for (const [label, ia, iv, ic] of JOINT_ANGLES) {
      if (minVis(pts, [ia, iv, ic]) < 0.3) continue;
      const deg = angleDeg(pts[ia], pts[iv], pts[ic]);
      if (!Number.isFinite(deg)) continue;
      const p = pts[iv];
      const text = `${label} ${Math.round(deg)}°`;
      // Labels are drawn unmirrored even on the front camera, or they read
      // backwards; only their anchor point is flipped.
      const x = mirrored ? canvas.width - p.x : p.x;
      g.save();
      if (mirrored) g.setTransform(1, 0, 0, 1, 0, 0);
      const w = g.measureText(text).width;
      g.fillStyle = 'rgba(0,0,0,0.65)';
      g.fillRect(x + 6, p.y - fontPx / 2 - 2, w + 8, fontPx + 4);
      g.fillStyle = '#fff';
      g.fillText(text, x + 10, p.y);
      g.restore();
    }
  }

  g.restore();
}
