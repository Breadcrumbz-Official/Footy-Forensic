/* biomechanics.js — the geometry the live overlay needs.
 *
 * This used to hold the whole measurement layer. That moved to the server
 * (server/biomechanics.py) when analysis did, so there is one copy of the maths
 * rather than two that drift apart. What is left is only what the browser needs
 * to draw the live preview: landmark indices, skeleton edges, the angle used by
 * the live joint labels, and torso length — which the live ball tracker uses to
 * size-gate candidates, exactly as the server does.
 *
 * "pts" is an array of 33 points already converted from MediaPipe's normalized
 * landmark space into ASPECT-CORRECTED pixel space ({x, y, v}).
 *
 * Why that conversion matters: MediaPipe normalizes x by frame width and y by
 * frame height independently, so on a non-square frame the normalized space is
 * anisotropically stretched and angles measured in it are simply wrong.
 */

// MediaPipe Pose landmark indices (33-point model).
export const LM = {
  NOSE: 0,
  L_SH: 11, R_SH: 12, L_EL: 13, R_EL: 14, L_WR: 15, R_WR: 16,
  L_HIP: 23, R_HIP: 24, L_KN: 25, R_KN: 26, L_AN: 27, R_AN: 28,
  L_HEEL: 29, R_HEEL: 30, L_FOOT: 31, R_FOOT: 32
};

// Skeleton edges used for the overlay drawing.
export const CONNECTIONS = [
  [LM.L_SH, LM.R_SH], [LM.L_HIP, LM.R_HIP],
  [LM.L_SH, LM.L_HIP], [LM.R_SH, LM.R_HIP],
  [LM.L_SH, LM.L_EL], [LM.L_EL, LM.L_WR],
  [LM.R_SH, LM.R_EL], [LM.R_EL, LM.R_WR],
  [LM.L_HIP, LM.L_KN], [LM.L_KN, LM.L_AN],
  [LM.L_AN, LM.L_HEEL], [LM.L_HEEL, LM.L_FOOT], [LM.L_AN, LM.L_FOOT],
  [LM.R_HIP, LM.R_KN], [LM.R_KN, LM.R_AN],
  [LM.R_AN, LM.R_HEEL], [LM.R_HEEL, LM.R_FOOT], [LM.R_AN, LM.R_FOOT]
];

export const clamp = (v, lo, hi) => Math.min(hi, Math.max(lo, v));
export const mid = (a, b) => ({ x: (a.x + b.x) / 2, y: (a.y + b.y) / 2, v: Math.min(a.v ?? 1, b.v ?? 1) });
export const dist = (a, b) => Math.hypot(a.x - b.x, a.y - b.y);

/** Convert a MediaPipe landmark list to aspect-corrected pixel points. */
export function toPixels(landmarks, w, h) {
  return landmarks.map(l => ({ x: l.x * w, y: l.y * h, v: l.visibility ?? 1 }));
}

/** Interior angle at vertex `b`, in degrees, formed by a-b-c. 180 = straight. */
export function angleDeg(a, b, c) {
  const v1x = a.x - b.x, v1y = a.y - b.y;
  const v2x = c.x - b.x, v2y = c.y - b.y;
  const n1 = Math.hypot(v1x, v1y), n2 = Math.hypot(v2x, v2y);
  if (n1 < 1e-6 || n2 < 1e-6) return NaN;
  const cos = clamp((v1x * v2x + v1y * v2y) / (n1 * n2), -1, 1);
  return Math.acos(cos) * 180 / Math.PI;
}

export const hipCenter = p => mid(p[LM.L_HIP], p[LM.R_HIP]);
export const shoulderCenter = p => mid(p[LM.L_SH], p[LM.R_SH]);

/**
 * Torso length (hip centre -> shoulder centre) in pixels — the unit for every
 * normalized distance in this app, on both sides of the wire. Falls back to hip
 * width * 2.2 if the torso is badly foreshortened.
 */
export function torsoScale(p) {
  const t = dist(hipCenter(p), shoulderCenter(p));
  const hipW = dist(p[LM.L_HIP], p[LM.R_HIP]);
  return Math.max(t, hipW * 2.2, 1e-3);
}

/** Minimum visibility across the given landmark indices — a confidence gate. */
export function minVis(p, idxs) {
  return idxs.reduce((m, i) => Math.min(m, p[i]?.v ?? 0), 1);
}

/**
 * How side-on the camera is, from the apparent shoulder span. Mirrors
 * server/biomechanics.py:view_quality so the live preview can warn the player
 * BEFORE they record, rather than the server telling them afterwards that the
 * angle was unusable. Seen side-on the shoulders are edge-on and their span
 * collapses; seen face-on they are at full width.
 */
const SIDE_ON_SPAN = 0.42;
const FRONT_ON_SPAN = 0.78;

export function viewQuality(p) {
  const scale = torsoScale(p);
  const span = scale > 0 ? dist(p[LM.L_SH], p[LM.R_SH]) / scale : NaN;
  if (!isFinite(span)) return { score: 0.5, span: NaN, label: 'unknown' };
  const score = span <= SIDE_ON_SPAN ? 1
    : span >= FRONT_ON_SPAN ? 0
    : (FRONT_ON_SPAN - span) / (FRONT_ON_SPAN - SIDE_ON_SPAN);
  const label = score >= 0.66 ? 'side-on' : score >= 0.33 ? 'angled' : 'face-on';
  return { score, span, label };
}
