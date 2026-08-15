/* biomechanics.js — pure geometry helpers.
 *
 * All functions operate on "pts": an array of 33 points already converted from
 * MediaPipe's normalized landmark space into ASPECT-CORRECTED pixel space
 * ({x: lm.x * frameW, y: lm.y * frameH, v: visibility}).
 *
 * Why the conversion matters: MediaPipe normalizes x by frame width and y by
 * frame height independently, so on a non-square frame the normalized space is
 * anisotropically stretched. Angles measured directly in that space are wrong.
 * Multiplying back by (W, H) restores a uniform scale so angles and distance
 * ratios are geometrically valid.
 *
 * Everything downstream is expressed as an angle or as a distance divided by
 * torso length, so results are independent of resolution, camera distance and
 * player size.
 */

// MediaPipe Pose landmark indices (33-point model).
export const LM = {
  NOSE: 0,
  L_SH: 11, R_SH: 12, L_EL: 13, R_EL: 14, L_WR: 15, R_WR: 16,
  L_HIP: 23, R_HIP: 24, L_KN: 25, R_KN: 26, L_AN: 27, R_AN: 28,
  L_HEEL: 29, R_HEEL: 30, L_FOOT: 31, R_FOOT: 32
};

// Per-side index bundles so analysis code can be written once and applied to
// whichever leg turns out to be the kicking / plant leg.
export const SIDES = {
  left:  { sh: LM.L_SH, el: LM.L_EL, wr: LM.L_WR, hip: LM.L_HIP, kn: LM.L_KN, an: LM.L_AN, heel: LM.L_HEEL, foot: LM.L_FOOT },
  right: { sh: LM.R_SH, el: LM.R_EL, wr: LM.R_WR, hip: LM.R_HIP, kn: LM.R_KN, an: LM.R_AN, heel: LM.R_HEEL, foot: LM.R_FOOT }
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
 * Torso length (hip centre -> shoulder centre) in pixels. This is the unit for
 * every normalized distance in the app: "0.4" means "0.4 torso lengths".
 * Falls back to hip width * 2.2 if the torso is badly foreshortened.
 */
export function torsoScale(p) {
  const t = dist(hipCenter(p), shoulderCenter(p));
  const hipW = dist(p[LM.L_HIP], p[LM.R_HIP]);
  return Math.max(t, hipW * 2.2, 1e-3);
}

/**
 * Signed torso lean from vertical, in degrees.
 * Positive = chest leaning FORWARD along the kick direction; negative = leaning back.
 * `dir` is +1 if the player kicks toward increasing image-x, else -1.
 */
export function torsoLeanDeg(p, dir) {
  const h = hipCenter(p), s = shoulderCenter(p);
  const forward = (s.x - h.x) * dir;   // horizontal displacement toward the target
  const up = h.y - s.y;                // image y grows downward, so this is positive when upright
  return Math.atan2(forward, up) * 180 / Math.PI;
}

/** Signed horizontal offset of point `pt` from `ref`, along the kick direction, in torso units. */
export function forwardOffset(pt, ref, dir, scale) {
  return ((pt.x - ref.x) * dir) / scale;
}

/** Height of `pt` above `ref` (positive = higher in the world), in torso units. */
export function heightAbove(pt, ref, scale) {
  return (ref.y - pt.y) / scale;
}

/** Minimum visibility across the given landmark indices — used as a confidence gate. */
export function minVis(p, idxs) {
  return idxs.reduce((m, i) => Math.min(m, p[i]?.v ?? 0), 1);
}

/**
 * Decide which leg kicks and which way the player faces.
 *
 * Kicking leg: across the contact and follow-through frames, the kicking ankle
 * is the one lifted highest relative to the hips. Summing both frames makes the
 * call robust when contact happens low to the ground.
 *
 * Direction: the kicking ankle travels toward the target from the backswing to
 * the follow-through, so the sign of that horizontal travel gives the facing.
 * If travel is tiny (near head-on camera), fall back to which way the plant
 * foot's toes point relative to its heel.
 */
export function deriveContext(frames) {
  const { plant, contact, followThrough } = frames;
  const lift = (p, side) => {
    const s = SIDES[side];
    return heightAbove(p[s.an], hipCenter(p), torsoScale(p));
  };
  const leftScore = lift(contact.pts, 'left') + lift(followThrough.pts, 'left');
  const rightScore = lift(contact.pts, 'right') + lift(followThrough.pts, 'right');
  const kickSide = leftScore > rightScore ? 'left' : 'right';
  const plantSide = kickSide === 'left' ? 'right' : 'left';

  const kAn = SIDES[kickSide].an;
  const travel = followThrough.pts[kAn].x - plant.pts[kAn].x;
  const scale = torsoScale(contact.pts);
  let dir;
  if (Math.abs(travel) / scale > 0.15) {
    dir = Math.sign(travel);
  } else {
    const ps = SIDES[plantSide];
    const toe = contact.pts[ps.foot].x - contact.pts[ps.heel].x;
    dir = Math.sign(toe) || 1;
  }

  return {
    kickSide, plantSide, dir,
    kick: SIDES[kickSide],
    plant: SIDES[plantSide],
    // How confidently we picked a leg — near-equal lift means a head-on camera.
    legConfidence: Math.min(1, Math.abs(leftScore - rightScore) / 0.5)
  };
}
