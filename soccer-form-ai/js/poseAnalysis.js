/* poseAnalysis.js — turns raw landmarks into soccer-relevant measurements,
 * and draws the skeleton overlay.
 *
 * MediaPipe answers "where are the body parts". This file answers "what is that
 * body position doing in a kick": every value here is either an angle or a
 * distance divided by torso length, so nothing depends on pixels.
 */

import {
  LM, CONNECTIONS, angleDeg, hipCenter, shoulderCenter, torsoScale,
  torsoLeanDeg, forwardOffset, heightAbove, minVis, dist
} from './biomechanics.js';

/* ── Measurements ───────────────────────────────────────── */

const m = (value, p, idxs) => ({ value, vis: minVis(p, idxs) });

/**
 * Rough 2D proxy for hip/shoulder separation. In a single camera view a rotated
 * segment appears narrower, so shoulderSpan/hipSpan rises as the shoulders open
 * relative to the hips. It is confounded by body type and camera angle, which is
 * why every metric built on it is flagged low-confidence in scoring.js.
 */
function rotationProxy(p) {
  const shSpan = dist(p[LM.L_SH], p[LM.R_SH]);
  const hipSpan = dist(p[LM.L_HIP], p[LM.R_HIP]);
  return m(hipSpan < 1e-3 ? NaN : shSpan / hipSpan, p, [LM.L_SH, LM.R_SH, LM.L_HIP, LM.R_HIP]);
}

function plantMetrics(p, c) {
  const s = torsoScale(p), hc = hipCenter(p), sc = shoulderCenter(p);
  const K = c.kick, P = c.plant;
  return {
    // Support foot fore/aft relative to the hips. Behind the hips means reaching.
    plantFootPlacement: m(forwardOffset(p[P.an], hc, c.dir, s), p, [P.an, LM.L_HIP, LM.R_HIP]),
    // Slight flex loads the support leg; locked out is brittle, deep is a collapse.
    plantKneeBend: m(angleDeg(p[P.hip], p[P.kn], p[P.an]), p, [P.hip, P.kn, P.an]),
    // Heel-toward-glutes cocks the leg; a straight-legged backswing loses power.
    backswingKneeFlex: m(angleDeg(p[K.hip], p[K.kn], p[K.an]), p, [K.hip, K.kn, K.an]),
    // How far the kicking foot is drawn behind the body.
    backswingReach: m(-forwardOffset(p[K.an], hc, c.dir, s), p, [K.an, LM.L_HIP, LM.R_HIP]),
    // Chest stacked over the support foot = balanced base.
    balance: m(Math.abs(p[P.an].x - sc.x) / s, p, [P.an, LM.L_SH, LM.R_SH]),
    torsoLean: m(torsoLeanDeg(p, c.dir), p, [LM.L_SH, LM.R_SH, LM.L_HIP, LM.R_HIP])
  };
}

function contactMetrics(p, c) {
  const s = torsoScale(p), hc = hipCenter(p), sc = shoulderCenter(p);
  const K = c.kick, P = c.plant;
  return {
    torsoLean: m(torsoLeanDeg(p, c.dir), p, [LM.L_SH, LM.R_SH, LM.L_HIP, LM.R_HIP]),
    kickLegExtension: m(angleDeg(p[K.hip], p[K.kn], p[K.an]), p, [K.hip, K.kn, K.an]),
    plantFootPlacement: m(forwardOffset(p[P.an], hc, c.dir, s), p, [P.an, LM.L_HIP, LM.R_HIP]),
    plantKneeBend: m(angleDeg(p[P.hip], p[P.kn], p[P.an]), p, [P.hip, P.kn, P.an]),
    balance: m(Math.abs(p[P.an].x - sc.x) / s, p, [P.an, LM.L_SH, LM.R_SH]),
    // Head position fore/aft over the support foot — the classic "over the ball" cue.
    headOverBall: m(forwardOffset(p[LM.NOSE], p[P.an], c.dir, s), p, [LM.NOSE, P.an]),
    // Shin-to-toe angle: near-straight means a locked, plantarflexed ankle.
    ankleLock: m(angleDeg(p[K.kn], p[K.an], p[K.foot]), p, [K.kn, K.an, K.foot]),
    hipRotation: rotationProxy(p)
  };
}

function followThroughMetrics(p, c) {
  const s = torsoScale(p), hc = hipCenter(p), sc = shoulderCenter(p);
  const K = c.kick, P = c.plant;
  return {
    kickLegExtension: m(angleDeg(p[K.hip], p[K.kn], p[K.an]), p, [K.hip, K.kn, K.an]),
    // How high the kicking foot finishes above hip level.
    followHeight: m(heightAbove(p[K.an], hc, s), p, [K.an, LM.L_HIP, LM.R_HIP]),
    // How far through the ball the leg continued.
    followReach: m(forwardOffset(p[K.an], hc, c.dir, s), p, [K.an, LM.L_HIP, LM.R_HIP]),
    balance: m(Math.abs(p[P.an].x - sc.x) / s, p, [P.an, LM.L_SH, LM.R_SH]),
    torsoLean: m(torsoLeanDeg(p, c.dir), p, [LM.L_SH, LM.R_SH, LM.L_HIP, LM.R_HIP]),
    // Support foot still under the body = able to recover and play on.
    support: m(Math.abs(p[P.an].x - hc.x) / s, p, [P.an, LM.L_HIP, LM.R_HIP]),
    hipRotation: rotationProxy(p)
  };
}

/** Compute every measurement for all three phases. */
export function computeMetrics(frames, ctx) {
  return {
    plant: plantMetrics(frames.plant.pts, ctx),
    contact: contactMetrics(frames.contact.pts, ctx),
    followThrough: followThroughMetrics(frames.followThrough.pts, ctx)
  };
}

/* ── Drawing ────────────────────────────────────────────── */

/** Landmarks emphasised per phase, resolved against the detected kick/plant side. */
export function highlightFor(phase, ctx) {
  const K = ctx.kick, P = ctx.plant;
  const torso = [LM.L_SH, LM.R_SH, LM.L_HIP, LM.R_HIP];
  const kickLeg = [K.hip, K.kn, K.an, K.heel, K.foot];
  const plantLeg = [P.hip, P.kn, P.an, P.heel, P.foot];
  if (phase === 'plant') return new Set([...kickLeg, ...plantLeg]);
  if (phase === 'contact') return new Set([...kickLeg, ...torso]);
  return new Set([...kickLeg, ...torso]);
}

/**
 * Draw the captured frame plus the skeleton into `out`.
 * `pts` are in the source frame's pixel space, so we scale by out/src.
 */
export function drawPose(out, frameCanvas, pts, highlight = null) {
  out.width = frameCanvas.width;
  out.height = frameCanvas.height;
  const g = out.getContext('2d');
  g.drawImage(frameCanvas, 0, 0);
  if (!pts) return;

  const unit = Math.max(2, Math.round(Math.min(out.width, out.height) / 160));
  const hot = i => !highlight || highlight.has(i);

  for (const [a, b] of CONNECTIONS) {
    const pa = pts[a], pb = pts[b];
    if (!pa || !pb || pa.v < 0.3 || pb.v < 0.3) continue;
    const on = hot(a) && hot(b);
    g.strokeStyle = on ? '#00e5ff' : 'rgba(255,255,255,0.45)';
    g.lineWidth = on ? unit * 1.6 : unit;
    g.beginPath();
    g.moveTo(pa.x, pa.y);
    g.lineTo(pb.x, pb.y);
    g.stroke();
  }

  for (let i = 0; i < pts.length; i++) {
    if (i > 0 && i < 11) continue;            // skip the dense face points
    const p = pts[i];
    if (!p || p.v < 0.3) continue;
    g.fillStyle = hot(i) ? '#ffdd00' : 'rgba(255,255,255,0.6)';
    g.beginPath();
    g.arc(p.x, p.y, hot(i) ? unit * 1.5 : unit, 0, Math.PI * 2);
    g.fill();
  }

  // Head marker
  const nose = pts[LM.NOSE];
  if (nose && nose.v >= 0.3) {
    g.strokeStyle = '#ffdd00';
    g.lineWidth = unit;
    g.beginPath();
    g.arc(nose.x, nose.y, unit * 3.5, 0, Math.PI * 2);
    g.stroke();
  }
}

// Joints worth labeling live: [label, pointA, vertex, pointC]. The label is
// drawn at `vertex`, where the interior angle is measured — the same
// angleDeg(a, vertex, c) used everywhere else in the app, so a live "142°" at
// the knee reads off the exact three landmark coordinates that would feed the
// backswingKneeFlex / kickLegExtension metrics once that frame gets selected.
const JOINT_ANGLES = [
  ['L knee', LM.L_HIP, LM.L_KN, LM.L_AN],
  ['R knee', LM.R_HIP, LM.R_KN, LM.R_AN],
  ['L hip', LM.L_SH, LM.L_HIP, LM.L_KN],
  ['R hip', LM.R_SH, LM.R_HIP, LM.R_KN],
  ['L ankle', LM.L_KN, LM.L_AN, LM.L_FOOT],
  ['R ankle', LM.R_KN, LM.R_AN, LM.R_FOOT]
];

/**
 * Draw a live overlay — skeleton and/or per-joint angle labels — on a
 * transparent canvas positioned over the camera preview. Unlike drawPose()
 * this never draws the frame image itself; the camera's own <video> element
 * shows that underneath.
 */
export function drawSkeletonOverlay(canvas, pts, { skeleton = true, angles = false } = {}) {
  const g = canvas.getContext('2d');
  g.clearRect(0, 0, canvas.width, canvas.height);
  if (!pts) return;

  const unit = Math.max(2, Math.round(Math.min(canvas.width, canvas.height) / 160));

  if (skeleton) {
    for (const [a, b] of CONNECTIONS) {
      const pa = pts[a], pb = pts[b];
      if (!pa || !pb || pa.v < 0.3 || pb.v < 0.3) continue;
      g.strokeStyle = '#00e5ff';
      g.lineWidth = unit;
      g.beginPath();
      g.moveTo(pa.x, pa.y);
      g.lineTo(pb.x, pb.y);
      g.stroke();
    }
    for (let i = 0; i < pts.length; i++) {
      if (i > 0 && i < 11) continue;
      const p = pts[i];
      if (!p || p.v < 0.3) continue;
      g.fillStyle = '#ffdd00';
      g.beginPath();
      g.arc(p.x, p.y, unit * 1.3, 0, Math.PI * 2);
      g.fill();
    }
  }

  if (angles) {
    const fontPx = Math.max(12, unit * 6);
    g.font = `bold ${fontPx}px system-ui, sans-serif`;
    g.textBaseline = 'middle';
    for (const [label, ia, iv, ic] of JOINT_ANGLES) {
      if (minVis(pts, [ia, iv, ic]) < 0.3) continue;
      const deg = angleDeg(pts[ia], pts[iv], pts[ic]);
      if (!isFinite(deg)) continue;
      drawAngleLabel(g, pts[iv], `${label} ${Math.round(deg)}°`, fontPx);
    }
  }
}

function drawAngleLabel(g, p, text, fontPx) {
  const padX = 4, padY = 2;
  const w = g.measureText(text).width;
  g.fillStyle = 'rgba(0,0,0,0.65)';
  g.fillRect(p.x + 6, p.y - fontPx / 2 - padY, w + padX * 2, fontPx + padY * 2);
  g.fillStyle = '#fff';
  g.fillText(text, p.x + 6 + padX, p.y);
}
