/* overlay.js — everything the client draws.
 *
 * This is all that is left of the browser's analysis code. Measuring and
 * scoring moved to the server (server/analysis.py, server/scoring.py) so there
 * is exactly one copy of the rules; the client's job is now to show the player
 * what the camera is seeing while they record, which has to be local because
 * a round trip per frame would never keep up.
 */

import { LM, CONNECTIONS, angleDeg, minVis } from './biomechanics.js';

const CYAN = '#00e5ff';
const YELLOW = '#ffdd00';
const PINK = '#ff3ba7';

// Joints worth labeling live: [label, pointA, vertex, pointC]. The label is
// drawn at `vertex`, where the interior angle is measured — the same
// angleDeg(a, vertex, c) the server uses for backswingKneeFlex /
// kickLegExtension / ankleLock, on the same three landmark coordinates. So a
// live "142°" at the knee is exactly what would score that joint if you picked
// that instant as a phase.
const JOINT_ANGLES = [
  ['L knee', LM.L_HIP, LM.L_KN, LM.L_AN],
  ['R knee', LM.R_HIP, LM.R_KN, LM.R_AN],
  ['L hip', LM.L_SH, LM.L_HIP, LM.L_KN],
  ['R hip', LM.R_SH, LM.R_HIP, LM.R_KN],
  ['L ankle', LM.L_KN, LM.L_AN, LM.L_FOOT],
  ['R ankle', LM.R_KN, LM.R_AN, LM.R_FOOT]
];

/**
 * Draw the live overlay on a transparent canvas positioned over the camera
 * preview. Never draws the frame image itself — the <video> shows that
 * underneath.
 *
 * @param {HTMLCanvasElement} canvas the overlay canvas
 * @param {Array|null} pts  landmarks in this canvas's pixel space
 * @param {{skeleton?:boolean, angles?:boolean, ball?:object|null}} opts
 */
export function drawOverlay(canvas, pts, { skeleton = true, angles = false, ball = null } = {}) {
  const g = canvas.getContext('2d');
  g.clearRect(0, 0, canvas.width, canvas.height);
  const unit = Math.max(2, Math.round(Math.min(canvas.width, canvas.height) / 160));

  if (ball) drawBall(g, ball, unit);
  if (!pts) return;

  if (skeleton) {
    for (const [a, b] of CONNECTIONS) {
      const pa = pts[a], pb = pts[b];
      if (!pa || !pb || pa.v < 0.3 || pb.v < 0.3) continue;
      g.strokeStyle = CYAN;
      g.lineWidth = unit;
      g.beginPath();
      g.moveTo(pa.x, pa.y);
      g.lineTo(pb.x, pb.y);
      g.stroke();
    }
    for (let i = 0; i < pts.length; i++) {
      if (i > 0 && i < 11) continue;           // skip the dense face points
      const p = pts[i];
      if (!p || p.v < 0.3) continue;
      g.fillStyle = YELLOW;
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

/** Ring the ball. Live, this is a framing aid: it shows you whether the ball is
 *  actually in shot and being seen, before you commit to a take. */
function drawBall(g, ball, unit) {
  g.save();
  g.strokeStyle = PINK;
  g.lineWidth = unit * 1.4;
  g.beginPath();
  g.arc(ball.x, ball.y, Math.max(ball.r, unit * 2), 0, Math.PI * 2);
  g.stroke();
  g.fillStyle = PINK;
  g.beginPath();
  g.arc(ball.x, ball.y, unit * 0.9, 0, Math.PI * 2);
  g.fill();
  g.restore();
}

function drawAngleLabel(g, p, text, fontPx) {
  const padX = 4, padY = 2;
  const w = g.measureText(text).width;
  g.fillStyle = 'rgba(0,0,0,0.65)';
  g.fillRect(p.x + 6, p.y - fontPx / 2 - padY, w + padX * 2, fontPx + padY * 2);
  g.fillStyle = '#fff';
  g.fillText(text, p.x + 6 + padX, p.y);
}
