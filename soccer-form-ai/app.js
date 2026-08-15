/* app.js — UI wiring for Soccer Form AI.
 * Everything (decode, pose inference, analysis) runs locally in this tab.
 */

import { Recorder, loadVideo, seekTo, captureFrame, captureWindow, captureRange } from './js/video.js';
import { initPose, detectPoseSequence, detectPoseLive, resetLiveClock, getBackend } from './js/mediapipe.js';
import { toPixels, deriveContext } from './js/biomechanics.js';
import { computeMetrics, drawPose, drawSkeletonOverlay, highlightFor } from './js/poseAnalysis.js';
import { scoreAll, grade } from './js/scoring.js';

const $ = s => document.querySelector(s);
const PHASES = ['plant', 'contact', 'followThrough'];
const PHASE_LABEL = { plant: 'Plant + Backswing', contact: 'Contact', followThrough: 'Follow-through' };

const video = $('#video');
const recorder = new Recorder();

/** frames[phase] = { time, canvas, pts, landmarks } */
let frames = { plant: null, contact: null, followThrough: null };
/** clipBounds[phase] = { start, end } in seconds, or nulls for "use the automatic window" */
let clipBounds = {
  plant: { start: null, end: null },
  contact: { start: null, end: null },
  followThrough: { start: null, end: null }
};
let lastResult = null;

/* ── Engine warm-up (starts immediately, never blocks the UI) ── */

initPose()
  .then(() => { $('#engineStatus').textContent = `Pose engine: ready (${getBackend()} backend)`; })
  .catch(err => {
    console.error(err);
    $('#engineStatus').textContent = 'Pose engine: failed to load — check your network connection.';
  });

/* ── Video input ─────────────────────────────────────────── */

$('#fileInput').addEventListener('change', async e => {
  const file = e.target.files?.[0];
  if (file) await useVideo(file);
});

$('#btnCam').addEventListener('click', async () => {
  try {
    await recorder.enableCamera($('#camPreview'));
    $('#camWrap').classList.toggle('mirrored', recorder.facingMode === 'user');
    $('#camOverlay').hidden = false;
    $('#btnRec').disabled = false;
    $('#btnFlip').disabled = false;
    $('#btnCam').disabled = true;
    msg('#inputMsg', 'Camera on. Nothing is being transmitted — the stream stays in this tab.', false);
    if (liveOverlayWanted()) startLiveLoop();
  } catch (err) {
    msg('#inputMsg', `Camera unavailable: ${err.message}`);
  }
});

$('#btnFlip').addEventListener('click', async () => {
  $('#btnFlip').disabled = true;
  try {
    const mode = await recorder.switchCamera($('#camPreview'));
    $('#camWrap').classList.toggle('mirrored', mode === 'user');
    sizeCamOverlay(); // front/back sensors can report different resolutions
    msg('#inputMsg', `Switched to the ${mode === 'user' ? 'front' : 'back'} camera.`, false);
  } catch (err) {
    msg('#inputMsg', `Could not switch camera — reconnected to the previous one. (${err.message})`);
  } finally {
    $('#btnFlip').disabled = false;
  }
});

$('#btnRec').addEventListener('click', async () => {
  $('#btnRec').disabled = true;
  $('#btnFlip').disabled = true; // switching mid-recording would kill the stream it's recording from
  $('#btnStop').disabled = false;
  msg('#inputMsg', '', false);

  const blob = await recorder.start(s => {
    $('#recTimer').textContent = `${s.toFixed(1)}s / 10.0s`;
    $('#btnStop').disabled = !recorder.canStop();
  });

  $('#btnStop').disabled = true;
  $('#recTimer').textContent = '';
  recorder.release();
  stopLiveLoop();
  $('#camPreview').hidden = true;
  $('#camOverlay').hidden = true;
  $('#camPreview').srcObject = null;
  $('#camWrap').classList.remove('mirrored');
  $('#btnCam').disabled = false;
  $('#btnFlip').disabled = true;
  await useVideo(blob);
});

$('#btnStop').addEventListener('click', () => recorder.stop());

/* ── Live skeleton/angle overlay (camera preview + while recording) ──────── */

// Off by default — this is the one place the app runs pose detection
// continuously rather than on 3 picked frames, so it's opt-in. The interval
// is a ceiling, not a guarantee: the busy guard below skips a tick outright
// if the previous detection hasn't finished, so a slower device naturally
// settles at whatever rate it can actually sustain instead of piling up
// overlapping detections.
const LIVE_INTERVAL_MS = 33; // ~30 detections/sec ceiling
let liveShowSkeleton = false;
let liveShowAngles = false;
let liveTimer = null;
let liveBusy = false;

const liveOverlayWanted = () => liveShowSkeleton || liveShowAngles;

function sizeCamOverlay() {
  const camPreview = $('#camPreview'), overlay = $('#camOverlay');
  if (camPreview.videoWidth && overlay.width !== camPreview.videoWidth) {
    overlay.width = camPreview.videoWidth;
    overlay.height = camPreview.videoHeight;
  }
}

function startLiveLoop() {
  if (liveTimer || !liveOverlayWanted()) return;
  resetLiveClock();
  liveBusy = false;
  liveTimer = setInterval(async () => {
    if (liveBusy) return; // previous detection still running — don't pile up
    const camPreview = $('#camPreview');
    if (camPreview.hidden || !camPreview.videoWidth) return;
    liveBusy = true;
    sizeCamOverlay();
    try {
      const frame = captureFrame(camPreview);
      const res = await detectPoseLive(frame);
      const overlay = $('#camOverlay');
      const pts = res ? toPixels(res.landmarks, overlay.width, overlay.height) : null;
      drawSkeletonOverlay(overlay, pts, { skeleton: liveShowSkeleton, angles: liveShowAngles });
    } catch (err) {
      console.error(err);
    } finally {
      liveBusy = false;
    }
  }, LIVE_INTERVAL_MS);
}

function stopLiveLoop() {
  clearInterval(liveTimer);
  liveTimer = null;
  const overlay = $('#camOverlay');
  overlay.getContext('2d').clearRect(0, 0, overlay.width, overlay.height);
}

$('#chkLiveSkeleton').addEventListener('change', e => {
  liveShowSkeleton = e.target.checked;
  liveOverlayWanted() ? startLiveLoop() : stopLiveLoop();
});
$('#chkLiveAngles').addEventListener('change', e => {
  liveShowAngles = e.target.checked;
  liveOverlayWanted() ? startLiveLoop() : stopLiveLoop();
});

async function useVideo(blob) {
  try {
    resetSelections();
    const duration = await loadVideo(video, blob);
    $('#secPlayer').hidden = false;
    $('#secResults').hidden = true;
    updateTime();
    await buildFilmstrip(duration);

    if (duration > 15) {
      msg('#inputMsg', `This clip is ${duration.toFixed(1)}s. Short clips (5–10s) are much easier to scrub — but you can still pick your three frames.`, false);
    } else {
      msg('#inputMsg', '', false);
    }
    $('#secPlayer').scrollIntoView({ behavior: 'smooth' });
  } catch (err) {
    msg('#inputMsg', err.message);
  }
}

/* ── Playback + scrubbing ────────────────────────────────── */

const frameStep = () => 1 / Math.max(1, Number($('#fps').value) || 30);

function updateTime() {
  const d = isFinite(video.duration) ? video.duration : 0;
  $('#timeLabel').textContent = `${video.currentTime.toFixed(3)} / ${d.toFixed(3)} s`;
  positionFilmstrip();
}

video.addEventListener('timeupdate', updateTime);
video.addEventListener('seeked', updateTime);
video.addEventListener('play', () => { $('#btnPlay').textContent = 'Pause'; });
video.addEventListener('pause', () => { $('#btnPlay').textContent = 'Play'; });

$('#btnPlay').addEventListener('click', () => video.paused ? video.play() : video.pause());

const step = async n => {
  video.pause();
  await seekTo(video, video.currentTime + n * frameStep());
  updateTime();
};
$('#btnBack10').addEventListener('click', () => step(-10));
$('#btnBack1').addEventListener('click', () => step(-1));
$('#btnFwd1').addEventListener('click', () => step(1));
$('#btnFwd10').addEventListener('click', () => step(10));

/* ── Filmstrip scrubbing (drag a strip of frame thumbnails, phone-gallery style) ── */

// Seeks are async and dragging fires far faster than they can resolve. Rather
// than queue every intermediate position (which would lag behind the pointer),
// we track only the latest requested time; each seek that finishes immediately
// starts the next one toward wherever the pointer is *now*, so it always
// catches up instead of playing back the drag in slow motion.
let scrubTarget = null;
let scrubBusy = false;
async function scrubTo(t) {
  const dur = isFinite(video.duration) ? video.duration : t;
  scrubTarget = Math.max(0, Math.min(t, Math.max(0, dur - 0.001)));
  if (scrubBusy) return;
  scrubBusy = true;
  while (scrubTarget !== null) {
    const target = scrubTarget;
    scrubTarget = null;
    await seekTo(video, target);
    updateTime();
  }
  scrubBusy = false;
}

const filmstrip = $('#filmstrip');
const filmstripTrack = $('#filmstripTrack');
let filmstripWidth = 0;   // total px width of the generated strip
let filmstripDuration = 0;

/**
 * Build the filmstrip: a row of small thumbnails spanning the whole video,
 * generated once per loaded video. The strip is positioned (via translateX)
 * so whichever thumbnail corresponds to the current time sits under the fixed
 * center playhead — dragging the strip left brings later thumbnails under
 * that playhead, exactly like the iOS/Android photo-gallery scrubber.
 */
async function buildFilmstrip(duration) {
  filmstripTrack.innerHTML = '';
  filmstripTrack.style.transform = '';
  filmstripWidth = 0;
  filmstripDuration = duration;
  if (!duration || !isFinite(duration)) return;

  const THUMB_H = 64;
  const COUNT = Math.min(60, Math.max(12, Math.round(duration * 6))); // ~6 thumbs/sec, capped
  const savedTime = video.currentTime;
  video.pause();

  msg('#filmstripMsg', `building preview strip… 0/${COUNT}`, false);
  for (let i = 0; i < COUNT; i++) {
    const t = (i / (COUNT - 1)) * Math.max(duration - 0.03, 0);
    await seekTo(video, t);
    const vw = video.videoWidth, vh = video.videoHeight;
    const w = Math.max(1, Math.round(THUMB_H * (vw && vh ? vw / vh : 0.75)));
    const c = document.createElement('canvas');
    c.width = w; c.height = THUMB_H;
    c.getContext('2d').drawImage(video, 0, 0, w, THUMB_H);
    filmstripTrack.appendChild(c);
    if (i % 5 === 0) msg('#filmstripMsg', `building preview strip… ${i}/${COUNT}`, false);
  }
  filmstripWidth = filmstripTrack.scrollWidth;

  await seekTo(video, savedTime); // restore the playhead to where it was before building
  updateTime();
  msg('#filmstripMsg', 'Drag the filmstrip below (or scroll over it) to scrub frame by frame.', false);
}

/** Slide the strip so the thumbnail at the current time sits under the center playhead. */
function positionFilmstrip() {
  if (!filmstripWidth || !filmstripDuration) return;
  const center = filmstrip.clientWidth / 2;
  const frac = Math.max(0, Math.min(1, video.currentTime / filmstripDuration));
  filmstripTrack.style.transform = `translateX(${center - frac * filmstripWidth}px)`;
}

let stripDragging = false, stripStartX = 0, stripStartTime = 0;

function stripDragBegin(clientX) {
  if (!filmstripWidth) return;
  stripDragging = true;
  video.pause();
  stripStartX = clientX;
  stripStartTime = video.currentTime;
  filmstrip.classList.add('scrubbing');
}
function stripDragMove(clientX) {
  if (!stripDragging) return;
  const pxPerSecond = filmstripWidth / filmstripDuration;
  // Dragging the strip left (negative dx) pulls later thumbnails under the
  // fixed center playhead — same convention as a native photo-gallery scrubber.
  const dt = -(clientX - stripStartX) / pxPerSecond;
  scrubTo(stripStartTime + dt);
}
function stripDragEnd() {
  stripDragging = false;
  filmstrip.classList.remove('scrubbing');
}

filmstrip.addEventListener('mousedown', e => { stripDragBegin(e.clientX); e.preventDefault(); });
window.addEventListener('mousemove', e => stripDragMove(e.clientX));
window.addEventListener('mouseup', stripDragEnd);

filmstrip.addEventListener('touchstart', e => stripDragBegin(e.touches[0].clientX), { passive: true });
filmstrip.addEventListener('touchmove', e => { e.preventDefault(); stripDragMove(e.touches[0].clientX); }, { passive: false });
filmstrip.addEventListener('touchend', stripDragEnd);
filmstrip.addEventListener('touchcancel', stripDragEnd);

filmstrip.addEventListener('wheel', e => {
  if (!filmstripWidth) return;
  e.preventDefault();
  video.pause();
  const dir = Math.sign(e.deltaY || e.deltaX);
  if (dir) scrubTo(video.currentTime + dir * frameStep());
}, { passive: false });

window.addEventListener('resize', positionFilmstrip);

// Arrow keys step frames when the player is on screen.
document.addEventListener('keydown', e => {
  if ($('#secPlayer').hidden || /^(INPUT|TEXTAREA)$/.test(e.target.tagName)) return;
  if (e.key === 'ArrowLeft') { e.preventDefault(); step(e.shiftKey ? -10 : -1); }
  if (e.key === 'ArrowRight') { e.preventDefault(); step(e.shiftKey ? 10 : 1); }
  if (e.key === ' ') { e.preventDefault(); video.paused ? video.play() : video.pause(); }
});

/* ── Frame selection ─────────────────────────────────────── */

function clipStatText(b) {
  if (b.start == null && b.end == null) return 'clip: auto (±0.13s around your pick)';
  const s = b.start == null ? '?' : `${b.start.toFixed(3)}s`;
  const e = b.end == null ? '?' : `${b.end.toFixed(3)}s`;
  return `clip: ${s} → ${e}`;
}

function refreshClipStat(phase, el) {
  const b = clipBounds[phase];
  el.querySelector('.clipStat').textContent = clipStatText(b);
  el.querySelector('.btnClipReset').disabled = b.start == null && b.end == null;
}

document.querySelectorAll('.phase').forEach(el => {
  const phase = el.dataset.phase;
  el.querySelector('.btnSel').addEventListener('click', () => selectFrame(phase, el));
  el.querySelector('.btnGoto').addEventListener('click', async () => {
    video.pause();
    await seekTo(video, frames[phase].time);
    updateTime();
  });
  el.querySelector('.btnClear').addEventListener('click', () => {
    frames[phase] = null;
    el.classList.remove('done');
    const stat = el.querySelector('.pstat');
    stat.textContent = 'not selected';
    stat.classList.remove('err');
    const c = el.querySelector('.thumb');
    c.getContext('2d').clearRect(0, 0, c.width, c.height);
    el.querySelector('.btnGoto').disabled = true;
    el.querySelector('.btnClear').disabled = true;
    refreshAnalyzeButton();
  });

  // Manual clip boundaries: scrub to a point, mark it as the start or end of
  // the clip that phase's pose reading is drawn from, instead of the
  // automatic ±0.13s window around the picked frame.
  el.querySelector('.btnClipStart').addEventListener('click', () => {
    clipBounds[phase].start = video.currentTime;
    refreshClipStat(phase, el);
  });
  el.querySelector('.btnClipEnd').addEventListener('click', () => {
    clipBounds[phase].end = video.currentTime;
    refreshClipStat(phase, el);
  });
  el.querySelector('.btnClipReset').addEventListener('click', () => {
    clipBounds[phase] = { start: null, end: null };
    refreshClipStat(phase, el);
  });
});

const CLIP_SPAN_MS = 260; // ~8 frames at 30fps either side of the picked instant

async function selectFrame(phase, el) {
  const stat = el.querySelector('.pstat');
  const btn = el.querySelector('.btnSel');
  btn.disabled = true;
  stat.classList.remove('err');
  video.pause();
  const centerTime = video.currentTime;
  const fps = Math.max(1, Number($('#fps').value) || 30);

  try {
    const b = clipBounds[phase];
    const manual = b.start != null && b.end != null && Math.abs(b.end - b.start) > 1e-3;

    const onProgress = (done, total) => { stat.textContent = `sampling frame ${done}/${total}…`; };
    stat.textContent = manual ? 'sampling your clip…' : 'sampling nearby frames…';
    const { frames: clip, centerIdx } = manual
      ? await captureRange(video, b.start, b.end, centerTime, fps, onProgress)
      : await captureWindow(video, centerTime, { spanMs: CLIP_SPAN_MS, fps, onProgress });

    stat.textContent = `reading pose across ${clip.length} frames…`;
    const results = await detectPoseSequence(clip);

    // Prefer the exact frame the user picked. If that one instant was too
    // blurred/occluded for a detection, fall back to the nearest frame in the
    // clip that did detect a person, rather than failing outright.
    let idx = centerIdx;
    if (!results[idx]) {
      let bestD = Infinity;
      results.forEach((r, i) => {
        if (!r) return;
        const d = Math.abs(i - centerIdx);
        if (d < bestD) { bestD = d; idx = i; }
      });
    }
    const res = results[idx];

    if (!res) {
      stat.textContent = manual
        ? 'no person detected anywhere in this clip — try a tighter start/end range or a moment where the whole body is visible'
        : 'no person detected in this clip — try a moment where the whole body is visible';
      stat.classList.add('err');
      return;
    }

    const canvas = clip[idx].canvas;
    const pts = toPixels(res.landmarks, canvas.width, canvas.height);
    frames[phase] = { time: clip[idx].time, canvas, pts, landmarks: res.landmarks };

    drawPose(el.querySelector('.thumb'), canvas, pts);
    const shifted = idx !== centerIdx
      ? ` (that exact instant was unreadable — shifted ${((idx - centerIdx) * (1000 / fps)).toFixed(0)}ms to a clean frame)`
      : '';
    stat.textContent = `selected at ${clip[idx].time.toFixed(3)}s${shifted}`;
    el.classList.add('done');
    el.querySelector('.btnGoto').disabled = false;
    el.querySelector('.btnClear').disabled = false;
  } catch (err) {
    console.error(err);
    stat.textContent = `error: ${err.message}`;
    stat.classList.add('err');
  } finally {
    btn.disabled = false;
    refreshAnalyzeButton();
  }
}

function refreshAnalyzeButton() {
  const ready = PHASES.every(p => frames[p]);
  $('#btnAnalyze').disabled = !ready;
  msg('#analyzeMsg', ready ? '' : `Select ${PHASES.filter(p => !frames[p]).map(p => PHASE_LABEL[p]).join(', ')}.`, false);
}

function resetSelections() {
  PHASES.forEach(p => {
    frames[p] = null;
    clipBounds[p] = { start: null, end: null };
    const el = document.querySelector(`.phase[data-phase="${p}"]`);
    el.classList.remove('done');
    const stat = el.querySelector('.pstat');
    stat.textContent = 'not selected';
    stat.classList.remove('err');
    el.querySelector('.btnGoto').disabled = true;
    el.querySelector('.btnClear').disabled = true;
    refreshClipStat(p, el);
    const c = el.querySelector('.thumb');
    c.width = c.height = 0;
  });
  refreshAnalyzeButton();
}

/* ── Analysis ────────────────────────────────────────────── */

$('#btnAnalyze').addEventListener('click', () => {
  try {
    // Frame ordering matters for direction inference, so warn if it looks off.
    if (!(frames.plant.time < frames.contact.time && frames.contact.time < frames.followThrough.time)) {
      msg('#analyzeMsg', 'Heads up: your frames are not in chronological order (plant → contact → follow-through). Results may be misleading.');
    }
    const ctx = deriveContext(frames);
    const metrics = computeMetrics(frames, ctx);
    lastResult = { ctx, ...scoreAll(metrics) };
    renderResults(lastResult);
    $('#secResults').hidden = false;
    $('#secResults').scrollIntoView({ behavior: 'smooth' });
  } catch (err) {
    console.error(err);
    msg('#analyzeMsg', `Analysis failed: ${err.message}`);
  }
});

function renderResults({ ctx, phases, overall }) {
  const lowLegConf = ctx.legConfidence < 0.35;
  $('#overall').innerHTML = `
    <div><span class="big">${overall ?? '—'}</span> / 100 overall</div>
    <p class="muted">
      Detected kicking leg: <strong>${ctx.kickSide}</strong>, support leg: <strong>${ctx.plantSide}</strong>.
      ${lowLegConf ? 'The two legs looked similar in this view, so that call is uncertain — a side-on camera angle gives a much better read.' : ''}
      Weighting: Plant 35% · Contact 40% · Follow-through 25%.
    </p>`;

  $('#phaseCards').innerHTML = '';
  for (const key of PHASES) {
    const p = phases[key];
    const card = document.createElement('div');
    card.className = 'card';
    card.innerHTML = `
      <h3>${grade(p.score)} ${p.label} — ${p.insufficient ? 'not enough measured' : `${p.score ?? '—'}/100`}
        <span class="muted">(${frames[key].time.toFixed(3)}s${p.skipped ? `, ${p.skipped} not scored` : ''})</span>
      </h3>
      ${p.insufficient
        ? `<p class="muted">⚠ Only ${Math.round(p.coverage * 100)}% of this phase could be measured, so no score is given and this phase is left out of the overall. Usually this means part of the body is out of frame, occluded, or too far away. Re-pick this frame, or re-shoot side-on with the whole body visible.</p>`
        : p.partial
          ? `<p class="muted">⚠ Only ${Math.round(p.coverage * 100)}% of this phase could be measured — treat this score as provisional. It counts proportionally less toward the overall.</p>`
          : ''}
      <div class="body"><div class="viz"></div><div class="metrics"></div></div>`;

    const viz = document.createElement('canvas');
    viz.style.width = '100%';
    drawPose(viz, frames[key].canvas, frames[key].pts, highlightFor(key, ctx));
    card.querySelector('.viz').appendChild(viz);

    const list = card.querySelector('.metrics');
    for (const mtr of p.metrics) list.appendChild(metricRow(mtr));

    $('#phaseCards').appendChild(card);
  }
}

function metricRow(mtr) {
  const div = document.createElement('div');
  div.className = 'metric';
  const val = isFinite(mtr.value) ? mtr.fmt(mtr.value) : 'n/a';
  const idealTxt = `ideal ${mtr.fmt.range(mtr.ideal[0], mtr.ideal[1])}`;

  if (mtr.uncertain) {
    div.innerHTML = `
      <div class="head"><span class="name">⚪ ${mtr.label}</span>
        <span class="val">${val} · ${idealTxt} · low confidence</span></div>
      <p class="muted">${mtr.reason}${mtr.caveat ? ' ' + mtr.caveat : ''}</p>
      ${mtr.feedback ? `<p>${mtr.feedback.what}</p>` : ''}`;
    return div;
  }

  const f = mtr.feedback;
  div.innerHTML = `
    <div class="head"><span class="name">${grade(mtr.score)} ${mtr.label}</span>
      <span class="val">${mtr.score}/100 · measured ${val} · ${idealTxt}</span></div>
    <div class="bar"><i style="width:${mtr.score}%"></i></div>
    <p>${f.what}</p>
    ${f.why ? `<p class="muted">${f.why}</p>` : ''}
    ${f.tip ? `<p class="tip">Try: ${f.tip}</p>` : ''}
    ${mtr.caveat ? `<p class="muted"><em>${mtr.caveat}</em></p>` : ''}`;
  return div;
}

/* ── Misc ────────────────────────────────────────────────── */

$('#btnRestart').addEventListener('click', () => {
  resetSelections();
  $('#secResults').hidden = true;
  $('#secPlayer').hidden = true;
  $('#fileInput').value = '';
  video.pause();
  filmstripTrack.innerHTML = '';
  filmstripTrack.style.transform = '';
  filmstripWidth = 0;
  filmstripDuration = 0;
  window.scrollTo({ top: 0, behavior: 'smooth' });
});

function msg(sel, text, isError = true) {
  const el = $(sel);
  el.textContent = text;
  el.style.color = isError ? '#a00' : '#555';
}

window.addEventListener('pagehide', () => {
  recorder.release();
  stopLiveLoop();
  if (video.dataset.objurl) URL.revokeObjectURL(video.dataset.objurl);
});
