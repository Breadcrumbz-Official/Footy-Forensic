import { Recorder, loadVideo, seekTo, captureFrame } from './js/video.js';
import { initPose, detectPoseLive, resetLiveClock, getBackend } from './js/mediapipe.js';
import { initBall, detectBallLive, resetBallLive, getBallBackend } from './js/ballDetection.js';
import { toPixels, torsoScale, viewQuality } from './js/biomechanics.js';
import { drawOverlay } from './js/overlay.js';
import * as api from './js/api.js';

const $ = s => document.querySelector(s);
const PHASES = ['plant', 'contact', 'followThrough'];
const PHASE_LABEL = { plant: 'Plant + Backswing', contact: 'Contact', followThrough: 'Follow-through' };

const video = $('#video');
const recorder = new Recorder();


let sourceBlob = null;

let picks = { plant: null, contact: null, followThrough: null };

let clipBounds = {
  plant: { start: null, end: null },
  contact: { start: null, end: null },
  followThrough: { start: null, end: null }
};
let serverInfo = null;


const engine = { pose: 'loading…', ball: 'loading…' };
const renderEngineStatus = () =>
  { $('#engineStatus').textContent = `Live preview — pose: ${engine.pose} · ball: ${engine.ball}`; };
renderEngineStatus();

initPose()
  .then(() => { engine.pose = `ready (${getBackend()})`; })
  .catch(err => { console.error(err); engine.pose = 'unavailable'; })
  .finally(renderEngineStatus);


initBall()
  .then(() => { engine.ball = `ready (${getBallBackend()})`; })
  .catch(err => { console.warn('Live ball detector unavailable:', err); engine.ball = 'unavailable'; })
  .finally(renderEngineStatus);


async function connectServer() {
  try {
    serverInfo = await api.checkHealth(api.getServerUrl());
    setServerStatus(`Analysis server connected · ${serverInfo.workers} workers`, 'ok');
  } catch (err) {
    serverInfo = null;
    setServerStatus(`Analysis server unreachable: ${err.message} Retrying…`, 'err');
    setTimeout(connectServer, 8000);
  }
  refreshAnalyzeButton();
}

function setServerStatus(text, kind) {
  const el = $('#serverStatus');
  el.textContent = text;
  el.className = kind === 'ok' ? 'ok' : kind === 'err' ? 'msg' : 'muted';
}

connectServer();


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
    msg('#inputMsg', 'Camera on. The preview and overlay stay in this tab — nothing is sent until you analyze.', false);
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
    sizeCamOverlay(); 
    msg('#inputMsg', `Switched to the ${mode === 'user' ? 'front' : 'back'} camera.`, false);
  } catch (err) {
    msg('#inputMsg', `Could not switch camera — reconnected to the previous one. (${err.message})`);
  } finally {
    $('#btnFlip').disabled = false;
  }
});

$('#btnRec').addEventListener('click', async () => {
  $('#btnRec').disabled = true;
  $('#btnFlip').disabled = true; 
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


const LIVE_INTERVAL_MS = 33;    
const BALL_INTERVAL_MS = 200;   
                                 
const BALL_EXTRAPOLATE_MAX_MS = 350; 
                                      
                                      
let liveShowSkeleton = false;
let liveShowAngles = false;
let liveShowBall = false;
let liveTimer = null;
let liveBusy = false;
let ballBusy = false;
let lastBallAt = 0;
let ballPrev = null;   
let ballCurr = null;   
let lastViewLabel = null;

const liveOverlayWanted = () => liveShowSkeleton || liveShowAngles || liveShowBall;

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
  resetBallLive();
  liveBusy = ballBusy = false;
  ballPrev = ballCurr = null;
  lastBallAt = 0;

  liveTimer = setInterval(async () => {
    if (liveBusy) return; 
    const camPreview = $('#camPreview');
    if (camPreview.hidden || !camPreview.videoWidth) return;
    liveBusy = true;
    sizeCamOverlay();
    try {
      const frame = captureFrame(camPreview);
      const overlay = $('#camOverlay');
      const res = await detectPoseLive(frame);
      const pts = res ? toPixels(res.landmarks, overlay.width, overlay.height) : null;
      const scale = pts ? torsoScale(pts) : null;

      if (pts) updateViewHint(pts);
      if (liveShowBall) maybeDetectBall(frame, scale, overlay);
      else ballPrev = ballCurr = null;

      const ball = liveShowBall ? scaleBall(getSmoothedBall(performance.now()), frame, overlay) : null;
      drawOverlay(overlay, pts, {
        skeleton: liveShowSkeleton, angles: liveShowAngles,
        ball
      });
    } catch (err) {
      console.error(err);
    } finally {
      liveBusy = false;
    }
  }, LIVE_INTERVAL_MS);
}


function maybeDetectBall(frame, scale, overlay) {
  const now = performance.now();
  if (ballBusy || now - lastBallAt < BALL_INTERVAL_MS) return;
  ballBusy = true;
  lastBallAt = now;
  detectBallLive(frame, scale ? scale * (frame.width / overlay.width) : null)
    .then(b => {
      const t = performance.now();
      if (b) { ballPrev = ballCurr; ballCurr = { ...b, t }; }
      
      
      else { ballPrev = ballCurr = null; }
    })
    .catch(err => { console.warn('live ball:', err); ballPrev = ballCurr = null; })
    .finally(() => { ballBusy = false; });
}


function getSmoothedBall(now) {
  if (!ballCurr) return null;
  if (!ballPrev) return ballCurr;
  const dt = ballCurr.t - ballPrev.t;
  const elapsed = now - ballCurr.t;
  if (dt <= 0 || elapsed < 0 || elapsed > BALL_EXTRAPOLATE_MAX_MS) return ballCurr;
  const vx = (ballCurr.x - ballPrev.x) / dt;
  const vy = (ballCurr.y - ballPrev.y) / dt;
  return { x: ballCurr.x + vx * elapsed, y: ballCurr.y + vy * elapsed, r: ballCurr.r };
}


function scaleBall(ball, frame, overlay) {
  if (!ball) return null;
  const k = overlay.width / frame.width;
  return { x: ball.x * k, y: ball.y * k, r: ball.r * k };
}


function updateViewHint(pts) {
  const v = viewQuality(pts);
  if (v.label === lastViewLabel) return;
  lastViewLabel = v.label;
  const el = $('#viewHint');
  if (v.label === 'side-on') {
    el.textContent = '📐 Camera angle looks side-on — good.';
    el.className = 'ok';
  } else if (v.label === 'angled') {
    el.textContent = '📐 Partly side-on. Rotate the camera square to the line of the shot for full accuracy.';
    el.className = 'muted';
  } else {
    el.textContent = '📐 This looks face-on. Fore/aft measurements cannot be read from here — move the camera to the side.';
    el.className = 'msg';
  }
}

function stopLiveLoop() {
  clearInterval(liveTimer);
  liveTimer = null;
  ballPrev = ballCurr = null;
  lastViewLabel = null;
  $('#viewHint').textContent = '';
  const overlay = $('#camOverlay');
  overlay.getContext('2d').clearRect(0, 0, overlay.width, overlay.height);
}

const bindLiveToggle = (sel, set) => $(sel).addEventListener('change', e => {
  set(e.target.checked);
  liveOverlayWanted() ? startLiveLoop() : stopLiveLoop();
});
bindLiveToggle('#chkLiveSkeleton', v => { liveShowSkeleton = v; });
bindLiveToggle('#chkLiveAngles', v => { liveShowAngles = v; });
bindLiveToggle('#chkLiveBall', v => { liveShowBall = v; });


async function useVideo(blob) {
  try {
    resetSelections();
    sourceBlob = blob;
    const duration = await loadVideo(video, blob);
    $('#secPlayer').hidden = false;
    $('#secResults').hidden = true;
    updateTime();
    await buildFilmstrip(duration);

    const mb = (blob.size / 1048576).toFixed(1);
    msg('#inputMsg',
      duration > 15
        ? `This clip is ${duration.toFixed(1)}s (${mb}MB). Short clips (5–10s) scrub more easily and upload faster.`
        : `Loaded ${duration.toFixed(1)}s (${mb}MB).`,
      false);
    $('#secPlayer').scrollIntoView({ behavior: 'smooth' });
  } catch (err) {
    msg('#inputMsg', err.message);
  }
}


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
let filmstripWidth = 0;
let filmstripDuration = 0;


async function buildFilmstrip(duration) {
  filmstripTrack.innerHTML = '';
  filmstripTrack.style.transform = '';
  filmstripWidth = 0;
  filmstripDuration = duration;
  if (!duration || !isFinite(duration)) return;

  const THUMB_H = 64;
  const COUNT = Math.min(60, Math.max(12, Math.round(duration * 6)));
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

  await seekTo(video, savedTime);
  updateTime();
  msg('#filmstripMsg', 'Drag the filmstrip below (or scroll over it) to scrub frame by frame.', false);
}

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
  
  
  scrubTo(stripStartTime - (clientX - stripStartX) / pxPerSecond);
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

document.addEventListener('keydown', e => {
  if ($('#secPlayer').hidden || /^(INPUT|TEXTAREA|SELECT)$/.test(e.target.tagName)) return;
  if (e.key === 'ArrowLeft') { e.preventDefault(); step(e.shiftKey ? -10 : -1); }
  if (e.key === 'ArrowRight') { e.preventDefault(); step(e.shiftKey ? 10 : 1); }
  if (e.key === ' ') { e.preventDefault(); video.paused ? video.play() : video.pause(); }
});


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

  
  el.querySelector('.btnSel').addEventListener('click', () => {
    video.pause();
    picks[phase] = { time: video.currentTime };
    const thumb = el.querySelector('.thumb');
    const frame = captureFrame(video);
    thumb.width = frame.width;
    thumb.height = frame.height;
    thumb.getContext('2d').drawImage(frame, 0, 0);
    el.querySelector('.pstat').textContent = `selected at ${video.currentTime.toFixed(3)}s`;
    el.querySelector('.pstat').classList.remove('err');
    el.classList.add('done');
    el.querySelector('.btnGoto').disabled = false;
    el.querySelector('.btnClear').disabled = false;
    refreshAnalyzeButton();
  });

  el.querySelector('.btnGoto').addEventListener('click', async () => {
    video.pause();
    await seekTo(video, picks[phase].time);
    updateTime();
  });

  el.querySelector('.btnClear').addEventListener('click', () => {
    picks[phase] = null;
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

function refreshAnalyzeButton() {
  const missing = PHASES.filter(p => !picks[p]);
  const ready = missing.length === 0 && !!serverInfo && !!sourceBlob;
  $('#btnAnalyze').disabled = !ready;
  if (!serverInfo) msg('#analyzeMsg', 'Waiting for the analysis server…', false);
  else if (missing.length) msg('#analyzeMsg', `Select ${missing.map(p => PHASE_LABEL[p]).join(', ')}.`, false);
  else msg('#analyzeMsg', '', false);
}

function resetSelections() {
  PHASES.forEach(p => {
    picks[p] = null;
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


$('#btnAnalyze').addEventListener('click', async () => {
  const btn = $('#btnAnalyze');
  btn.disabled = true;
  msg('#analyzeMsg', '', false);
  $('#progressWrap').hidden = false;
  setProgress(0, 'preparing…');

  const spec = {
    phases: Object.fromEntries(PHASES.map(p => [p, {
      time: picks[p].time,
      clip: (clipBounds[p].start != null && clipBounds[p].end != null)
        ? { start: clipBounds[p].start, end: clipBounds[p].end } : null
    }])),
    fps: Math.max(1, Number($('#fps').value) || 30),
    footedness: $('#footedness').value
  };

  try {
    const result = await api.analyze(api.getServerUrl(), sourceBlob, spec, (pct, phase) => {
      if (phase === 'upload') setProgress(pct * 0.6, `uploading… ${Math.round(pct * 100)}%`);
      else setProgress(0.6, 'analyzing on the server — pose, ball tracking and scoring…');
    });
    setProgress(1, 'done');
    renderResults(result);
    $('#secResults').hidden = false;
    $('#secResults').scrollIntoView({ behavior: 'smooth' });
  } catch (err) {
    console.error(err);
    msg('#analyzeMsg', `Analysis failed: ${err.message}`);
  } finally {
    $('#progressWrap').hidden = true;
    btn.disabled = false;
    refreshAnalyzeButton();
  }
});

function setProgress(frac, text) {
  $('#progressBar').style.width = `${Math.round(frac * 100)}%`;
  $('#progressMsg').textContent = text;
}


const grade = s => s == null ? '⚪' : s >= 85 ? '🟢' : s >= 70 ? '🟡' : '🔴';

const DIR_SOURCE_TEXT = {
  ball: 'from the ball’s own travel — the most direct evidence available',
  ankle: 'from the kicking ankle’s travel (no ball tracked across these frames)',
  toe: 'from the plant foot’s toe direction — the weakest of the three, so a near head-on camera may have confused it'
};

function renderResults(r) {
  const ctx = r.context;
  const ballsFound = PHASES.filter(p => r.frames[p]?.ball).length;
  const legText = ctx.legSource === 'stated'
    ? `as you selected (<strong>${ctx.kickSide}</strong>-footed)`
    : `detected as <strong>${ctx.kickSide}</strong>`;

  $('#overall').innerHTML = `
    <div><span class="big">${r.overall ?? '—'}</span> / 100 overall</div>
    <p class="muted">
      Kicking leg ${legText}, support leg <strong>${ctx.plantSide}</strong>.
      Camera angle: <strong>${ctx.view.label}</strong> (shoulder span ${ctx.view.shoulderRatio.toFixed(2)} × torso).
      Kick direction taken ${DIR_SOURCE_TEXT[ctx.dirSource] || 'from body movement'}.
      Ball tracked in <strong>${ballsFound}</strong> of 3 moments.
      Weighting: Plant 35% · Contact 40% · Follow-through 25%.
      <span class="mono">${r.timing.totalMs}ms server-side</span>
    </p>`;

  $('#warnings').innerHTML = r.warnings?.length
    ? `<ul class="warnList">${r.warnings.map(w => `<li>${escapeHtml(w)}</li>`).join('')}</ul>`
    : '';

  $('#phaseCards').innerHTML = '';
  for (const key of PHASES) {
    const p = r.phases[key];
    const f = r.frames[key];
    const card = document.createElement('div');
    card.className = 'card';
    card.innerHTML = `
      <h3>${grade(p.score)} ${p.label} — ${p.insufficient ? 'not enough measured' : `${p.score ?? '—'}/100`}
        <span class="muted">(${f.time.toFixed(3)}s${p.skipped ? `, ${p.skipped} not scored` : ''}${
          f.ballFramesFound ? `, ball in ${f.ballFramesFound}/${f.clipFrames} clip frames` : ', no ball'})</span>
      </h3>
      ${p.insufficient
        ? `<p class="muted">⚠ Only ${Math.round(p.coverage * 100)}% of this phase could be measured, so no score is given and this phase is left out of the overall. Usually this means part of the body is out of frame, occluded, or the camera is not side-on enough.</p>`
        : p.partial
          ? `<p class="muted">⚠ Only ${Math.round(p.coverage * 100)}% of this phase could be measured — treat this score as provisional. It counts proportionally less toward the overall.</p>`
          : ''}
      <div class="body"><div class="viz"></div><div class="metrics"></div></div>`;

    const img = new Image();
    img.src = f.image;
    img.alt = `${p.label} frame with detected skeleton and ball`;
    img.style.width = '100%';
    card.querySelector('.viz').appendChild(img);

    const list = card.querySelector('.metrics');
    for (const mtr of p.metrics) list.appendChild(metricRow(mtr));
    $('#phaseCards').appendChild(card);
  }
}

function metricRow(mtr) {
  const div = document.createElement('div');
  div.className = 'metric';
  const val = mtr.valueText ?? 'n/a';
  const idealTxt = `ideal ${mtr.idealText}`;

  if (mtr.uncertain) {
    div.innerHTML = `
      <div class="head"><span class="name">⚪ ${escapeHtml(mtr.label)}</span>
        <span class="val">${val} · ${idealTxt} · low confidence</span></div>
      <p class="muted">${escapeHtml(mtr.reason || '')}${mtr.caveat ? ' ' + escapeHtml(mtr.caveat) : ''}</p>
      ${mtr.feedback ? `<p>${escapeHtml(mtr.feedback.what)}</p>` : ''}`;
    return div;
  }

  const f = mtr.feedback || {};
  div.innerHTML = `
    <div class="head"><span class="name">${grade(mtr.score)} ${escapeHtml(mtr.label)}</span>
      <span class="val">${mtr.score}/100 · measured ${val} · ${idealTxt}</span></div>
    <div class="bar"><i style="width:${mtr.score}%"></i></div>
    <p>${escapeHtml(f.what || '')}</p>
    ${f.why ? `<p class="muted">${escapeHtml(f.why)}</p>` : ''}
    ${f.tip ? `<p class="tip">Try: ${escapeHtml(f.tip)}</p>` : ''}
    ${mtr.caveat ? `<p class="muted"><em>${escapeHtml(mtr.caveat)}</em></p>` : ''}`;
  return div;
}


function escapeHtml(s) {
  return String(s).replace(/[&<>"']/g, c =>
    ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));
}


$('#btnRestart').addEventListener('click', () => {
  resetSelections();
  sourceBlob = null;
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
