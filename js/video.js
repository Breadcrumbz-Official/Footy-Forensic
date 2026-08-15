/* video.js — video input (upload + record), precise seeking, frame extraction. */

const MAX_CAPTURE_EDGE = 720;   // cap extracted frames; MediaPipe downscales anyway
const REC_LIMIT_MS = 10000;     // hard stop at 10s
const REC_MIN_MS = 1500;

/* ── Recording ──────────────────────────────────────────── */

export class Recorder {
  constructor() {
    this.stream = null;
    this.rec = null;
    this.chunks = [];
    this.startedAt = 0;
    this.timer = null;
    this.facingMode = 'environment';
  }

  /** Pick a container/codec this browser can actually produce. */
  static pickMime() {
    const candidates = [
      'video/webm;codecs=vp9', 'video/webm;codecs=vp8',
      'video/webm', 'video/mp4;codecs=avc1', 'video/mp4'
    ];
    return candidates.find(t => window.MediaRecorder?.isTypeSupported?.(t)) || '';
  }

  async enableCamera(previewEl, facingMode = this.facingMode) {
    if (!navigator.mediaDevices?.getUserMedia) throw new Error('Camera API not available in this browser.');
    // Drop any existing stream first — most devices only allow one active
    // capture of a given camera at a time, and this is also how switching
    // cameras releases the one we're about to stop using.
    this.stream?.getTracks().forEach(t => t.stop());
    this.stream = await navigator.mediaDevices.getUserMedia({
      video: { facingMode: { ideal: facingMode }, width: { ideal: 1280 }, height: { ideal: 720 } },
      audio: false
    });
    this.facingMode = facingMode;
    previewEl.srcObject = this.stream;
    previewEl.hidden = false;
    await previewEl.play().catch(() => {});
    return this.stream;
  }

  /**
   * Switch between front ('user') and back ('environment') camera. Not safe
   * to call while actively recording — the in-progress MediaRecorder is bound
   * to the current stream's tracks, and this stops them. Callers should keep
   * the flip control disabled during recording.
   */
  async switchCamera(previewEl) {
    const prev = this.facingMode;
    const next = prev === 'environment' ? 'user' : 'environment';
    try {
      await this.enableCamera(previewEl, next);
      return next;
    } catch (err) {
      // Likely a single-camera device (e.g. a laptop webcam) rejecting the
      // facingMode it doesn't have — reconnect to the one that was working
      // rather than leaving the camera off.
      await this.enableCamera(previewEl, prev).catch(() => {});
      throw err;
    }
  }

  /**
   * Start recording. `onTick(seconds)` fires ~10x/s. Resolves with a Blob when
   * the user stops or the 10s cap is reached.
   */
  start(onTick) {
    if (!this.stream) throw new Error('Camera not enabled.');
    const mimeType = Recorder.pickMime();
    this.chunks = [];
    this.rec = new MediaRecorder(this.stream, mimeType ? { mimeType } : undefined);

    const done = new Promise((resolve, reject) => {
      this.rec.ondataavailable = e => { if (e.data.size) this.chunks.push(e.data); };
      this.rec.onerror = e => reject(e.error || new Error('Recording failed.'));
      this.rec.onstop = () => {
        clearInterval(this.timer);
        resolve(new Blob(this.chunks, { type: this.rec.mimeType || 'video/webm' }));
      };
    });

    this.rec.start(100);
    this.startedAt = performance.now();
    this.timer = setInterval(() => {
      const s = (performance.now() - this.startedAt) / 1000;
      onTick?.(s);
      if (s * 1000 >= REC_LIMIT_MS) this.stop();
    }, 100);

    return done;
  }

  elapsedMs() { return performance.now() - this.startedAt; }
  canStop() { return this.elapsedMs() >= REC_MIN_MS; }

  stop() {
    if (this.rec && this.rec.state !== 'inactive') this.rec.stop();
  }

  release() {
    this.stop();
    this.stream?.getTracks().forEach(t => t.stop());
    this.stream = null;
  }
}

/* ── Loading ────────────────────────────────────────────── */

/**
 * Load a Blob/File into a <video> and guarantee a finite duration.
 *
 * MediaRecorder WebM files ship without a duration in the header, so
 * video.duration reads Infinity. The standard workaround is to seek far past
 * the end: the browser then clamps currentTime to the true end and backfills
 * duration. We restore currentTime to 0 afterwards.
 */
export async function loadVideo(videoEl, blob) {
  if (videoEl.dataset.objurl) URL.revokeObjectURL(videoEl.dataset.objurl);
  const url = URL.createObjectURL(blob);
  videoEl.dataset.objurl = url;
  videoEl.srcObject = null;
  videoEl.src = url;
  videoEl.load();

  await new Promise((resolve, reject) => {
    videoEl.onloadedmetadata = () => resolve();
    videoEl.onerror = () => reject(new Error('Could not decode this video format in your browser.'));
  });

  if (!isFinite(videoEl.duration) || videoEl.duration === 0) {
    await new Promise(resolve => {
      const onUpdate = () => {
        if (isFinite(videoEl.duration) && videoEl.duration > 0) {
          videoEl.removeEventListener('timeupdate', onUpdate);
          videoEl.currentTime = 0;
          resolve();
        } else {
          videoEl.currentTime = 1e6;   // keep pushing past the end
        }
      };
      videoEl.addEventListener('timeupdate', onUpdate);
      videoEl.currentTime = 1e6;
      setTimeout(resolve, 3000);       // never hang forever
    });
  }

  await seekTo(videoEl, 0);
  return videoEl.duration;
}

/* ── Seeking ────────────────────────────────────────────── */

/**
 * Seek and wait until the new frame is actually decoded and paintable.
 * `seeked` alone is enough on most browsers; when requestVideoFrameCallback
 * exists we also wait for one presented frame, which removes the occasional
 * off-by-one-frame capture on Chrome.
 *
 * Browsers do not reliably fire `seeked` for a "redundant" seek — a target
 * time that lands on the frame already displayed (common when sampling a clip
 * at a finer step than the source's real frame spacing, e.g. a burst of
 * frames for motion-blur handling). Without a fast path for that, every one
 * of those seeks would sit out the full safety timeout below, which turns a
 * multi-frame clip capture into a multi-second stall per frame. The poll
 * catches "already there" almost immediately instead.
 */
export function seekTo(videoEl, t) {
  const dur = isFinite(videoEl.duration) ? videoEl.duration : t;
  const target = Math.max(0, Math.min(t, Math.max(0, dur - 0.001)));

  if (Math.abs(videoEl.currentTime - target) < 1e-4 && videoEl.readyState >= 2) {
    return Promise.resolve(videoEl.currentTime);
  }

  return new Promise(resolve => {
    let settled = false;
    const cleanup = () => {
      videoEl.removeEventListener('seeked', onSeeked);
      clearInterval(poll);
      clearTimeout(safety);
    };
    const finish = () => {
      if (settled) return;
      settled = true;
      cleanup();
      resolve(videoEl.currentTime);
    };
    const onSeeked = () => {
      if (videoEl.requestVideoFrameCallback) {
        videoEl.requestVideoFrameCallback(() => finish());
        setTimeout(finish, 80);   // rVFC does not fire while fully paused on some builds
      } else {
        finish();
      }
    };
    videoEl.addEventListener('seeked', onSeeked);
    videoEl.currentTime = target;

    let stableTicks = 0;
    const poll = setInterval(() => {
      if (Math.abs(videoEl.currentTime - target) < 0.02 && videoEl.readyState >= 2) {
        if (++stableTicks >= 2) finish();   // two consecutive ticks so we don't fire mid-seek
      } else {
        stableTicks = 0;
      }
    }, 30);

    const safety = setTimeout(finish, 900); // never hang forever
  });
}

/* ── Frame extraction ───────────────────────────────────── */

/**
 * Copy the currently displayed video frame into a fresh canvas, downscaled so
 * the long edge is at most MAX_CAPTURE_EDGE. One copy per selected frame — we
 * reuse this canvas for detection, the thumbnail and the results view.
 */
export function captureFrame(videoEl) {
  const vw = videoEl.videoWidth, vh = videoEl.videoHeight;
  if (!vw || !vh) throw new Error('Video frame not ready yet.');
  const k = Math.min(1, MAX_CAPTURE_EDGE / Math.max(vw, vh));
  const canvas = document.createElement('canvas');
  canvas.width = Math.round(vw * k);
  canvas.height = Math.round(vh * k);
  canvas.getContext('2d', { willReadFrequently: false })
        .drawImage(videoEl, 0, 0, canvas.width, canvas.height);
  return canvas;
}

/* Clip extraction used to live here, sampling frames by repeatedly seeking the
 * <video> element. That moved to the server (server/video.py), which cuts the
 * clips out of the ORIGINAL upload in a single sequential decode — more exact
 * than browser seeking, at full resolution rather than the 720p this file
 * captures at, and it costs the user no waiting at pick time. All the browser
 * sends now is the timestamps. */
