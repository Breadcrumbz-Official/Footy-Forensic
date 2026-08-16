const MAX_CAPTURE_EDGE = 720;   
const REC_LIMIT_MS = 10000;     
const REC_MIN_MS = 1500;


export class Recorder {
  constructor() {
    this.stream = null;
    this.rec = null;
    this.chunks = [];
    this.startedAt = 0;
    this.timer = null;
    this.facingMode = 'environment';
  }

  
  static pickMime() {
    const candidates = [
      'video/webm;codecs=vp9', 'video/webm;codecs=vp8',
      'video/webm', 'video/mp4;codecs=avc1', 'video/mp4'
    ];
    return candidates.find(t => window.MediaRecorder?.isTypeSupported?.(t)) || '';
  }

  async enableCamera(previewEl, facingMode = this.facingMode) {
    if (!navigator.mediaDevices?.getUserMedia) throw new Error('Camera API not available in this browser.');
    
    
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

  
  async switchCamera(previewEl) {
    const prev = this.facingMode;
    const next = prev === 'environment' ? 'user' : 'environment';
    try {
      await this.enableCamera(previewEl, next);
      return next;
    } catch (err) {
      
      
      await this.enableCamera(previewEl, prev).catch(() => {});
      throw err;
    }
  }

  
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
          videoEl.currentTime = 1e6;   
        }
      };
      videoEl.addEventListener('timeupdate', onUpdate);
      videoEl.currentTime = 1e6;
      setTimeout(resolve, 3000);       
    });
  }

  await seekTo(videoEl, 0);
  return videoEl.duration;
}


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
        setTimeout(finish, 80);   
      } else {
        finish();
      }
    };
    videoEl.addEventListener('seeked', onSeeked);
    videoEl.currentTime = target;

    let stableTicks = 0;
    const poll = setInterval(() => {
      if (Math.abs(videoEl.currentTime - target) < 0.02 && videoEl.readyState >= 2) {
        if (++stableTicks >= 2) finish();   
      } else {
        stableTicks = 0;
      }
    }, 30);

    const safety = setTimeout(finish, 900); 
  });
}


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
