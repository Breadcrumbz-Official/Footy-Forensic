export const REC_LIMIT_MS = 10000;

export const REC_MIN_MS = 1500;

export type FacingMode = 'environment' | 'user';

export class Recorder {
  stream: MediaStream | null = null;
  private rec: MediaRecorder | null = null;
  private chunks: Blob[] = [];
  private startedAt = 0;
  private timer: number | null = null;
  facingMode: FacingMode = 'environment';

  
  static pickMime(): string {
    const candidates = [
      'video/webm;codecs=vp9', 'video/webm;codecs=vp8',
      'video/webm', 'video/mp4;codecs=avc1', 'video/mp4',
    ];
    return candidates.find(t => window.MediaRecorder?.isTypeSupported?.(t)) || '';
  }

  static supported(): boolean {
    
    
    return typeof navigator.mediaDevices?.getUserMedia === "function"
        && typeof window.MediaRecorder === "function";
  }

  async enableCamera(previewEl: HTMLVideoElement, facingMode: FacingMode = this.facingMode) {
    if (!navigator.mediaDevices?.getUserMedia) {
      throw new Error('Camera API not available in this browser.');
    }
    
    
    this.stream?.getTracks().forEach(t => t.stop());
    this.stream = await navigator.mediaDevices.getUserMedia({
      video: { facingMode: { ideal: facingMode }, width: { ideal: 1280 }, height: { ideal: 720 } },
      audio: false,
    });
    this.facingMode = facingMode;
    previewEl.srcObject = this.stream;
    await previewEl.play().catch(() => {});
    return this.stream;
  }

  
  async switchCamera(previewEl: HTMLVideoElement): Promise<FacingMode> {
    const prev = this.facingMode;
    const next: FacingMode = prev === 'environment' ? 'user' : 'environment';
    try {
      await this.enableCamera(previewEl, next);
      return next;
    } catch (err) {
      
      
      await this.enableCamera(previewEl, prev).catch(() => {});
      throw err;
    }
  }

  
  start(onTick?: (seconds: number) => void): Promise<Blob> {
    if (!this.stream) throw new Error('Camera not enabled.');
    const mimeType = Recorder.pickMime();
    this.chunks = [];
    this.rec = new MediaRecorder(this.stream, mimeType ? { mimeType } : undefined);

    const done = new Promise<Blob>((resolve, reject) => {
      this.rec!.ondataavailable = e => { if (e.data.size) this.chunks.push(e.data); };
      this.rec!.onerror = () => reject(new Error('Recording failed.'));
      this.rec!.onstop = () => {
        if (this.timer !== null) clearInterval(this.timer);
        resolve(new Blob(this.chunks, { type: this.rec?.mimeType || 'video/webm' }));
      };
    });

    this.rec.start(100);
    this.startedAt = performance.now();
    this.timer = window.setInterval(() => {
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


export function ensureFiniteDuration(videoEl: HTMLVideoElement): Promise<number> {
  if (Number.isFinite(videoEl.duration) && videoEl.duration > 0) {
    return Promise.resolve(videoEl.duration);
  }

  return new Promise<number>(resolve => {
    let settled = false;
    const finish = () => {
      if (settled) return;
      settled = true;
      videoEl.removeEventListener('timeupdate', onUpdate);
      clearTimeout(safety);
      try { videoEl.currentTime = 0; } catch {  }
      resolve(Number.isFinite(videoEl.duration) ? videoEl.duration : 0);
    };
    const onUpdate = () => {
      if (Number.isFinite(videoEl.duration) && videoEl.duration > 0) finish();
      else videoEl.currentTime = 1e6;   
    };

    videoEl.addEventListener('timeupdate', onUpdate);
    videoEl.currentTime = 1e6;
    const safety = setTimeout(finish, 3000);   
  });
}
