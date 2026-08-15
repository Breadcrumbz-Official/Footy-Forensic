/* api.ts — talks to the analysis server.
 *
 * The browser scores nothing. It captures the video and lets the user pick
 * three moments; the server decodes at full resolution, runs the heavy pose
 * model and the ball tracker, and returns the finished analysis.
 * See server/main.py and server/scoring.py for the shapes below.
 *
 * The page is served by that same server, so the default base URL is empty —
 * every request is same-origin and there is nothing for the user to configure.
 * The localStorage override exists for `vite dev`, where the page is on :5173
 * and the API is not.
 */

const KEY = 'sfai.serverUrl';

/** Empty string means same origin, which is the normal case in production. */
export const getServerUrl = () => localStorage.getItem(KEY) || '';

export function setServerUrl(url: string) {
  const clean = String(url || '').trim().replace(/\/+$/, '');
  if (clean) localStorage.setItem(KEY, clean);
  else localStorage.removeItem(KEY);
  return clean;
}

/* ── Server response shapes ──────────────────────────────────────────────── */

export type PhaseKey = 'plant' | 'contact' | 'followThrough';

export const PHASE_KEYS: PhaseKey[] = ['plant', 'contact', 'followThrough'];

export interface Feedback {
  what: string;
  why?: string;
  tip?: string;
}

export interface Metric {
  id: string;
  label: string;
  weight: number;
  value: number | null;
  vis: number;
  caveat: string | null;
  ideal: [number, number];
  idealText: string;
  valueText: string | null;
  /** Absent when the metric could not be scored. */
  score?: number;
  /** Reported but excluded from the phase score. */
  uncertain?: boolean;
  reason?: string | null;
  feedback?: Feedback;
}

export interface Phase {
  key: PhaseKey;
  label: string;
  weight: number;
  metrics: Metric[];
  /** null when coverage was too low to publish a number at all. */
  score: number | null;
  counted: number;
  skipped: number;
  coverage: number;
  insufficient: boolean;
  partial: boolean;
}

export interface Ball {
  x: number;
  y: number;
  r: number;
  score: number;
  interpolated: boolean;
  rescued: boolean;
}

export interface FrameOut {
  time: number;
  shiftedMs: number;
  ball: Ball | null;
  ballFramesFound: number;
  clipFrames: number;
  /** data:image/jpeg;base64,... — the annotated frame the server drew. */
  image: string;
}

export interface Context {
  kickSide: string;
  plantSide: string;
  dir: number;
  dirSource: string;
  legConfidence: number;
  legSource: string;
  /** shoulderRatio is null when the torso was too degenerate to measure. */
  view: { score: number; shoulderRatio: number | null; label: string };
}

export interface AnalysisResult {
  overall: number | null;
  phases: Record<PhaseKey, Phase>;
  context: Context;
  frames: Record<PhaseKey, FrameOut>;
  warnings: string[];
  video: Record<string, unknown>;
  timing: { decodeMs: number; detectMs: number; totalMs: number };
}

export interface Health {
  ok: boolean;
  workers: number;
  poseModel: string;
  ballModel: string;
  maxUploadMb: number;
  maxEdge: number;
}

export interface AnalyseSpec {
  phases: Record<PhaseKey, { time: number; clip?: { start: number; end: number } }>;
  fps: number;
  footedness: 'auto' | 'left' | 'right';
  /**
   * The duration this browser measured for the video. The server lays the
   * decoded frames evenly across it instead of trusting the container's
   * declared frame rate, which is what keeps the picked times meaning the same
   * thing on both sides. Without it, picks near the end of a variable-rate
   * recording fall off the end of the server's timeline and return no frames.
   */
  duration: number;
}

/* ── Calls ───────────────────────────────────────────────────────────────── */

export class ApiError extends Error {
  status: number;
  detail: unknown;
  constructor(message: string, { status = 0, detail = null }: { status?: number; detail?: unknown } = {}) {
    super(message);
    this.name = 'ApiError';
    this.status = status;
    this.detail = detail;
  }
}

// ngrok's free tier serves a browser interstitial unless this is set, which
// would otherwise come back as HTML and fail JSON parsing with a confusing
// error. Harmless on any other host.
const NGROK_HEADER = { 'ngrok-skip-browser-warning': 'true' } as const;

/** Ask the server what it is and whether it is up. Short timeout so a wrong
 *  URL fails fast rather than hanging. */
export async function checkHealth(url = getServerUrl(), { timeoutMs = 8000 } = {}): Promise<Health> {
  const ctrl = new AbortController();
  const timer = setTimeout(() => ctrl.abort(), timeoutMs);
  try {
    const res = await fetch(`${url}/health`, { signal: ctrl.signal, headers: NGROK_HEADER });
    if (!res.ok) throw new ApiError(`Server replied ${res.status}.`, { status: res.status });
    return (await res.json()) as Health;
  } catch (err) {
    if (err instanceof ApiError) throw err;
    if ((err as Error).name === 'AbortError') throw new ApiError('Server did not respond in time.');
    throw new ApiError(`Could not reach the analysis server. ${(err as Error).message}`);
  } finally {
    clearTimeout(timer);
  }
}

/**
 * Upload the video and the three picks, get the analysis back.
 *
 * XMLHttpRequest rather than fetch purely for `upload.onprogress` — over a
 * tunnel the upload is the slow part, and a progress bar is the difference
 * between "working" and "frozen" from the user's side.
 */
export function analyze(
  videoBlob: Blob,
  spec: AnalyseSpec,
  onProgress?: (pct: number, stage: 'upload' | 'analyze') => void,
  url = getServerUrl(),
): Promise<AnalysisResult> {
  const form = new FormData();
  form.append('spec', JSON.stringify(spec));
  form.append('video', videoBlob, 'kick.webm');

  return new Promise((resolve, reject) => {
    const xhr = new XMLHttpRequest();
    xhr.open('POST', `${url}/analyze`);
    xhr.responseType = 'json';
    xhr.setRequestHeader('ngrok-skip-browser-warning', 'true');
    xhr.timeout = 10 * 60 * 1000;   // the heavy model on a cold worker is not fast

    xhr.upload.onprogress = e => {
      if (e.lengthComputable) onProgress?.(e.loaded / e.total, 'upload');
    };
    // Upload finished; everything from here is the server thinking.
    xhr.upload.onload = () => onProgress?.(1, 'analyze');

    xhr.onload = () => {
      const body = xhr.response;
      if (xhr.status >= 200 && xhr.status < 300) return resolve(body as AnalysisResult);
      // FastAPI puts the readable part in `detail`.
      const detail = body?.detail ?? body;
      reject(new ApiError(
        typeof detail === 'string' ? detail : `Server error ${xhr.status}.`,
        { status: xhr.status, detail }));
    };
    xhr.onerror = () => reject(new ApiError(
      'Upload failed — check the analysis server is reachable and the tunnel is still open.'));
    xhr.ontimeout = () => reject(new ApiError('The server took too long to respond.'));
    xhr.send(form);
  });
}
