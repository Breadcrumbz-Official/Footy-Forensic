const KEY = 'sfai.serverUrl';


export const getServerUrl = () => localStorage.getItem(KEY) || '';

export function setServerUrl(url: string) {
  const clean = String(url || '').trim().replace(/\/+$/, '');
  if (clean) localStorage.setItem(KEY, clean);
  else localStorage.removeItem(KEY);
  return clean;
}


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
  
  score?: number;
  
  uncertain?: boolean;
  reason?: string | null;
  feedback?: Feedback;
}

export interface Phase {
  key: PhaseKey;
  label: string;
  weight: number;
  metrics: Metric[];
  
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
  
  reposed?: boolean;
  
  image: string;
}

export interface Context {
  kickSide: string;
  plantSide: string;
  dir: number;
  dirSource: string;
  legConfidence: number;
  legSource: string;
  
  view: { score: number; shoulderRatio: number | null; label: string };
}

export interface AnalysisResult {
  overall: number | null;
  phases: Record<PhaseKey, Phase>;
  context: Context;
  frames: Record<PhaseKey, FrameOut>;
  warnings: string[];
  video: Record<string, unknown>;
  
  aiFeedback: Partial<Record<PhaseKey, string>> | null;
  timing: { decodeMs: number; detectMs: number; verifyMs?: number; aiMs?: number; totalMs: number };
}

export interface Health {
  ok: boolean;
  workers: number;
  poseModel: string;
  ballModel: string;
  maxUploadMb: number;
  maxEdge: number;
  
  aiFeedback: boolean;
}

export interface AnalyseSpec {
  phases: Record<PhaseKey, { time: number; clip?: { start: number; end: number } }>;
  fps: number;
  footedness: 'auto' | 'left' | 'right';
  
  duration: number;
}


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


const NGROK_HEADER = { 'ngrok-skip-browser-warning': 'true' } as const;


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
    xhr.timeout = 10 * 60 * 1000;   

    xhr.upload.onprogress = e => {
      if (e.lengthComputable) onProgress?.(e.loaded / e.total, 'upload');
    };
    
    xhr.upload.onload = () => onProgress?.(1, 'analyze');

    xhr.onload = () => {
      const body = xhr.response;
      if (xhr.status >= 200 && xhr.status < 300) return resolve(body as AnalysisResult);
      
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
