/* api.js — talks to the analysis server.
 *
 * The browser no longer scores anything. It captures the video, shows the live
 * overlay, and lets the user pick three moments; the server decodes the video
 * at full resolution, runs the heavy pose model and the ball tracker, and
 * returns the finished analysis. See server/main.py.
 *
 * The server address is fixed rather than user-entered — there is exactly one
 * deployment this app talks to. Update SERVER_URL when the tunnel changes.
 */

const SERVER_URL = 'https://emote-galore-panther.ngrok-free.dev';

export const getServerUrl = () => SERVER_URL;

export class ApiError extends Error {
  constructor(message, { status = 0, detail = null } = {}) {
    super(message);
    this.name = 'ApiError';
    this.status = status;
    this.detail = detail;
  }
}

/** Ask the server what it is and whether it is up. Short timeout: this runs on
 *  a URL the user just typed, and a wrong one should fail fast, not hang. */
export async function checkHealth(url, { timeoutMs = 8000 } = {}) {
  if (!url) throw new ApiError('No server URL set.');
  const ctrl = new AbortController();
  const timer = setTimeout(() => ctrl.abort(), timeoutMs);
  try {
    const res = await fetch(`${url}/health`, {
      signal: ctrl.signal,
      // ngrok's free tier serves a browser interstitial unless this is set,
      // which would otherwise come back as HTML and fail JSON parsing with a
      // confusing error.
      headers: { 'ngrok-skip-browser-warning': 'true' }
    });
    if (!res.ok) throw new ApiError(`Server replied ${res.status}.`, { status: res.status });
    return await res.json();
  } catch (err) {
    if (err.name === 'AbortError') throw new ApiError('Server did not respond in time.');
    if (err instanceof ApiError) throw err;
    throw new ApiError(`Could not reach the server. ${err.message}`);
  } finally {
    clearTimeout(timer);
  }
}

/**
 * Upload the video and the three picks, get the analysis back.
 *
 * XMLHttpRequest rather than fetch purely for `upload.onprogress` — the video
 * is the slow part over a tunnel, and a progress bar is the difference between
 * "working" and "frozen" from the user's side.
 *
 * @param {string} url        server base URL
 * @param {Blob}   videoBlob  the recorded or uploaded video, as-is
 * @param {object} spec       { phases: {name: {time, clip}}, fps, footedness }
 * @param {(pct:number, phase:string) => void} onProgress
 */
export function analyze(url, videoBlob, spec, onProgress) {
  if (!url) return Promise.reject(new ApiError('No server URL set.'));

  const form = new FormData();
  form.append('spec', JSON.stringify(spec));
  form.append('video', videoBlob, 'kick.webm');

  return new Promise((resolve, reject) => {
    const xhr = new XMLHttpRequest();
    xhr.open('POST', `${url}/analyze`);
    xhr.responseType = 'json';
    xhr.setRequestHeader('ngrok-skip-browser-warning', 'true');
    xhr.timeout = 10 * 60 * 1000;   // heavy model on a cold worker is not fast

    xhr.upload.onprogress = e => {
      if (e.lengthComputable) onProgress?.(e.loaded / e.total, 'upload');
    };
    // Upload finished; everything from here is the server thinking.
    xhr.upload.onload = () => onProgress?.(1, 'analyze');

    xhr.onload = () => {
      const body = xhr.response;
      if (xhr.status >= 200 && xhr.status < 300) return resolve(body);
      // FastAPI puts the readable part in `detail`.
      const detail = body?.detail ?? body;
      reject(new ApiError(
        typeof detail === 'string' ? detail : `Server error ${xhr.status}.`,
        { status: xhr.status, detail }));
    };
    xhr.onerror = () => reject(new ApiError(
      'Upload failed — check the server URL is reachable and the tunnel is still open.'));
    xhr.ontimeout = () => reject(new ApiError('The server took too long to respond.'));
    xhr.send(form);
  });
}
