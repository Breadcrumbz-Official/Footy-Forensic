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


export async function checkHealth(url, { timeoutMs = 8000 } = {}) {
  if (!url) throw new ApiError('No server URL set.');
  const ctrl = new AbortController();
  const timer = setTimeout(() => ctrl.abort(), timeoutMs);
  try {
    const res = await fetch(`${url}/health`, {
      signal: ctrl.signal,
      
      
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
    xhr.timeout = 10 * 60 * 1000;   

    xhr.upload.onprogress = e => {
      if (e.lengthComputable) onProgress?.(e.loaded / e.total, 'upload');
    };
    
    xhr.upload.onload = () => onProgress?.(1, 'analyze');

    xhr.onload = () => {
      const body = xhr.response;
      if (xhr.status >= 200 && xhr.status < 300) return resolve(body);
      
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
