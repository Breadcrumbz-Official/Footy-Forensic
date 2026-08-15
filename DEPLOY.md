# Soccer Form AI Server — Deployment Guide

## What You Need

Copy these **11 Python files** to a new machine:

```
server/
├── main.py                 (FastAPI app)
├── worker.py               (per-phase analysis)
├── video.py                (decode + clip extraction)
├── pose.py                 (PoseLandmarker wrapper)
├── ball_detection.py       (ObjectDetector + Viterbi)
├── biomechanics.py         (geometry + view quality)
├── analysis.py             (measurements + frame annotation)
├── scoring.py              (thresholds + coaching text)
├── models_cache.py         (model download/cache)
├── check_env.py            (preflight verification)
└── requirements.txt        (Python dependencies)
```

**Note on models/**: Don't copy the 54MB `models/` directory. The server downloads on first run if missing. If you want to pre-cache, copy `server/models/` over.

## Setup (5 min)

### 1. Create a Python 3.10+ virtualenv

```bash
python -m venv sfai_env
sfai_env\Scripts\activate          # Windows
source sfai_env/bin/activate       # macOS/Linux
```

### 2. Install dependencies

```bash
cd server
pip install -r requirements.txt
```

### 3. Run the preflight check

```bash
python check_env.py
```

Should show:
```
[ok]   Python version
[ok]   fastapi                 0.110.0
[ok]   uvicorn[standard]       0.27.0
[ok]   python-multipart        0.0.9
[ok]   mediapipe               1.0.1
[ok]   opencv-python           4.9.0
[ok]   numpy                   1.26.0
[ok]   pose graph      built 1.2s, 1 frame 142ms
[ok]   ball graph      built 0.8s, 1 frame 89ms
  12 logical cores, SFAI_WORKERS=3
 RESULT: ready. Start with:
   python -m uvicorn main:app --host 0.0.0.0 --port 8000
```

If **Intel Mac**: you'll see a warning about mediapipe 1.x. Use `pip install 'mediapipe==0.10.14'` instead.

If **headless Linux**: use `opencv-python-headless` instead of `opencv-python`.

## Run It

```bash
python -m uvicorn main:app --host 0.0.0.0 --port 8000
```

First run: ~7–8 seconds (downloads + builds models).
Subsequent runs: ~3–4 seconds.

### Expose it to the browser

```bash
ngrok http 8000
```

Copy the `https://...` URL from ngrok and paste it into the **Server URL** box on the browser page.

## Environment Variables (optional)

```bash
SFAI_WORKERS=3              # processes per-phase (default 3, try 1 on slow machines)
SFAI_MAX_EDGE=1920          # max frame width/height in pixels (default 1920)
SFAI_MAX_UPLOAD_MB=200      # max video file size (default 200)
SFAI_ALLOW_ORIGINS=*        # CORS origins (default *; restrict for production)
SFAI_MODEL_DIR=./models     # cache directory (default ./models/)
```

Example:
```bash
set SFAI_WORKERS=1 SFAI_MAX_EDGE=1280
python -m uvicorn main:app --host 0.0.0.0 --port 8000
```

## Capacity

- **Cores**: Server uses `SFAI_WORKERS` processes (default 3) to parallelize phases. On a 4-core machine, try `SFAI_WORKERS=1`.
- **VRAM**: MediaPipe graphs are CPU; GPU delegate available only on Linux. Typical memory use: ~500MB per worker.
- **Network**: Models (~54MB) download on first run if not pre-cached. Video uploads stream; no disk retention after analysis.

## Troubleshooting

**"ModuleNotFoundError: No module named 'mediapipe'"**
→ `pip install -r requirements.txt` not run, or wrong virtualenv activated.

**"libGL.so.1 not found" (Linux)**
→ `pip uninstall opencv-python && pip install opencv-python-headless`

**"No module named 'fastapi'" after reboot**
→ Virtualenv not activated. Run `source sfai_env/bin/activate` (Linux/Mac) or `sfai_env\Scripts\activate` (Windows).

**Server starts but `/health` returns 500 after 10s**
→ Model download timing out. Check internet, or `curl -v http://localhost:8000/health` to see the error.

**Browser won't connect to ngrok URL**
→ ngrok tunnel closed. Restart: `ngrok http 8000`. Paste new URL into Server URL box.

**"You selected left-footed, but in these frames the right leg is the one swinging"**
→ This is correct! Footedness detection inferred wrong. The server is using your override anyway, as intended.

## Files — What They Do

| File | Responsibility |
| --- | --- |
| `main.py` | FastAPI routes (`/health`, `/analyze`), CORS, multipart upload, response JSON assembly |
| `worker.py` | Work scheduler — runs one phase's pose + ball in ProcessPoolExecutor |
| `video.py` | Video decode (any codec), clip extraction by frame index, JPEG encoding |
| `pose.py` | PoseLandmarker wrapper, VIDEO mode, rescues small players at 2× scale |
| `ball_detection.py` | ObjectDetector, Viterbi track selection, size/shape gates, gap filling, rescue crop |
| `biomechanics.py` | Pixel↔normalized conversion, torso normalization, angles, view-quality detection |
| `analysis.py` | Phase metrics (angles, distances, lean), frame annotation (skeleton + ball rings) |
| `scoring.py` | **Authoritative rules.** All thresholds, weights, coaching text. Defines every metric. |
| `models_cache.py` | Downloads models on first use, caches to disk, no re-downloads. |
| `check_env.py` | Preflight: verifies Python, dependencies, model download, graph build, capacity. |
| `requirements.txt` | pip install target. Pinned versions for reproducibility. |
