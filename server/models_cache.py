"""Downloads and caches the MediaPipe model files next to the server.

Kept separate so both detectors share one cache directory and one download
path, and so a cold start fetches each asset exactly once even when several
worker processes come up at the same time.
"""

from __future__ import annotations

import os
import tempfile
import urllib.request
from pathlib import Path

CACHE_DIR = Path(os.environ.get("SFAI_MODEL_DIR", Path(__file__).parent / "models"))

# Server-side we deliberately run heavier models than the browser can afford.
# The browser uses pose_landmarker_full and EfficientDet-Lite0-uint8 (4.5MB)
# because they have to be downloaded by a phone on a mobile connection; here
# the assets are local and the machine has cores to spare, so accuracy wins.
POSE_HEAVY = ("pose_landmarker_heavy.task",
              "https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
              "pose_landmarker_heavy/float16/1/pose_landmarker_heavy.task")
POSE_FULL = ("pose_landmarker_full.task",
             "https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
             "pose_landmarker_full/float16/1/pose_landmarker_full.task")
DETECTOR_LITE2 = ("efficientdet_lite2.tflite",
                  "https://storage.googleapis.com/mediapipe-models/object_detector/"
                  "efficientdet_lite2/float32/1/efficientdet_lite2.tflite")
DETECTOR_LITE0 = ("efficientdet_lite0.tflite",
                  "https://storage.googleapis.com/mediapipe-models/object_detector/"
                  "efficientdet_lite0/float32/1/efficientdet_lite0.tflite")


def ensure(asset: tuple[str, str]) -> str:
    """Return a local path to `asset`, downloading it on first use."""
    name, url = asset
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    dest = CACHE_DIR / name
    if dest.exists() and dest.stat().st_size > 0:
        return str(dest)

    # Download to a temp file in the same directory and rename, so a partial
    # download can never be mistaken for a usable model — and so two workers
    # racing on a cold start cannot hand each other a half-written file.
    fd, tmp = tempfile.mkstemp(dir=str(CACHE_DIR), suffix=".part")
    os.close(fd)
    try:
        with urllib.request.urlopen(url, timeout=120) as r, open(tmp, "wb") as f:
            while chunk := r.read(1 << 20):
                f.write(chunk)
        os.replace(tmp, dest)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)
    return str(dest)


def prefetch_all() -> dict:
    """Warm the cache at startup so the first request is not paying for it."""
    return {
        "pose": ensure(POSE_HEAVY),
        "detector": ensure(DETECTOR_LITE2),
    }
