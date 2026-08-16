from __future__ import annotations

import os
import tempfile
import urllib.request
from pathlib import Path

CACHE_DIR = Path(os.environ.get("SFAI_MODEL_DIR", Path(__file__).parent / "models"))

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

YOLO11X = ("yolo11x.pt",
           "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11x.pt")

YOLO11X_POSE = ("yolo11x-pose.pt",
                "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11x-pose.pt")


def ensure(asset: tuple[str, str]) -> str:
    name, url = asset
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    dest = CACHE_DIR / name
    if dest.exists() and dest.stat().st_size > 0:
        return str(dest)

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


def ball_asset() -> tuple[str, str]:
    if os.environ.get("SFAI_BALL_BACKEND", "yolo").strip().lower() == "efficientdet":
        return DETECTOR_LITE2
    return YOLO11X


def prefetch_all() -> dict:
    return {
        "pose": ensure(POSE_HEAVY),
        "detector": ensure(ball_asset()),
    }
