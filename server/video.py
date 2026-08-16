from __future__ import annotations

import math
import os

import cv2
import numpy as np

MAX_EDGE = int(os.environ.get("SFAI_MAX_EDGE", "1920"))
MAX_CLIP_FRAMES = 60


class VideoError(RuntimeError):
    pass


def probe(path: str) -> dict:
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise VideoError("Could not open this video — the format may be unsupported.")
    try:
        fps = _finite_or(cap.get(cv2.CAP_PROP_FPS), 0.0)
        count = _finite_or(cap.get(cv2.CAP_PROP_FRAME_COUNT), 0.0)
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    finally:
        cap.release()
    duration = (count / fps) if fps > 0 and count > 0 else 0.0
    return {"fps": fps, "frames": int(count), "width": w, "height": h, "duration": duration}


def _finite_or(value, default: float) -> float:
    try:
        v = float(value)
    except (TypeError, ValueError):
        return default
    return v if math.isfinite(v) else default


def _downscale(frame: np.ndarray) -> np.ndarray:
    h, w = frame.shape[:2]
    k = MAX_EDGE / max(w, h)
    if k >= 1.0:
        return frame
    return cv2.resize(frame, (round(w * k), round(h * k)), interpolation=cv2.INTER_AREA)


def count_frames(path: str) -> int:
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise VideoError("Could not open this video — the format may be unsupported.")
    n = 0
    try:
        while cap.grab():
            n += 1
    finally:
        cap.release()
    return n


def extract_clips(path: str, windows: dict[str, tuple[float, float]],
                  fallback_fps: float = 30.0,
                  source_duration: float | None = None) -> dict[str, list[dict]]:
    fps = 0.0
    if source_duration and math.isfinite(source_duration) and source_duration > 0:
        n = count_frames(path)
        if n > 1:
            fps = n / source_duration

    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise VideoError("Could not open this video — the format may be unsupported.")

    if not (fps > 0) or not math.isfinite(fps):
        fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
    if not (fps > 0) or not math.isfinite(fps):
        fps = fallback_fps

    latest = max((w[1] for w in windows.values()), default=0.0)
    out: dict[str, list[dict]] = {name: [] for name in windows}

    try:
        idx = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            t = idx / fps
            idx += 1
            if t > latest + (1.0 / fps):
                break
            for name, (start, end) in windows.items():
                if start - 1e-9 <= t <= end + 1e-9:
                    out[name].append({"image": _downscale(frame), "time": t})
                    break
    finally:
        cap.release()

    for name, frames in out.items():
        if len(frames) > MAX_CLIP_FRAMES:
            step = (len(frames) - 1) / (MAX_CLIP_FRAMES - 1)
            out[name] = [frames[round(i * step)] for i in range(MAX_CLIP_FRAMES)]
    return out


def nearest_index(frames: list[dict], t: float) -> int:
    if not frames:
        return 0
    return min(range(len(frames)), key=lambda i: abs(frames[i]["time"] - t))


def encode_jpeg(image: np.ndarray, quality: int = 82) -> bytes:
    ok, buf = cv2.imencode(".jpg", image, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    if not ok:
        raise VideoError("Could not encode the annotated frame.")
    return buf.tobytes()
