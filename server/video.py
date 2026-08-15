"""Decoding and clip extraction.

Frames are pulled in ONE sequential pass rather than by seeking. Per-frame
seeking with OpenCV is unreliable across containers and codecs — with a
MediaRecorder WebM in particular, `CAP_PROP_POS_MSEC` seeks land on the wrong
frame or silently no-op — and the browser has already paid that cost once to
let the user pick their moments. A single forward decode keeping only the
frames that fall inside a requested window is both exact and cheap, because the
windows together cover well under a second of a clip that is at most ten.
"""

from __future__ import annotations

import math
import os

import cv2
import numpy as np

# Guard against someone uploading 4K: MediaPipe downsamples internally anyway,
# so beyond this we are paying memory and decode time for nothing. Still far
# above the 720p the browser build captured at.
MAX_EDGE = int(os.environ.get("SFAI_MAX_EDGE", "1920"))
# Safety cap if a user drags a very wide manual range.
MAX_CLIP_FRAMES = 60


class VideoError(RuntimeError):
    pass


def probe(path: str) -> dict:
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise VideoError("Could not open this video — the format may be unsupported.")
    try:
        fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
        count = cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0.0
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    finally:
        cap.release()
    duration = (count / fps) if fps > 0 and count > 0 else 0.0
    return {"fps": fps, "frames": int(count), "width": w, "height": h, "duration": duration}


def _downscale(frame: np.ndarray) -> np.ndarray:
    h, w = frame.shape[:2]
    k = MAX_EDGE / max(w, h)
    if k >= 1.0:
        return frame
    return cv2.resize(frame, (round(w * k), round(h * k)), interpolation=cv2.INTER_AREA)


def extract_clips(path: str, windows: dict[str, tuple[float, float]],
                  fallback_fps: float = 30.0) -> dict[str, list[dict]]:
    """Pull the frames falling inside each named [start, end] window.

    Returns {name: [{"image": BGR ndarray, "time": seconds}, ...]} ordered by
    time. A window that captured nothing comes back as an empty list rather
    than raising, so one bad phase pick does not lose the other two.
    """
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise VideoError("Could not open this video — the format may be unsupported.")

    fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
    if not (fps > 0) or not math.isfinite(fps):
        fps = fallback_fps

    # Clamp to the widest span we actually need, so we can stop decoding early.
    latest = max((w[1] for w in windows.values()), default=0.0)
    out: dict[str, list[dict]] = {name: [] for name in windows}

    try:
        idx = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            # Frame index / fps is stable where CAP_PROP_POS_MSEC is not: some
            # WebM files report 0 for every frame's timestamp.
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
    """Index of the frame closest to `t`; 0 for an empty list."""
    if not frames:
        return 0
    return min(range(len(frames)), key=lambda i: abs(frames[i]["time"] - t))


def encode_jpeg(image: np.ndarray, quality: int = 82) -> bytes:
    ok, buf = cv2.imencode(".jpg", image, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    if not ok:
        raise VideoError("Could not encode the annotated frame.")
    return buf.tobytes()
