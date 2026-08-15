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
        # `x or 0.0` is not enough: OpenCV reports NaN for these on some
        # containers (MediaRecorder WebM among them), and NaN is truthy, so it
        # would pass straight through and later fail JSON encoding.
        fps = _finite_or(cap.get(cv2.CAP_PROP_FPS), 0.0)
        count = _finite_or(cap.get(cv2.CAP_PROP_FRAME_COUNT), 0.0)
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    finally:
        cap.release()
    duration = (count / fps) if fps > 0 and count > 0 else 0.0
    return {"fps": fps, "frames": int(count), "width": w, "height": h, "duration": duration}


def _finite_or(value, default: float) -> float:
    """OpenCV property reads come back as NaN or inf on some containers."""
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
    """Number of frames actually decodable from this file.

    `grab()` advances without converting the frame to a numpy array, so this is
    far cheaper than a real decode — worth paying to learn the true length of a
    file whose metadata lies.
    """
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
    """Pull the frames falling inside each named [start, end] window.

    Returns {name: [{"image": BGR ndarray, "time": seconds}, ...]} ordered by
    time. A window that captured nothing comes back as an empty list rather
    than raising, so one bad phase pick does not lose the other two.

    `source_duration` is the length the BROWSER measured, and the requested
    windows are points on that timeline. It matters because this function builds
    its own timeline as `frame_index / fps`, and the two disagree whenever the
    container's declared frame rate is not the rate it was actually captured at
    — which is the normal case for MediaRecorder WebM, where the camera is free
    to deliver frames irregularly while the header still claims a flat 30fps.
    Twenty seconds of real footage then reads as fifteen here, every pick lands
    earlier than the user intended, and picks near the end fall off the end of
    the timeline entirely and return no frames at all.

    Given the browser's duration we can sidestep the declared rate completely:
    count the frames that actually decode, and lay them evenly across that
    duration. That is exact for constant-rate video and much closer than the
    header for variable-rate video.
    """
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
