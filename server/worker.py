from __future__ import annotations

import numpy as np

import ball_detection
import pose as pose_mod
import video as video_mod
from biomechanics import to_pixels, torso_scale

_pose: pose_mod.PoseSession | None = None
_ball: ball_detection.BallSession | None = None
_ypose = None


def _sessions():
    global _pose, _ball
    if _pose is None:
        _pose = pose_mod.PoseSession(heavy=True)
    if _ball is None:
        _ball = ball_detection.BallSession()
    return _pose, _ball


def repose_frame(image):
    global _ypose
    import pose_yolo
    if _ypose is None:
        _ypose = pose_yolo.YoloPoseSession()
    try:
        return _ypose.detect(image)
    except Exception:
        return None


def analyse_phase(clip: list[dict], center_time: float) -> dict:
    pose_s, ball_s = _sessions()

    landmark_sets = pose_s.detect_sequence(clip)
    all_pts = [
        to_pixels(lms, c["image"].shape[1], c["image"].shape[0]) if lms is not None else None
        for lms, c in zip(landmark_sets, clip)
    ]
    scales = [torso_scale(p) if p is not None else None for p in all_pts]

    center_idx = video_mod.nearest_index(clip, center_time)
    idx = center_idx
    if all_pts[idx] is None:
        candidates = [i for i, p in enumerate(all_pts) if p is not None]
        if not candidates:
            return {"pts": None, "ball": None, "time": center_time,
                    "clip_len": len(clip), "ball_found": 0, "shifted_ms": 0}
        idx = min(candidates, key=lambda i: abs(i - center_idx))

    track = [None] * len(clip)
    try:
        track = ball_s.detect_sequence(clip, scales, all_pts)
    except Exception:
        pass

    shifted_ms = 0
    if idx != center_idx and len(clip) > 1:
        shifted_ms = round((clip[idx]["time"] - clip[center_idx]["time"]) * 1000)

    return {
        "pts": all_pts[idx],
        "ball": track[idx],
        "image": clip[idx]["image"],
        "time": clip[idx]["time"],
        "clip_len": len(clip),
        "ball_found": sum(1 for b in track if b),
        "shifted_ms": shifted_ms,
    }
