"""What one worker process does for one phase: pose, then ball, then pick a frame.

This lives in its own module because ProcessPoolExecutor has to import the
target by name in the child process. Each child builds its own PoseSession and
BallSession lazily on first use and then keeps them for the life of the
process — constructing a MediaPipe graph costs far more than running one, so
rebuilding per request would dominate the response time.
"""

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
    """Re-detect one frame's landmarks with the fallback pose model.

    Called only for frames the vision check flagged as having a skeleton that
    is not on the player. Loaded lazily because most runs never need it, and
    the weights are ~118MB per worker process.
    """
    global _ypose
    import pose_yolo
    if _ypose is None:
        _ypose = pose_yolo.YoloPoseSession()
    try:
        return _ypose.detect(image)
    except Exception:
        return None


def analyse_phase(clip: list[dict], center_time: float) -> dict:
    """Run pose and ball tracking across one clip and return the scored frame.

    Prefer the exact frame the user picked. If that instant was too blurred or
    occluded for a detection, fall back to the nearest frame in the clip that
    did read — and report the shift, rather than silently substituting a
    different moment.
    """
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

    # Ball tracking is strictly additive: a failure here must never cost the
    # user their frame, so it is contained.
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
