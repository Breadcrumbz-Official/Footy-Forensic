"""Fallback pose estimator: YOLO11x-pose.

MediaPipe is the primary and usually right. It has one failure mode that its
own confidence numbers do not catch: locking onto the wrong subject — a
bystander, a shadow, a half-body ghost — and reporting high visibility for a
skeleton that is nowhere near the player. Every downstream angle is then
measured off the wrong body while looking perfectly well-formed.

That is what this exists for. When the vision check in gemini_feedback says the
drawn skeleton does not sit on the player, the frame is re-posed with a model
that has completely different failure modes, on the theory that two unrelated
detectors are unlikely to be wrong the same way.

COCO-17 vs MediaPipe-33
-----------------------
YOLO pose emits the 17 COCO keypoints. MediaPipe emits 33, and the whole
analysis pipeline indexes into that 33-slot layout, so results are returned in
MediaPipe's shape with the 16 keypoints COCO does not have left at visibility
0. That is not a fudge — visibility 0 is the same signal a genuinely occluded
landmark carries, and scoring already drops any metric whose landmarks fall
below VIS_GATE.

The consequence worth knowing: COCO has no heel or foot-index keypoints, so
`ankleLock` (which needs knee-ankle-foot) cannot be measured from a re-posed
frame and is dropped from that phase's score. Everything else — all knee and
hip angles, torso lean, balance, plant placement — is fully available, because
none of it reaches past the ankles.
"""

from __future__ import annotations

import os

import numpy as np

import models_cache

# COCO keypoint index -> MediaPipe landmark index. The 16 MediaPipe slots absent
# here (eye corners, mouth, hands, heels, foot indices) stay at visibility 0.
COCO_TO_MP = {
    0: 0,    # nose
    1: 2,    # left eye
    2: 5,    # right eye
    3: 7,    # left ear
    4: 8,    # right ear
    5: 11,   # left shoulder
    6: 12,   # right shoulder
    7: 13,   # left elbow
    8: 14,   # right elbow
    9: 15,   # left wrist
    10: 16,  # right wrist
    11: 23,  # left hip
    12: 24,  # right hip
    13: 25,  # left knee
    14: 26,  # right knee
    15: 27,  # left ankle
    16: 28,  # right ankle
}

MP_LANDMARKS = 33
IMGSZ = int(os.environ.get("SFAI_POSE_IMGSZ", "640"))


class YoloPoseSession:
    """One YOLO pose model. Not thread-safe — give each worker its own."""

    def __init__(self, model_path: str | None = None):
        import torch
        from ultralytics import YOLO
        from ultralytics.utils import SETTINGS

        try:
            SETTINGS.update({"sync": False})
        except Exception:
            pass

        workers = max(1, int(os.environ.get("SFAI_WORKERS", "3")))
        torch.set_num_threads(max(1, (os.cpu_count() or 4) // workers))
        self.model = YOLO(model_path or models_cache.ensure(models_cache.YOLO11X_POSE))
        self.name = os.path.basename(models_cache.YOLO11X_POSE[0])

    def detect(self, bgr: np.ndarray) -> np.ndarray | None:
        """Landmarks for the most prominent person, in MediaPipe's (33, 3)
        [x_px, y_px, visibility] shape. None when no person was found."""
        results = self.model.predict(bgr, classes=[0], imgsz=IMGSZ, verbose=False)
        if not results:
            return None
        r = results[0]
        kp = getattr(r, "keypoints", None)
        if kp is None or kp.xy is None or len(kp.xy) == 0:
            return None

        # Largest detected box wins. The player being scored fills far more of
        # the frame than a bystander on the touchline, and picking by detection
        # confidence instead would happily choose a crisp distant spectator over
        # a motion-blurred player mid-strike.
        idx = 0
        boxes = getattr(r, "boxes", None)
        if boxes is not None and len(boxes) > 1:
            areas = [(float(b[2] - b[0]) * float(b[3] - b[1]))
                     for b in boxes.xyxy]
            idx = int(np.argmax(areas))

        xy = kp.xy[idx].cpu().numpy() if hasattr(kp.xy[idx], "cpu") else np.asarray(kp.xy[idx])
        conf = None
        if getattr(kp, "conf", None) is not None:
            c = kp.conf[idx]
            conf = c.cpu().numpy() if hasattr(c, "cpu") else np.asarray(c)

        pts = np.zeros((MP_LANDMARKS, 3), dtype=np.float64)
        for coco_i, mp_i in COCO_TO_MP.items():
            if coco_i >= len(xy):
                continue
            x, y = float(xy[coco_i][0]), float(xy[coco_i][1])
            v = float(conf[coco_i]) if conf is not None and coco_i < len(conf) else 1.0
            # YOLO reports (0, 0) for a keypoint it could not place at all.
            if x <= 0 and y <= 0:
                v = 0.0
            pts[mp_i] = (x, y, v)

        # A skeleton with no torso is not usable as a replacement: torso_scale
        # is the denominator of every normalised measurement in the app.
        if pts[11][2] < 0.3 and pts[12][2] < 0.3:
            return None
        if pts[23][2] < 0.3 and pts[24][2] < 0.3:
            return None
        return pts
