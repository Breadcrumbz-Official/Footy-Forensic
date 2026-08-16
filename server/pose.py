from __future__ import annotations

import os

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision

import models_cache

_CLIP_GAP_MS = 200


class PoseSession:

    def __init__(self, model_path: str | None = None, heavy: bool = True):
        asset = models_cache.POSE_HEAVY if heavy else models_cache.POSE_FULL
        path = model_path or models_cache.ensure(asset)
        options = vision.PoseLandmarkerOptions(
            base_options=mp_python.BaseOptions(model_asset_path=path),
            running_mode=vision.RunningMode.VIDEO,
            num_poses=1,
            min_pose_detection_confidence=0.5,
            min_pose_presence_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        self.landmarker = vision.PoseLandmarker.create_from_options(options)
        self.clock_ms = 0
        self.model = os.path.basename(path)

    def close(self):
        try:
            self.landmarker.close()
        except Exception:
            pass

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    def _detect(self, bgr: np.ndarray, ts_ms: int):
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        return self.landmarker.detect_for_video(image, ts_ms)

    def detect_sequence(self, frames: list[dict], rescue: bool = True) -> list:
        out = []
        prev_t = None
        for f in frames:
            dt = 33 if prev_t is None else max(1, round((f["time"] - prev_t) * 1000))
            prev_t = f["time"]
            self.clock_ms += dt
            res = self._detect(f["image"], self.clock_ms)
            landmarks = res.pose_landmarks[0] if res.pose_landmarks else None

            if landmarks is None and rescue:
                landmarks = self._rescue(f["image"])
            out.append(landmarks)

        self.clock_ms += _CLIP_GAP_MS
        return out

    def _rescue(self, bgr: np.ndarray):
        h, w = bgr.shape[:2]
        big = cv2.resize(bgr, (w * 2, h * 2), interpolation=cv2.INTER_CUBIC)
        self.clock_ms += 1
        res = self._detect(big, self.clock_ms)
        return res.pose_landmarks[0] if res.pose_landmarks else None
