"""PoseLandmarker wrapper — the server-side counterpart of js/mediapipe.js.

Runs in VIDEO mode over short ordered clips rather than single stills. The
moment with the most motion blur in a kick — the foot at contact — is exactly
the one carrying the most weight in the score, and a single blurred frame can
put landmarks in the wrong place while still reporting high visibility. VIDEO
mode gives the model temporal context across the clip instead of asking it to
judge one instant alone.

Server-side additions over the browser build:
  * the `heavy` model instead of `full`
  * frames at the video's native resolution, not downscaled to 720p
  * a rescue pass that retries a failed frame on an upscaled crop
"""

from __future__ import annotations

import os

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision

import models_cache

# detect_for_video needs strictly increasing timestamps for the life of an
# instance. Phases are separate clips that are not chronological relative to
# each other (a user may re-pick an earlier phase after a later one), so we run
# our own virtual clock rather than using real video time. Deltas *within* a
# clip still track real elapsed time, because the filter's smoothing depends on
# realistic inter-frame spacing.
_CLIP_GAP_MS = 200


class PoseSession:
    """One landmarker plus its clock. Not thread-safe — give each worker its own."""

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
        """Run pose across one ordered clip.

        `frames` is [{"image": ndarray BGR, "time": seconds}], ascending in time.
        Returns one entry per frame: a landmark list, or None where no person
        was found even after the rescue pass.
        """
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
        """Second chance for a frame the model missed.

        A person small in frame is the common cause — the model downsamples its
        input, and a distant player can fall below the size it can resolve.
        Re-running at 2x on the same frame costs one extra inference and
        recovers a useful share of them. Coordinates come back normalized, so
        they need no rescaling to map onto the original frame.
        """
        h, w = bgr.shape[:2]
        big = cv2.resize(bgr, (w * 2, h * 2), interpolation=cv2.INTER_CUBIC)
        self.clock_ms += 1
        res = self._detect(big, self.clock_ms)
        return res.pose_landmarks[0] if res.pose_landmarks else None
