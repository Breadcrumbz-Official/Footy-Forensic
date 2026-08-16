from __future__ import annotations

import os

import numpy as np

import models_cache

COCO_TO_MP = {
    0: 0,
    1: 2,
    2: 5,
    3: 7,
    4: 8,
    5: 11,
    6: 12,
    7: 13,
    8: 14,
    9: 15,
    10: 16,
    11: 23,
    12: 24,
    13: 25,
    14: 26,
    15: 27,
    16: 28,
}

MP_LANDMARKS = 33
IMGSZ = int(os.environ.get("SFAI_POSE_IMGSZ", "640"))


class YoloPoseSession:

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
        results = self.model.predict(bgr, classes=[0], imgsz=IMGSZ, verbose=False)
        if not results:
            return None
        r = results[0]
        kp = getattr(r, "keypoints", None)
        if kp is None or kp.xy is None or len(kp.xy) == 0:
            return None

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
            if x <= 0 and y <= 0:
                v = 0.0
            pts[mp_i] = (x, y, v)

        if pts[11][2] < 0.3 and pts[12][2] < 0.3:
            return None
        if pts[23][2] < 0.3 and pts[24][2] < 0.3:
            return None
        return pts
