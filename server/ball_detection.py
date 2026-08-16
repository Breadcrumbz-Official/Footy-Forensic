from __future__ import annotations

import math
import os

import cv2
import numpy as np

import models_cache
from biomechanics import LM, X, Y, V

BALL_CLASS_ID = 32
BALL_CATEGORY = "sports ball"

SCORE_FLOOR = float(os.environ.get("SFAI_BALL_SCORE_FLOOR", "0.15"))

IMGSZ = int(os.environ.get("SFAI_BALL_IMGSZ", "640"))

MIN_ROUNDNESS = 0.5
MIN_DIAM_TORSO = 0.18
MAX_DIAM_TORSO = 0.95

MISS_COST = 0.8
STEP_WEIGHT = 0.6
MAX_STEP_TORSO = 2.5


def _thread_budget() -> int:
    workers = max(1, int(os.environ.get("SFAI_WORKERS", "3")))
    return max(1, (os.cpu_count() or 4) // workers)


class _YoloBackend:

    def __init__(self, model_path: str | None = None):
        import torch
        from ultralytics import YOLO
        from ultralytics.utils import SETTINGS

        try:
            SETTINGS.update({"sync": False})
        except Exception:
            pass

        torch.set_num_threads(_thread_budget())
        self.model = YOLO(model_path or models_cache.ensure(models_cache.YOLO11X))

    def raw_boxes(self, images: list[np.ndarray]) -> list[list[dict]]:
        results = self.model.predict(images, classes=[BALL_CLASS_ID],
                                     conf=SCORE_FLOOR, imgsz=IMGSZ, verbose=False)
        out = []
        for r in results:
            frame = []
            for b in r.boxes:
                x1, y1, x2, y2 = (float(v) for v in b.xyxy[0].tolist())
                frame.append({"x": (x1 + x2) / 2, "y": (y1 + y2) / 2,
                              "w": x2 - x1, "h": y2 - y1, "score": float(b.conf[0])})
            out.append(frame)
        return out

    def close(self):
        pass


class _EfficientDetBackend:

    def __init__(self, model_path: str | None = None):
        import mediapipe as mp
        from mediapipe.tasks import python as mp_python
        from mediapipe.tasks.python import vision
        self._mp = mp

        path = model_path or models_cache.ensure(models_cache.DETECTOR_LITE2)
        self.detector = vision.ObjectDetector.create_from_options(
            vision.ObjectDetectorOptions(
                base_options=mp_python.BaseOptions(model_asset_path=path),
                running_mode=vision.RunningMode.IMAGE,
                category_allowlist=[BALL_CATEGORY],
                score_threshold=SCORE_FLOOR,
                max_results=8,
            ))

    def raw_boxes(self, images: list[np.ndarray]) -> list[list[dict]]:
        out = []
        for bgr in images:
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            image = self._mp.Image(image_format=self._mp.ImageFormat.SRGB, data=rgb)
            frame = []
            for d in getattr(self.detector.detect(image), "detections", None) or []:
                b = d.bounding_box
                if b is None or b.width <= 0 or b.height <= 0:
                    continue
                frame.append({"x": b.origin_x + b.width / 2, "y": b.origin_y + b.height / 2,
                              "w": float(b.width), "h": float(b.height),
                              "score": float(d.categories[0].score if d.categories else 0.0)})
            out.append(frame)
        return out

    def close(self):
        try:
            self.detector.close()
        except Exception:
            pass


class BallSession:

    def __init__(self, model_path: str | None = None):
        backend = os.environ.get("SFAI_BALL_BACKEND", "yolo").strip().lower()
        self.backend_name = "efficientdet" if backend == "efficientdet" else "yolo11x"
        self._backend = (_EfficientDetBackend(model_path)
                         if self.backend_name == "efficientdet"
                         else _YoloBackend(model_path))

    def close(self):
        self._backend.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    def detect_sequence(self, frames: list[dict], scales, all_pts=None) -> list:
        n = len(frames)
        filled = _fill_scales(scales, n)

        raw = self._backend.raw_boxes([f["image"] for f in frames])
        per_frame = [_candidates(raw[i], filled[i]) for i in range(n)]

        for i in range(n):
            if not per_frame[i]:
                pts = all_pts[i] if all_pts is not None else None
                per_frame[i] = self._rescue(frames[i]["image"], pts, filled[i])

        return _fill_gaps(_best_track(per_frame, filled))

    def _rescue(self, bgr, pts, scale):
        if pts is None or scale <= 0:
            return []
        h, w = bgr.shape[:2]
        feet = [pts[i] for i in (LM.L_AN, LM.R_AN, LM.L_FOOT, LM.R_FOOT) if pts[i][V] >= 0.3]
        if not feet:
            return []

        cx = float(np.mean([p[X] for p in feet]))
        cy = float(np.mean([p[Y] for p in feet]))
        pad = scale * 1.8
        x0, y0 = max(0, int(cx - pad)), max(0, int(cy - pad))
        x1, y1 = min(w, int(cx + pad)), min(h, int(cy + pad))
        if x1 - x0 < 16 or y1 - y0 < 16:
            return []

        crop = bgr[y0:y1, x0:x1]
        k = min(4.0, IMGSZ / max(crop.shape[0], crop.shape[1]))
        if k > 1.05:
            crop = cv2.resize(crop, None, fx=k, fy=k, interpolation=cv2.INTER_CUBIC)
        else:
            k = 1.0

        cands = _candidates(self._backend.raw_boxes([crop])[0], scale * k)
        for c in cands:
            c["x"] = c["x"] / k + x0
            c["y"] = c["y"] / k + y0
            c["r"] = c["r"] / k
            c["rescued"] = True
        return cands


def _fill_scales(scales, n):
    known = [s for s in (scales or []) if s is not None and math.isfinite(s) and s > 0]
    median = sorted(known)[len(known) // 2] if known else 0.0
    out = []
    for i in range(n):
        s = scales[i] if scales is not None and i < len(scales) else None
        out.append(s if (s is not None and math.isfinite(s) and s > 0) else median)
    return out


def _candidates(boxes, scale):
    out = []
    for b in boxes or []:
        w, h = b["w"], b["h"]
        if w <= 0 or h <= 0:
            continue
        if min(w, h) / max(w, h) < MIN_ROUNDNESS:
            continue

        diam = (w + h) / 2
        if scale and scale > 0:
            rel = diam / scale
            if rel < MIN_DIAM_TORSO or rel > MAX_DIAM_TORSO:
                continue

        out.append({"x": b["x"], "y": b["y"], "r": diam / 2,
                    "score": float(b["score"]), "rescued": False})
    return out


def _best_track(per_frame, scales):
    n = len(per_frame)
    if n == 0:
        return []
    MISS = -1

    def states_at(i):
        return list(range(len(per_frame[i]))) + [MISS]

    def emission(i, k):
        return MISS_COST if k == MISS else 1 - per_frame[i][k]["score"]

    def transition(i, frm, to):
        if frm == MISS or to == MISS:
            return 0.0
        a, b = per_frame[i - 1][frm], per_frame[i][to]
        s = scales[i] or scales[i - 1] or 0
        if not s:
            return 0.0
        step = math.hypot(a["x"] - b["x"], a["y"] - b["y"]) / s
        return STEP_WEIGHT * min(step, MAX_STEP_TORSO)

    states = states_at(0)
    costs = [emission(0, k) for k in states]
    paths = [[k] for k in states]

    for i in range(1, n):
        nxt = states_at(i)
        new_costs, new_paths = [], []
        for to in nxt:
            best, best_j = math.inf, 0
            for j, frm in enumerate(states):
                c = costs[j] + transition(i, frm, to)
                if c < best:
                    best, best_j = c, j
            new_costs.append(best + emission(i, to))
            new_paths.append(paths[best_j] + [to])
        states, costs, paths = nxt, new_costs, new_paths

    winner = costs.index(min(costs))
    return [None if k == MISS else per_frame[i][k] for i, k in enumerate(paths[winner])]


def _fill_gaps(track):
    known = [i for i, b in enumerate(track) if b]
    if not known:
        return [None] * len(track)

    out = []
    for i, b in enumerate(track):
        if b:
            out.append({**b, "interpolated": False})
            continue
        before = max((k for k in known if k < i), default=None)
        after = min((k for k in known if k > i), default=None)
        if before is None or after is None:
            out.append(None)
            continue
        t = (i - before) / (after - before)
        a, c = track[before], track[after]
        out.append({
            "x": a["x"] + (c["x"] - a["x"]) * t,
            "y": a["y"] + (c["y"] - a["y"]) * t,
            "r": a["r"] + (c["r"] - a["r"]) * t,
            "score": min(a["score"], c["score"]),
            "rescued": False,
            "interpolated": True,
        })
    return out
