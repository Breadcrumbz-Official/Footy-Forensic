"""Finds the ball in a clip — the server-side counterpart of js/ballDetection.js.

One frame's raw detection is not trustworthy for this. The COCO "sports ball"
class also fires on round-ish background clutter (in testing, on a stretched
patch of grass at 0.33 confidence), and the frame that matters most — contact —
is exactly where the ball is fastest, blurriest and most likely to be missed.
Either failure would quietly corrupt a metric.

So we read the WHOLE clip and pick the single most temporally coherent track
through it, by Viterbi over each frame's candidates plus a "not visible here"
state. A real ball moves a short smooth distance frame to frame; a false
positive appears once somewhere unrelated and vanishes.

Every gate is expressed relative to the player's own torso length as measured
by the pose model on that same frame, so there are no pixel thresholds here any
more than anywhere else in this app.

Server-side additions over the browser build:
  * EfficientDet-Lite2 (23MB) instead of Lite0-uint8 (4.5MB)
  * a crop-and-upscale rescue pass around the player's feet, which makes a
    small ball far larger relative to the detector's fixed 320x320 input
"""

from __future__ import annotations

import math

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision

import models_cache
from biomechanics import LM, X, Y, V

BALL_CATEGORY = "sports ball"
SCORE_FLOOR = 0.2

# ── Plausibility gates (all scale-free) ─────────────────────────────────────
# A ball's bounding box is near-square from any angle. Long thin boxes are line
# markings, shins or smeared grass. Loose, because a fast ball smears to an oval.
MIN_ROUNDNESS = 0.5
# A size-5 ball is ~22cm across and hip-to-shoulder is ~50cm, so a ball is ~0.45
# torso lengths wide. The band is generous because the ball can sit nearer to or
# further from the camera than the player.
MIN_DIAM_TORSO = 0.18
MAX_DIAM_TORSO = 0.95

# ── Track-selection costs ───────────────────────────────────────────────────
# Cost of declaring "no ball in this frame". Must sit above the emission cost of
# a plausible detection (1 - score, so ~0.3 at score 0.7) or the track would
# rather see nothing than accept a real ball.
MISS_COST = 0.8
STEP_WEIGHT = 0.6
# A struck ball leaves at ~25m/s; at 30fps that is ~0.8m, or ~1.7 torso lengths,
# in one frame. Displacement beyond this is capped rather than rejected, so a
# genuinely fast ball is penalised but still reachable.
MAX_STEP_TORSO = 2.5

_CLIP_GAP_MS = 200


class BallSession:
    """One detector plus its clock. Not thread-safe — give each worker its own."""

    def __init__(self, model_path: str | None = None):
        path = model_path or models_cache.ensure(models_cache.DETECTOR_LITE2)
        options = vision.ObjectDetectorOptions(
            base_options=mp_python.BaseOptions(model_asset_path=path),
            running_mode=vision.RunningMode.VIDEO,
            # Ask the model for nothing else. Cheaper than filtering after the
            # fact, and a stray "person" can never reach us.
            category_allowlist=[BALL_CATEGORY],
            score_threshold=SCORE_FLOOR,
            max_results=8,
        )
        self.detector = vision.ObjectDetector.create_from_options(options)
        self.clock_ms = 0

    def close(self):
        try:
            self.detector.close()
        except Exception:
            pass

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    def _detect(self, bgr: np.ndarray, ts_ms: int):
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        return self.detector.detect_for_video(image, ts_ms)

    def detect_sequence(self, frames: list[dict], scales, all_pts=None) -> list:
        """Find the ball across one ordered clip.

        `frames`  [{"image": BGR ndarray, "time": seconds}] ascending in time
        `scales`  torso length in px per frame (None where pose failed)
        `all_pts` per-frame landmark arrays, used only to aim the rescue crop
        Returns one entry per frame: {"x","y","r","score","interpolated"} or None.
        """
        n = len(frames)
        filled = _fill_scales(scales, n)
        per_frame = []
        prev_t = None
        for i, f in enumerate(frames):
            dt = 33 if prev_t is None else max(1, round((f["time"] - prev_t) * 1000))
            prev_t = f["time"]
            self.clock_ms += dt
            cands = _candidates(self._detect(f["image"], self.clock_ms), filled[i])

            if not cands:
                pts = all_pts[i] if all_pts is not None else None
                cands = self._rescue(f["image"], pts, filled[i])
            per_frame.append(cands)

        self.clock_ms += _CLIP_GAP_MS
        return _fill_gaps(_best_track(per_frame, filled))

    def _rescue(self, bgr, pts, scale):
        """Second chance on a crop around the feet, upscaled.

        The detector resizes whatever it is given to a fixed 320x320 input, so a
        ball that is 40px wide in a 1080p frame arrives about 12px wide — near
        the limit of what it can resolve. Cropping to the region the ball can
        plausibly be in and upscaling that gives the same ball several times the
        pixels, at the cost of one extra inference on frames that found nothing
        anyway.
        """
        if pts is None or scale <= 0:
            return []
        h, w = bgr.shape[:2]
        feet = [pts[i] for i in (LM.L_AN, LM.R_AN, LM.L_FOOT, LM.R_FOOT) if pts[i][V] >= 0.3]
        if not feet:
            return []

        cx = float(np.mean([p[X] for p in feet]))
        cy = float(np.mean([p[Y] for p in feet]))
        # The ball sits within roughly a torso length of the feet at the moments
        # this app scores; pad generously and let the gates do the filtering.
        pad = scale * 1.8
        x0, y0 = max(0, int(cx - pad)), max(0, int(cy - pad))
        x1, y1 = min(w, int(cx + pad)), min(h, int(cy + pad))
        if x1 - x0 < 16 or y1 - y0 < 16:
            return []

        crop = bgr[y0:y1, x0:x1]
        k = min(4.0, 640 / max(crop.shape[0], crop.shape[1]))
        if k > 1.05:
            crop = cv2.resize(crop, None, fx=k, fy=k, interpolation=cv2.INTER_CUBIC)
        else:
            k = 1.0

        self.clock_ms += 1
        cands = _candidates(self._detect(crop, self.clock_ms), scale * k)
        # Map back into full-frame coordinates.
        for c in cands:
            c["x"] = c["x"] / k + x0
            c["y"] = c["y"] / k + y0
            c["r"] = c["r"] / k
            c["rescued"] = True
        return cands


def _fill_scales(scales, n):
    """Torso length is the yardstick for every gate, but pose may have failed on
    some frames. Borrow the median of the frames it did read — the player's size
    does not meaningfully change within ~0.3s."""
    known = [s for s in (scales or []) if s is not None and math.isfinite(s) and s > 0]
    median = sorted(known)[len(known) // 2] if known else 0.0
    out = []
    for i in range(n):
        s = scales[i] if scales is not None and i < len(scales) else None
        out.append(s if (s is not None and math.isfinite(s) and s > 0) else median)
    return out


def _candidates(result, scale):
    """Detections from one frame, reduced to plausible ball candidates."""
    out = []
    for d in getattr(result, "detections", None) or []:
        b = d.bounding_box
        if b is None or b.width <= 0 or b.height <= 0:
            continue
        if min(b.width, b.height) / max(b.width, b.height) < MIN_ROUNDNESS:
            continue

        diam = (b.width + b.height) / 2
        # Only size-gate when we know how big the player is; with no pose on any
        # frame of the clip we would be inventing a reference.
        if scale and scale > 0:
            rel = diam / scale
            if rel < MIN_DIAM_TORSO or rel > MAX_DIAM_TORSO:
                continue

        score = d.categories[0].score if d.categories else 0.0
        out.append({"x": b.origin_x + b.width / 2, "y": b.origin_y + b.height / 2,
                    "r": diam / 2, "score": float(score), "rescued": False})
    return out


def _best_track(per_frame, scales):
    """Pick the most coherent path through the per-frame candidates (Viterbi).

    Each frame's states are its candidates plus a "not visible here" state, so
    the track survives the ball being missed mid-clip without being forced onto
    whatever junk the detector did return. State cost is unconfidence
    (1 - score); moving between two real candidates costs how far the ball would
    have had to travel, in torso lengths. A lone false positive far from
    everything else therefore loses to the miss state.
    """
    n = len(per_frame)
    if n == 0:
        return []
    MISS = -1

    def states_at(i):
        return list(range(len(per_frame[i]))) + [MISS]

    def emission(i, k):
        return MISS_COST if k == MISS else 1 - per_frame[i][k]["score"]

    def transition(i, frm, to):
        # Nothing to compare across a gap — moving in or out of the miss state
        # is free, and charging for it twice would just bias against gaps.
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
    """Fill frames missed *between* two good ones. Over the ~0.3s of a clip the
    ball's path is close enough to straight for this to beat having no reading.
    We deliberately do not extrapolate past the ends of the track — that would
    be a guess, not an interpolation — and everything filled is flagged."""
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
