"""Pure geometry — the server-side counterpart of the browser's js/biomechanics.js.

Points ("pts") are a numpy array of shape (33, 3): x, y, visibility, already
converted from MediaPipe's normalized landmark space into ASPECT-CORRECTED
pixel space (x * frame_w, y * frame_h).

Why that conversion matters: MediaPipe normalizes x by frame width and y by
frame height independently, so on a non-square frame the normalized space is
anisotropically stretched and angles measured in it are simply wrong.

Everything downstream is an angle or a distance divided by torso length, so
results are independent of resolution, camera distance and player size.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

X, Y, V = 0, 1, 2


class LM:
    """MediaPipe Pose landmark indices (33-point model)."""
    NOSE = 0
    L_SH, R_SH = 11, 12
    L_EL, R_EL = 13, 14
    L_WR, R_WR = 15, 16
    L_HIP, R_HIP = 23, 24
    L_KN, R_KN = 25, 26
    L_AN, R_AN = 27, 28
    L_HEEL, R_HEEL = 29, 30
    L_FOOT, R_FOOT = 31, 32


# Per-side index bundles so analysis is written once and applied to whichever
# leg turns out to be the kicking / plant leg.
SIDES = {
    "left":  {"sh": LM.L_SH, "el": LM.L_EL, "wr": LM.L_WR, "hip": LM.L_HIP,
              "kn": LM.L_KN, "an": LM.L_AN, "heel": LM.L_HEEL, "foot": LM.L_FOOT},
    "right": {"sh": LM.R_SH, "el": LM.R_EL, "wr": LM.R_WR, "hip": LM.R_HIP,
              "kn": LM.R_KN, "an": LM.R_AN, "heel": LM.R_HEEL, "foot": LM.R_FOOT},
}

CONNECTIONS = [
    (LM.L_SH, LM.R_SH), (LM.L_HIP, LM.R_HIP),
    (LM.L_SH, LM.L_HIP), (LM.R_SH, LM.R_HIP),
    (LM.L_SH, LM.L_EL), (LM.L_EL, LM.L_WR),
    (LM.R_SH, LM.R_EL), (LM.R_EL, LM.R_WR),
    (LM.L_HIP, LM.L_KN), (LM.L_KN, LM.L_AN),
    (LM.L_AN, LM.L_HEEL), (LM.L_HEEL, LM.L_FOOT), (LM.L_AN, LM.L_FOOT),
    (LM.R_HIP, LM.R_KN), (LM.R_KN, LM.R_AN),
    (LM.R_AN, LM.R_HEEL), (LM.R_HEEL, LM.R_FOOT), (LM.R_AN, LM.R_FOOT),
]


def to_pixels(landmarks, w: int, h: int) -> np.ndarray:
    """Convert a MediaPipe NormalizedLandmark list to aspect-corrected pixels."""
    return np.array(
        [[lm.x * w, lm.y * h, getattr(lm, "visibility", 1.0)] for lm in landmarks],
        dtype=np.float64,
    )


def dist(a, b) -> float:
    return float(math.hypot(a[X] - b[X], a[Y] - b[Y]))


def mid(a, b) -> np.ndarray:
    return np.array([(a[X] + b[X]) / 2, (a[Y] + b[Y]) / 2, min(a[V], b[V])])


def angle_deg(a, b, c) -> float:
    """Interior angle at vertex `b`, in degrees, formed by a-b-c. 180 = straight."""
    v1 = (a[X] - b[X], a[Y] - b[Y])
    v2 = (c[X] - b[X], c[Y] - b[Y])
    n1, n2 = math.hypot(*v1), math.hypot(*v2)
    if n1 < 1e-6 or n2 < 1e-6:
        return float("nan")
    cos = max(-1.0, min(1.0, (v1[0] * v2[0] + v1[1] * v2[1]) / (n1 * n2)))
    return math.degrees(math.acos(cos))


def hip_center(p: np.ndarray) -> np.ndarray:
    return mid(p[LM.L_HIP], p[LM.R_HIP])


def shoulder_center(p: np.ndarray) -> np.ndarray:
    return mid(p[LM.L_SH], p[LM.R_SH])


def torso_scale(p: np.ndarray) -> float:
    """Torso length (hip centre -> shoulder centre) in px: the unit for every
    normalized distance. Falls back to hip width * 2.2 if badly foreshortened."""
    t = dist(hip_center(p), shoulder_center(p))
    hip_w = dist(p[LM.L_HIP], p[LM.R_HIP])
    return max(t, hip_w * 2.2, 1e-3)


def torso_lean_deg(p: np.ndarray, direction: int) -> float:
    """Signed torso lean from vertical. Positive = chest leaning FORWARD along
    the kick direction. `direction` is +1 if the player kicks toward increasing x."""
    h, s = hip_center(p), shoulder_center(p)
    forward = (s[X] - h[X]) * direction
    up = h[Y] - s[Y]              # image y grows downward, so this is + when upright
    return math.degrees(math.atan2(forward, up))


def forward_offset(pt, ref, direction: int, scale: float) -> float:
    """Signed horizontal offset of `pt` from `ref` along the kick direction, in torso units."""
    return ((pt[X] - ref[X]) * direction) / scale


def height_above(pt, ref, scale: float) -> float:
    """Height of `pt` above `ref` (positive = higher in the world), in torso units."""
    return (ref[Y] - pt[Y]) / scale


def min_vis(p: np.ndarray, idxs) -> float:
    """Minimum visibility across the given landmarks — the confidence gate."""
    return float(min(p[i][V] for i in idxs)) if len(idxs) else 0.0


# ── View quality ────────────────────────────────────────────────────────────

# Seen side-on, the shoulders and hips are edge-on to the camera, so their
# apparent span collapses to a fraction of torso length. Seen face-on they are
# at full width. Measured across a range of stills: side-on lands near
# 0.25-0.45, front-on near 0.70-0.95.
SIDE_ON_SPAN = 0.42     # at or below this the view is treated as fully side-on
FRONT_ON_SPAN = 0.78    # at or above this it is treated as fully face-on


@dataclass
class ViewQuality:
    """How side-on the camera is. 1.0 = ideal side-on, 0.0 = face-on."""
    score: float
    shoulder_ratio: float
    label: str

    def as_dict(self) -> dict:
        # shoulder_ratio is NaN when the torso was too degenerate to measure
        # (see view_quality below). round() preserves NaN, and NaN is not valid
        # JSON, so it becomes null here rather than killing the whole response.
        ratio = self.shoulder_ratio
        return {"score": round(self.score, 3),
                "shoulderRatio": round(ratio, 3) if math.isfinite(ratio) else None,
                "label": self.label}


def view_quality(p: np.ndarray) -> ViewQuality:
    """Judge how close to side-on this frame is.

    This app's geometry is built for a side-on camera: torso lean, fore/aft
    plant placement and backswing reach are all measured in the image plane, so
    a face-on camera projects them to near zero and quietly reports a good kick
    as a bad one. Detecting that is more useful than silently scoring it.
    """
    scale = torso_scale(p)
    span = dist(p[LM.L_SH], p[LM.R_SH]) / scale if scale > 0 else float("nan")
    if not math.isfinite(span):
        return ViewQuality(0.5, float("nan"), "unknown")

    if span <= SIDE_ON_SPAN:
        score = 1.0
    elif span >= FRONT_ON_SPAN:
        score = 0.0
    else:
        score = (FRONT_ON_SPAN - span) / (FRONT_ON_SPAN - SIDE_ON_SPAN)

    label = "side-on" if score >= 0.66 else ("angled" if score >= 0.33 else "face-on")
    return ViewQuality(score, span, label)


# ── Context ─────────────────────────────────────────────────────────────────

@dataclass
class Context:
    kick_side: str
    plant_side: str
    direction: int
    dir_source: str
    leg_confidence: float
    leg_source: str
    view: ViewQuality
    kick: dict = field(default_factory=dict)
    plant: dict = field(default_factory=dict)

    def as_dict(self) -> dict:
        return {
            "kickSide": self.kick_side,
            "plantSide": self.plant_side,
            "dir": self.direction,
            "dirSource": self.dir_source,
            "legConfidence": round(self.leg_confidence, 3),
            "legSource": self.leg_source,
            "view": self.view.as_dict(),
        }


def derive_context(frames: dict, footedness: str = "auto") -> Context:
    """Decide which leg kicks and which way the player faces.

    Kicking leg, best source first:
      1. `footedness` if the user stated it. A right-footed player shoots with
         the right leg, and them telling us beats any amount of inference —
         this is the single most common way a 2D read goes wrong, because from
         a side-on camera the near and far leg overlap constantly.
      2. Otherwise: across the contact and follow-through frames, the kicking
         ankle is the one lifted highest relative to the hips.

    Direction, best source first:
      1. Where the BALL went between plant and follow-through. It is the thing
         being aimed, rather than a proxy for it.
      2. The kicking ankle's travel over the same interval.
      3. Which way the plant foot's toes point — last resort for a head-on view.
    """
    plant, contact, follow = frames["plant"], frames["contact"], frames["followThrough"]

    def lift(p, side):
        s = SIDES[side]
        return height_above(p[s["an"]], hip_center(p), torso_scale(p))

    left_score = lift(contact["pts"], "left") + lift(follow["pts"], "left")
    right_score = lift(contact["pts"], "right") + lift(follow["pts"], "right")

    if footedness in ("left", "right"):
        kick_side, leg_source = footedness, "stated"
        # Still report how strongly the frames agree, so a mismatch is visible
        # rather than silently overridden.
        leg_confidence = min(1.0, abs(left_score - right_score) / 0.5)
    else:
        kick_side = "left" if left_score > right_score else "right"
        leg_source = "inferred"
        leg_confidence = min(1.0, abs(left_score - right_score) / 0.5)

    plant_side = "right" if kick_side == "left" else "left"

    scale = torso_scale(contact["pts"])
    k_an = SIDES[kick_side]["an"]
    ankle_travel = follow["pts"][k_an][X] - plant["pts"][k_an][X]
    ball_travel = None
    if plant.get("ball") and follow.get("ball"):
        ball_travel = follow["ball"]["x"] - plant["ball"]["x"]

    if ball_travel is not None and abs(ball_travel) / scale > 0.15:
        direction, dir_source = int(math.copysign(1, ball_travel)), "ball"
    elif abs(ankle_travel) / scale > 0.15:
        direction, dir_source = int(math.copysign(1, ankle_travel)), "ankle"
    else:
        ps = SIDES[plant_side]
        toe = contact["pts"][ps["foot"]][X] - contact["pts"][ps["heel"]][X]
        direction = int(math.copysign(1, toe)) if toe else 1
        dir_source = "toe"

    # View quality is judged at contact: the frame carrying the most weight in
    # the score, and the one the camera is usually squarest to.
    return Context(
        kick_side=kick_side,
        plant_side=plant_side,
        direction=direction,
        dir_source=dir_source,
        leg_confidence=leg_confidence,
        leg_source=leg_source,
        view=view_quality(contact["pts"]),
        kick=SIDES[kick_side],
        plant=SIDES[plant_side],
    )


def inferred_kick_side(frames: dict) -> str:
    """What the geometry alone would have picked, ignoring any stated footedness.
    Used to tell the user when their stated foot disagrees with the video."""
    def lift(p, side):
        s = SIDES[side]
        return height_above(p[s["an"]], hip_center(p), torso_scale(p))
    left = lift(frames["contact"]["pts"], "left") + lift(frames["followThrough"]["pts"], "left")
    right = lift(frames["contact"]["pts"], "right") + lift(frames["followThrough"]["pts"], "right")
    return "left" if left > right else "right"
