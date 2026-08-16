from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

X, Y, V = 0, 1, 2


class LM:
    NOSE = 0
    L_SH, R_SH = 11, 12
    L_EL, R_EL = 13, 14
    L_WR, R_WR = 15, 16
    L_HIP, R_HIP = 23, 24
    L_KN, R_KN = 25, 26
    L_AN, R_AN = 27, 28
    L_HEEL, R_HEEL = 29, 30
    L_FOOT, R_FOOT = 31, 32


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
    return np.array(
        [[lm.x * w, lm.y * h, getattr(lm, "visibility", 1.0)] for lm in landmarks],
        dtype=np.float64,
    )


def dist(a, b) -> float:
    return float(math.hypot(a[X] - b[X], a[Y] - b[Y]))


def mid(a, b) -> np.ndarray:
    return np.array([(a[X] + b[X]) / 2, (a[Y] + b[Y]) / 2, min(a[V], b[V])])


def angle_deg(a, b, c) -> float:
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
    t = dist(hip_center(p), shoulder_center(p))
    hip_w = dist(p[LM.L_HIP], p[LM.R_HIP])
    return max(t, hip_w * 2.2, 1e-3)


def torso_lean_deg(p: np.ndarray, direction: int) -> float:
    h, s = hip_center(p), shoulder_center(p)
    forward = (s[X] - h[X]) * direction
    up = h[Y] - s[Y]
    return math.degrees(math.atan2(forward, up))


def forward_offset(pt, ref, direction: int, scale: float) -> float:
    return ((pt[X] - ref[X]) * direction) / scale


def height_above(pt, ref, scale: float) -> float:
    return (ref[Y] - pt[Y]) / scale


def min_vis(p: np.ndarray, idxs) -> float:
    return float(min(p[i][V] for i in idxs)) if len(idxs) else 0.0


SIDE_ON_SPAN = 0.42
FRONT_ON_SPAN = 0.78


@dataclass
class ViewQuality:
    score: float
    shoulder_ratio: float
    label: str

    def as_dict(self) -> dict:
        ratio = self.shoulder_ratio
        return {"score": round(self.score, 3),
                "shoulderRatio": round(ratio, 3) if math.isfinite(ratio) else None,
                "label": self.label}


def view_quality(p: np.ndarray) -> ViewQuality:
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
    plant, contact, follow = frames["plant"], frames["contact"], frames["followThrough"]

    def lift(p, side):
        s = SIDES[side]
        return height_above(p[s["an"]], hip_center(p), torso_scale(p))

    left_score = lift(contact["pts"], "left") + lift(follow["pts"], "left")
    right_score = lift(contact["pts"], "right") + lift(follow["pts"], "right")

    if footedness in ("left", "right"):
        kick_side, leg_source = footedness, "stated"
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
    def lift(p, side):
        s = SIDES[side]
        return height_above(p[s["an"]], hip_center(p), torso_scale(p))
    left = lift(frames["contact"]["pts"], "left") + lift(frames["followThrough"]["pts"], "left")
    right = lift(frames["contact"]["pts"], "right") + lift(frames["followThrough"]["pts"], "right")
    return "left" if left > right else "right"
