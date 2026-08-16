from __future__ import annotations
import math
import cv2
import numpy as np

from biomechanics import (
    LM, SIDES, CONNECTIONS, X, Y, V,
    angle_deg, dist, forward_offset, height_above, hip_center,
    min_vis, shoulder_center, torso_scale,
)

NAN = float("nan")


def _m(value, p, idxs):
    return {"value": value, "vis": min_vis(p, idxs)}

def _m_ball(value, p, idxs, ball):
    if not ball:
        return {"value": NAN, "vis": 0.0}
    ball_vis = 0.55 if ball.get("interpolated") else 1.0
    return {"value": value, "vis": min(min_vis(p, idxs), ball_vis)}


def _rotation_proxy(p):
    sh_span = dist(p[LM.L_SH], p[LM.R_SH])
    hip_span = dist(p[LM.L_HIP], p[LM.R_HIP])
    value = NAN if hip_span < 1e-3 else sh_span / hip_span
    return _m(value, p, [LM.L_SH, LM.R_SH, LM.L_HIP, LM.R_HIP])

def _ball_pt(ball):
    return np.array([ball["x"], ball["y"], 1.0]) if ball else None

def plant_metrics(p, c, ball):
    s, hc, sc = torso_scale(p), hip_center(p), shoulder_center(p)
    K, P = c.kick, c.plant
    b = _ball_pt(ball)
    return {
        "plantFootPlacement": _m(forward_offset(p[P["an"]], hc, c.direction, s), p,
                                 [P["an"], LM.L_HIP, LM.R_HIP]),
        "plantBallOffset": _m_ball(
            forward_offset(p[P["an"]], b, c.direction, s) if b is not None else NAN,
            p, [P["an"]], ball),
        "plantKneeBend": _m(angle_deg(p[P["hip"]], p[P["kn"]], p[P["an"]]), p,
                            [P["hip"], P["kn"], P["an"]]),
        "backswingKneeFlex": _m(angle_deg(p[K["hip"]], p[K["kn"]], p[K["an"]]), p,
                                [K["hip"], K["kn"], K["an"]]),
        "backswingReach": _m(-forward_offset(p[K["an"]], hc, c.direction, s), p,
                             [K["an"], LM.L_HIP, LM.R_HIP]),
        "balance": _m(abs(p[P["an"]][X] - sc[X]) / s, p, [P["an"], LM.L_SH, LM.R_SH]),
        "torsoLean": _m(_lean(p, c), p, [LM.L_SH, LM.R_SH, LM.L_HIP, LM.R_HIP]),
    }

def contact_metrics(p, c, ball):
    s, hc, sc = torso_scale(p), hip_center(p), shoulder_center(p)
    K, P = c.kick, c.plant
    b = _ball_pt(ball)
    return {
        "torsoLean": _m(_lean(p, c), p, [LM.L_SH, LM.R_SH, LM.L_HIP, LM.R_HIP]),
        "kickLegExtension": _m(angle_deg(p[K["hip"]], p[K["kn"]], p[K["an"]]), p,
                               [K["hip"], K["kn"], K["an"]]),
        "plantBallOffset": _m_ball(
            forward_offset(p[P["an"]], b, c.direction, s) if b is not None else NAN,
            p, [P["an"]], ball),
        "plantFootPlacement": _m(forward_offset(p[P["an"]], hc, c.direction, s), p,
                                 [P["an"], LM.L_HIP, LM.R_HIP]),
        "plantKneeBend": _m(angle_deg(p[P["hip"]], p[P["kn"]], p[P["an"]]), p,
                            [P["hip"], P["kn"], P["an"]]),
        "balance": _m(abs(p[P["an"]][X] - sc[X]) / s, p, [P["an"], LM.L_SH, LM.R_SH]),
        "headOverBall": _m(forward_offset(p[LM.NOSE], p[P["an"]], c.direction, s), p,
                           [LM.NOSE, P["an"]]),
        "ankleLock": _m(angle_deg(p[K["kn"]], p[K["an"]], p[K["foot"]]), p,
                        [K["kn"], K["an"], K["foot"]]),
        "hipRotation": _rotation_proxy(p),
    }

def follow_through_metrics(p, c, ball=None):
    s, hc, sc = torso_scale(p), hip_center(p), shoulder_center(p)
    K, P = c.kick, c.plant
    return {
        "kickLegExtension": _m(angle_deg(p[K["hip"]], p[K["kn"]], p[K["an"]]), p,
                               [K["hip"], K["kn"], K["an"]]),
        "followHeight": _m(height_above(p[K["an"]], hc, s), p,
                           [K["an"], LM.L_HIP, LM.R_HIP]),
        "followReach": _m(forward_offset(p[K["an"]], hc, c.direction, s), p,
                          [K["an"], LM.L_HIP, LM.R_HIP]),
        "balance": _m(abs(p[P["an"]][X] - sc[X]) / s, p, [P["an"], LM.L_SH, LM.R_SH]),
        "torsoLean": _m(_lean(p, c), p, [LM.L_SH, LM.R_SH, LM.L_HIP, LM.R_HIP]),
        "support": _m(abs(p[P["an"]][X] - hc[X]) / s, p, [P["an"], LM.L_HIP, LM.R_HIP]),
        "hipRotation": _rotation_proxy(p),
    }

def _lean(p, c):
    from biomechanics import torso_lean_deg
    return torso_lean_deg(p, c.direction)

def compute_metrics(frames: dict, ctx) -> dict:
    return {
        "plant": plant_metrics(frames["plant"]["pts"], ctx, frames["plant"].get("ball")),
        "contact": contact_metrics(frames["contact"]["pts"], ctx, frames["contact"].get("ball")),
        "followThrough": follow_through_metrics(frames["followThrough"]["pts"], ctx),
    }


def foot_to_ball_distance(pts, ball):
    if ball is None or pts is None:
        return None
    scale = torso_scale(pts)
    b = _ball_pt(ball)
    best = math.inf
    for side in SIDES.values():
        for idx in (side["foot"], side["an"]):
            p = pts[idx]
            if p[V] >= 0.3:
                best = min(best, dist(p, b) / scale)
    return None if math.isinf(best) else best


CYAN = (255, 229, 0)
DIM = (200, 200, 200)
YELLOW = (0, 221, 255)
PINK = (167, 59, 255)

GOOD = (84, 223, 0)
WARN = (0, 184, 255)
BAD = (60, 60, 224)
UNSCORED = (175, 163, 156)

ANGLE_JOINTS = {
    "plantKneeBend": ("plant", "kn"),
    "backswingKneeFlex": ("kick", "kn"),
    "kickLegExtension": ("kick", "kn"),
    "ankleLock": ("kick", "an"),
}


def angle_labels(phase: dict, ctx) -> list[dict]:
    out = []
    for m in phase["metrics"]:
        spec = ANGLE_JOINTS.get(m["id"])
        if spec is None or not m.get("valueText"):
            continue
        side = ctx.kick if spec[0] == "kick" else ctx.plant
        out.append({"idx": side[spec[1]], "text": m["valueText"],
                    "score": m.get("score"), "uncertain": bool(m.get("uncertain"))})
    return out


def highlight_for(phase: str, ctx) -> set:
    K, P = ctx.kick, ctx.plant
    torso = {LM.L_SH, LM.R_SH, LM.L_HIP, LM.R_HIP}
    kick_leg = {K["hip"], K["kn"], K["an"], K["heel"], K["foot"]}
    plant_leg = {P["hip"], P["kn"], P["an"], P["heel"], P["foot"]}
    if phase == "plant":
        return kick_leg | plant_leg
    return kick_leg | torso


def draw_pose(image: np.ndarray, pts, highlight=None, ball=None, angles=None) -> np.ndarray:
    out = image.copy()
    h, w = out.shape[:2]
    unit = max(2, round(min(w, h) / 160))

    if ball:
        _draw_ball(out, ball, unit)
    if pts is None:
        return out

    hot = (lambda i: True) if highlight is None else (lambda i: i in highlight)

    for a, b in CONNECTIONS:
        pa, pb = pts[a], pts[b]
        if pa[V] < 0.3 or pb[V] < 0.3:
            continue
        on = hot(a) and hot(b)
        cv2.line(out, (int(pa[X]), int(pa[Y])), (int(pb[X]), int(pb[Y])),
                 CYAN if on else DIM, int(unit * 1.6) if on else unit, cv2.LINE_AA)

    for i, p in enumerate(pts):
        if 0 < i < 11 or p[V] < 0.3:
            continue
        r = int(unit * 1.5) if hot(i) else unit
        cv2.circle(out, (int(p[X]), int(p[Y])), r, YELLOW if hot(i) else DIM, -1, cv2.LINE_AA)

    nose = pts[LM.NOSE]
    if nose[V] >= 0.3:
        cv2.circle(out, (int(nose[X]), int(nose[Y])), int(unit * 3.5), YELLOW, unit, cv2.LINE_AA)

    if angles:
        _draw_angles(out, pts, angles, unit)
    return out


def _angle_color(a: dict):
    if a.get("uncertain") or a.get("score") is None:
        return UNSCORED
    return GOOD if a["score"] >= 80 else (WARN if a["score"] >= 60 else BAD)


def _draw_angles(img, pts, angles, unit):
    h, w = img.shape[:2]
    scale = max(0.8, unit * 0.30)
    thick = max(2, round(unit * 0.55))
    pad = max(3, round(unit * 0.9))
    placed: list[tuple[int, int, int, int]] = []

    for a in angles:
        p = pts[a["idx"]]
        if p[V] < 0.3:
            continue
        (tw, th), base = cv2.getTextSize(a["text"], cv2.FONT_HERSHEY_SIMPLEX, scale, thick)
        bw, bh = tw + pad * 2, th + base + pad * 2

        off = int(unit * 2.5)
        for dx, dy in ((off, -off - bh), (-off - bw, -off - bh), (off, off), (-off - bw, off)):
            x, y = int(p[X]) + dx, int(p[Y]) + dy
            if not (0 <= x and x + bw <= w and 0 <= y and y + bh <= h):
                continue
            if any(x < px + pw and px < x + bw and y < py + ph and py < y + bh
                   for px, py, pw, ph in placed):
                continue
            break
        else:
            continue

        colour = _angle_color(a)
        cv2.rectangle(img, (x, y), (x + bw, y + bh), (20, 20, 20), -1)
        cv2.rectangle(img, (x, y), (x + bw, y + bh), colour, max(1, thick - 1))
        cv2.putText(img, a["text"], (x + pad, y + pad + th),
                    cv2.FONT_HERSHEY_SIMPLEX, scale, colour, thick, cv2.LINE_AA)
        cv2.line(img, (int(p[X]), int(p[Y])),
                 (x + bw // 2, y + bh if dy < 0 else y),
                 colour, max(1, thick - 1), cv2.LINE_AA)
        placed.append((x, y, bw, bh))


def _draw_ball(img, ball, unit):
    cx, cy = int(ball["x"]), int(ball["y"])
    r = int(max(ball["r"], unit * 2))
    if ball.get("interpolated"):
        for start in range(0, 360, 30):
            cv2.ellipse(img, (cx, cy), (r, r), 0, start, start + 16, PINK,
                        int(unit * 1.4), cv2.LINE_AA)
    else:
        cv2.circle(img, (cx, cy), r, PINK, int(unit * 1.4), cv2.LINE_AA)
    cv2.circle(img, (cx, cy), max(1, int(unit * 0.9)), PINK, -1, cv2.LINE_AA)
