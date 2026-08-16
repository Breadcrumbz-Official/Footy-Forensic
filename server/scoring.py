from __future__ import annotations

VIS_GATE = 0.4
MIN_COVERAGE = 0.35

SIDE_VIEW_HARD = 0.35
SIDE_VIEW_SOFT = 0.66


def band(value: float, ideal, tol: float) -> int:
    lo, hi = ideal
    if lo <= value <= hi:
        return 100
    d = (lo - value) if value < lo else (value - hi)
    return max(20, round(100 - 60 * (d / tol)))


def _deg(v: float) -> str:
    return f"{v:.0f}°"


def _deg_range(a: float, b: float) -> str:
    return f"{a:.0f} to {b:.0f}°"


def _tor(v: float) -> str:
    return f"{v:.2f} × torso"


def _tor_range(a: float, b: float) -> str:
    return f"{a:.2f} to {b:.2f} × torso"


def _rat(v: float) -> str:
    return f"{v:.2f}"


def _rat_range(a: float, b: float) -> str:
    return f"{a:.2f} to {b:.2f}"


FMT = {
    "deg": (_deg, _deg_range),
    "tor": (_tor, _tor_range),
    "rat": (_rat, _rat_range),
}


RULES = {
    "plant": {
        "label": "Plant + Backswing",
        "weight": 0.35,
        "metrics": [
            {
                "id": "plantFootPlacement", "label": "Plant foot placement", "weight": 1.2,
                "ideal": (-0.05, 0.40), "tol": 0.40, "fmt": "tor", "side_view": True,
                "good": "Support foot level with your hips.",
                "low": {"what": "Support foot lands behind your hips.",
                        "tip": "Take a longer final stride to arrive level."},
                "high": {"what": "Support foot well ahead of your hips.",
                         "tip": "Shorten the last step a little."},
            },
            {
                "id": "plantBallOffset", "label": "Plant foot vs ball", "weight": 1.2,
                "ideal": (-0.30, 0.30), "tol": 0.45, "fmt": "tor", "side_view": True,
                "caveat": "Fore/aft only — a side-on camera can't read sideways offset.",
                "good": "Support foot level with the ball.",
                "low": {"what": "Support foot lands behind the ball.",
                        "tip": "Lengthen your stride to arrive level with it."},
                "high": {"what": "Support foot lands past the ball.",
                         "tip": "Shorten the last step."},
            },
            {
                "id": "plantKneeBend", "label": "Plant-leg loading", "weight": 1.0,
                "ideal": (150, 170), "tol": 22, "fmt": "deg",
                "good": "Support knee softly flexed — stable landing.",
                "low": {"what": "Support knee bent deep.",
                        "tip": "Land a touch taller."},
                "high": {"what": "Support leg nearly locked straight.",
                         "tip": "Allow a little bend on landing."},
            },
            {
                "id": "backswingKneeFlex", "label": "Backswing knee cock", "weight": 1.1,
                "ideal": (70, 108), "tol": 35, "fmt": "deg",
                "good": "Kicking knee well cocked, heel toward glutes.",
                "low": {"what": "Knee folded very tight.",
                        "tip": "Swing the whole leg back from the hip."},
                "high": {"what": "Leg taken back nearly straight.",
                         "tip": "Let the heel rise, then snap through."},
            },
            {
                "id": "backswingReach", "label": "Backswing length", "weight": 1.0,
                "ideal": (0.30, 1.05), "tol": 0.40, "fmt": "tor", "side_view": True,
                "good": "Good backswing — plenty of runway to accelerate.",
                "low": {"what": "Kicking foot hasn't travelled far back.",
                        "tip": "Let the leg swing further behind you."},
                "high": {"what": "Backswing is unusually long.",
                         "tip": "Trim it — accelerate through the ball instead."},
            },
            {
                "id": "balance", "label": "Body balance", "weight": 1.0,
                "ideal": (0, 0.35), "tol": 0.35, "fmt": "tor", "side_view": True,
                "good": "Chest stacked nicely over the support foot.",
                "low": None,
                "high": {"what": "Upper body offset from support foot.",
                         "tip": "Approach slower, arrive stacked over it."},
            },
            {
                "id": "torsoLean", "label": "Torso angle", "weight": 0.9,
                "ideal": (3, 28), "tol": 20, "fmt": "deg", "side_view": True,
                "good": "Slight forward lean into the plant — good.",
                "low": {"what": "Leaning backwards already at the plant.",
                        "tip": "Keep your chest travelling toward the target."},
                "high": {"what": "Leaning a long way forward.",
                         "tip": "Stay tall; lean from the whole body."},
            },
        ],
    },

    "contact": {
        "label": "Contact",
        "weight": 0.40,
        "metrics": [
            {
                "id": "torsoLean", "label": "Torso position", "weight": 1.3,
                "ideal": (3, 25), "tol": 18, "fmt": "deg", "side_view": True,
                "good": "Chest over the ball — keeps the shot down.",
                "low": {"what": "Leaning backwards at contact.",
                        "tip": "Keep your chest slightly forward, over the ball."},
                "high": {"what": "Folded well forward over the ball.",
                         "tip": "Stay tall through contact."},
            },
            {
                "id": "kickLegExtension", "label": "Kicking-leg extension", "weight": 1.2,
                "ideal": (130, 155), "tol": 25, "fmt": "deg",
                "good": "Leg extending strongly through the ball.",
                "low": {"what": "Knee still noticeably bent at contact.",
                        "tip": "Let the shin snap through at impact."},
                "high": {"what": "Knee locked out at contact.",
                         "tip": "Keep a hint of bend, extend fully after."},
            },
            {
                "id": "plantBallOffset", "label": "Plant foot vs ball", "weight": 1.2,
                "ideal": (-0.30, 0.30), "tol": 0.45, "fmt": "tor", "side_view": True,
                "caveat": "Fore/aft only — a side-on camera can't read sideways offset.",
                "good": "Support foot level with the ball at strike.",
                "low": {"what": "Ball still ahead of your support foot.",
                        "tip": "Let the ball come level before you swing."},
                "high": {"what": "You've run past the ball.",
                         "tip": "Check your run-up length."},
            },
            {
                "id": "plantFootPlacement", "label": "Plant foot position", "weight": 1.1,
                "ideal": (-0.10, 0.35), "tol": 0.35, "fmt": "tor", "side_view": True,
                "good": "Support foot well placed at contact.",
                "low": {"what": "Support foot sits behind your hips.",
                        "tip": "Aim to plant level with the ball."},
                "high": {"what": "Support foot well ahead of your hips.",
                         "tip": "Get it down beside the ball, not past it."},
            },
            {
                "id": "plantKneeBend", "label": "Plant-leg stability", "weight": 0.9,
                "ideal": (115, 140), "tol": 24, "fmt": "deg",
                "good": "Support knee loaded well — powering the strike.",
                "low": {"what": "Support knee collapses too far.",
                        "tip": "Brace it — think 'post you swing around.'"},
                "high": {"what": "Support leg too straight to load power.",
                         "tip": "Sit into the plant a touch more."},
            },
            {
                "id": "balance", "label": "Body balance", "weight": 1.0,
                "ideal": (0, 0.30), "tol": 0.30, "fmt": "tor", "side_view": True,
                "good": "Mass over your base at the moment of contact.",
                "low": None,
                "high": {"what": "Upper body off to one side.",
                         "tip": "Slow the last two steps, land chest over base."},
            },
            {
                "id": "headOverBall", "label": "Head position", "weight": 0.9,
                "ideal": (-0.15, 0.35), "tol": 0.35, "fmt": "tor", "side_view": True,
                "good": "Head over your base — steady and stable.",
                "low": {"what": "Head behind your support foot.",
                        "tip": "Keep your eyes down over the ball a beat longer."},
                "high": {"what": "Head thrown well past your support foot.",
                         "tip": "Let the leg travel; keep the head still."},
            },
            {
                "id": "ankleLock", "label": "Ankle / foot lock", "weight": 0.7,
                "ideal": (112, 135), "tol": 28, "fmt": "deg",
                "caveat": "Foot landmarks are the noisiest part of the model — indicative only.",
                "good": "Ankle firm and pointed — solid strike surface.",
                "low": {"what": "Ankle relaxed, toes up at contact.",
                        "tip": "Point the toes down before the foot arrives."},
                "high": {"what": "Foot over-extended past the shin line.",
                         "tip": "Firm, not maximal — strike with the laces."},
            },
            {
                "id": "hipRotation", "label": "Hip/shoulder separation", "weight": 0.5,
                "uncertain": True,
                "caveat": "2D proxy from shoulder/hip width, not true rotation.",
                "ideal": (1.05, 1.55), "tol": 0.45, "fmt": "rat",
                "good": "Shoulders leading the hips — good separation.",
                "low": {"what": "Shoulders and hips turning as one block.",
                        "tip": "Let shoulders open while hips stay closed a beat."},
                "high": {"what": "Shoulders far more open than hips.",
                         "tip": "Drive the hip through with the leg."},
            },
        ],
    },

    "followThrough": {
        "label": "Follow-through",
        "weight": 0.25,
        "metrics": [
            {
                "id": "kickLegExtension", "label": "Leg extension", "weight": 1.2,
                "ideal": (152, 175), "tol": 24, "fmt": "deg",
                "good": "Full extension through the ball.",
                "low": {"what": "Kicking leg stays bent after contact.",
                        "tip": "Let the leg keep straightening past the ball."},
                "high": None,
            },
            {
                "id": "followHeight", "label": "Follow-through height", "weight": 1.0,
                "ideal": (0.30, 1.20), "tol": 0.40, "fmt": "tor",
                "good": "Foot finishes high — swung through, not at it.",
                "low": {"what": "Follow-through cut short and low.",
                        "tip": "Swing through; let momentum carry the foot up."},
                "high": {"what": "Foot finishes extremely high.",
                         "tip": "Keep the chest over the ball as you finish."},
            },
            {
                "id": "followReach", "label": "Follow-through direction", "weight": 1.0,
                "ideal": (0.25, 1.00), "tol": 0.40, "fmt": "tor", "side_view": True,
                "good": "Leg continues toward the target.",
                "low": {"what": "Foot hasn't travelled far past your body.",
                        "tip": "Finish with the foot pointing at the target."},
                "high": {"what": "Leg swings a very long way past your body.",
                         "tip": "Let the swing decelerate naturally."},
            },
            {
                "id": "balance", "label": "Balance", "weight": 1.0,
                "ideal": (0, 0.45), "tol": 0.40, "fmt": "tor", "side_view": True,
                "good": "Balanced through the finish.",
                "low": None,
                "high": {"what": "Upper body falls away from support foot.",
                         "tip": "Strengthen the plant leg; land under control."},
            },
            {
                "id": "torsoLean", "label": "Torso control", "weight": 0.9,
                "ideal": (0, 30), "tol": 22, "fmt": "deg", "side_view": True,
                "good": "Torso stays over the strike as you finish.",
                "low": {"what": "Falling backwards through the follow-through.",
                        "tip": "Keep the chest travelling toward the target."},
                "high": {"what": "Torso pitches sharply forward on the finish.",
                         "tip": "Let the leg finish while the torso stays tall."},
            },
            {
                "id": "support", "label": "Recovery / stability", "weight": 0.9,
                "ideal": (0, 0.50), "tol": 0.40, "fmt": "tor", "side_view": True,
                "good": "Support foot still under you — ready to react.",
                "low": None,
                "high": {"what": "Support foot ends up far from under your hips.",
                         "tip": "Land and recover your base in one step."},
            },
            {
                "id": "hipRotation", "label": "Hip rotation", "weight": 0.5,
                "uncertain": True,
                "caveat": "2D proxy from shoulder/hip width, not true rotation.",
                "ideal": (0.95, 1.45), "tol": 0.45, "fmt": "rat",
                "good": "Hips and shoulders rotated through together.",
                "low": {"what": "Hips appear not to have rotated through.",
                        "tip": "Drive the kicking hip toward the target."},
                "high": {"what": "Shoulders appear to finish well ahead of hips.",
                         "tip": "Let hip and shoulder rotate together."},
            },
        ],
    },
}


def _score_metric(rule: dict, meas, view_score: float) -> dict:
    fmt_v, fmt_r = FMT[rule["fmt"]]
    raw = None if meas is None else meas["value"]
    vis = 0.0 if meas is None else meas["vis"]
    out = {
        "id": rule["id"], "label": rule["label"], "weight": rule["weight"],
        "value": raw if (raw is not None and _finite(raw)) else None,
        "vis": vis if _finite(vis) else 0.0,
        "caveat": rule.get("caveat"),
        "ideal": list(rule["ideal"]),
        "idealText": fmt_r(*rule["ideal"]),
        "valueText": None,
    }

    if meas is None or meas["value"] is None or not _finite(meas["value"]):
        return {**out, "uncertain": True,
                "reason": "Could not measure this from the selected frame."}

    out["valueText"] = fmt_v(meas["value"])

    if meas["vis"] < VIS_GATE:
        return {**out, "uncertain": True,
                "reason": "The body parts this depends on are hidden or out of frame."}

    if rule.get("side_view") and view_score < SIDE_VIEW_HARD:
        return {**out, "uncertain": True,
                "reason": ("This is measured along the camera's fore/aft axis, and this "
                           "shot is too close to face-on for that to mean anything. "
                           "Re-shoot side-on to score it.")}

    score = band(meas["value"], rule["ideal"], rule["tol"])
    if meas["value"] < rule["ideal"][0]:
        side = "low"
    elif meas["value"] > rule["ideal"][1]:
        side = "high"
    else:
        side = "good"
    msg = {"what": rule["good"]} if side == "good" else (rule.get(side) or {"what": rule["good"]})

    caveat = out["caveat"]
    if rule.get("side_view") and view_score < SIDE_VIEW_SOFT:
        extra = ("The camera is only partly side-on, so this fore/aft reading is "
                 "compressed and reads lower than reality.")
        caveat = f"{caveat} {extra}" if caveat else extra

    return {**out, "caveat": caveat, "score": score,
            "uncertain": bool(rule.get("uncertain")),
            "reason": "Low-confidence proxy — reported but not scored." if rule.get("uncertain") else None,
            "feedback": msg}


def _finite(v) -> bool:
    return v == v and v not in (float("inf"), float("-inf"))


def _weighted_mean(items):
    w = sum(i["weight"] for i in items)
    return round(sum(i["score"] * i["weight"] for i in items) / w) if w > 0 else None


def score_all(metrics: dict, view_score: float = 1.0) -> dict:
    phases = {}
    for key, rule in RULES.items():
        scored = [_score_metric(r, metrics[key].get(r["id"]), view_score) for r in rule["metrics"]]
        counted = [s for s in scored if not s.get("uncertain") and s.get("score") is not None]

        total_w = sum(r["weight"] for r in rule["metrics"])
        coverage = (sum(i["weight"] for i in counted) / total_w) if total_w > 0 else 0.0

        insufficient = coverage < MIN_COVERAGE
        phases[key] = {
            "key": key, "label": rule["label"], "weight": rule["weight"],
            "metrics": scored,
            "score": None if insufficient else _weighted_mean(counted),
            "counted": len(counted),
            "skipped": len(scored) - len(counted),
            "coverage": round(coverage, 3),
            "insufficient": insufficient,
            "partial": (not insufficient) and coverage < 0.6,
        }

    usable = [p for p in phases.values() if p["score"] is not None]
    w_sum = sum(p["weight"] * p["coverage"] for p in usable)
    overall = (round(sum(p["score"] * p["weight"] * p["coverage"] for p in usable) / w_sum)
               if w_sum > 0 else None)
    return {"phases": phases, "overall": overall}


def grade(s) -> str:
    if s is None:
        return "⚪"
    return "\U0001F7E2" if s >= 85 else ("\U0001F7E1" if s >= 70 else "\U0001F534")
