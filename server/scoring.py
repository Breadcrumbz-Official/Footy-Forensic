"""Thresholds, weights and coaching text — the single source of truth for scoring.

Every rule below is data, not logic: change `ideal`, `tol` or `weight` to retune
the model without touching any other file. Scores are a transparent function of
the measurement; there are no free-floating numbers.

This moved out of the browser when analysis moved to the server, so there is
exactly one copy of it. The client no longer scores anything.
"""

from __future__ import annotations

VIS_GATE = 0.4        # below this the landmark is too occluded to judge
MIN_COVERAGE = 0.35   # below this share of a phase's weight, publish no score

# Metrics flagged `side_view` are measured along the image's horizontal axis —
# fore/aft placement, reach, lean. A face-on camera projects that axis to almost
# nothing, so those numbers collapse toward zero and would report a fine kick as
# a bad one. Below HARD they are dropped outright; between HARD and SOFT they
# are kept but marked.
SIDE_VIEW_HARD = 0.35
SIDE_VIEW_SOFT = 0.66


def band(value: float, ideal, tol: float) -> int:
    """Map a measurement to 0-100. Inside [lo, hi] -> 100. Beyond it the score
    falls linearly, reaching 40 once `tol` outside the band, floored at 20."""
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
                "good": "Your support foot lands roughly level with your hips — a solid base to strike from.",
                "low": {"what": "Your support foot is planted behind your hips.",
                        "why": "Planting behind your body pushes your weight backwards, so you end up reaching for the ball and lifting it.",
                        "tip": "Take a slightly longer final stride so the support foot arrives alongside the ball, not trailing it."},
                "high": {"what": "Your support foot is well ahead of your hips.",
                         "why": "Over-striding stalls your momentum and forces the kicking leg to catch up, which costs power.",
                         "tip": "Shorten the last step a little so you arrive balanced rather than braking."},
            },
            {
                "id": "plantBallOffset", "label": "Plant foot vs ball", "weight": 1.2,
                "ideal": (-0.30, 0.30), "tol": 0.45, "fmt": "tor", "side_view": True,
                "caveat": "Measured from the tracked ball. Only the fore/aft component is judged — how far the plant foot sits to the SIDE of the ball cannot be read from a side-on camera.",
                "good": "Your support foot lands level with the ball — the base the whole strike swings around.",
                "low": {"what": "Your support foot lands behind the ball.",
                        "why": "Planting short leaves you reaching forward for the ball, which lifts your chest and opens the strike face upward.",
                        "tip": "Lengthen the final stride so the support foot arrives level with the ball, not short of it."},
                "high": {"what": "Your support foot lands past the ball.",
                         "why": "Over-running the ball means you strike behind your own base, so the leg has to pull back through and loses speed.",
                         "tip": "Shorten the last step and let the ball stay level with your plant foot as you strike."},
            },
            {
                "id": "plantKneeBend", "label": "Plant-leg loading", "weight": 1.0,
                "ideal": (148, 172), "tol": 22, "fmt": "deg",
                "good": "Your support knee is softly flexed — that absorbs the landing and keeps you stable.",
                "low": {"what": "Your support knee is bent deeply.",
                        "why": "Sinking too low drops your hips and makes it hard to hold the base while the kicking leg accelerates.",
                        "tip": "Land taller. Think “quiet, springy knee” rather than sitting into the plant."},
                "high": {"what": "Your support leg is almost locked straight.",
                         "why": "A locked knee cannot absorb the landing, so the impact goes into your hip and your base wobbles.",
                         "tip": "Allow a small amount of bend as the foot lands."},
            },
            {
                "id": "backswingKneeFlex", "label": "Backswing knee cock", "weight": 1.1,
                "ideal": (55, 110), "tol": 35, "fmt": "deg",
                "good": "Your kicking knee is well cocked — the heel is loaded toward your glutes ready to whip through.",
                "low": {"what": "Your kicking knee is folded very tightly.",
                        "why": "An extreme fold usually means the leg is being pulled up rather than swung back, which shortens the strike path.",
                        "tip": "Let the whole leg swing back from the hip instead of snapping the heel up."},
                "high": {"what": "You are taking the leg back nearly straight.",
                         "why": "Power in a kick comes from unfolding the knee at the last moment. A straight-legged backswing removes that whip.",
                         "tip": "Let the heel travel up toward your backside on the way back, then snap the lower leg through."},
            },
            {
                "id": "backswingReach", "label": "Backswing length", "weight": 1.0,
                "ideal": (0.30, 1.05), "tol": 0.40, "fmt": "tor", "side_view": True,
                "good": "Good backswing length — enough runway to accelerate the leg into the ball.",
                "low": {"what": "Your kicking foot has not travelled far behind your body.",
                        "why": "A short backswing gives the leg very little distance to build speed, so the strike is mostly a push.",
                        "tip": "Let the leg swing further back before you drive it forward — feel the hip open behind you."},
                "high": {"what": "Your backswing is unusually long.",
                         "why": "An over-long swing is hard to time and often pulls your torso backwards as you strike.",
                         "tip": "Trim the backswing slightly and focus on accelerating through the ball rather than winding up."},
            },
            {
                "id": "balance", "label": "Body balance", "weight": 1.0,
                "ideal": (0, 0.35), "tol": 0.35, "fmt": "tor", "side_view": True,
                "good": "Your chest is stacked nicely over the support foot.",
                "low": None,
                "high": {"what": "Your upper body is offset from your support foot.",
                         "why": "When the chest is not over the base, the plant leg fights to stay upright instead of anchoring the strike.",
                         "tip": "Approach the ball a touch slower and let your chest arrive over the plant foot as it lands."},
            },
            {
                "id": "torsoLean", "label": "Torso angle", "weight": 0.9,
                "ideal": (3, 28), "tol": 20, "fmt": "deg", "side_view": True,
                "good": "Slight forward lean into the plant — exactly what you want before striking.",
                "low": {"what": "You are already leaning backwards at the plant.",
                        "why": "Leaning back this early almost guarantees you get under the ball and skew the strike upward.",
                        "tip": "Keep your chest travelling toward the target as the support foot lands."},
                "high": {"what": "You are leaning a long way forward.",
                         "why": "Too much forward lean shortens the leg swing and can drag the strike low and across the ball.",
                         "tip": "Stay tall through the plant; lean from the whole body, not by folding at the waist."},
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
                "good": "Your chest is over the ball at contact — this is what keeps a shot down and driven.",
                "low": {"what": "Your upper body is leaning backwards at contact.",
                        "why": "A backward chest opens the strike face upward, which is the most common reason shots fly over the bar.",
                        "tip": "Try keeping your chest slightly forward, over the ball, as your foot arrives."},
                "high": {"what": "You are folded well forward over the ball.",
                         "why": "Excessive forward lean cramps the hip and cuts the leg swing short, so you lose power and often slice.",
                         "tip": "Stay tall. Lean with your whole torso rather than bending at the waist."},
            },
            {
                "id": "kickLegExtension", "label": "Kicking-leg extension", "weight": 1.2,
                "ideal": (138, 172), "tol": 25, "fmt": "deg",
                "good": "Your kicking leg is extending strongly through contact without locking out.",
                "low": {"what": "Your knee is still noticeably bent at contact.",
                        "why": "Striking with a folded knee means the lower leg has not finished accelerating, so much of the available speed never reaches the ball.",
                        "tip": "Let the shin snap through so the leg is close to straight as the foot meets the ball."},
                "high": {"what": "Your knee is locked out at contact.",
                         "why": "A fully locked leg strikes with a rigid limb and offers no give, which is both weaker and harder on the joint.",
                         "tip": "Keep a small amount of bend at impact and extend fully during the follow-through instead."},
            },
            {
                "id": "plantBallOffset", "label": "Plant foot vs ball", "weight": 1.2,
                "ideal": (-0.30, 0.30), "tol": 0.45, "fmt": "tor", "side_view": True,
                "caveat": "Measured from the tracked ball. Only the fore/aft component is judged — sideways distance from the ball cannot be read from a side-on camera.",
                "good": "Your support foot is level with the ball at the moment you strike it.",
                "low": {"what": "The ball is ahead of your support foot at contact.",
                        "why": "Striking a ball that is still in front of your base means your weight is travelling away from it, which is where scuffed and skied shots come from.",
                        "tip": "Let the ball come level with your plant foot before you swing through it."},
                "high": {"what": "You have run past the ball by the time you strike.",
                         "why": "With the ball behind your plant foot the kicking leg has to reach backwards, cutting the swing short right at impact.",
                         "tip": "Check your run-up length so you arrive beside the ball rather than beyond it."},
            },
            {
                "id": "plantFootPlacement", "label": "Plant foot position", "weight": 1.1,
                "ideal": (-0.10, 0.35), "tol": 0.35, "fmt": "tor", "side_view": True,
                "good": "Your support foot is positioned well relative to your body at contact.",
                "low": {"what": "Your support foot sits behind your hips at contact.",
                        "why": "With the plant behind you, your weight falls away from the ball and the strike loses its base.",
                        "tip": "Aim to place the support foot level with the ball as you strike."},
                "high": {"what": "Your support foot is well ahead of your hips at contact.",
                         "why": "Reaching past the ball means you strike behind your own base and have to pull the leg back through.",
                         "tip": "Get the plant foot down beside the ball rather than past it."},
            },
            {
                "id": "plantKneeBend", "label": "Plant-leg stability", "weight": 0.9,
                "ideal": (150, 176), "tol": 22, "fmt": "deg",
                "good": "Your support leg is holding firm through the strike.",
                "low": {"what": "Your support knee collapses as you strike.",
                        "why": "If the base sinks during contact, energy leaks downward instead of into the ball.",
                        "tip": "Brace the support leg — think of it as a post you swing around."},
                "high": {"what": "Your support leg is locked rigid.",
                         "why": "A locked plant leg cannot absorb rotation, so your hips stall instead of turning through.",
                         "tip": "Keep a hint of bend so the hip is free to rotate."},
            },
            {
                "id": "balance", "label": "Body balance", "weight": 1.0,
                "ideal": (0, 0.30), "tol": 0.30, "fmt": "tor", "side_view": True,
                "good": "Well balanced — your mass is over your base at the moment of contact.",
                "low": None,
                "high": {"what": "Your upper body is off to one side of your support foot.",
                         "why": "Striking off-balance means the leg has to correct mid-swing, which pulls the contact point around.",
                         "tip": "Slow the last two steps and land with your chest over the plant foot."},
            },
            {
                "id": "headOverBall", "label": "Head position", "weight": 0.9,
                "ideal": (-0.15, 0.35), "tol": 0.35, "fmt": "tor", "side_view": True,
                "good": "Your head is over your base — steady eyes and a stable strike.",
                "low": {"what": "Your head is behind your support foot.",
                        "why": "The head leads the torso. Hanging back tilts the whole body away from the ball.",
                        "tip": "Keep your eyes down over the ball a moment longer through contact."},
                "high": {"what": "Your head is thrown well past your support foot.",
                         "why": "Lunging the head forward drags your weight past the ball and shortens the strike.",
                         "tip": "Let the leg travel to the ball; keep the head still."},
            },
            {
                "id": "ankleLock", "label": "Ankle / foot lock", "weight": 0.7,
                "ideal": (128, 170), "tol": 30, "fmt": "deg",
                "caveat": "Foot landmarks are the noisiest part of the pose model — treat this as indicative.",
                "good": "Your ankle looks firm and pointed, giving a solid striking surface.",
                "low": {"what": "Your ankle appears relaxed / toes up at contact.",
                        "why": "A loose ankle absorbs the impact instead of transferring it, and the strike surface moves on you.",
                        "tip": "Point the toes down and lock the ankle before the foot arrives."},
                "high": {"what": "Your foot is extended very far in line with the shin.",
                         "why": "Over-pointing can move contact onto the toes rather than the laces.",
                         "tip": "Firm rather than maximal — aim to strike with the laces, ankle locked."},
            },
            {
                "id": "hipRotation", "label": "Hip/shoulder separation", "weight": 0.5,
                "uncertain": True,
                "caveat": "A 2D proxy from apparent shoulder vs hip width. True 3D hip rotation cannot be measured from one camera.",
                "ideal": (1.05, 1.55), "tol": 0.45, "fmt": "rat",
                "good": "Shoulders appear to lead the hips, which is the pattern you want going into contact.",
                "low": {"what": "Your shoulders and hips appear to be turning as one block.",
                        "why": "Separation between the upper and lower body is what stores elastic energy for the strike.",
                        "tip": "Let the shoulders open toward the target while the hips stay closed a fraction longer."},
                "high": {"what": "Your shoulders appear far more open than your hips.",
                         "why": "This can mean the upper body is spinning out ahead of the strike, which pulls the contact across the ball.",
                         "tip": "Drive the hip through with the leg instead of leading with the shoulder."},
            },
        ],
    },

    "followThrough": {
        "label": "Follow-through",
        "weight": 0.25,
        "metrics": [
            {
                "id": "kickLegExtension", "label": "Leg extension", "weight": 1.2,
                "ideal": (148, 180), "tol": 25, "fmt": "deg",
                "good": "Full extension through the ball — the leg finished the job.",
                "low": {"what": "Your kicking leg stays bent after contact.",
                        "why": "Pulling the leg back early means you decelerated into the ball rather than through it.",
                        "tip": "Let the leg keep straightening after contact instead of stopping at the ball."},
                "high": None,
            },
            {
                "id": "followHeight", "label": "Follow-through height", "weight": 1.0,
                "ideal": (0.30, 1.20), "tol": 0.40, "fmt": "tor",
                "good": "The kicking foot finishes high — a sign you swung through and not at the ball.",
                "low": {"what": "Your follow-through is cut short and low.",
                        "why": "A stopped follow-through almost always means you were slowing down before impact.",
                        "tip": "Swing through the ball and let the momentum carry your foot up naturally."},
                "high": {"what": "Your foot finishes extremely high.",
                         "why": "A very high finish often comes with leaning back, which sends the ball up.",
                         "tip": "Keep the chest over the ball; the finish should be high but driven forward, not lifted."},
            },
            {
                "id": "followReach", "label": "Follow-through direction", "weight": 1.0,
                "ideal": (0.25, 1.00), "tol": 0.40, "fmt": "tor", "side_view": True,
                "good": "The leg continues toward the target — energy went into the ball, not sideways.",
                "low": {"what": "The kicking foot has not travelled far past your body.",
                        "why": "A short forward path means the swing stopped at the ball or wrapped across it.",
                        "tip": "Finish with the foot pointing where you want the ball to go."},
                "high": {"what": "The leg swings a very long way past your body.",
                         "why": "An over-long finish can pull you off balance and delay your recovery.",
                         "tip": "Let the swing decelerate naturally rather than throwing the leg."},
            },
            {
                "id": "balance", "label": "Balance", "weight": 1.0,
                "ideal": (0, 0.45), "tol": 0.40, "fmt": "tor", "side_view": True,
                "good": "You stay balanced through the finish.",
                "low": None,
                "high": {"what": "Your upper body falls away from your support foot after contact.",
                         "why": "Losing balance on the finish means the strike was not supported — and you cannot react to the rebound.",
                         "tip": "Strengthen the plant leg and try to land the follow-through under control."},
            },
            {
                "id": "torsoLean", "label": "Torso control", "weight": 0.9,
                "ideal": (0, 30), "tol": 22, "fmt": "deg", "side_view": True,
                "good": "Your torso stays over the strike as you finish.",
                "low": {"what": "You fall backwards through the follow-through.",
                        "why": "Falling away is the classic ball-over-the-bar pattern — the body opens up as the foot arrives.",
                        "tip": "Keep the chest travelling toward the target after contact."},
                "high": {"what": "Your torso pitches sharply forward on the finish.",
                         "why": "Folding forward after contact usually means you lunged rather than swung.",
                         "tip": "Let the leg finish while the torso stays tall."},
            },
            {
                "id": "support", "label": "Recovery / stability", "weight": 0.9,
                "ideal": (0, 0.50), "tol": 0.40, "fmt": "tor", "side_view": True,
                "good": "Your support foot is still under you — you could play the next action immediately.",
                "low": None,
                "high": {"what": "Your support foot ends up far from under your hips.",
                         "why": "A base scattered away from your body means you need extra steps before you can react.",
                         "tip": "Aim to land the follow-through and recover your base in one step."},
            },
            {
                "id": "hipRotation", "label": "Hip rotation", "weight": 0.5,
                "uncertain": True,
                "caveat": "A 2D proxy from apparent shoulder vs hip width. True 3D hip rotation cannot be measured from one camera.",
                "ideal": (0.95, 1.45), "tol": 0.45, "fmt": "rat",
                "good": "Hips and shoulders appear to have rotated through together on the finish.",
                "low": {"what": "Your hips appear not to have rotated through.",
                        "why": "If the hip stays closed, the kick is powered by the leg alone rather than the whole body.",
                        "tip": "Think about driving the kicking hip toward the target as the leg swings through."},
                "high": {"what": "Your shoulders appear to finish far ahead of your hips.",
                         "why": "The upper body may be spinning off rather than rotating with the strike.",
                         "tip": "Let hip and shoulder rotate together through the finish."},
            },
        ],
    },
}


def _score_metric(rule: dict, meas, view_score: float) -> dict:
    """Score one measurement against its rule.

    A metric is "uncertain" — reported but never counted — when its landmarks
    are occluded, when the geometry is degenerate, when the rule is a flagged
    low-confidence proxy, or when it reads along the image's fore/aft axis and
    the camera is not side-on enough for that axis to mean anything.
    """
    fmt_v, fmt_r = FMT[rule["fmt"]]
    out = {
        "id": rule["id"], "label": rule["label"], "weight": rule["weight"],
        "value": None if meas is None else meas["value"],
        "vis": 0.0 if meas is None else meas["vis"],
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
    """Score every phase and roll up to an overall figure."""
    phases = {}
    for key, rule in RULES.items():
        scored = [_score_metric(r, metrics[key].get(r["id"]), view_score) for r in rule["metrics"]]
        counted = [s for s in scored if not s.get("uncertain") and s.get("score") is not None]

        # Coverage = share of this phase's intended weight we could actually
        # measure. A phase judged on a third of its metrics should not speak as
        # loudly as a fully measured one.
        total_w = sum(r["weight"] for r in rule["metrics"])
        coverage = (sum(i["weight"] for i in counted) / total_w) if total_w > 0 else 0.0

        # Below MIN_COVERAGE we publish no number at all: "100/100" off a single
        # measurable metric reads as a confident verdict and is worse than
        # saying nothing.
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
