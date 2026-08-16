"""Soccer Form AI — analysis server.

The browser captures the video, shows the live skeleton/ball overlay, and lets
the user pick three moments. Everything after that happens here: decoding at
native resolution, pose across each clip with the heavy model, ball tracking,
biomechanics and scoring.

Run:
    python -m uvicorn main:app --host 0.0.0.0 --port 8000
    (from inside the server/ directory)

Then expose it:
    ngrok http 8000

PRIVACY: unlike the original browser-only build, video uploaded here leaves the
user's device. Uploads are written to a temp file, used, and deleted in a
finally block — nothing is retained after the response is sent — but the client
UI must say plainly that the video is being sent to a server, and it does.

If GEMINI_API_KEY is set (see gemini_feedback.py), the three annotated phase
frames are ALSO sent to Google's Gemini API for freeform coaching text. This
is a second, separate destination for that image data and the client UI's
privacy line reflects that when the /health check reports it enabled.
"""

from __future__ import annotations

import asyncio
import base64
import json
import math
import os
import tempfile
import time
from concurrent.futures import ProcessPoolExecutor
from contextlib import asynccontextmanager


def _load_dotenv(path: str = ".env") -> None:
    """Dependency-free .env loader, so a machine run by hand (`python -m
    uvicorn ...`) doesn't need GEMINI_API_KEY exported by the shell every
    time. A real environment variable always wins — this only fills in what
    is missing. Never commit the .env file itself (see .gitignore)."""
    if not os.path.isfile(path):
        return
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, _, v = line.partition("=")
            os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


_load_dotenv()

import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field, ValidationError

import analysis
import biomechanics
import gemini_feedback
import models_cache
import scoring
import video
from worker import analyse_phase

PHASES = ("plant", "contact", "followThrough")
PHASE_LABEL = {"plant": "Plant + Backswing", "contact": "Contact",
               "followThrough": "Follow-through"}

# Automatic clip span when the user did not set bounds by hand: ~8 frames at
# 30fps either side of the picked instant.
AUTO_SPAN_S = 0.26
# At a real contact frame a foot is touching the ball, so the gap is about the
# ball's own radius (~0.2 torso). Well past that and the picked frame is not the
# strike, which matters because every contact metric assumes it is.
CONTACT_NEAR_TORSO = 0.55
MAX_UPLOAD_MB = int(os.environ.get("SFAI_MAX_UPLOAD_MB", "200"))

# One worker per phase. Each holds its own MediaPipe graphs, which are neither
# picklable nor thread-safe, so this is processes rather than threads. Three
# leaves plenty of the 12 cores for the TFLite/XNNPACK thread pools inside each.
WORKERS = int(os.environ.get("SFAI_WORKERS", "3"))

_pool: ProcessPoolExecutor | None = None


@asynccontextmanager
async def lifespan(_app: FastAPI):
    global _pool
    models_cache.prefetch_all()   # never make the first request pay for the download
    _pool = ProcessPoolExecutor(max_workers=WORKERS)
    try:
        yield
    finally:
        _pool.shutdown(wait=False, cancel_futures=True)


app = FastAPI(title="Soccer Form AI", version="2.0", lifespan=lifespan)

# The browser page is served from somewhere else entirely (a local static
# server, GitHub Pages, wherever) and reaches this through an ngrok URL, so the
# origin never matches. Tighten SFAI_ALLOW_ORIGINS in any real deployment.
app.add_middleware(
    CORSMiddleware,
    allow_origins=os.environ.get("SFAI_ALLOW_ORIGINS", "*").split(","),
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ClipBounds(BaseModel):
    start: float | None = None
    end: float | None = None


class PhasePick(BaseModel):
    time: float
    clip: ClipBounds | None = None


class AnalyseSpec(BaseModel):
    phases: dict[str, PhasePick]
    fps: float = 30.0
    footedness: str = Field(default="auto", pattern="^(auto|left|right)$")
    # The duration the BROWSER measured for this video. The picked times are
    # points on that timeline, and it is not always the timeline the container's
    # own metadata implies — see video.extract_clips.
    duration: float | None = None

    def window(self, name: str) -> tuple[float, float]:
        pick = self.phases[name]
        c = pick.clip
        if c and c.start is not None and c.end is not None and abs(c.end - c.start) > 1e-3:
            return (min(c.start, c.end), max(c.start, c.end))
        return (pick.time - AUTO_SPAN_S / 2, pick.time + AUTO_SPAN_S / 2)


@app.get("/health")
def health():
    return {
        "ok": True,
        "workers": WORKERS,
        "poseModel": os.path.basename(models_cache.POSE_HEAVY[0]),
        "ballModel": os.path.basename(models_cache.ball_asset()[0]),
        "maxUploadMb": MAX_UPLOAD_MB,
        "maxEdge": video.MAX_EDGE,
        "aiFeedback": gemini_feedback.enabled(),
    }


@app.post("/analyze")
async def analyze(video_file: UploadFile = File(..., alias="video"),
                  spec: str = Form(...)):
    t_start = time.perf_counter()
    try:
        parsed = AnalyseSpec.model_validate_json(spec)
    except ValidationError as e:
        raise HTTPException(422, f"Bad spec: {e.errors()[:3]}")
    missing = [p for p in PHASES if p not in parsed.phases]
    if missing:
        raise HTTPException(422, f"Missing phase picks: {', '.join(missing)}")

    data = await video_file.read()
    if not data:
        raise HTTPException(400, "Empty upload.")
    if len(data) > MAX_UPLOAD_MB * 1024 * 1024:
        raise HTTPException(413, f"Video larger than {MAX_UPLOAD_MB}MB.")

    suffix = os.path.splitext(video_file.filename or "")[1] or ".webm"
    fd, path = tempfile.mkstemp(suffix=suffix)
    os.close(fd)
    try:
        with open(path, "wb") as f:
            f.write(data)
        result = await _run(path, parsed)
    finally:
        # The upload exists on disk only for as long as it takes to read the
        # frames out of it.
        try:
            os.unlink(path)
        except OSError:
            pass

    result["timing"]["totalMs"] = round((time.perf_counter() - t_start) * 1000)
    return _json_safe(result)


def _json_safe(obj):
    """Make a response payload survive JSON encoding.

    NaN and Infinity are not valid JSON and json.dumps refuses them outright, so
    one unmeasurable number anywhere in a response turns a complete analysis
    into a 500 with nothing in it — the user waited for the upload and the heavy
    model and gets a stack trace. The known sources are fixed where they arise
    (scoring._score_metric, biomechanics.ViewQuality.as_dict, video.probe);
    this is the net under them, and it also flattens numpy scalars, which the
    encoder cannot serialise either.
    """
    if isinstance(obj, np.generic):
        return _json_safe(obj.item())
    if isinstance(obj, float):
        return obj if math.isfinite(obj) else None
    if isinstance(obj, dict):
        return {k: _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(v) for v in obj]
    return obj


# ── The browser client ──────────────────────────────────────────────────────
# Serving the page from here too means one ngrok tunnel covers both it and the
# API, and the page then talks to its own origin — no pasting a URL that changes
# every time the tunnel restarts.
#
# This points at ui/dist, the Vite build output, and nothing else: everything
# under it is generated for the browser, so mounting the whole directory
# publishes no source. Pointing it at the repo root would expose server/ and the
# 54MB models/ over a public URL. Build it with `npm install && npm run build`
# in ui/. The mount is declared after /health and /analyze, so those still win.
_REPO_ROOT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
CLIENT_DIR = os.environ.get("SFAI_CLIENT_DIR", os.path.join(_REPO_ROOT, "ui", "dist"))

if os.path.isfile(os.path.join(CLIENT_DIR, "index.html")):
    app.mount("/", StaticFiles(directory=CLIENT_DIR, html=True), name="client")
else:
    @app.get("/", include_in_schema=False)
    def client_missing():
        raise HTTPException(
            503, f"No built client at {CLIENT_DIR}. Run: cd ui && npm install && npm run build")


async def _run(path: str, spec: AnalyseSpec) -> dict:
    t0 = time.perf_counter()
    try:
        info = video.probe(path)
        clips = video.extract_clips(path, {p: spec.window(p) for p in PHASES},
                                    fallback_fps=spec.fps,
                                    source_duration=spec.duration)
    except video.VideoError as e:
        raise HTTPException(400, str(e))
    decode_ms = round((time.perf_counter() - t0) * 1000)

    empty = [p for p in PHASES if not clips[p]]
    if empty:
        raise HTTPException(
            422,
            f"No frames found for: {', '.join(PHASE_LABEL[p] for p in empty)}. "
            "The picked times may be past the end of the video.")

    # Pose + ball for the three phases run in parallel, one process each.
    t1 = time.perf_counter()
    loop = asyncio.get_running_loop()
    jobs = [loop.run_in_executor(_pool, analyse_phase, clips[p], spec.phases[p].time)
            for p in PHASES]
    try:
        done = await asyncio.gather(*jobs)
    except Exception as e:
        raise HTTPException(500, f"Analysis failed: {type(e).__name__}: {e}")
    detect_ms = round((time.perf_counter() - t1) * 1000)

    per_phase = dict(zip(PHASES, done))
    unreadable = [p for p in PHASES if per_phase[p].get("pts") is None]
    if unreadable:
        raise HTTPException(
            422,
            "No person detected in: "
            + ", ".join(PHASE_LABEL[p] for p in unreadable)
            + ". Pick a moment where the whole body is visible.")

    frames = {p: {"pts": per_phase[p]["pts"],
                  "ball": per_phase[p].get("ball"),
                  "time": per_phase[p]["time"]} for p in PHASES}

    ctx = biomechanics.derive_context(frames, footedness=spec.footedness)
    scored = scoring.score_all(analysis.compute_metrics(frames, ctx), ctx.view.score)

    warnings = _warnings(spec, ctx, frames, per_phase)

    out_frames = {}
    jpeg_bytes = {}
    for p in PHASES:
        img = per_phase[p]["image"]
        drawn = analysis.draw_pose(img, frames[p]["pts"],
                                  analysis.highlight_for(p, ctx), frames[p]["ball"],
                                  angles=analysis.angle_labels(scored["phases"][p], ctx))
        jb = video.encode_jpeg(drawn)
        jpeg_bytes[p] = jb
        out_frames[p] = {
            "time": round(frames[p]["time"], 3),
            "shiftedMs": per_phase[p].get("shifted_ms", 0),
            "ball": _ball_out(frames[p]["ball"]),
            "ballFramesFound": per_phase[p].get("ball_found", 0),
            "clipFrames": per_phase[p].get("clip_len", 0),
            "image": "data:image/jpeg;base64," + base64.b64encode(jb).decode("ascii"),
        }

    ai_ms_t0 = time.perf_counter()
    ai_feedback = await _ai_feedback(jpeg_bytes, scored, ctx)
    ai_ms = round((time.perf_counter() - ai_ms_t0) * 1000)

    return {
        "overall": scored["overall"],
        "phases": scored["phases"],
        "context": ctx.as_dict(),
        "frames": out_frames,
        "warnings": warnings,
        "video": info,
        "aiFeedback": ai_feedback,
        "timing": {"decodeMs": decode_ms, "detectMs": detect_ms, "aiMs": ai_ms},
    }


async def _ai_feedback(jpeg_bytes: dict[str, bytes], scored: dict, ctx) -> dict | None:
    """One short Gemini paragraph per phase, run concurrently across the three
    phases so the wait is roughly one call's latency, not three. Off unless
    GEMINI_API_KEY is set; any failure here — network, quota, a malformed
    response — must never cost the user the rule-based analysis they already
    have, which is why every failure mode inside gemini_feedback.py resolves
    to None rather than raising."""
    if not gemini_feedback.enabled():
        return None
    try:
        # One shared connection pool across the three, which is worth an order
        # of magnitude here — see gemini_feedback.batch().
        texts = await gemini_feedback.batch([
            (PHASE_LABEL[p], jpeg_bytes[p], scored["phases"][p]["metrics"], ctx.kick_side)
            for p in PHASES
        ])
    except Exception:
        return None
    out = dict(zip(PHASES, texts))
    return out if any(out.values()) else None


def _ball_out(ball):
    if not ball:
        return None
    return {"x": round(ball["x"], 1), "y": round(ball["y"], 1), "r": round(ball["r"], 1),
            "score": round(ball["score"], 3),
            "interpolated": bool(ball.get("interpolated")),
            "rescued": bool(ball.get("rescued"))}


def _warnings(spec: AnalyseSpec, ctx, frames, per_phase) -> list[str]:
    out = []

    times = [spec.phases[p].time for p in PHASES]
    if not (times[0] < times[1] < times[2]):
        out.append("Moments aren't in order (plant → contact → follow-through) — "
                   "direction and leg detection may be off.")

    if ctx.view.label == "face-on":
        out.append(
            f"Close to face-on (shoulder span {ctx.view.shoulder_ratio:.2f}× torso) — "
            "fore/aft metrics left unscored. Film side-on for a full read.")
    elif ctx.view.label == "angled":
        out.append("Only partly side-on — fore/aft metrics read lower than reality. "
                   "Square up to the line of the shot.")

    if ctx.leg_source == "stated":
        inferred = biomechanics.inferred_kick_side(frames)
        if inferred != ctx.kick_side and ctx.leg_confidence > 0.5:
            out.append(
                f"You picked {ctx.kick_side}-footed, but the {inferred} leg is the "
                "one swinging here. Using your selection.")
    elif ctx.leg_confidence < 0.35:
        out.append("Legs looked similar, so the kicking-leg call is uncertain — "
                   "set your strong foot explicitly.")

    contact_ball = frames["contact"]["ball"]
    if contact_ball:
        d = analysis.foot_to_ball_distance(frames["contact"]["pts"], contact_ball)
        if d is not None and d > CONTACT_NEAR_TORSO:
            out.append(
                f"Nearest foot is {d:.2f} torso lengths from the ball at contact — "
                "may not be the actual strike frame. Worth re-picking.")

    # "Plant foot vs ball" is scored at both plant and contact; either one
    # missing the ball just drops that one metric, not the phase — the rest of
    # the score comes from pose alone. Named here so that is visible rather
    # than left to be inferred from a quietly absent metric.
    no_ball = [p for p in ("plant", "contact") if not frames[p]["ball"]]
    if len(no_ball) == 2:
        out.append("No ball tracked at plant or contact — those two ball-relative "
                   "metrics were skipped, scored on body position alone.")
    elif no_ball:
        out.append(f"No ball tracked at {PHASE_LABEL[no_ball[0]].lower()} — its "
                   "'plant foot vs ball' metric was skipped.")

    for p in PHASES:
        shifted = per_phase[p].get("shifted_ms", 0)
        if shifted:
            out.append(f"{PHASE_LABEL[p]}: picked instant was unreadable, used "
                       f"nearest clean frame ({shifted:+d}ms).")
    return out
