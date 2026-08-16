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
from worker import analyse_phase, repose_frame

PHASES = ("plant", "contact", "followThrough")
PHASE_LABEL = {"plant": "Plant + Backswing", "contact": "Contact",
               "followThrough": "Follow-through"}

AUTO_SPAN_S = 0.26
CONTACT_NEAR_TORSO = 0.55
MAX_UPLOAD_MB = int(os.environ.get("SFAI_MAX_UPLOAD_MB", "200"))

WORKERS = int(os.environ.get("SFAI_WORKERS", "3"))

_pool: ProcessPoolExecutor | None = None


@asynccontextmanager
async def lifespan(_app: FastAPI):
    global _pool
    models_cache.prefetch_all()
    _pool = ProcessPoolExecutor(max_workers=WORKERS)
    try:
        yield
    finally:
        _pool.shutdown(wait=False, cancel_futures=True)


app = FastAPI(title="Foot Form", version="2.0", lifespan=lifespan)

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
        try:
            os.unlink(path)
        except OSError:
            pass

    result["timing"]["totalMs"] = round((time.perf_counter() - t_start) * 1000)
    return _json_safe(result)


def _json_safe(obj):
    if isinstance(obj, np.generic):
        return _json_safe(obj.item())
    if isinstance(obj, float):
        return obj if math.isfinite(obj) else None
    if isinstance(obj, dict):
        return {k: _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(v) for v in obj]
    return obj


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

    t2 = time.perf_counter()
    reposed = await _verify_and_repose(per_phase, frames)
    verify_ms = round((time.perf_counter() - t2) * 1000)

    ctx = biomechanics.derive_context(frames, footedness=spec.footedness)
    scored = scoring.score_all(analysis.compute_metrics(frames, ctx), ctx.view.score)

    warnings = _warnings(spec, ctx, frames, per_phase, reposed)

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
            "reposed": bool(reposed.get(p)),
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
        "timing": {"decodeMs": decode_ms, "detectMs": detect_ms,
                   "verifyMs": verify_ms, "aiMs": ai_ms},
    }


async def _verify_and_repose(per_phase: dict, frames: dict) -> dict[str, bool]:
    out = {p: False for p in PHASES}
    if not gemini_feedback.enabled():
        return out

    shots = []
    for p in PHASES:
        drawn = analysis.draw_pose(per_phase[p]["image"], frames[p]["pts"],
                                   None, frames[p]["ball"])
        shots.append(video.encode_jpeg(drawn))

    try:
        verdicts = await gemini_feedback.verify_batch(shots)
    except Exception:
        return out

    bad = [p for p, ok in zip(PHASES, verdicts) if ok is False]
    if not bad:
        return out

    loop = asyncio.get_running_loop()
    try:
        results = await asyncio.gather(*[
            loop.run_in_executor(_pool, repose_frame, per_phase[p]["image"]) for p in bad
        ])
    except Exception:
        return out

    for p, pts in zip(bad, results):
        if pts is not None:
            frames[p]["pts"] = pts
            out[p] = True
    return out


async def _ai_feedback(jpeg_bytes: dict[str, bytes], scored: dict, ctx) -> dict | None:
    if not gemini_feedback.enabled():
        return None
    try:
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


def _warnings(spec: AnalyseSpec, ctx, frames, per_phase, reposed=None) -> list[str]:
    out = []

    for p in PHASES:
        if (reposed or {}).get(p):
            out.append(f"{PHASE_LABEL[p]}: the first skeleton didn't sit on your "
                       "body, so this frame was re-detected with the backup pose "
                       "model. Ankle lock isn't measurable from it and was skipped.")

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
