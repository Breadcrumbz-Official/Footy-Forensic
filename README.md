# Soccer Form AI

Record or upload a video of a soccer kick, pick three key moments, and get
biomechanical analysis and coaching feedback.

> **Where your video goes.** The browser runs no model of its own — nothing is
> inferred on your device. The camera preview stays local until you act, and the
> **video file itself is uploaded** to the analysis server when you press
> *Analyze technique*. The server writes it to a temp file, reads the frames it
> needs, and deletes it in a `finally` block before responding; nothing is
> retained. This is a change from earlier versions of this app, which did
> everything in the browser and sent nothing.

## Architecture

```
BROWSER                                    SERVER (FastAPI)
capture / upload video
scrub, pick 3 moments  ── video + times ─▶  decode at native resolution
                                            pose (heavy model) per clip
                                            ball tracking per clip
                                            biomechanics + scoring
render results         ◀── JSON + JPEGs ──  annotate frames
```

The browser measures and scores **nothing**. All of that lives in `server/`, so
there is exactly one copy of the rules rather than two that drift apart.

## Run it

**1. Start the server** (on whatever machine does the analysis):

```bash
cd soccer-form-ai/server && pip install -r requirements.txt
```

```bash
cd soccer-form-ai/server && python -m uvicorn main:app --host 0.0.0.0 --port 8000
```

First start downloads ~54MB of models into `server/models/` and caches them.

**2. Expose it** — the browser page and the server are usually not on the same
machine (phone in the garden, laptop indoors):

```bash
ngrok http 8000
```

**3. Serve the page** and paste the ngrok https URL into the Server URL box:

```bash
npx serve soccer-form-ai
```

Camera capture requires `https://` or `localhost`.

> The tunnel is public while it is open — anyone with the URL can post video to
> your machine. `SFAI_ALLOW_ORIGINS` narrows CORS, but for anything beyond a
> demo put real auth in front of `/analyze`.

## Files

| File | Role |
| --- | --- |
| `index.html` | Markup |
| `style.css` | Minimal layout (intentionally unstyled) |
| `app.js` | UI wiring, capture, scrubbing, results rendering |
| `js/api.js` | Server connection + upload |
| `js/video.js` | Recording, loading, precise seeking, frame capture |
| `server/main.py` | FastAPI: `/health`, `/analyze` |
| `server/worker.py` | One process's work for one phase |
| `server/video.py` | Decode + clip extraction |
| `server/pose.py` | PoseLandmarker (heavy) |
| `server/ball_detection.py` | ObjectDetector + coherent ball track |
| `server/biomechanics.py` | Geometry, kick-side, view quality |
| `server/analysis.py` | Measurements + frame annotation |
| `server/scoring.py` | Thresholds, weights, coaching text |

## Which foot

`Auto-detect` infers the kicking leg from which ankle lifts higher across the
contact and follow-through frames. From a side-on camera the near and far leg
overlap constantly, and getting this wrong mirrors every result — so you can
state **left** or **right** instead, and that always wins. If your selection
disagrees with what the frames show, the server says so in the warnings and
still uses your selection.

## Film side-on

Every fore/aft measurement — plant placement, plant-vs-ball, backswing reach,
follow-through direction, torso lean, balance — is read across the image plane.
A face-on camera projects that axis to nearly nothing, so those numbers collapse
toward zero and a fine kick reads as a bad one.

Rather than silently scoring that, the server measures how side-on the camera
actually is, from apparent shoulder span over torso length (edge-on shoulders
collapse to ~0.25–0.45 × torso; face-on is ~0.70–0.95):

| View | Shoulder span | What happens |
| --- | --- | --- |
| side-on | ≤ 0.42 | everything scored normally |
| angled | 0.42 – 0.78 | scored, but fore/aft metrics carry a "compressed, reads low" caveat |
| face-on | ≥ 0.78 | fore/aft metrics dropped as unscoreable, with a warning |

Angles that live in the sagittal plane anyway — knee flex, ankle lock — are
unaffected.

## Motion blur

The moment with the most motion blur — the foot at contact — is exactly the one
carrying the most weight in the score, and `visibility` alone does not catch it.

So each pick is read from a short **clip**, not one still: the server cuts the
frames either side of your pick out of the original upload and runs them through
MediaPipe's **VIDEO** mode in order, so the model has temporal context rather
than judging one instant alone. If the exact frame you picked has no detectable
person, it falls back to the nearest clean frame in the clip and says so. A
frame that still fails gets one retry at 2× scale, which recovers players who
are small in frame.

By default the clip is ±0.13s around your pick; you can set the start and end
yourself per phase.

## Ball tracking

MediaPipe `ObjectDetector` filtered to the COCO `sports ball` class,
EfficientDet-Lite2, on the server.

A single frame's detection is not good enough to build a metric on. In testing
the model put a confident `sports ball` box on a stretched patch of grass, and
missed the ball entirely in a crowded scene. So the server reads the whole clip
and picks the most temporally coherent track through it, by Viterbi over each
frame's candidates plus a "not visible here" state:

- **Shape gate** — a ball's box is near-square; long thin boxes are line
  markings or smeared grass.
- **Size gate** — a ball is ~0.45 torso lengths across, checked against the
  player's own torso length *on that same frame*. No pixel thresholds.
- **Motion cost** — moving between candidates costs how far the ball would have
  travelled, in torso lengths. A lone false positive loses to the miss state.
- **Gap filling** — frames missed *inside* a good track are interpolated and
  flagged; past either end they stay unknown, because that is extrapolation.
- **Rescue crop** — a frame that found nothing is retried on a crop around the
  feet, upscaled. The detector resizes its input to 320×320, so a 40px ball in
  1080p arrives ~12px wide; cropping gives it several times the pixels.

What it buys:

| | Without ball | With ball |
| --- | --- | --- |
| Plant foot placement | vs. the player's hips | vs. **the ball** (`plantBallOffset`) |
| Kick direction | kicking ankle's travel | **the ball's own travel** |
| Contact frame | trusted as picked | flagged if no foot is near the ball |

The ball is drawn as a pink ring on every returned frame, dashed when
interpolated — deliberately, because it is the one part of the pipeline you can
verify at a glance.

If the ball is never found, nothing breaks: ball-relative metrics report as
unmeasured and drop out of the score exactly like an occluded landmark, and
direction falls back to ankle travel.

## Using all the machine

- **Heavy pose model** server-side (`pose_landmarker_heavy`), vs `full` in the
  browser, and **EfficientDet-Lite2** vs Lite0.
- **Native resolution** frames (capped at 1920px long edge) rather than the
  720p the browser captures at.
- **Three worker processes**, one per phase, running pose and ball in parallel.
  Processes rather than threads because MediaPipe graphs are neither picklable
  nor thread-safe; three leaves the rest of the cores for the XNNPACK pools
  inside each. Sessions are built once per process and reused — on a 12-core
  machine a 3s clip goes ~7.5s cold, ~3.4s warm.

MediaPipe's Python GPU delegate is Linux-only, so on Windows this is CPU
throughout. Tunables: `SFAI_WORKERS`, `SFAI_MAX_EDGE`, `SFAI_MAX_UPLOAD_MB`,
`SFAI_ALLOW_ORIGINS`, `SFAI_MODEL_DIR`.

## Tuning the model

Everything is data in `server/scoring.py`. Each metric declares an `ideal`
range, a tolerance `tol`, a `weight`, and its feedback text:

```python
{
  "id": "torsoLean", "label": "Torso position", "weight": 1.3,
  "ideal": (3, 25), "tol": 18, "fmt": "deg", "side_view": True,
  "good": "…", "low": {...}, "high": {...}
}
```

Scoring is `band(value, ideal, tol)`: 100 inside the ideal range, falling
linearly to 40 at `tol` outside it, floored at 20. Phase score = weighted mean
of its metrics. Overall = Plant 35% · Contact 40% · Follow-through 25%.

**Coverage guard.** A metric is dropped when its landmarks are occluded
(`visibility < VIS_GATE`), when it is a flagged low-confidence proxy, or when it
needs a side-on view the camera did not have. Each phase reports the share of
its intended weight that survived. Below `MIN_COVERAGE` (35%) the phase
publishes **no score at all** — a "100/100" derived from one measurable metric
reads as a confident verdict and would be worse than saying nothing. Between 35%
and 60% the score is shown but marked provisional, and every phase is weighted
into the overall in proportion to its coverage.

## Normalization

Landmarks are converted out of MediaPipe's normalized space into
aspect-corrected pixels (`x * frameWidth`, `y * frameHeight`) before any
geometry runs — MediaPipe normalizes x and y by different divisors, so angles
measured in raw normalized space are wrong on non-square frames. Every distance
is then divided by **torso length** (hip centre to shoulder centre), so results
are independent of resolution, camera distance and player size. No pixel
thresholds are hard-coded anywhere.

## Camera

"Flip camera" switches between back (`environment`, the default) and front
(`user`). The front preview is mirrored for natural selfie framing, but that is
cosmetic CSS only (`.camWrap.mirrored`) — the recorded frames are never
mirrored, so kick-side detection is unaffected by which camera you used. Flip is
disabled during recording, since the in-progress capture is bound to the current
camera's tracks. If a device has only one camera, a failed flip reconnects to
the one that was working.

## What it can and cannot tell you

**Reasonably detectable:** joint angles, torso lean from vertical, limb
positions relative to the hips, whether body mass is stacked over the support
foot, where the ball is, and the plant foot's fore/aft position relative to it.

**Not measured / unreliable:** the exact contact patch on the boot, ball speed
or spin, where the shot ended up, sideways plant-foot distance from the ball,
true 3D hip rotation from one camera, and anything depending on depth toward or
away from the camera. Metrics resting on those — plus any metric whose landmarks
are occluded, or which needs a side-on view the camera did not have — are
labelled **low confidence** and excluded from the score rather than presented as
precise.

Best results come from a **side-on camera** with the whole body in frame.
