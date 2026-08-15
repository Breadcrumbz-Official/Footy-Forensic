# Soccer Form AI

Upload or record a video of a soccer kick, pick three key moments, and get on-device
biomechanical analysis and coaching feedback. **No backend. No upload. No database.**
Your video never leaves your device.

## How it works

```
Upload / Record  →  Scrub frame-by-frame  →  Pick 3 moments  →  MediaPipe Pose  →  Geometry  →  Score + Feedback
```

Only **three short clips** (~0.26s, roughly 8 frames each) around the three moments you
pick are ever passed to the pose model — never the whole video. That keeps it fast and
cheap on phones, while still giving MediaPipe enough temporal context to smooth out a
single blurry instant (see "Motion blur" below).

## Stack

- Vanilla HTML/CSS/JS, ES modules, no framework, no build step
- [`@mediapipe/tasks-vision@0.10.35`](https://www.npmjs.com/package/@mediapipe/tasks-vision)
  `PoseLandmarker` (Tasks Vision web API) running in **VIDEO** mode via `detectForVideo()`,
  `pose_landmarker_full` model, GPU delegate with automatic CPU fallback
- `getUserMedia` + `MediaRecorder` for recording, `<canvas>` for frame extraction,
  `requestVideoFrameCallback` for accurate seeks

## Run locally

ES modules require HTTP — opening `index.html` from the filesystem will not work.

```bash
npx serve soccer-form-ai
```

Then open the printed URL. Camera capture also requires `https://` or `localhost`.

## Deploy

It is a static site. Push the `soccer-form-ai/` folder to GitHub Pages, Vercel, Netlify
or Replit static hosting as-is — there is nothing to build and no server to run.

## Files

| File | Role |
| --- | --- |
| `index.html` | Markup for the three steps |
| `style.css` | Minimal layout (intentionally unstyled) |
| `app.js` | UI wiring and orchestration |
| `js/video.js` | Recording, loading, precise seeking, frame capture |
| `js/mediapipe.js` | PoseLandmarker init + detection |
| `js/biomechanics.js` | Geometry primitives, landmark indices, kick-side inference |
| `js/poseAnalysis.js` | Phase measurements + skeleton drawing |
| `js/scoring.js` | Thresholds, weights, coaching text |

## Tuning the model

Everything is data in `js/scoring.js`. Each metric declares an `ideal` range, a
tolerance `tol`, a `weight`, and its feedback text:

```js
{
  id: 'torsoLean', label: 'Torso position', weight: 1.3,
  ideal: [3, 25], tol: 18, fmt: deg,
  good: '…', low: { what, why, tip }, high: { what, why, tip }
}
```

Scoring is `band(value, ideal, tol)`: 100 inside the ideal range, falling linearly to 40
at `tol` outside it, floored at 20. Phase score = weighted mean of its metrics.
Overall = Plant 35% · Contact 40% · Follow-through 25%.

**Coverage guard.** A metric is dropped when its landmarks are occluded (`visibility <
VIS_GATE`) or when it is a flagged low-confidence proxy. Each phase then reports the
share of its intended weight that survived. Below `MIN_COVERAGE` (35%) the phase publishes
**no score at all** and is excluded from the overall — a "100/100" derived from one
measurable metric reads as a confident verdict and would be worse than saying nothing.
Between 35% and 60% the score is shown but marked provisional, and every phase is
weighted into the overall in proportion to its coverage.

## Normalization

Landmarks are converted out of MediaPipe's normalized space into aspect-corrected pixels
(`x * frameWidth`, `y * frameHeight`) before any geometry runs — MediaPipe normalizes x
and y by different divisors, so angles measured in raw normalized space are wrong on
non-square frames. Every distance is then divided by **torso length** (hip centre to
shoulder centre), so results are independent of resolution, camera distance and player
size. No pixel thresholds are hard-coded anywhere.

The kicking leg is inferred as the ankle lifted highest relative to the hips across the
contact and follow-through frames; kick direction from the horizontal travel of that
ankle between the backswing and follow-through.

## Motion blur

The moment with the most motion blur in a kick — the foot at contact — is exactly the
one carrying the most weight in the score. A single still frame there is often blurred
enough that pose landmarks land in the wrong place while still reporting high confidence,
so `visibility` alone can't be trusted to catch it.

Instead of reading one still, each pick captures a short clip (`captureWindow` in
`js/video.js`) of real frames either side of it and runs the whole clip through
MediaPipe's **VIDEO** running mode (`detectForVideo` in `js/mediapipe.js`) in order, so
the model has temporal context rather than judging one instant in isolation. If the exact
frame you picked has no detectable person — peak blur can do that — it falls back to the
nearest frame in the clip that does, and says so in the UI rather than silently
substituting a different moment.

This reduces landmark jitter; it does not eliminate blur. For the sharpest source video:
shoot at 60fps or higher if your phone supports it, film from ~3–4m away, and prefer more
light over less (faster shutter speed).

## Camera

"Flip camera" switches between back (`environment`, the default — best for filming
someone else) and front (`user`) facing modes. The front camera preview is mirrored for
natural selfie framing, but that's cosmetic CSS only (`.camWrap.mirrored`) — the actual
video frames used for recording and pose analysis are never mirrored, so kick-side
detection isn't affected by which camera you used. Flip is disabled during active
recording, since the in-progress capture is bound to the current camera's tracks and
switching would cut the recording off; if a device only has one camera, a failed flip
reconnects to the one that was working instead of leaving the preview dead.

## Live preview overlay

Two checkboxes above the camera preview — **Show skeleton** and **Show joint angles** —
turn on a live pose overlay while the camera is on and while recording. This is the one
place the app runs pose detection continuously instead of on 3 picked frames, so it's
off by default. It targets ~30 detections/sec (`LIVE_INTERVAL_MS` in `app.js`) but that's
a ceiling, not a guarantee — a busy guard skips a tick outright if the previous detection
hasn't finished, so a slower device settles at whatever rate it can actually sustain
instead of piling up overlapping detections.

The angle labels are not a separate calculation — they read the same
`angleDeg(a, vertex, c)` call, on the same three landmark coordinates, that
`js/poseAnalysis.js` uses for `backswingKneeFlex` / `kickLegExtension` / `ankleLock`. What
you see live at the knee is exactly what would score that joint if you picked that instant
as a phase.

## What it can and cannot tell you

**Reasonably detectable:** joint angles, torso lean from vertical, limb positions
relative to the hips, whether body mass is stacked over the support foot.

**Not measured / unreliable:** exact ball-contact point, foot-to-ball distance (the ball
is not tracked), ball speed or direction, true 3D hip rotation from one camera, and
anything depending on depth toward or away from the camera. Metrics resting on those —
plus any metric whose landmarks are occluded — are labelled **low confidence** in the UI
and excluded from the score rather than being presented as precise.

Best results come from a **side-on camera** with the whole body in frame.
