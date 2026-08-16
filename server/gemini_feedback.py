"""Optional AI coaching layer.

The rule-based scorer in scoring.py is the source of truth for every number
and score in the app — nothing here changes that. This module is purely
additive: it hands the same annotated frame the user sees, plus the exact
measurements already taken from it, to Gemini and asks for one short
freeform paragraph of coaching per phase. It exists because a fixed rule set
can tell a player their knee angle is off; it can't look at the picture and
say what it actually looks like they're doing.

Fully optional. With no GEMINI_API_KEY set, `enabled()` is False and
`phase_feedback()` is never called. If the call fails for any reason —
network, quota, a bad response — it returns None and the analysis response
is unaffected; a player never loses their score because an LLM call timed
out.

Set the key via a real environment variable, or drop a `.env` file next to
main.py with a line `GEMINI_API_KEY=...` (see main.py's tiny loader). Never
commit that file.
"""

from __future__ import annotations

import asyncio
import base64
import os

import httpx

# gemini-2.5-flash still appears in the models listing but returns 404 "no
# longer available to new users" for keys issued recently, and phase_feedback
# swallows that into None — the feature simply produced nothing, silently.
# Pinned to an explicit version rather than an alias like gemini-flash-latest,
# which floats to whatever is newest and took over 45s to answer when tried.
# Override with SFAI_GEMINI_MODEL.
DEFAULT_MODEL = "gemini-3.7-flash"


def _api_key() -> str:
    return os.environ.get("GEMINI_API_KEY", "").strip()


def _model() -> str:
    return os.environ.get("SFAI_GEMINI_MODEL", DEFAULT_MODEL)


def _timeout_s() -> float:
    # Measured latency for one image + paragraph call swings between about 3
    # and 20 seconds, the slow end being a cold connection. At the old 20s
    # ceiling the first of the three concurrent phase calls lost that race and
    # came back None, so a report would arrive with one phase silently missing
    # its paragraph. The three run concurrently, so a longer ceiling costs
    # nothing on the normal path — it only decides how long a stuck call waits
    # before being given up on.
    return float(os.environ.get("SFAI_GEMINI_TIMEOUT_S", "45"))


def enabled() -> bool:
    return bool(_api_key())


_SYSTEM = (
    "You are a soccer shooting-technique coach reviewing one still frame from "
    "an amateur player's kick, captured at a specific phase of the swing. "
    "You are given the exact measurements a computer-vision pipeline already "
    "took from this frame — joint angles, distances in torso lengths, and how "
    "each scored against an elite reference — plus the image itself. Use the "
    "measurements as ground truth; do not invent or restate numbers that were "
    "not given to you, and do not guess angles from the picture yourself. "
    "Look at the image for things the numbers can't capture — balance, "
    "posture, where the eyes are, overall shape — and combine that with the "
    "measurements into ONE short paragraph, three to five sentences, no "
    "headers, no bullet points, no markdown. Talk directly to the player as "
    "'you'. Be specific and concrete, not generic. If the frame looks clean, "
    "say so plainly rather than manufacturing a criticism."
)


def _metrics_block(metrics: list[dict]) -> str:
    lines = []
    for m in metrics:
        if m.get("uncertain") or m.get("valueText") is None:
            continue
        score = m.get("score")
        lines.append(f"- {m['label']}: {m['valueText']} "
                     f"(ideal {m['idealText']}{f', scored {score}/100' if score is not None else ''})")
    return "\n".join(lines) if lines else "(no measurements available for this frame)"


async def batch(items: list[tuple[str, bytes, list[dict], str]]) -> list[str | None]:
    """Run several phase calls over ONE connection pool.

    Each call used to open its own AsyncClient, so firing the three phases
    concurrently meant three separate TLS handshakes racing each other. Measured
    against this API that was pathological — three calls took 46s and two of
    them timed out, where the same three over a shared client finish in about
    1.4s each. Same requests, same concurrency; the only difference is the
    connection pool.
    """
    if not _api_key():
        return [None] * len(items)
    async with httpx.AsyncClient(timeout=_timeout_s()) as client:
        return list(await asyncio.gather(
            *[phase_feedback(*item, client=client) for item in items]))


async def phase_feedback(phase_label: str, image_bytes: bytes, metrics: list[dict],
                         kick_side: str, client: httpx.AsyncClient | None = None) -> str | None:
    """One short coaching paragraph for one phase, or None on any failure.

    Pass `client` to reuse an existing connection pool — see batch() for why
    that matters. Without one a private client is opened for this call alone.
    """
    key = _api_key()
    if not key:
        return None

    prompt = (
        f"Phase of the kick: {phase_label}.\n"
        f"Kicking foot: {kick_side}.\n"
        f"Measurements from this exact frame:\n{_metrics_block(metrics)}\n\n"
        "Write the paragraph now."
    )
    body = {
        "system_instruction": {"parts": [{"text": _SYSTEM}]},
        "contents": [{
            "role": "user",
            "parts": [
                {"text": prompt},
                {"inline_data": {"mime_type": "image/jpeg",
                                 "data": base64.b64encode(image_bytes).decode("ascii")}},
            ],
        }],
        # Gemini 3.x are thinking models, and thinking tokens are charged
        # against maxOutputTokens. Left alone, a 350 budget was spent almost
        # entirely on thoughts (336 of 350) and the reply came back truncated
        # after ten tokens with finishReason MAX_TOKENS — which phase_feedback
        # then returned as a half-sentence. There is nothing here worth
        # thinking about: the measurements are given and the task is to write
        # one paragraph, so thinking is switched off. The larger ceiling is
        # headroom in case a future model ignores thinkingConfig.
        "generationConfig": {
            "temperature": 0.6,
            "maxOutputTokens": 700,
            "thinkingConfig": {"thinkingBudget": 0},
        },
    }
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{_model()}:generateContent"

    try:
        if client is not None:
            resp = await client.post(url, params={"key": key}, json=body)
        else:
            async with httpx.AsyncClient(timeout=_timeout_s()) as own:
                resp = await own.post(url, params={"key": key}, json=body)
        resp.raise_for_status()
        data = resp.json()
        parts = data["candidates"][0]["content"]["parts"]
        text = "".join(p.get("text", "") for p in parts if "text" in p).strip()
        return text or None
    except Exception:
        return None
