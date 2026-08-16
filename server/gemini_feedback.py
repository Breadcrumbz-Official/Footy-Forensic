from __future__ import annotations

import asyncio
import base64
import os

import httpx

DEFAULT_MODEL = "gemini-3.7-flash"


def _api_key() -> str:
    return os.environ.get("GEMINI_API_KEY", "").strip()


def _model() -> str:
    return os.environ.get("SFAI_GEMINI_MODEL", DEFAULT_MODEL)


def _timeout_s() -> float:
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


_VERIFY_SYSTEM = (
    "You are checking a computer-vision overlay for gross errors. The image is "
    "a video frame of a person kicking a ball, with a detected skeleton drawn "
    "over it as cyan/grey lines and yellow joint dots, and possibly a pink ring "
    "around the ball. Your only job is to judge whether that skeleton is drawn "
    "on the kicking player's actual body.\n\n"
    "Answer NO only for a gross failure: the skeleton is on a different person, "
    "on empty background, badly offset from the body, or its limbs clearly do "
    "not follow the player's limbs.\n\n"
    "Answer YES if the skeleton broadly follows the player, even if it is "
    "imperfect — a joint a little off, a foot or hand slightly misplaced, or "
    "limbs missing where they leave frame or are hidden are all normal and "
    "still YES. A blurry or awkward-looking pose is not itself a failure.\n\n"
    "Reply with exactly one word: YES or NO."
)


async def verify_skeleton(image_bytes: bytes,
                          client: httpx.AsyncClient | None = None) -> bool | None:
    key = _api_key()
    if not key:
        return None
    body = {
        "system_instruction": {"parts": [{"text": _VERIFY_SYSTEM}]},
        "contents": [{
            "role": "user",
            "parts": [
                {"text": "Is the drawn skeleton on the player's body? YES or NO."},
                {"inline_data": {"mime_type": "image/jpeg",
                                 "data": base64.b64encode(image_bytes).decode("ascii")}},
            ],
        }],
        "generationConfig": {"temperature": 0.0, "maxOutputTokens": 200,
                             "thinkingConfig": {"thinkingBudget": 0}},
    }
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{_model()}:generateContent"
    try:
        if client is not None:
            resp = await client.post(url, params={"key": key}, json=body)
        else:
            async with httpx.AsyncClient(timeout=_timeout_s()) as own:
                resp = await own.post(url, params={"key": key}, json=body)
        resp.raise_for_status()
        parts = resp.json()["candidates"][0]["content"]["parts"]
        text = "".join(p.get("text", "") for p in parts if "text" in p).strip().upper()
        if not text:
            return None
        if "NO" in text:
            return False
        if "YES" in text:
            return True
        return None
    except Exception:
        return None


async def verify_batch(images: list[bytes]) -> list[bool | None]:
    if not _api_key():
        return [None] * len(images)
    async with httpx.AsyncClient(timeout=_timeout_s()) as client:
        return list(await asyncio.gather(
            *[verify_skeleton(img, client=client) for img in images]))


async def batch(items: list[tuple[str, bytes, list[dict], str]]) -> list[str | None]:
    if not _api_key():
        return [None] * len(items)
    async with httpx.AsyncClient(timeout=_timeout_s()) as client:
        return list(await asyncio.gather(
            *[phase_feedback(*item, client=client) for item in items]))


async def phase_feedback(phase_label: str, image_bytes: bytes, metrics: list[dict],
                         kick_side: str, client: httpx.AsyncClient | None = None) -> str | None:
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
