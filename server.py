"""TostAI Voice Studio -- a web app for Breeze TTS 2.

Wraps every capability of the Breeze TTS 2 model behind one small HTTP surface:

* plain text-to-speech            -> template ``tts_plain``
* voice design (instruction only) -> template ``tts_instruction``
* voice clone (reference only)    -> template ``ref_clone_tata``
* voice direction (reference+ins) -> template ``ref_edit_tata``
* bilingual English / Chinese, inline vocal events, seed and CFG control
* true chunk-by-chunk streaming (24 kHz s16le PCM) and buffered WAV export

Requests are forwarded to a running Breeze inference server
(``python -m breeze_infer.api ../breeze-tts-2``). When that server is not
reachable the app falls back to a clearly-labelled offline demo synth so the
whole interface stays explorable without a GPU.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import shutil
import struct
import subprocess
import sys
import tarfile
import tempfile
import threading
import time
import traceback
import urllib.error
import urllib.request
from array import array
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, NamedTuple

import httpx
from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import FileResponse, JSONResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles

APP_DIR = Path(__file__).resolve().parent
STATIC_DIR = APP_DIR / "static"
# Every finished take is written here, so the shelf in the UI survives a reload
# and a restart. Override with BREEZE_OUTPUTS_DIR (the image points it at a
# directory that can be mounted as a volume).
#
# `or` rather than a get() default, in both this and DEFAULT_UPSTREAM below: a
# variable that is SET BUT EMPTY -- which is what sourcing a .env line like
# BREEZE_OUTPUTS_DIR=${BREEZE_OUTPUTS_DIR:-} leaves behind -- would otherwise
# win, and Path("") is the current directory.
OUTPUTS_DIR = Path(os.environ.get("BREEZE_OUTPUTS_DIR") or (APP_DIR / "outputs"))
SAFE_NAME = re.compile(r"^[A-Za-z0-9._-]+\.wav$")
DEFAULT_UPSTREAM = os.environ.get("BREEZE_API_URL") or "http://127.0.0.1:7860"
DEFAULT_SAMPLE_RATE = 24000
UPSTREAM_HEALTH_TIMEOUT = 1.5

SAMPLE_RATE_HEADER = "X-Sample-Rate"
TEMPLATE_HEADER = "X-Breeze-Template"
DEMO_HEADER = "X-Breeze-Demo"
ELAPSED_HEADER = "X-Breeze-Elapsed-Ms"
BYTES_HEADER = "X-Breeze-Audio-Bytes"
WARNING_HEADER = "X-Breeze-Warning"
OUTPUT_HEADER = "X-Breeze-Output"
OPTIONAL_AUDIO_FILE = File(None)


class Inference(NamedTuple):
    """One prepared inference: either a real model stream or demo audio."""

    chunks: Any
    started: float
    template: str
    demo: bool
    sample_rate: int
    warning: str
    request: dict[str, Any]


# --------------------------------------------------------------------------- #
# Catalog: the single source of truth for everything the app can do.
# --------------------------------------------------------------------------- #

MODES: list[dict[str, Any]] = [
    {
        "id": "design",
        "name": "Voice Design",
        "tagline": "Invent a voice from a description",
        "template": "tts_instruction",
        "needs_reference": False,
        "needs_instruction": True,
        "default_cfg": 4.0,
        "hint": "No reference audio. Describe the voice you want and match the "
        "instruction language to the text.",
    },
    {
        "id": "clone",
        "name": "Voice Clone",
        "tagline": "Copy a speaker's timbre",
        "template": "ref_clone_tata",
        "needs_reference": True,
        "needs_instruction": False,
        "default_cfg": 1.0,
        "hint": "Upload clean speech plus its exact transcript. The instruction "
        "is ignored in this mode.",
    },
    {
        "id": "direct",
        "name": "Voice Direction",
        "tagline": "Steer tone and delivery",
        "template": "ref_edit_tata",
        "needs_reference": True,
        "needs_instruction": True,
        "default_cfg": 4.0,
        "hint": "Keep a reference speaker's identity while directing emotion, "
        "pace and delivery.",
    },
    {
        "id": "plain",
        "name": "Plain TTS",
        "tagline": "Read text in the model's default voice",
        "template": "tts_plain",
        "needs_reference": False,
        "needs_instruction": False,
        "default_cfg": 1.0,
        "hint": "Just text. The fastest path with no reference and no guidance.",
    },
]

VOCAL_EVENTS: dict[str, list[dict[str, str]]] = {
    "en": [
        {"token": "(laugh)", "label": "Laugh"},
        {"token": "(cough)", "label": "Cough"},
        {"token": "(clears throat)", "label": "Clears throat"},
        {"token": "(sigh)", "label": "Sigh"},
    ],
    "zh": [
        {"token": "[笑]", "label": "Laugh"},
        {"token": "[咳嗽]", "label": "Cough"},
        {"token": "[清嗓子]", "label": "Clears throat"},
        {"token": "[叹气]", "label": "Sigh"},
    ],
}

EXAMPLES: dict[str, dict[str, str]] = {
    "en": {
        "text": "(sigh) It is good to hear your voice again after all this time.",
        "design": "A warm, thoughtful young woman with a clear voice and a calm, "
        "reflective delivery.",
        "direction": "Speak slowly with a restrained, serious tone.",
    },
    "zh": {
        "text": "[叹气] 没想到过了这么久，你还记得我的声音。",
        "design": "一位温柔自信的年轻女性，声音清晰，语气亲切，表达轻快而富有感染力。",
        "direction": "语速缓慢，语气克制而严肃。",
    },
}

VOICE_PRESETS: dict[str, list[dict[str, str]]] = {
    "en": [
        {
            "label": "Warm narrator",
            "instruction": "A warm, thoughtful young woman with a clear voice and "
            "a calm, reflective delivery.",
        },
        {
            "label": "Gravelly veteran",
            "instruction": "An older man with a deep, gravelled voice, measured "
            "pace and a weary but kind tone.",
        },
        {
            "label": "Bright host",
            "instruction": "An energetic, bright young man with crisp articulation "
            "and an upbeat, playful delivery.",
        },
        {
            "label": "Whisper close",
            "instruction": "A soft, breathy voice speaking very close to the "
            "microphone, intimate and quiet.",
        },
        {
            "label": "News anchor",
            "instruction": "A neutral, authoritative female news anchor with even "
            "pacing and precise enunciation.",
        },
        {
            "label": "Storybook",
            "instruction": "A playful storyteller voice, expressive and animated, "
            "suitable for reading to children.",
        },
    ],
    "zh": [
        {
            "label": "温柔女声",
            "instruction": "一位温柔自信的年轻女性，声音清晰，语气亲切，表达轻快而富有感染力。",
        },
        {
            "label": "沉稳男声",
            "instruction": "一位沉稳的中年男性，声音低沉浑厚，语速平缓，语气可信而庄重。",
        },
        {
            "label": "新闻播报",
            "instruction": "一位标准的新闻播音员，吐字清晰，语速均匀，语气客观而权威。",
        },
        {
            "label": "活泼少女",
            "instruction": "一位活泼开朗的少女，音色明亮，语速偏快，情绪饱满。",
        },
        {
            "label": "耳语",
            "instruction": "轻声细语，气息声明显，仿佛在耳边低语。",
        },
        {
            "label": "说书人",
            "instruction": "一位富有感染力的说书人，抑扬顿挫，情绪起伏明显。",
        },
    ],
}

DIRECTION_PRESETS: dict[str, list[dict[str, str]]] = {
    "en": [
        {"label": "Serious", "instruction": "Speak slowly with a restrained, serious tone."},
        {"label": "Excited", "instruction": "Speak quickly and excitedly, with rising energy."},
        {"label": "Whisper", "instruction": "Whisper the whole line, quiet and intimate."},
        {"label": "Sad", "instruction": "A quiet, melancholy delivery with long pauses."},
        {"label": "Angry", "instruction": "A tight, angry delivery with clipped consonants."},
        {"label": "Announcer", "instruction": "Project like a stadium announcer, bold and proud."},
    ],
    "zh": [
        {"label": "严肃", "instruction": "语速缓慢，语气克制而严肃。"},
        {"label": "兴奋", "instruction": "语速加快，情绪兴奋，音调上扬。"},
        {"label": "耳语", "instruction": "整句话都用气声耳语，轻声而亲密。"},
        {"label": "悲伤", "instruction": "语气低沉悲伤，停顿较长。"},
        {"label": "愤怒", "instruction": "语气愤怒，咬字用力，节奏紧凑。"},
    ],
}

FAST_STAGES: list[dict[str, str]] = [
    {
        "flag": "--fast-text-encoder",
        "stage": "Text encoder",
        "detail": "Static CUDA Graph selected by CFG shape and text-length bucket",
    },
    {
        "flag": "--fast-backbone-prefill",
        "stage": "Backbone prefill",
        "detail": "CUDA Graph selected by CFG shape and prompt-length bucket",
    },
    {
        "flag": "--fast-backbone-decode",
        "stage": "Backbone decode",
        "detail": "StaticCache-backed graph selected by CFG shape",
    },
    {
        "flag": "--fast-depth-decoder",
        "stage": "Depth decoder",
        "detail": "Full-graph compilation with CFG-shape CUDA Graphs",
    },
    {
        "flag": "--fast-codec",
        "stage": "Codec",
        "detail": "Single-request streaming CUDA Graph with one-frame chunks",
    },
]

MODEL_FACTS: dict[str, Any] = {
    "name": "Breeze TTS 2",
    "sample_rate": DEFAULT_SAMPLE_RATE,
    "channels": 1,
    "format": "s16le",
    "languages": ["en", "zh"],
    "gpu_memory_gib": {"eager": 7.7, "fast_all": 14.4},
    "ttfa_ms": 40,
    "rtf": 0.32,
    "license": "BreezeBlue Research and Non-Commercial License",
}


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #


def _select_template(instruction: str | None, has_reference: bool) -> str:
    """Mirror ``breeze_infer.templates.select_template_name``."""
    if has_reference:
        return "ref_edit_tata" if instruction else "ref_clone_tata"
    return "tts_instruction" if instruction else "tts_plain"


def _validate(cfg_scale: float, has_audio: bool, ref_text: str) -> None:
    if not math.isfinite(cfg_scale) or cfg_scale <= 0:
        raise HTTPException(status_code=400, detail="cfg_scale must be greater than 0.")
    if has_audio != bool(ref_text):
        raise HTTPException(
            status_code=400,
            detail="Reference audio and its transcript must be provided together "
            "or both omitted.",
        )


def _wav_header(data_len: int, sample_rate: int) -> bytes:
    return struct.pack(
        "<4sI4s4sIHHIIHH4sI",
        b"RIFF",
        36 + data_len,
        b"WAVE",
        b"fmt ",
        16,
        1,  # PCM
        1,  # mono
        sample_rate,
        sample_rate * 2,  # byte rate
        2,  # block align
        16,  # bits per sample
        b"data",
        data_len,
    )


def _to_little_endian(samples: array) -> bytes:
    if sys.byteorder == "big":
        samples = array("h", samples)
        samples.byteswap()
    return samples.tobytes()


def _demo_pcm(text: str, lang: str) -> bytes:
    """A small formant-ish placeholder synth so the UI works without a GPU.

    It is intentionally obvious that this is *not* the model: it is a buzzy
    robotic warble whose length tracks the text.
    """
    sample_rate = DEFAULT_SAMPLE_RATE
    base = 158.0 if lang == "zh" else 126.0
    samples = array("h")
    for char in text:
        if char.isspace():
            samples.extend([0] * int(sample_rate * 0.045))
            continue
        code = ord(char)
        length = int(sample_rate * (0.08 + (code % 7) * 0.008))
        freq = base + (code % 17) * 6.5 + (26.0 if lang == "zh" else 0.0)
        attack = max(1, int(sample_rate * 0.012))
        release = max(1, int(sample_rate * 0.02))
        for i in range(length):
            t = i / sample_rate
            env = min(1.0, i / attack, (length - i) / release)
            vibrato = 1.0 + 0.02 * math.sin(2 * math.pi * 5.0 * t)
            value = (
                math.sin(2 * math.pi * freq * vibrato * t) * 0.6
                + math.sin(2 * math.pi * freq * 2.0 * t) * 0.22
                + math.sin(2 * math.pi * freq * 3.0 * t) * 0.12
            )
            samples.append(int(max(-1.0, min(1.0, value * env * 0.5)) * 32767))
        samples.extend([0] * int(sample_rate * 0.012))

    minimum = sample_rate // 4
    maximum = sample_rate * 12
    if len(samples) < minimum:
        samples.extend([0] * (minimum - len(samples)))
    if len(samples) > maximum:
        del samples[maximum:]
    return _to_little_endian(samples)


async def _demo_stream(text: str, lang: str) -> Any:
    pcm = _demo_pcm(text, lang)
    frame = max(2, len(pcm) // 6)
    for start in range(0, len(pcm), frame):
        yield pcm[start : start + frame]


# --------------------------------------------------------------------------- #
# Output shelf
# --------------------------------------------------------------------------- #

TEMPLATE_MODES = {
    "tts_plain": "plain",
    "tts_instruction": "design",
    "ref_clone_tata": "clone",
    "ref_edit_tata": "direct",
}


def _output_path(name: str) -> Path:
    """Resolve a requested output name, refusing anything but a bare .wav.

    The name arrives from the browser, so it is matched against a strict
    pattern (no separators, no dots beyond the extension) and the resolved path
    is then re-checked to be inside ``OUTPUTS_DIR``. Both, not either: the
    pattern is what rejects ``..\\..\\secrets.wav`` early, and the containment
    check is what still holds if the pattern is ever loosened.
    """
    if not SAFE_NAME.match(name):
        raise HTTPException(status_code=400, detail="Not a valid output name.")
    path = (OUTPUTS_DIR / name).resolve()
    if path.parent != OUTPUTS_DIR.resolve() or not path.is_file():
        raise HTTPException(status_code=404, detail=f"No such output: {name}")
    return path


def _save_output(
    pcm: bytes,
    sample_rate: int,
    request: dict[str, Any],
    *,
    demo: bool = False,
    ttfa_ms: float | None = None,
    total_ms: float | None = None,
) -> dict[str, Any]:
    """Write one take as WAV plus a JSON sidecar, and return its record.

    The sidecar is what makes the shelf survivable: the list endpoint is built
    from these files, so a take outlives the browser tab, the server process and
    a container restart without any database.
    """
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d-%H%M%S")
    stem = f"breeze-{stamp}-{request.get('mode', 'take')}-seed{request.get('seed', 0)}"
    path = OUTPUTS_DIR / f"{stem}.wav"
    counter = 2
    while path.exists():  # two takes in the same second are normal, not an error
        path = OUTPUTS_DIR / f"{stem}-{counter}.wav"
        counter += 1

    path.write_bytes(_wav_header(len(pcm), sample_rate) + pcm)
    record = {
        **request,
        "name": path.name,
        "url": f"/api/outputs/{path.name}",
        "sample_rate": sample_rate,
        "channels": 1,
        "audio_bytes": len(pcm),
        "duration": len(pcm) / 2 / sample_rate,
        "demo": demo,
        "ttfa_ms": ttfa_ms,
        "total_ms": total_ms,
        "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "created_ts": time.time(),
    }
    path.with_suffix(".json").write_text(
        json.dumps(record, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return record


def _load_outputs(limit: int = 200) -> list[dict[str, Any]]:
    """Every saved take, newest first, skipping sidecars with no audio beside them."""
    if not OUTPUTS_DIR.is_dir():
        return []
    records: list[dict[str, Any]] = []
    for sidecar in OUTPUTS_DIR.glob("*.json"):
        if not sidecar.with_suffix(".wav").is_file():
            continue  # audio deleted by hand: the sidecar is now a ghost
        try:
            record = json.loads(sidecar.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        # Defaults, not just for tidiness: a sidecar written by an older build
        # is still a valid take, and the shelf must not reject it for a field
        # that did not exist when it was produced.
        record.setdefault("name", sidecar.with_suffix(".wav").name)
        record.setdefault("url", f"/api/outputs/{record['name']}")
        record.setdefault("demo", False)
        record.setdefault("duration", 0.0)
        record.setdefault("ttfa_ms", None)
        record.setdefault("total_ms", None)
        records.append(record)
    records.sort(key=lambda r: r.get("created_ts", 0), reverse=True)
    return records[:limit]


# --------------------------------------------------------------------------- #
# App
# --------------------------------------------------------------------------- #

client: httpx.AsyncClient | None = None
UPSTREAM = DEFAULT_UPSTREAM
FORCE_DEMO = False
AUTO_DEMO = True


@asynccontextmanager
async def _lifespan(_: FastAPI):
    global client
    client = httpx.AsyncClient(timeout=httpx.Timeout(600.0, connect=5.0))
    try:
        yield
    finally:
        await client.aclose()
        client = None


app = FastAPI(title="TostAI Voice Studio", version="1.0.0", lifespan=_lifespan)
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.get("/")
async def index() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/api/catalog")
async def catalog() -> JSONResponse:
    return JSONResponse(
        {
            "modes": MODES,
            "vocal_events": VOCAL_EVENTS,
            "examples": EXAMPLES,
            "voice_presets": VOICE_PRESETS,
            "direction_presets": DIRECTION_PRESETS,
            "fast_stages": FAST_STAGES,
            "model": MODEL_FACTS,
        }
    )


@app.get("/api/status")
async def status() -> JSONResponse:
    if FORCE_DEMO:
        return JSONResponse(
            {
                "connected": False,
                "demo": True,
                "fallback": AUTO_DEMO,
                "upstream": UPSTREAM,
                "sample_rate": DEFAULT_SAMPLE_RATE,
                "detail": "Forced demo mode.",
            }
        )
    assert client is not None
    started = time.perf_counter()
    try:
        response = await client.get(
            f"{UPSTREAM}/health", timeout=UPSTREAM_HEALTH_TIMEOUT
        )
        payload = response.json() if response.status_code < 500 else {}
        connected = response.status_code == 200 and payload.get("status") == "ok"
        return JSONResponse(
            {
                "connected": connected,
                "demo": not connected,
                "fallback": AUTO_DEMO,
                "upstream": UPSTREAM,
                "sample_rate": int(payload.get("sample_rate", DEFAULT_SAMPLE_RATE)),
                "upstream_status": response.status_code,
                "latency_ms": round((time.perf_counter() - started) * 1000, 1),
                "detail": payload.get("status", "unknown"),
            }
        )
    except Exception as exc:  # noqa: BLE001 - surfaced to the UI verbatim
        return JSONResponse(
            {
                "connected": False,
                "demo": True,
                "fallback": AUTO_DEMO,
                "upstream": UPSTREAM,
                "sample_rate": DEFAULT_SAMPLE_RATE,
                "latency_ms": round((time.perf_counter() - started) * 1000, 1),
                "detail": f"{type(exc).__name__}: {exc}",
            }
        )


async def _run_inference(
    *,
    text: str,
    instruction: str,
    cfg_scale: float,
    seed: int,
    ref_text: str,
    ref_audio: UploadFile | None,
    lang: str,
) -> Any:
    """Yield an :class:`Inference` -- from Breeze, or from demo fallback."""
    text = text.strip()
    if not text:
        raise HTTPException(status_code=400, detail="Text cannot be empty.")
    instruction = (instruction or "").strip()
    ref_text = (ref_text or "").strip()
    has_audio = ref_audio is not None and bool(ref_audio.filename)
    _validate(cfg_scale, has_audio, ref_text)

    template = _select_template(instruction or None, has_audio)
    started = time.perf_counter()

    record_request = {
        "text": text,
        "instruction": instruction,
        "lang": lang,
        "seed": seed,
        "cfg_scale": cfg_scale,
        "template": template,
        "mode": TEMPLATE_MODES.get(template, "take"),
        "has_reference": has_audio,
        "ref_text": ref_text,
    }

    if FORCE_DEMO:
        yield Inference(
            _demo_stream(text, lang),
            started,
            template,
            True,
            DEFAULT_SAMPLE_RATE,
            "Demo mode was requested with --demo.",
            record_request,
        )
        return

    assert client is not None
    data: dict[str, Any] = {
        "text": text,
        "cfg_scale": str(cfg_scale),
        "seed": str(seed),
    }
    if instruction:
        data["instruction"] = instruction
    files: dict[str, Any] | None = None
    if has_audio:
        assert ref_audio is not None
        payload = await ref_audio.read()
        if not payload:
            raise HTTPException(status_code=400, detail="Reference audio is empty.")
        data["ref_text"] = ref_text
        files = {
            "ref_audio": (
                ref_audio.filename or "reference.wav",
                payload,
                ref_audio.content_type or "audio/wav",
            )
        }

    try:
        request = client.build_request(
            "POST", f"{UPSTREAM}/v1/audio/speech", data=data, files=files
        )
        upstream = await client.send(request, stream=True)
    except Exception as exc:
        if AUTO_DEMO:
            yield Inference(
                _demo_stream(text, lang),
                started,
                template,
                True,
                DEFAULT_SAMPLE_RATE,
                f"Breeze server at {UPSTREAM} is unreachable "
                f"({type(exc).__name__}), serving demo audio instead.",
                record_request,
            )
            return
        raise HTTPException(
            status_code=502,
            detail=f"Cannot reach the Breeze inference server at {UPSTREAM} "
            f"({type(exc).__name__}: {exc}).",
        ) from exc

    if upstream.status_code >= 400:
        body = await upstream.aread()
        await upstream.aclose()
        detail = body.decode("utf-8", "replace")[:400] or "upstream error"
        raise HTTPException(status_code=upstream.status_code, detail=detail)

    sample_rate = int(upstream.headers.get(SAMPLE_RATE_HEADER, DEFAULT_SAMPLE_RATE))

    async def stream() -> Any:
        try:
            async for chunk in upstream.aiter_bytes(4096):
                if chunk:
                    yield chunk
        finally:
            await upstream.aclose()

    yield Inference(stream(), started, template, False, sample_rate, "", record_request)


def _common_headers(
    template: str,
    started: float,
    demo: bool,
    sample_rate: int,
    size: int | None,
    warning: str = "",
) -> dict[str, str]:
    headers = {
        TEMPLATE_HEADER: template,
        DEMO_HEADER: "1" if demo else "0",
        ELAPSED_HEADER: str(round((time.perf_counter() - started) * 1000, 1)),
        SAMPLE_RATE_HEADER: str(sample_rate),
        "Cache-Control": "no-store",
    }
    if size is not None:
        headers[BYTES_HEADER] = str(size)
    if warning:
        headers[WARNING_HEADER] = warning.encode("ascii", "replace").decode("ascii")
    return headers


@app.post("/api/generate")
async def generate(
    text: str = Form(...),
    instruction: str = Form(""),
    cfg_scale: float = Form(1.0),
    seed: int = Form(42),
    ref_text: str = Form(""),
    lang: str = Form("en"),
    ref_audio: UploadFile | None = OPTIONAL_AUDIO_FILE,
) -> Response:
    """Buffered synthesis returned as a WAV file (easy playback + download)."""
    async for result in _run_inference(
        text=text,
        instruction=instruction,
        cfg_scale=cfg_scale,
        seed=seed,
        ref_text=ref_text,
        ref_audio=ref_audio,
        lang=lang,
    ):
        pcm = b"".join([chunk async for chunk in result.chunks])
        headers = _common_headers(
            result.template,
            result.started,
            result.demo,
            result.sample_rate,
            len(pcm),
            result.warning,
        )
        if pcm:
            elapsed = round((time.perf_counter() - result.started) * 1000, 1)
            record = _save_output(
                pcm,
                result.sample_rate,
                result.request,
                demo=result.demo,
                # Buffered by definition: nothing plays until the whole clip
                # exists, so time-to-first-audio is the total.
                ttfa_ms=elapsed,
                total_ms=elapsed,
            )
            headers[OUTPUT_HEADER] = record["name"]
        return Response(
            content=_wav_header(len(pcm), result.sample_rate) + pcm,
            media_type="audio/wav",
            headers=headers,
        )
    raise HTTPException(status_code=500, detail="No audio produced.")


@app.post("/api/stream")
async def stream(
    text: str = Form(...),
    instruction: str = Form(""),
    cfg_scale: float = Form(1.0),
    seed: int = Form(42),
    ref_text: str = Form(""),
    lang: str = Form("en"),
    ref_audio: UploadFile | None = OPTIONAL_AUDIO_FILE,
) -> StreamingResponse:
    """Raw streaming PCM (mono, s16le) exactly as the Breeze API emits it."""
    async for result in _run_inference(
        text=text,
        instruction=instruction,
        cfg_scale=cfg_scale,
        seed=seed,
        ref_text=ref_text,
        ref_audio=ref_audio,
        lang=lang,
    ):
        headers = _common_headers(
            result.template,
            result.started,
            result.demo,
            result.sample_rate,
            None,
            result.warning,
        )
        chunks = result.chunks
        rate = result.sample_rate
        meta = result.request
        started_at = result.started
        demo = result.demo

        async def body(
            source: Any = chunks,
            rate: int = rate,
            meta: dict = meta,
            started_at: float = started_at,
            demo: bool = demo,
        ) -> Any:
            buffer = bytearray()
            first_chunk_at: float | None = None
            async for chunk in source:
                if first_chunk_at is None:
                    first_chunk_at = time.perf_counter()
                buffer.extend(chunk)
                yield chunk
            # Only reached when the model stream ended on its own. A client that
            # disconnects closes this generator at the yield instead, and half a
            # take is not worth keeping on the shelf.
            if buffer:
                total = round((time.perf_counter() - started_at) * 1000, 1)
                ttfa = (
                    round((first_chunk_at - started_at) * 1000, 1)
                    if first_chunk_at
                    else total
                )
                # Off the event loop: writing a 30-second WAV is disk work, and
                # the shelf endpoint shares this loop.
                await run_in_threadpool(
                    _save_output,
                    bytes(buffer),
                    rate,
                    meta,
                    demo=demo,
                    ttfa_ms=ttfa,
                    total_ms=total,
                )

        return StreamingResponse(body(), media_type="audio/pcm", headers=headers)
    raise HTTPException(status_code=500, detail="No audio produced.")


@app.get("/api/outputs")
async def list_outputs() -> JSONResponse:
    """Every saved take, newest first -- this is what the UI's shelf renders."""
    outputs = _load_outputs()
    return JSONResponse(
        {"dir": str(OUTPUTS_DIR), "count": len(outputs), "outputs": outputs}
    )


@app.get("/api/outputs/{name}")
async def get_output(name: str) -> FileResponse:
    return FileResponse(
        _output_path(name),
        media_type="audio/wav",
        headers={"Cache-Control": "no-store"},
    )


@app.delete("/api/outputs/{name}")
async def delete_output(name: str) -> JSONResponse:
    path = _output_path(name)
    path.unlink()
    path.with_suffix(".json").unlink(missing_ok=True)
    return JSONResponse({"deleted": name})


# --------------------------------------------------------------------------- #
# self-update: pull the latest source from this app's own repo, then restart
# --------------------------------------------------------------------------- #
#
# Why a token is typed in rather than configured: the build's GITHUB_TOKEN is a
# secret MOUNT, so it is deliberately gone by the time this code runs, and the
# repo is private. Baking a PAT into the image would hand it to anyone who pulls
# the image. Asking for it per-update keeps it in memory for one request.
#
# Why the files are copied over the live tree rather than the container being
# rebuilt: this is a dev convenience -- click, get latest, keep working. It is
# explicitly NOT durable. The installed source lives in the container and dies
# with it; a rebuild replaces it. The durable path is still a rebuild.
#
# Why the restart is a re-exec: `index.html`/`app.js` are re-read per request,
# but `server.py` is Python held in memory, so new code on disk is invisible
# until the process restarts. `os.execv` replaces the process image in place,
# which keeps the same PID 1 and the same container -- a plain `sys.exit` would
# stop the container instead (the default restart policy is `no`).

APP_REPO = os.environ.get("TOSTAI_APP_REPO", "camenduru/TostAI-Voice-Studio")
APP_REV_FILE = APP_DIR / ".tostai_rev"
UPDATE_BACKUP = APP_DIR / ".update_backup"

# Never overwritten by an update. `outputs/` is this container's own state and
# is gitignored upstream, so it should never appear in the tarball at all --
# this is belt and braces, because the cost of being wrong is somebody's take.
# `.update_backup` is excluded so a backup never contains itself.
UPDATE_KEEP = ("outputs", ".update_backup", ".git", "__pycache__")

_UA = {"User-Agent": "tostai-voice-studio-update",
       "Accept": "application/vnd.github+json"}


def _gh_json(url: str, token: str) -> tuple[Any, str | None]:
    """GET a GitHub API URL. Returns (data, None) or (None, human_error)."""
    req = urllib.request.Request(
        url, headers=dict(_UA, Authorization="Bearer " + token))
    try:
        with urllib.request.urlopen(req, timeout=30) as r:
            return json.loads(r.read().decode("utf-8")), None
    except urllib.error.HTTPError as e:
        if e.code == 401:
            return None, ("GitHub rejected the token (401). It may have expired, "
                          "or lack `repo` scope for a private repository.")
        if e.code == 404:
            return None, ("GitHub returned 404 for %s. Either the repository name "
                          "is wrong, or the token cannot see it -- a private repo "
                          "404s rather than 403s for an unauthorised token." % url)
        if e.code == 403:
            return None, ("GitHub returned 403 (rate limit or blocked). If the "
                          "token is a fine-grained PAT it needs Contents: read.")
        return None, "GitHub returned HTTP %d." % e.code
    except Exception as ex:                                     # noqa: BLE001
        return None, "%s: %s" % (type(ex).__name__, ex)


def _current_rev() -> str:
    try:
        return APP_REV_FILE.read_text(encoding="utf-8").strip()
    except OSError:
        return ""


def _restart_soon(delay: float = 1.5) -> None:
    """Replace this process with a fresh one, once the response has flushed.

    The delay is load-bearing: both paths below tear the server down, so doing it
    before the response is written turns a successful update into a network error
    on the user's screen.
    """
    def _go() -> None:
        time.sleep(delay)
        argv = [sys.executable, os.path.abspath(__file__)] + sys.argv[1:]
        try:
            if os.name == "nt":
                # Windows: os.execv is NOT usable from here. Measured in
                # sprite_studio and reproducible in isolation -- a uvicorn server
                # that execv's itself from a background thread dies outright: the
                # in-flight response is cut off mid-body, no replacement process is
                # ever started, and the port goes dead. So spawn a detached
                # replacement and leave; exiting is safe because there is no PID 1
                # to keep alive.
                flags = (subprocess.DETACHED_PROCESS
                         | subprocess.CREATE_NEW_PROCESS_GROUP)
                subprocess.Popen(argv, cwd=str(APP_DIR), close_fds=True,
                                 creationflags=flags)
                os._exit(0)
            else:
                # POSIX, and specifically a container: execve(2) swaps the image
                # inside the SAME pid, so PID 1 stays PID 1 and the container is
                # not stopped. Exiting would end the container, and the default
                # restart policy is `no`.
                os.execv(sys.executable, argv)
        except Exception:                                       # noqa: BLE001
            # Restart failed, so this process is still serving the OLD code. Say
            # so rather than exiting: a stopped container is worse than a stale one
            # that still works.
            traceback.print_exc()
    threading.Thread(target=_go, daemon=True, name="tostai-restart").start()


# The commit subject the running process started from. Set on a successful POST
# and read by the GET below, so the dialog can name what is installed rather
# than only its hash.
_rev_subject = ""


@app.get("/api/update")
async def update_status() -> JSONResponse:
    """What is installed, so the dialog can say so before anything is typed."""
    rev = _current_rev()
    return JSONResponse({"ok": True, "repo": APP_REPO, "rev": rev, "short": rev[:10],
                         "subject": _rev_subject})


@app.post("/api/update")
async def update_apply(req: Request) -> JSONResponse:
    global _rev_subject
    body = await req.json() or {}
    token = (body.get("token") or "").strip()
    if not token:
        return JSONResponse({"ok": False, "error": "no GitHub token given"},
                            status_code=400)

    # 1. What is upstream right now?
    commits, err = _gh_json(
        "https://api.github.com/repos/%s/commits?per_page=1" % APP_REPO, token)
    if err:
        return JSONResponse({"ok": False, "error": err}, status_code=400)
    if not commits:
        return JSONResponse({"ok": False, "error": "the repository has no commits"},
                            status_code=400)
    latest = commits[0]["sha"]
    subject = (commits[0].get("commit", {}).get("message") or "").splitlines()[0]
    current = _current_rev()

    if latest == current:
        _rev_subject = subject
        return JSONResponse({"ok": True, "updated": False, "rev": latest,
                             "subject": subject,
                             "message": "already up to date at %s" % latest[:10]})

    tmp = tempfile.mkdtemp(prefix="tostai_update_")
    try:
        # 2. Fetch that exact tree. The API redirects to codeload with a signed
        #    token in the URL, so the redirect is followed without the header.
        tarball = os.path.join(tmp, "src.tar.gz")
        dl = urllib.request.Request(
            "https://api.github.com/repos/%s/tarball/%s" % (APP_REPO, latest),
            headers=dict(_UA, Authorization="Bearer " + token))
        try:
            with urllib.request.urlopen(dl, timeout=180) as r, \
                    open(tarball, "wb") as f:
                shutil.copyfileobj(r, f)
        except Exception as ex:                                 # noqa: BLE001
            return JSONResponse({"ok": False, "error": "download failed: %s: %s"
                                 % (type(ex).__name__, ex)}, status_code=400)

        root = os.path.join(tmp, "x")
        os.makedirs(root)
        with tarfile.open(tarball, "r:gz") as tf:
            try:
                tf.extractall(root, filter="data")              # py3.12+
            except TypeError:
                tf.extractall(root)                             # py3.10/3.11

        # GitHub wraps the tree in a single <owner>-<repo>-<shortsha> directory.
        entries = sorted(e for e in os.listdir(root) if not e.startswith("."))
        if len(entries) != 1 or not os.path.isdir(os.path.join(root, entries[0])):
            return JSONResponse({"ok": False, "error":
                                 "unexpected tarball layout: %r" % entries},
                                status_code=400)
        src = os.path.join(root, entries[0])

        if not os.path.isfile(os.path.join(src, "server.py")):
            return JSONResponse({"ok": False, "error":
                                 "refusing to install: the new tree has no server.py"},
                                status_code=400)

        # 3. Parse every new .py BEFORE anything is swapped in. A file that fails
        #    to parse would be re-exec'd into a container that never comes back,
        #    and the default restart policy is `no`.
        bad = []
        for dirpath, dirnames, filenames in os.walk(src):
            dirnames[:] = [d for d in dirnames if d not in UPDATE_KEEP]
            for fn in filenames:
                if not fn.endswith(".py"):
                    continue
                p = os.path.join(dirpath, fn)
                try:
                    with open(p, "rb") as fh:
                        compile(fh.read(), p, "exec")
                except SyntaxError as ex:
                    bad.append("%s: line %s: %s" % (os.path.relpath(p, src),
                                                    ex.lineno, ex.msg))
                except (OSError, ValueError) as ex:
                    bad.append("%s: %s" % (os.path.relpath(p, src), ex))
        if bad:
            return JSONResponse({"ok": False, "error":
                                 "refusing to install -- the new source does not "
                                 "compile:\n" + "\n".join(bad[:10])},
                                status_code=400)

        # 4. Back up the current tree, then copy the new one over it. Copy, not
        #    replace, so files deleted upstream linger harmlessly rather than
        #    `outputs/` being swept away with them.
        if UPDATE_BACKUP.is_dir():
            shutil.rmtree(UPDATE_BACKUP, ignore_errors=True)
        try:
            shutil.copytree(APP_DIR, UPDATE_BACKUP,
                            ignore=shutil.ignore_patterns(*UPDATE_KEEP))
        except Exception as ex:                                 # noqa: BLE001
            return JSONResponse({"ok": False, "error":
                                 "could not write a backup, so nothing was "
                                 "changed: %s" % ex}, status_code=400)

        copied = 0
        for dirpath, dirnames, filenames in os.walk(src):
            dirnames[:] = [d for d in dirnames if d not in UPDATE_KEEP]
            rel = os.path.relpath(dirpath, src)
            dst_dir = APP_DIR if rel == "." else APP_DIR / rel
            os.makedirs(dst_dir, exist_ok=True)
            for fn in filenames:
                shutil.copy2(os.path.join(dirpath, fn), os.path.join(dst_dir, fn))
                copied += 1

        APP_REV_FILE.write_text(latest + "\n", encoding="utf-8")
        _rev_subject = subject

        _restart_soon()
        return JSONResponse({"ok": True, "updated": True, "rev": latest,
                             "subject": subject, "files": copied,
                             "message": "updated to %s, restarting" % latest[:10]})
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Serve TostAI Voice Studio")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument(
        "--upstream",
        default=DEFAULT_UPSTREAM,
        help="Base URL of the Breeze inference server (default: %(default)s)",
    )
    parser.add_argument(
        "--demo",
        action="store_true",
        help="Never call the model; always return demo audio.",
    )
    parser.add_argument(
        "--no-demo-fallback",
        action="store_true",
        help="Return a 502 instead of demo audio when the model server is down.",
    )
    args = parser.parse_args()

    global UPSTREAM, FORCE_DEMO, AUTO_DEMO
    UPSTREAM = args.upstream.rstrip("/")
    FORCE_DEMO = args.demo
    AUTO_DEMO = not args.no_demo_fallback

    import uvicorn

    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
