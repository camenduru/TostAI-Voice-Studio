"""Exercise every mode of TostAI Voice Studio against a running app.

One request per template, asserting the things that actually distinguish the
four modes rather than just "did it return audio":

  plain    no reference, no instruction -> tts_plain
  design   instruction, no reference    -> tts_instruction
  clone    reference, no instruction    -> ref_clone_tata
  direct   reference AND instruction    -> ref_edit_tata

Clone and Direction need a reference clip plus its EXACT transcript, so rather
than shipping a fixture this borrows the newest take in the output folder and
reads the transcript out of that take's sidecar -- the sidecar exists precisely
to record what was said. Pass --reference/--ref-text to use something else.

Run against a live server:

    .venv/Scripts/python.exe smoke_modes.py                     # all four
    .venv/Scripts/python.exe smoke_modes.py --only plain design # a subset

Exits non-zero if any mode fails, so it is usable as a gate.
"""

from __future__ import annotations

import argparse
import struct
import sys
import time
from pathlib import Path
from typing import Any

import httpx

# Short lines on purpose: this is a correctness probe, not a benchmark, and a
# clone/direction request has to prefill the whole reference clip first.
TEXTS = {
    "plain": "Plain text to speech check.",
    "design": "A voice built from a description.",
    "clone": "Cloning this speaker now.",
    "direct": "Directed delivery check.",
}
INSTRUCTIONS = {
    "design": "A warm, thoughtful young woman with a clear voice and a calm delivery.",
    "direct": "Speak slowly with a restrained, serious tone.",
}
# What the template each mode must resolve to.
EXPECTED_TEMPLATE = {
    "plain": "tts_plain",
    "design": "tts_instruction",
    "clone": "ref_clone_tata",
    "direct": "ref_edit_tata",
}
# Breeze has no negative prompt for plain/clone, so CFG must stay at 1.0 there.
CFG = {"plain": 1.0, "design": 4.0, "clone": 1.0, "direct": 4.0}
MODE_ORDER = ("plain", "design", "clone", "direct")


class Result:
    def __init__(self, mode: str) -> None:
        self.mode = mode
        self.ok = True
        self.notes: list[str] = []
        self.template = "-"
        self.demo = "-"
        self.seconds = 0.0
        self.elapsed = 0.0
        self.saved = "-"

    def fail(self, note: str) -> None:
        self.ok = False
        self.notes.append(note)

    def note(self, note: str) -> None:
        self.notes.append(note)


def wav_duration(data: bytes) -> tuple[float, int, int]:
    """Duration, channels and sample rate from a RIFF/WAVE blob."""
    if len(data) < 44 or data[:4] != b"RIFF" or data[8:12] != b"WAVE":
        raise ValueError("not a RIFF/WAVE file")
    channels = sample_rate = bits = None
    offset = 12
    while offset + 8 <= len(data):
        chunk_id, size = struct.unpack_from("<4sI", data, offset)
        body = data[offset + 8 : offset + 8 + size]
        if chunk_id == b"fmt ":
            _, channels, sample_rate, _, _, bits = struct.unpack_from("<HHIIHH", body, 0)
        elif chunk_id == b"data":
            if not (channels and sample_rate and bits):
                raise ValueError("data chunk before fmt chunk")
            return len(body) / (sample_rate * channels * bits // 8), channels, sample_rate
        offset += 8 + size + (size % 2)
    raise ValueError("no data chunk")


def fetch_reference(client: httpx.Client, base: str, explicit: Path | None, text: str) -> tuple[Path, str]:
    """A reference clip and its transcript: the argument, or the newest take."""
    if explicit is not None:
        if not explicit.is_file():
            raise SystemExit(f"reference clip not found: {explicit}")
        if not text:
            raise SystemExit("--reference needs --ref-text")
        return explicit, text

    listing = client.get(f"{base}/api/outputs").json()
    outputs = listing.get("outputs") or []
    if not outputs:
        raise SystemExit(
            "no saved takes to borrow a reference from. Run a plain/design "
            "generation first, or pass --reference and --ref-text."
        )
    # Longest take wins: a clone has more speaker to learn from a longer clip.
    newest = max(outputs, key=lambda o: o.get("duration") or 0)
    path = Path(listing["dir"]) / newest["name"]
    transcript = (newest.get("text") or "").strip()
    if not path.is_file() or not transcript:
        raise SystemExit(f"borrowed take is unusable: {path} / {transcript!r}")
    return path, transcript


def run_mode(
    client: httpx.Client,
    base: str,
    mode: str,
    reference: tuple[Path, str] | None,
) -> Result:
    result = Result(mode)
    fields: dict[str, str] = {"text": TEXTS[mode], "cfg_scale": str(CFG[mode]), "seed": "99"}
    files: dict[str, Any] | None = None
    if mode in ("design", "direct"):
        fields["instruction"] = INSTRUCTIONS[mode]
    if mode in ("clone", "direct"):
        assert reference is not None
        path, transcript = reference
        fields["ref_text"] = transcript
        files = {"ref_audio": (path.name, path.read_bytes(), "audio/wav")}

    started = time.perf_counter()
    try:
        response = client.post(f"{base}/api/generate", data=fields, files=files)
    except Exception as exc:  # noqa: BLE001 - reported as a failure, not a crash
        result.fail(f"request raised {type(exc).__name__}: {exc}")
        return result
    result.elapsed = time.perf_counter() - started

    if response.status_code != 200:
        result.fail(f"HTTP {response.status_code}: {response.text[:200]}")
        return result

    result.template = response.headers.get("X-Breeze-Template", "-")
    result.demo = response.headers.get("X-Breeze-Demo", "-")
    result.saved = response.headers.get("X-Breeze-Output", "-")

    if result.template != EXPECTED_TEMPLATE[mode]:
        result.fail(f"template {result.template!r}, expected {EXPECTED_TEMPLATE[mode]!r}")

    if result.demo != "0":
        result.fail("served demo audio, not the model (is the model server up?)")

    try:
        result.seconds, channels, sample_rate = wav_duration(response.content)
    except ValueError as exc:
        result.fail(f"bad WAV: {exc}")
        return result

    if channels != 1 or sample_rate != 24000:
        result.fail(f"{channels}ch {sample_rate}Hz, expected 1ch 24000Hz")
    # A runaway take is the failure mode to catch here: Breeze emits up to
    # MAX_NEW_TOKENS if it never finds EOS, which is ~47s of audio.
    if not 0.3 <= result.seconds <= 30:
        result.fail(f"implausible duration {result.seconds:.2f}s")
    if result.saved == "-":
        result.fail("no X-Breeze-Output header: the take was not saved")

    return result


def check_shelf(client: httpx.Client, base: str, results: list[Result]) -> list[str]:
    """The shelf must contain every take we just made, with matching metadata."""
    problems: list[str] = []
    listing = client.get(f"{base}/api/outputs").json()
    outputs = {o["name"]: o for o in listing.get("outputs") or []}
    for result in results:
        if result.saved == "-" or not result.ok:
            continue
        record = outputs.get(result.saved)
        if record is None:
            problems.append(f"{result.mode}: {result.saved} is not in GET /api/outputs")
            continue
        if record.get("template") != result.template:
            problems.append(
                f"{result.mode}: sidecar template {record.get('template')!r} "
                f"!= header {result.template!r}"
            )
        if record.get("mode") != result.mode:
            problems.append(
                f"{result.mode}: sidecar mode {record.get('mode')!r} != {result.mode!r}"
            )
        if abs((record.get("duration") or 0) - result.seconds) > 0.05:
            problems.append(
                f"{result.mode}: sidecar duration {record.get('duration')} "
                f"!= WAV {result.seconds:.2f}"
            )
        # A reference-less mode must not have recorded one, and vice versa.
        wants_reference = result.mode in ("clone", "direct")
        if bool(record.get("has_reference")) != wants_reference:
            problems.append(f"{result.mode}: has_reference is wrong")
    return problems


def check_stream(client: httpx.Client, base: str) -> tuple[bool, str]:
    """The streaming path must return PCM and still save the take."""
    before = len(client.get(f"{base}/api/outputs").json().get("outputs") or [])
    payload = {"text": "Streaming path check.", "cfg_scale": "1.0", "seed": "98"}
    with client.stream("POST", f"{base}/api/stream", data=payload) as response:
        if response.status_code != 200:
            return False, f"HTTP {response.status_code}"
        template = response.headers.get("X-Breeze-Template", "-")
        demo = response.headers.get("X-Breeze-Demo", "-")
        sample_rate = response.headers.get("X-Sample-Rate", "-")
        chunks = 0
        size = 0
        first_at = None
        for chunk in response.iter_bytes(4096):
            if chunk and first_at is None:
                first_at = time.perf_counter()
            chunks += 1
            size += len(chunk)
    after = client.get(f"{base}/api/outputs").json().get("outputs") or []
    size = size - (size % 2)

    if size == 0:
        return False, "no PCM returned"
    if size % 2:  # pragma: no cover - guarded by the subtraction above
        return False, "odd byte count is not s16le"
    if template != "tts_plain":
        return False, f"template {template!r}, expected 'tts_plain'"
    if demo != "0":
        return False, "served demo audio"
    if sample_rate != "24000":
        return False, f"X-Sample-Rate {sample_rate!r}"
    if len(after) != before + 1:
        return False, f"shelf grew by {len(after) - before}, expected 1"
    if not first_at:
        return False, "no timing observed"
    seconds = size / 2 / 24000
    return True, f"{chunks} chunks, {size / 1024:.0f} KiB, {seconds:.2f}s, saved"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--base", default="http://127.0.0.1:8000")
    parser.add_argument("--reference", type=Path, default=None, help="reference clip for clone/direction")
    parser.add_argument("--ref-text", default="", help="exact transcript of --reference")
    parser.add_argument("--only", nargs="+", choices=MODE_ORDER, default=list(MODE_ORDER))
    parser.add_argument("--skip-stream", action="store_true")
    args = parser.parse_args()

    base = args.base.rstrip("/")
    try:
        status = httpx.get(f"{base}/api/status", timeout=10).json()
    except Exception as exc:
        raise SystemExit(f"no app server at {base}: {exc}") from exc
    print(f"app    : {base}")
    print(f"model  : {'online' if status['connected'] else 'OFFLINE -- expecting demo audio'} "
          f"({status['upstream']}, {status.get('detail')})")

    with httpx.Client(timeout=900.0) as client:
        reference = None
        if any(m in args.only for m in ("clone", "direct")):
            reference = fetch_reference(client, base, args.reference, args.ref_text)
            print(f"voice  : {reference[0].name}")
            print(f"spoken : {reference[1]!r}")
        print()

        results: list[Result] = []
        for mode in args.only:
            result = run_mode(client, base, mode, reference)
            results.append(result)
            flag = "PASS" if result.ok else "FAIL"
            print(
                f"{flag}  {mode:<7} {result.template:<16} demo={result.demo} "
                f"{result.seconds:5.2f}s  {result.elapsed:6.1f}s wall  {result.saved}"
            )
            for note in result.notes:
                print(f"        - {note}")

        problems = check_shelf(client, base, results)
        print()
        print("output shelf:")
        for problem in problems:
            print(f"  FAIL {problem}")
        if not problems:
            checked = sum(1 for r in results if r.ok)
            print(f"  ok   {checked} take(s) present with matching metadata")

        stream_ok = True
        stream_detail = "skipped"
        if not args.skip_stream and "plain" in args.only:
            stream_ok, stream_detail = check_stream(client, base)
            print(f"streaming: {'PASS' if stream_ok else 'FAIL'}  {stream_detail}")

    failed = [r.mode for r in results if not r.ok] + problems
    print()
    if failed or not stream_ok:
        print(f"FAILED: {failed if failed else ''} {'' if stream_ok else 'streaming'}")
        return 1
    print("ALL MODES PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
