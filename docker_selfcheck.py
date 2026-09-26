"""Build-time proof for the TostAI Voice Studio image.

Run by the Dockerfile after everything has landed, so a truncated weight
download or a broken clone fails the BUILD rather than 500ing on the first
palette click. Returns non-zero on the first category of failure that matters.

Two things it deliberately does NOT do:

* load the model or touch the GPU. A build must not depend on one, and
  `load_runtime` would exit the build on a machine without a driver. The model
  is exercised on the first real request instead -- which is what the
  HEALTHCHECK's start-period is for.
* reach the network. The image is built with HF_HUB_OFFLINE=1, and a check that
  phones home would pass in CI and fail in an air-gapped rebuild.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

CODE_DIR = Path(os.environ.get("BREEZE_CODE_DIR", "/app/breeze-tts"))
MODEL_DIR = Path(os.environ.get("BREEZE_MODEL_DIR", "/app/breeze-tts-2"))
STUDIO_DIR = Path(__file__).resolve().parent

REQUIRED_CHECKPOINT_FILES = (
    "config.json",
    "generation_config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "model.safetensors.index.json",
)
EXPECTED_TEMPLATES = {
    "tts_plain",
    "tts_instruction",
    "ref_clone_tata",
    "ref_edit_tata",
}
MIN_CHECKPOINT_BYTES = 5 * 1024**3  # the published checkpoint is ~7.2 GB

failures: list[str] = []


def check(label: str, ok: bool, detail: str = "") -> None:
    suffix = f" -- {detail}" if detail else ""
    print(f"  {'ok  ' if ok else 'FAIL'} {label}{suffix}")
    if not ok:
        failures.append(label)


print("TostAI Voice Studio self-check")
print("=" * 60)

# --------------------------------------------------------------------------- #
# 1. The inference checkout
# --------------------------------------------------------------------------- #
check("inference checkout present", CODE_DIR.is_dir(), str(CODE_DIR))
sys.path.insert(0, str(CODE_DIR))
try:
    from breeze_infer.templates import TEMPLATES, select_template_name

    check(
        "breeze_infer.templates imports",
        set(TEMPLATES) == EXPECTED_TEMPLATES,
        ", ".join(sorted(TEMPLATES)),
    )
    # The studio's mode routing mirrors this function; if upstream ever renames
    # a template, that has to be noticed here rather than in the UI.
    check(
        "template routing agrees with the studio",
        select_template_name({"text": "x", "instruction": "y"}) == "tts_instruction"
        and select_template_name({"text": "x"}) == "tts_plain",
    )
except Exception as exc:  # noqa: BLE001 - reported, then the build stops
    check("breeze_infer.templates imports", False, repr(exc))

# --------------------------------------------------------------------------- #
# 2. The weights -- the failure this is really here for
# --------------------------------------------------------------------------- #
for name in REQUIRED_CHECKPOINT_FILES:
    check(f"checkpoint file {name}", (MODEL_DIR / name).is_file())

index_path = MODEL_DIR / "model.safetensors.index.json"
if index_path.is_file():
    try:
        shards = sorted(set(json.loads(index_path.read_text())["weight_map"].values()))
    except Exception as exc:  # noqa: BLE001
        shards = []
        check("shard index parses", False, repr(exc))
    total = 0
    for shard in shards:
        path = MODEL_DIR / shard
        size = path.stat().st_size if path.is_file() else 0
        total += size
        # A truncated shard is the classic silent failure: the file exists, the
        # import works, and loading dies much later with a cryptic error.
        check(f"shard {shard} complete", size > 1024**3, f"{size / 1024**3:.2f} GiB")
    check(
        f"checkpoint total (>= {MIN_CHECKPOINT_BYTES / 1024**3:.0f} GiB)",
        total >= MIN_CHECKPOINT_BYTES,
        f"{total / 1024**3:.2f} GiB",
    )

tokenizer_dir = MODEL_DIR / "audio_tokenizer"
check(
    "audio tokenizer weights",
    tokenizer_dir.is_dir()
    and (tokenizer_dir / "model.safetensors").is_file()
    and (tokenizer_dir / "config.json").is_file(),
    str(tokenizer_dir),
)

# --------------------------------------------------------------------------- #
# 3. The CUDA build of torch
# --------------------------------------------------------------------------- #
try:
    import torch

    check("torch imports", True, torch.__version__)
    check(
        "torch is a CUDA build",
        torch.version.cuda is not None,
        f"cuda {torch.version.cuda}",
    )
except Exception as exc:  # noqa: BLE001
    check("torch imports", False, repr(exc))

# --------------------------------------------------------------------------- #
# 4. The studio answers its own routes
# --------------------------------------------------------------------------- #
try:
    from fastapi.testclient import TestClient

    sys.path.insert(0, str(STUDIO_DIR))
    import server as studio

    with TestClient(studio.app) as client:
        home = client.get("/")
        check("GET /", home.status_code == 200, str(home.status_code))
        catalog = client.get("/api/catalog").json()
        check(
            "GET /api/catalog",
            len(catalog["modes"]) == 4,
            f"{len(catalog['modes'])} modes, {len(catalog['vocal_events']['en'])} events",
        )
        check("catalog carries the model facts", catalog["model"]["sample_rate"] == 24000)
        # /api/status reports the model's reachability rather than requiring it,
        # so it must answer 200 even with nothing listening on 7860.
        status = client.get("/api/status")
        check("GET /api/status", status.status_code == 200, str(status.status_code))
        outputs = client.get("/api/outputs").json()
        check("GET /api/outputs", "outputs" in outputs, f"dir={outputs.get('dir')}")
        # A traversal attempt must not resolve to a file outside outputs/.
        check(
            "output names are guarded",
            client.get("/api/outputs/..%2Fserver.py").status_code in (400, 404),
        )
except Exception as exc:  # noqa: BLE001
    check("studio routes", False, repr(exc))

print("=" * 60)
if failures:
    print(f"SELF-CHECK FAILED: {len(failures)} problem(s):")
    for failure in failures:
        print(f"  - {failure}")
    sys.exit(1)
print("SELF-CHECK PASSED")
