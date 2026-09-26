# syntax=docker/dockerfile:1.10
#
# 1.10, not 1.7: `env=` on a secret mount (used below to make an optional
# HF_TOKEN visible to one RUN) is rejected by older frontends.
#
# ===========================================================================
# TostAI Voice Studio -- self-contained image
#
# One stage, linear, everything fetched during the build: the inference code is
# cloned from GitHub, the weights are downloaded from HuggingFace, and the
# studio itself is cloned from its own GitHub repository. The build context
# supplies only docker_selfcheck.py.
#
#   docker build -t tostai-voice-studio .
#
#   docker run --rm --gpus all -p 8000:8000 tostai-voice-studio
#
# then open http://127.0.0.1:8000.
#
# THE BUILD CONTEXT IS THIS DIRECTORY (tostai-voice-studio/), not the repo root.
# The model repo beside it is 7.2 GB and the inference checkout is another clone
# -- neither is read from the context, so neither should be uploaded to the
# builder. `.dockerignore` therefore ignores everything except the studio's
# build-time proof.
#
# ---------------------------------------------------------------------------
# WHAT IS DOWNLOADED, AND WHERE IT LANDS
#
#   https://github.com/breezeblue-ai/breeze-tts   -> /app/breeze-tts
#   https://huggingface.co/BreezeBlue/Breeze-TTS-2 -> /app/breeze-tts-2
#   https://github.com/camenduru/TostAI-Voice-Studio -> /app/tostai-voice-studio
#
# Those three paths are what the app expects: the studio's `--upstream` is the
# model server on loopback, and `python -m breeze_infer.api` is run with the
# code directory as its working directory (it imports `breeze_infer` and
# `models` as top-level packages).
#
# ---------------------------------------------------------------------------
# TOKEN REQUIREMENTS
#
#   * BreezeBlue/Breeze-TTS-2 answers `resolve/main/config.json` with 200 and no
#     Authorization header, i.e. the model repo is public -- HF_TOKEN is
#     OPTIONAL.
#   * breezeblue-ai/breeze-tts is public, so that clone needs no credential.
#   * camenduru/TostAI-Voice-Studio is PRIVATE, so GITHUB_TOKEN is REQUIRED for
#     the studio clone below. The mount is marked `required=true` and carries a
#     `-z` guard, so a build without it fails with a clear message rather than a
#     git authentication error.
#
# GITHUB_TOKEN additionally lifts GitHub's anonymous rate limit on a build farm.
# Both are passed the same way the reference RunPod-style images do it:
#
#   docker build \
#     --secret id=hf_token,env=HF_TOKEN \
#     --secret id=gh_token,env=GITHUB_TOKEN \
#     -t tostai-voice-studio .
#
# Docker reads neither your shell environment nor `.env` on its own: the
# `env=NAME` on each `--secret` is what lifts the value out of the process
# environment the build is running in. That is why this directory ships a
# gitignored `.env` holding nothing but these two tokens, each guarded as
# NAME=${NAME:-} so a value already in your environment wins:
#
#   set -a; . ./.env; set +a
#
# `set -a` is required -- without it the values stay shell-local and the build
# sees nothing. `.env` is also in `.dockerignore`, so it never reaches the
# builder's context.
#
# `--secret ...,env=NAME` lifts the value out of the caller's environment and
# `--mount=type=secret,...,env=NAME` exposes it to that ONE RUN. It is NOT an
# ARG and NOT an ENV, so it never reaches `docker history` or `.Config.Env`,
# and it is gone from the next layer. Nothing is written to disk.
#
# `required=true` is deliberately NOT set: it would abort every build that does
# not pass the flag, including the public one that needs no token. The
# `-z` guard inside the RUN covers the other case -- flag passed, value empty.
#
# ---------------------------------------------------------------------------
# NO CUDA TOOLKIT, AND WHY THAT IS FINE
#
# There is no `cuda_*.run --silent --toolkit` in here: it is a ~4 GB download
# that inference does not need. The pip torch wheels bring their own CUDA
# runtime (nvidia-cuda-runtime-cu13, nvidia-cudnn-cu13, ...) and only the HOST
# driver has to exist. That is why `--gpus all` is the whole GPU story.
#
# The image is therefore arch-independent at BUILD time and needs sm70+ at RUN
# time. It was developed against torch 2.9.0+cu130 on an RTX 3090 (sm86) driven
# by driver 610.62; `--fast-all` uses CUDA Graphs and is NOT enabled by
# default, matching the upstream recommendation of the eager path.
#
# ---------------------------------------------------------------------------
# WHY THE PINS RESOLVE (checked against the indexes, not guessed)
#
#   torch==2.9.1 / torchaudio==2.9.1  -> both published as +cu130 wheels on
#     https://download.pytorch.org/whl/cu130, so `--extra-index-url` is enough.
#     --extra-index-url, NOT --index-url: that index hosts torch but not every
#     transitive dependency, so PyPI has to stay reachable. The +cu130 builds
#     sort above the plain PyPI ones, so pip picks them.
#   qwen-tts==0.1.1 -> on PyPI.
#
# numpy>=2.0 in the upstream requirements is a RANGE, not a pin, which matters
# on this base: ubuntu 22.04 ships Python 3.10 and numpy >= 2.3 refuses to
# install below 3.11, so pip resolves the newest 3.10-compatible 2.2.x on its
# own. No override needed -- unlike the reference image, whose explicit
# `numpy==2.3.4` had to be walked back to 2.2.6 for exactly this reason.
# ===========================================================================
FROM ubuntu:22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=True \
    PYTHONDONTWRITEBYTECODE=True

# ---------------------------------------------------------------------------
# System packages and a non-root user
#
# ffmpeg is here for torchaudio/soundfile format coverage -- reference audio can
# arrive as mp3/m4a, not just wav. git-lfs because a checkout that declares LFS
# filters can fail on the smudge step without it.
#
# build-essential is deliberately absent: every wheel below is prebuilt, and it
# is ~200 MB of compiler nobody uses at runtime.
# ---------------------------------------------------------------------------
RUN apt-get update && apt-get install -y --no-install-recommends \
        python3 python3-pip python3-venv python3-dev \
        git git-lfs curl ca-certificates ffmpeg \
        tini \
    && rm -rf /var/lib/apt/lists/* \
    && useradd --create-home --shell /bin/bash camenduru \
    && mkdir -p /app /opt \
    && chown -R camenduru:camenduru /app /opt

# ---------------------------------------------------------------------------
# The virtualenv, used by EVERY python process in this image
#
# Both the model server and the studio run out of /opt/venv/bin, and PATH is
# set once here so nothing downstream has to remember. Keeping it out of the
# system interpreter also means `docker run --rm tostai-voice-studio python ...` gets
# the same environment the entrypoint does.
# ---------------------------------------------------------------------------
RUN python3 -m venv /opt/venv \
    && /opt/venv/bin/python -m pip install --no-cache-dir --upgrade pip

ENV PATH="/opt/venv/bin:${PATH}" \
    VIRTUAL_ENV=/opt/venv \
    BREEZE_CODE_DIR=/app/breeze-tts \
    BREEZE_MODEL_DIR=/app/breeze-tts-2 \
    BREEZE_API_URL=http://127.0.0.1:7860

# ---------------------------------------------------------------------------
# The inference checkout
#
# Placed above the torch install so that a change to the code does not force a
# ~2.5 GB torch re-download: this layer is a few MB and rebuilds in seconds.
# ---------------------------------------------------------------------------
# The token, when supplied, goes into the clone's remote URL rather than an
# extraheader -- and `.git` is removed in THIS SAME RUN, so the credentialed URL
# never reaches a layer. The instruction text in `docker history` shows the
# shell variable, not the value, because the value arrives from the secret mount
# at build time and is never interpolated by the Dockerfile parser.
RUN --mount=type=secret,id=gh_token,env=GITHUB_TOKEN \
    set -eu; \
    if [ -n "${GITHUB_TOKEN:-}" ]; then \
        echo "cloning with a GITHUB_TOKEN"; \
        url="https://x-access-token:${GITHUB_TOKEN}@github.com/breezeblue-ai/breeze-tts.git"; \
    else \
        echo "cloning anonymously (the repo is public)"; \
        url="https://github.com/breezeblue-ai/breeze-tts.git"; \
    fi; \
    git clone --depth 1 "$url" /app/breeze-tts; \
    rm -rf /app/breeze-tts/.git; \
    test -f /app/breeze-tts/breeze_infer/api.py

# ---------------------------------------------------------------------------
# Dependencies, split so the big one caches on its own
#
# torch + torchaudio first, in their own layer (~2.5 GB): a change to any pin
# below must not re-download them. Then the repo's requirements, whose torch
# lines are already satisfied and therefore no-ops.
#
# httpx and python-multipart are added on top of requirements.txt: the studio
# proxies multipart uploads to the model server, and requirements.txt does not
# list httpx (nothing upstream calls out over HTTP).
# ---------------------------------------------------------------------------
RUN --mount=type=cache,target=/root/.cache/pip \
    /opt/venv/bin/python -m pip install \
        --extra-index-url https://download.pytorch.org/whl/cu130 \
        "torch==2.9.1" "torchaudio==2.9.1"

RUN --mount=type=cache,target=/root/.cache/pip \
    /opt/venv/bin/python -m pip install \
        --extra-index-url https://download.pytorch.org/whl/cu130 \
        -r /app/breeze-tts/requirements.txt \
        "httpx>=0.27"

# ---------------------------------------------------------------------------
# The weights
#
# Only the files inference reads are fetched. LICENSE, README.md and assets/
# are skipped: they are not opened at runtime, and assets/ alone is a handful
# of megabytes of README images.
#
# snapshot_download, rather than a list of aria2c/curl calls: it comes with
# transformers (already installed), it resumes, and it verifies. The reference
# image needed hand-rolled retries around aria2c because a single transient TLS
# reset threw away a build that had already paid for apt, torch and a gigabyte
# of CUDA wheels; this does not have that failure mode.
#
# The revision is pinned rather than tracking main, so an image rebuilt in six
# months is the same model. Override with --build-arg BREEZE_REV=<sha|branch>.
#
# *_OFFLINE is set AFTER the download, for the obvious reason: it is what stops
# the running container from reaching the Hub at all.
# ---------------------------------------------------------------------------
ARG BREEZE_REV=main

RUN --mount=type=secret,id=hf_token,env=HF_TOKEN \
    set -eu; \
    if [ -n "${HF_TOKEN:-}" ]; then echo "using the supplied HF_TOKEN"; else echo "no HF_TOKEN: expecting a public repo"; fi; \
    HF_TOKEN="${HF_TOKEN:-}" /opt/venv/bin/python - <<'PY'
import os
from huggingface_hub import snapshot_download

path = snapshot_download(
    repo_id="BreezeBlue/Breeze-TTS-2",
    revision=os.environ.get("BREEZE_REV", "main"),
    local_dir="/app/breeze-tts-2",
    allow_patterns=[
        "*.json",
        "tokenizer*",
        "*.safetensors",
        "audio_tokenizer/*",
    ],
    token=os.environ.get("HF_TOKEN") or None,
    max_workers=4,
)
print("downloaded to", path)
PY

ENV HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1

# ---------------------------------------------------------------------------
# The studio -- cloned from its own repository
#
#   https://github.com/camenduru/TostAI-Voice-Studio -> /app/tostai-voice-studio
#
# Cloned rather than copied out of the build context, so the image is
# reproducible from the repository alone. The token goes into the clone URL, so
# `.git` is removed in THIS SAME RUN and the credentialed URL never reaches a
# layer. The URL form (x-access-token:<token>@github.com) works for both classic
# PATs and `gh` OAuth tokens.
#
# CACHEBUST IS NOT OPTIONAL IN PRACTICE. BuildKit caches this clone under a key
# that ignores what the branch points at now, so without the flag a rebuild
# silently re-serves the first snapshot. Pass `--build-arg CACHEBUST=$(date
# +%s)`; it invalidates only the clone and the cheap layers after it, so the
# apt/pip/weights layers stay cached. If the image id does not change after a
# rebuild, nothing was rebuilt.
#
# The resolved commit is written to .tostai_rev so the running app can report
# what it is (GET /api/update). `docker_selfcheck.py` is copied from the build
# context rather than taken from the clone, so the build-time proof may be newer
# than what is committed.
#
# `--chown` is load-bearing, not tidiness: `USER camenduru` is set below, and a
# root-owned COPY turned the reference image's update endpoint into a 500 that
# the UI could not even parse.
#
# outputs/ is created here so the container starts with the folder present even
# when the app is pointed at BREEZE_OUTPUTS_DIR elsewhere.
# ---------------------------------------------------------------------------
ARG CACHEBUST=0

RUN --mount=type=secret,id=gh_token,env=GITHUB_TOKEN,required=true \
    set -eu; \
    if [ -z "${GITHUB_TOKEN:-}" ]; then echo "GITHUB_TOKEN is empty" >&2; exit 1; fi; \
    git clone --depth 1 \
      "https://x-access-token:${GITHUB_TOKEN}@github.com/camenduru/TostAI-Voice-Studio.git" \
      /app/tostai-voice-studio; \
    git -C /app/tostai-voice-studio rev-parse HEAD > /app/tostai-voice-studio/.tostai_rev; \
    rm -rf /app/tostai-voice-studio/.git; \
    test -f /app/tostai-voice-studio/server.py; \
    echo "cloned camenduru/TostAI-Voice-Studio at $(cat /app/tostai-voice-studio/.tostai_rev)"

COPY --chown=camenduru:camenduru docker_selfcheck.py /app/tostai-voice-studio/docker_selfcheck.py

# smoke_modes.py comes with the clone so a running container can be verified in
# place:  docker exec <id> python /app/tostai-voice-studio/smoke_modes.py
# It drives the studio over HTTP and needs no GPU of its own.
WORKDIR /app/tostai-voice-studio
RUN chmod +x /app/tostai-voice-studio/docker-entrypoint.sh \
    && mkdir -p /app/tostai-voice-studio/outputs \
    && chown -R camenduru:camenduru /app/tostai-voice-studio

USER camenduru

# ---------------------------------------------------------------------------
# Build-time proof
#
# Asserts the code imports, every template is registered, the two shards really
# arrived (a truncated download is the failure this catches), and the studio
# answers its own routes. It does NOT load the model: that needs a GPU, and a
# build must not depend on one. The model is exercised on first request
# instead, which is what the HEALTHCHECK's start-period covers.
# ---------------------------------------------------------------------------
RUN /opt/venv/bin/python /app/tostai-voice-studio/docker_selfcheck.py

EXPOSE 8000 7860

# tini reaps the backgrounded model server; without an init, PID 1's orphaned
# children accumulate and `docker stop` can hang on the shutdown path.
ENTRYPOINT ["/usr/bin/tini", "--"]

# The health endpoint never touches the GPU: /api/status reports the model's
# reachability rather than requiring it, so a container with a still-loading
# model is "healthy" and the UI says "model offline" instead of the healthcheck
# flapping. start-period covers interpreter + server startup.
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/api/status', timeout=4)"

CMD ["/app/tostai-voice-studio/docker-entrypoint.sh"]
