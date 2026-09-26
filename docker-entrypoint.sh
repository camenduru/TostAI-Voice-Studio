#!/usr/bin/env bash
#
# TostAI Voice Studio entrypoint.
#
# Two processes, one container:
#
#   breeze_infer.api   the model server, on loopback only (7860)
#   server.py          the studio, on 0.0.0.0 (8000)
#
# The studio is deliberately in the FOREGROUND: it is the process whose exit
# should end the container, and it is the one the HEALTHCHECK probes. When it
# goes down the container should die rather than keep a GPU pinned by an
# orphaned model server.
#
# BREEZE_SERVE_MODEL=0 skips the model server entirely, for the case where the
# model lives elsewhere and this container is only the UI. Point it at that
# server with BREEZE_API_URL (or --upstream, which this passes through).
set -euo pipefail

CODE_DIR="${BREEZE_CODE_DIR:-/app/breeze-tts}"
MODEL_DIR="${BREEZE_MODEL_DIR:-/app/breeze-tts-2}"
STUDIO_DIR="${BREEZE_STUDIO_DIR:-/app/breeze-app}"
MODEL_PORT="${BREEZE_MODEL_PORT:-7860}"
STUDIO_PORT="${BREEZE_STUDIO_PORT:-8000}"
UPSTREAM="${BREEZE_API_URL:-http://127.0.0.1:${MODEL_PORT}}"
# Word-split on purpose: this is how `--fast-all` gets through.
MODEL_FLAGS="${BREEZE_MODEL_FLAGS:-}"

model_pid=""
studio_pid=""

shutdown() {
  # SIGTERM the model server and let it release its CUDA context; without this
  # `docker stop` waits out the full timeout on every run.
  if [ -n "${model_pid}" ] && kill -0 "${model_pid}" 2>/dev/null; then
    kill "${model_pid}" 2>/dev/null || true
  fi
  if [ -n "${studio_pid}" ] && kill -0 "${studio_pid}" 2>/dev/null; then
    kill "${studio_pid}" 2>/dev/null || true
  fi
}
trap shutdown TERM INT EXIT

if [ "${BREEZE_SERVE_MODEL:-1}" = "1" ]; then
  if [ ! -d "${MODEL_DIR}" ]; then
    echo "studio: no model directory at ${MODEL_DIR}" >&2
    exit 1
  fi
  echo "studio: model server -> 127.0.0.1:${MODEL_PORT} (flags: ${MODEL_FLAGS:-none})"
  # `cd` because breeze_infer and models are top-level packages in that checkout.
  # The model is loaded during this process's startup, which is why the studio
  # is started immediately after instead of waiting: the UI comes up instantly
  # and reports the model as offline until /health answers.
  (
    cd "${CODE_DIR}"
    # shellcheck disable=SC2086
    exec python -m breeze_infer.api "${MODEL_DIR}" \
      --host 127.0.0.1 --port "${MODEL_PORT}" ${MODEL_FLAGS}
  ) &
  model_pid=$!
else
  echo "studio: BREEZE_SERVE_MODEL=0, expecting a model server at ${UPSTREAM}"
fi

echo "studio: UI -> 0.0.0.0:${STUDIO_PORT} (upstream ${UPSTREAM})"
cd "${STUDIO_DIR}"
python server.py --host 0.0.0.0 --port "${STUDIO_PORT}" --upstream "${UPSTREAM}" &
studio_pid=$!

# `wait` on a specific pid, not `wait -n`: the exit status here is the studio's,
# and that is what decides the container's.
wait "${studio_pid}"
