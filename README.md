# TostAI Voice Studio

A web app for **Breeze TTS 2** that exposes every capability the model has —
voice design, voice clone, voice direction, plain speech, bilingual EN/ZH
output, inline vocal events, seed/CFG control and true low-latency streaming —
behind one flat, light/dark interface. Every take is written to an output folder
and listed in the UI.

```
tostai-voice-studio/
├── server.py              FastAPI app: proxies Breeze, muxes WAV, output shelf
├── requirements.txt       Web-app dependencies (the model's deps live in ../breeze-tts)
├── Dockerfile             self-contained image: clones the code, pulls the weights
├── docker-entrypoint.sh   model server + studio in one container
├── docker_selfcheck.py    build-time proof that the image is complete
├── outputs/               every finished take: <name>.wav + <name>.json
└── static/
    ├── index.html         interface, with the pre-paint theme script
    ├── styles.css         both palettes, one set of variable names
    ├── app.js             streaming Web Audio player, visualiser, output shelf
    ├── logo.png           the TostAI mark from the toolbar's favicon
    └── favicon.ico        same mark, 16/32/48 px
```

## Run it

The app has its own virtualenv, so its four dependencies never touch the
interpreter the model runs on:

```bash
cd tostai-voice-studio
python -m venv .venv                                   # once
.venv/Scripts/python.exe -m pip install -r requirements.txt   # once (POSIX: .venv/bin/python)
.venv/Scripts/python.exe server.py --port 8000
```

Open <http://127.0.0.1:8000>.

### Give it a model

The app auto-detects the model server. To serve real audio, start the Breeze
inference API from the sibling `breeze-tts` checkout (needs a CUDA GPU):

```bash
cd ../breeze-tts
python -m breeze_infer.api ../breeze-tts-2 --host 127.0.0.1 --port 7860
# add --fast-all for the warmed CUDA-Graph fast path (~40s cold start)
```

The header badge flips from **demo audio (model offline)** to **model online**
within one of the app's 20-second status polls, or immediately if you click it.
Loaded on an RTX 3090 the eager path takes about 30 s to become healthy.

Point the app elsewhere with `--upstream http://host:port`.

### No GPU? It still works

If the model server is unreachable the app **falls back to a clearly-labelled
demo synth** so you can explore the whole interface; demo takes are badged
`demo` in the shelf and the player (`X-Breeze-Demo: 1`). Force it with
`--demo`, or make an unreachable model a hard 502 with `--no-demo-fallback`.

## The output folder

Every finished take is written to `outputs/` as a WAV **plus a JSON sidecar**
holding the text, instruction, mode, template, seed, CFG, sample rate, duration,
time-to-first-audio and the demo flag. The shelf in the UI is a *view of that
folder* — it is re-read from disk, not mirrored in the browser — so takes
survive a page reload, a server restart and a container restart, and you can
also just open the folder.

* WAV is mono 24 kHz 16-bit, exactly what the model emits.
* `<name>` is `breeze-<timestamp>-<mode>-seed<seed>.wav`.
* Override the location with `BREEZE_OUTPUTS_DIR` (the image mounts it).
* Delete a take with the ✕ in the shelf, or delete the pair by hand. A sidecar
  whose WAV has gone is skipped rather than shown as broken.

## Light and dark

Both themes use the same CSS variable names, so no rule knows which is active:
`:root` holds the light palette and `html[data-theme="dark"]` overrides it, and
`color-scheme` hands the browser's own controls over to the same decision.

The choice is made by an inline script in `index.html` **before the first
paint** — otherwise a dark page flashes the light palette and repaints. A stored
choice in `localStorage` (`tostai.voice.theme`) wins; with none, the OS decides. The
button names the theme it switches *to* ("Dark" on a light page), and the
canvas visualiser re-reads `--acc` whenever the theme changes, since a canvas
cannot use CSS variables.

The palette, the component shapes and the app icon are taken from
TostAI Sprite Studio (`ui.html`): the same `--bd/--tx/--mut/--acc/--accbg/`
`--acctext/--acch` names, the same flat 1px-bordered surfaces, the same `+`/`–`
`<details>` panels, and the same mark — `static/logo.png` is that page's inline
header logo and `static/favicon.ico` is that project's toolbar icon. To retheme,
edit the two variable blocks at the top of `styles.css` and nothing else.

## Docker

Pull the published image — no build, no model download, no tokens:

```bash
docker pull camenduru/tostai-voice-studio
docker run --rm --gpus all -p 8000:8000 camenduru/tostai-voice-studio
```

then open <http://127.0.0.1:8000>.

One self-contained image: it clones the inference code, clones the studio from
its own GitHub repo, and downloads the weights during the build, so it needs no
local model, and it runs both processes. To build it yourself:

```bash
cd tostai-voice-studio
docker build --build-arg CACHEBUST=$(date +%s) -t camenduru/tostai-voice-studio .
docker run --rm --gpus all -p 8000:8000 camenduru/tostai-voice-studio
```

The studio repository (`camenduru/TostAI-Voice-Studio`) is **private**, so the
build needs `GITHUB_TOKEN`; the inference and model repos are public, so
`HF_TOKEN` is optional. See [Configuration](#configuration) for the
`set -a; . ./.env` step and the token build command.

| | |
| --- | --- |
| Inference code | `git clone https://github.com/breezeblue-ai/breeze-tts` → `/app/breeze-tts` |
| Studio | `git clone https://github.com/camenduru/TostAI-Voice-Studio` → `/app/tostai-voice-studio` |
| Weights | `BreezeBlue/Breeze-TTS-2` (7.2 GB) → `/app/breeze-tts-2` |
| Studio ports | 8000 (UI), 7860 (model API, loopback only) |
| User | `camenduru` (non-root) |
| Python | `/opt/venv`, a virtualenv, used by both processes |

The build context is **this directory**, not the repo root — `.dockerignore`
keeps it to about 1 kB by excluding `.venv/`, caches and local audio. The
studio's files come from the clone, not the context; only `docker_selfcheck.py`
is copied from the context.

Notes worth knowing before you build:

* **`GITHUB_TOKEN` is required** for the private studio repo. `HF_TOKEN` is
  optional (the model repo is public). Both are supplied as secret mounts
  (`--secret id=hf_token,env=HF_TOKEN --secret id=gh_token,env=GITHUB_TOKEN`) —
  never as `ARG` or `ENV`, so they stay out of `docker history`.
* **`CACHEBUST` is not optional in practice.** BuildKit caches the studio clone
  under a key that ignores what the branch points at now, so without
  `--build-arg CACHEBUST=$(date +%s)` a rebuild silently re-serves the first
  snapshot. The flag only invalidates the clone and the cheap layers after it.
* **No CUDA toolkit is installed.** The pip torch wheels carry their own CUDA
  runtime; only the host driver is required, which is why `--gpus all` is the
  whole GPU story.
* The build runs `docker_selfcheck.py`, which asserts the checkout imports,
  every template is registered, both 3.6 GB shards are complete, torch is a
  CUDA build, and the studio answers its own routes. It does **not** load the
  model — a build must not depend on a GPU.
* `BREEZE_MODEL_FLAGS="--fast-all"` enables the fast path at run time;
  `BREEZE_SERVE_MODEL=0` makes the container UI-only, pointing at
  `BREEZE_API_URL` instead.

### Updating a running container

The **Update** button in the header pulls the latest studio source from
`camenduru/TostAI-Voice-Studio` and restarts the server in place, with no
rebuild. Type a GitHub token with read access to the repo when the dialog asks;
it is used for that one fetch and never written to disk. This is a dev
convenience — the files it writes live in the container and die with it. The
durable path is still a rebuild.

## Configuration

`.env` holds **credentials only** — `HF_TOKEN` and `GITHUB_TOKEN` — and nothing
else. It is gitignored and excluded from the Docker build context, so nothing in
it is committed or uploaded to the builder.

`GITHUB_TOKEN` is **required**: `camenduru/TostAI-Voice-Studio` is private, so
the Dockerfile clones it with that token. `HF_TOKEN` is **optional**:
`BreezeBlue/Breeze-TTS-2` and `breezeblue-ai/breeze-tts` are public, so they need
no credential. `GITHUB_TOKEN` also lifts GitHub's anonymous clone rate limit.

```bash
set -a; . ./.env; set +a        # `set -a` is required: it marks values for export
docker build \
  --secret id=hf_token,env=HF_TOKEN \
  --secret id=gh_token,env=GITHUB_TOKEN \
  --build-arg CACHEBUST=$(date +%s) \
  -t camenduru/tostai-voice-studio .
```

### Publishing to Docker Hub (`camenduru/tostai-voice-studio`)

```bash
docker login
docker build \
  --secret id=hf_token,env=HF_TOKEN \
  --secret id=gh_token,env=GITHUB_TOKEN \
  --build-arg CACHEBUST=$(date +%s) \
  -t camenduru/tostai-voice-studio:latest .
docker push camenduru/tostai-voice-studio:latest
# optional version tag:
# docker tag camenduru/tostai-voice-studio:latest camenduru/tostai-voice-studio:<version>
# docker push camenduru/tostai-voice-studio:<version>
```

The tokens arrive as secret mounts, never as `ARG` or `ENV`, so they stay out of
`docker history` and the image config. Each assignment is guarded as
`NAME=${NAME:-}`, so a value already in your environment wins — keep the guard
if you edit the file.

The app's own behaviour is set with **ordinary environment variables**, which
belong in your shell or in `docker run -e` rather than in a credentials file:

| Variable | Default | Effect |
| --- | --- | --- |
| `BREEZE_API_URL` | `http://127.0.0.1:7860` | Model server the studio talks to |
| `BREEZE_OUTPUTS_DIR` | `./outputs` | Where takes are written; point it at a volume |
| `BREEZE_SERVE_MODEL` | `1` | `0` runs the UI alone against `BREEZE_API_URL` |
| `BREEZE_MODEL_FLAGS` | *(empty)* | Extra flags for `breeze_infer.api`, e.g. `--fast-all` |
| `BREEZE_MODEL_PORT` | `7860` | Model server port (inside the container) |
| `BREEZE_STUDIO_PORT` | `8000` | UI port |
| `BREEZE_REV` | `main` | Checkpoint revision baked into the image (build-time) |
| `TOSTAI_APP_REPO` | `camenduru/TostAI-Voice-Studio` | Repo the studio's Update button pulls from |

An **empty** value is treated as unset by the studio, so `BREEZE_OUTPUTS_DIR=`
keeps the default folder.

## HTTP surface

| Method | Path | Purpose |
| --- | --- | --- |
| `GET` | `/` | The app |
| `GET` | `/api/catalog` | Modes, vocal events, presets, fast-path flags, model facts |
| `GET` | `/api/status` | Probes the Breeze server, reports sample rate + demo fallback |
| `POST` | `/api/generate` | Form in, **WAV** out (buffered); saves the take |
| `POST` | `/api/stream` | Form in, **streaming s16le PCM** out; saves the take |
| `GET` | `/api/outputs` | Every saved take, newest first, with its metadata |
| `GET` | `/api/outputs/{name}` | One saved take as a WAV |
| `DELETE` | `/api/outputs/{name}` | Remove a take (audio + sidecar) |
| `GET` | `/api/update` | The installed studio revision (for the Update dialog) |
| `POST` | `/api/update` | Pull latest source with a GitHub token, then restart |

Both POST endpoints accept `text`, `instruction`, `cfg_scale`, `seed`,
`ref_text`, `lang` and an optional `ref_audio` file, mirroring
`/v1/audio/speech` on the Breeze API.

## What the app covers

| Breeze feature | Where |
| --- | --- |
| **Plain TTS** (`tts_plain`) | *Plain TTS* mode |
| **Voice Design** (`tts_instruction`) | *Voice Design* mode + preset voice descriptions |
| **Voice Clone** (`ref_clone_tata`) | *Voice Clone* mode (reference + transcript) |
| **Voice Direction** (`ref_edit_tata`) | *Voice Direction* mode + delivery presets |
| **Vocal events** `(laugh)`, `(cough)`, `(clears throat)`, `(sigh)` | Insert chips; Chinese `[笑]`/`[咳嗽]`/`[清嗓子]`/`[叹气]` |
| **Bilingual EN / 中文** | Language toggle swaps examples, placeholders and event set |
| **Seed control** | Advanced panel, with randomiser |
| **CFG scale** | Advanced panel — locked to 1.0 for templates Breeze has no negative prompt for |
| **Real-time streaming PCM** | "Stream audio as it is generated" — off by default; enable for chunk-by-chunk Web Audio playback |
| **Buffered WAV export** | The default: streaming off returns one WAV, or use the WAV/PCM buttons and the shelf |
| **Fast-path flags** | Listed with each stage so you know what to pass to `breeze_infer.api` |
| **Model facts** | Header → *Model facts* (sample rate, format, GPU memory, TTFA, RTF, licence) |
| **In-place update** | Header → *Update* — pulls latest source with a GitHub token and restarts |

Also: latency stats (time to first audio, total, real-time factor), an audio
visualiser, a reference-waveform preview, a light/dark theme, and a shelf of
previous takes you can replay, download, reuse or delete.

## License

The app code here is yours to use. Breeze TTS 2 **model weights and
self-hosted outputs are research / non-commercial only** — see the licence in
`../breeze-tts-2`. Only clone voices you have consent to use.
