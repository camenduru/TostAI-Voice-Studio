/* TostAI Voice Studio — front-end for the Breeze TTS 2 web app. */
'use strict';

const $ = (sel) => document.querySelector(sel);
const $$ = (sel) => Array.from(document.querySelectorAll(sel));

/* ────────────────────────── theme ────────────────────────── */

// The palette itself lives in styles.css: :root is light and
// html[data-theme="dark"] overrides the same variable names. The inline script
// in index.html has already picked one before the first paint, so setTheme()
// here only has to keep the button honest and re-read the few colours the
// canvas needs (a canvas cannot use CSS variables).
const THEME_KEY = 'tostai.voice.theme';
// A canvas cannot use CSS variables, so the one accent it draws with is read
// back out of the stylesheet whenever the theme changes. One colour, not a
// gradient: this UI is flat, and so is the visualiser.
const themeColor = { acc: [14, 116, 144] };

function hexToRgb(value) {
  const hex = (value || '').trim().replace('#', '');
  if (hex.length === 3) {
    return [0, 1, 2].map((i) => parseInt(hex[i] + hex[i], 16));
  }
  if (hex.length >= 6) {
    return [0, 2, 4].map((i) => parseInt(hex.slice(i, i + 2), 16));
  }
  return null;
}

function rgba(triple, alpha) {
  return `rgba(${triple[0]}, ${triple[1]}, ${triple[2]}, ${alpha})`;
}

function refreshThemeColors() {
  const rgb = hexToRgb(getComputedStyle(document.documentElement).getPropertyValue('--acc'));
  if (rgb && rgb.every((n) => Number.isFinite(n))) themeColor.acc = rgb;
}

function setTheme(theme) {
  document.documentElement.dataset.theme = theme;
  try {
    localStorage.setItem(THEME_KEY, theme);
  } catch (err) {
    /* private mode: the choice just will not persist */
  }
  refreshThemeColors();
  const button = $('#theme-toggle');
  // The label names the theme it switches TO, so "Dark" on a light page can
  // never be read as "the page is dark".
  button.textContent = theme === 'dark' ? 'Light' : 'Dark';
  button.title = theme === 'dark' ? 'switch to the light theme' : 'switch to the dark theme';
}

function toggleTheme() {
  setTheme(document.documentElement.dataset.theme === 'dark' ? 'light' : 'dark');
}

const state = {
  catalog: null,
  mode: 'design',
  lang: 'en',
  refFile: null,
  abort: null,
  busy: false,
  audio: null, // { float32, sampleRate }
  outputs: [],
  activeOutput: null,
  streaming: false,
};

// One audio element for the whole shelf: starting a take must stop the previous
// one, and a per-row element cannot know about the others.
let shelfAudio = null;
let shelfPlaying = null;

/* ────────────────────────── audio engine ────────────────────────── */

const engine = {
  ctx: null,
  analyser: null,
  master: null,
  sources: new Set(),
  liveChunks: [],
  nextTime: 0,
  liveStart: 0,
  liveEnd: 0,
  buffer: null,
  playing: false,
  startCtxTime: 0,
  startOffset: 0,
  pauseOffset: 0,
  current: null,
  sampleRate: 24000,
  requested: null,
};

function ensureCtx(sampleRate) {
  if (engine.ctx && (!sampleRate || engine.requested === sampleRate)) {
    return engine.ctx;
  }
  if (engine.ctx) engine.ctx.close();
  let ctx;
  try {
    ctx = sampleRate ? new AudioContext({ sampleRate }) : new AudioContext();
  } catch (err) {
    ctx = new AudioContext();
  }
  engine.requested = sampleRate || ctx.sampleRate;
  const analyser = ctx.createAnalyser();
  analyser.fftSize = 512;
  analyser.smoothingTimeConstant = 0.72;
  const master = ctx.createGain();
  master.gain.value = 0.92;
  master.connect(analyser);
  analyser.connect(ctx.destination);
  engine.ctx = ctx;
  engine.analyser = analyser;
  engine.master = master;
  engine.sampleRate = ctx.sampleRate;
  engine.sources = new Set();
  return ctx;
}

function resample(input, from, to) {
  if (from === to) return input;
  const ratio = to / from;
  const length = Math.max(1, Math.round(input.length * ratio));
  const output = new Float32Array(length);
  for (let i = 0; i < length; i++) {
    const pos = i / ratio;
    const lo = Math.floor(pos);
    const hi = Math.min(lo + 1, input.length - 1);
    const frac = pos - lo;
    output[i] = input[lo] * (1 - frac) + input[hi] * frac;
  }
  return output;
}

function stopSources() {
  for (const source of engine.sources) {
    try {
      source.onended = null;
      source.stop();
    } catch (err) {
      /* already stopped */
    }
  }
  engine.sources.clear();
}

function resetEngine(sampleRate) {
  stopSources();
  engine.buffer = null;
  engine.playing = false;
  engine.pauseOffset = 0;
  engine.liveChunks = [];
  engine.liveStart = 0;
  engine.liveEnd = 0;
  engine.nextTime = 0;
  engine.current = null;
  ensureCtx(sampleRate || engine.sampleRate);
}

function pushLive(float32, sampleRate) {
  const ctx = ensureCtx(sampleRate);
  if (ctx.state === 'suspended') ctx.resume();
  const data = resample(float32, sampleRate, ctx.sampleRate);
  engine.liveChunks.push(data);
  const buffer = ctx.createBuffer(1, data.length, ctx.sampleRate);
  buffer.copyToChannel(data, 0);
  const source = ctx.createBufferSource();
  source.buffer = buffer;
  source.connect(engine.master);
  const startAt = Math.max(ctx.currentTime + 0.04, engine.nextTime);
  source.start(startAt);
  if (!engine.liveStart) engine.liveStart = startAt;
  engine.nextTime = startAt + buffer.duration;
  engine.liveEnd = engine.nextTime;
  engine.sources.add(source);
  source.onended = () => engine.sources.delete(source);
}

function makeBuffer(float32, sampleRate) {
  const ctx = ensureCtx(sampleRate);
  const data = resample(float32, sampleRate, ctx.sampleRate);
  const buffer = ctx.createBuffer(1, data.length, ctx.sampleRate);
  buffer.copyToChannel(data, 0);
  return buffer;
}

function playBuffer(offset) {
  if (!engine.buffer) return;
  const ctx = ensureCtx();
  ctx.resume();
  stopSources();
  const start = Math.min(Math.max(0, offset || 0), Math.max(0, engine.buffer.duration - 0.02));
  const source = ctx.createBufferSource();
  source.buffer = engine.buffer;
  source.connect(engine.master);
  source.onended = () => {
    engine.sources.delete(source);
    if (engine.current === source) {
      engine.playing = false;
      engine.pauseOffset = 0;
      engine.current = null;
      syncTransport();
    }
  };
  source.start(0, start);
  engine.sources.add(source);
  engine.current = source;
  engine.playing = true;
  engine.startCtxTime = ctx.currentTime;
  engine.startOffset = start;
  syncTransport();
}

function pauseBuffer() {
  if (!engine.playing) return;
  engine.pauseOffset = position();
  const source = engine.current;
  engine.playing = false;
  engine.current = null;
  if (source) {
    try {
      source.onended = null;
      source.stop();
    } catch (err) {
      /* ignore */
    }
    engine.sources.delete(source);
  }
  syncTransport();
}

function position() {
  if (!engine.ctx || !engine.buffer) return engine.pauseOffset || 0;
  if (engine.playing && engine.current) {
    return Math.min(
      engine.buffer.duration,
      engine.startOffset + (engine.ctx.currentTime - engine.startCtxTime)
    );
  }
  return Math.min(engine.pauseOffset || 0, engine.buffer.duration);
}

/* ────────────────────────── encoding helpers ────────────────────────── */

function encodeWav(float32, sampleRate) {
  const length = float32.length;
  const view = new DataView(new ArrayBuffer(44 + length * 2));
  const write = (offset, text) => {
    for (let i = 0; i < text.length; i++) view.setUint8(offset + i, text.charCodeAt(i));
  };
  write(0, 'RIFF');
  view.setUint32(4, 36 + length * 2, true);
  write(8, 'WAVE');
  write(12, 'fmt ');
  view.setUint32(16, 16, true);
  view.setUint16(20, 1, true);
  view.setUint16(22, 1, true);
  view.setUint32(24, sampleRate, true);
  view.setUint32(28, sampleRate * 2, true);
  view.setUint16(32, 2, true);
  view.setUint16(34, 16, true);
  write(36, 'data');
  view.setUint32(40, length * 2, true);
  let offset = 44;
  for (let i = 0; i < length; i++) {
    const s = Math.max(-1, Math.min(1, float32[i]));
    view.setInt16(offset, s < 0 ? s * 0x8000 : s * 0x7fff, true);
    offset += 2;
  }
  return new Blob([view.buffer], { type: 'audio/wav' });
}

function encodePcm(float32) {
  const out = new DataView(new ArrayBuffer(float32.length * 2));
  for (let i = 0; i < float32.length; i++) {
    const s = Math.max(-1, Math.min(1, float32[i]));
    out.setInt16(i * 2, s < 0 ? s * 0x8000 : s * 0x7fff, true);
  }
  return new Blob([out.buffer], { type: 'audio/pcm' });
}

function formatTime(seconds) {
  if (!Number.isFinite(seconds) || seconds < 0) seconds = 0;
  const m = Math.floor(seconds / 60);
  const s = Math.floor(seconds % 60);
  return `${m}:${String(s).padStart(2, '0')}`;
}

function download(source, name) {
  // A saved take is already a URL on the server; a fresh one is still a Blob.
  const objectUrl = typeof source === 'string';
  const url = objectUrl ? source : URL.createObjectURL(source);
  const a = document.createElement('a');
  a.href = url;
  a.download = name;
  document.body.appendChild(a);
  a.click();
  a.remove();
  if (!objectUrl) setTimeout(() => URL.revokeObjectURL(url), 4000);
}

/* ────────────────────────── catalog rendering ────────────────────────── */

function modeById(id) {
  return state.catalog.modes.find((m) => m.id === id) || state.catalog.modes[0];
}

function renderModes() {
  const container = $('#modes');
  container.innerHTML = '';
  for (const mode of state.catalog.modes) {
    const button = document.createElement('button');
    button.type = 'button';
    button.className = 'mode-tab' + (mode.id === state.mode ? ' is-active' : '');
    button.dataset.mode = mode.id;
    button.setAttribute('role', 'tab');
    button.innerHTML = `<strong>${mode.name}</strong><span>${mode.tagline}</span>`;
    button.addEventListener('click', () => selectMode(mode.id));
    container.appendChild(button);
  }
}

function renderEventChips() {
  const container = $('#event-chips');
  container.innerHTML = '';
  const events = state.catalog.vocal_events[state.lang] || [];
  for (const event of events) {
    const button = document.createElement('button');
    button.type = 'button';
    button.className = 'chip-btn mono';
    button.textContent = event.token;
    button.title = `Insert ${event.label} at the cursor`;
    button.addEventListener('click', () => insertAtCursor($('#text'), `${event.token} `));
    container.appendChild(button);
  }
}

function renderPresets() {
  const container = $('#preset-chips');
  container.innerHTML = '';
  const mode = modeById(state.mode);
  const source =
    mode.id === 'clone'
      ? []
      : mode.id === 'direct'
        ? state.catalog.direction_presets[state.lang]
        : state.catalog.voice_presets[state.lang];
  for (const preset of source || []) {
    const button = document.createElement('button');
    button.type = 'button';
    button.className = 'chip-btn';
    button.textContent = preset.label;
    button.title = preset.instruction;
    button.addEventListener('click', () => {
      $('#instruction').value = preset.instruction;
    });
    container.appendChild(button);
  }
}

function renderFastStages() {
  const list = $('#fast-stages');
  list.innerHTML = '';
  for (const stage of state.catalog.fast_stages) {
    const li = document.createElement('li');
    li.innerHTML = `<b>${stage.stage}</b> <code>${stage.flag}</code> <span>${stage.detail}</span>`;
    list.appendChild(li);
  }
}

function renderFacts() {
  const model = state.catalog.model;
  const body = $('#facts-body');
  const rows = [
    ['Model', model.name],
    ['Sample rate', `${model.sample_rate} Hz`],
    ['Channels / format', `${model.channels} · ${model.format}`],
    ['Languages', model.languages.join(', ')],
    ['GPU memory (eager)', `${model.gpu_memory_gib.eager} GiB`],
    ['GPU memory (fast path)', `${model.gpu_memory_gib.fast_all} GiB`],
    ['Time to first audio', `~${model.ttfa_ms} ms (H100, warmed)`],
    ['Real-time factor', `${model.rtf}× real time`],
    ['License', model.license],
  ];
  body.innerHTML = rows
    .map(([k, v]) => `<div class="fact-row"><span>${k}</span><span>${v}</span></div>`)
    .join('');
}

/* ────────────────────────── mode / language ────────────────────────── */

function selectMode(id) {
  state.mode = id;
  const mode = modeById(id);
  $$('.mode-tab').forEach((tab) => tab.classList.toggle('is-active', tab.dataset.mode === id));
  $('#mode-hint').textContent = mode.hint;
  $('#instruction-field').hidden = !mode.needs_instruction;
  $('#reference-field').hidden = !mode.needs_reference;
  $('#instruction-tag').textContent = mode.template === 'ref_edit_tata' ? 'direction' : 'design';

  const cfg = $('#cfg');
  const usesCfg = mode.template === 'tts_instruction' || mode.template === 'ref_edit_tata';
  cfg.disabled = !usesCfg;
  cfg.value = usesCfg ? String(mode.default_cfg) : '1';
  $('#cfg-value').textContent = Number(cfg.value).toFixed(1);
  $('#cfg-note').textContent = usesCfg
    ? '4.0 strengthens instruction-following.'
    : 'Breeze ignores CFG here — kept at 1.0.';
  renderPresets();
  $('#meta-template').textContent = mode.template;
}

function selectLang(lang) {
  const previous = state.lang;
  state.lang = lang;
  $$('.lang-btn').forEach((btn) => btn.classList.toggle('is-active', btn.dataset.lang === lang));
  renderEventChips();
  renderPresets();

  const example = state.catalog.examples[lang];
  const priorExample = state.catalog.examples[previous];
  const text = $('#text');
  const instruction = $('#instruction');
  if (!text.value.trim() || text.value.trim() === (priorExample.text || '').trim()) {
    text.value = example.text;
  }
  if (!instruction.value.trim() || instruction.value.trim() === (priorExample.design || '').trim()) {
    instruction.value = example.design;
  }
  $('#text').placeholder = lang === 'zh' ? '输入要朗读的文字…' : 'Type what the voice should say…';
  $('#ref-text').placeholder =
    lang === 'zh'
      ? '参考音频的准确文字稿（必须与音频完全一致）'
      : 'Exact transcript of the reference clip (must match the audio)';
  updateCounter();
}

function insertAtCursor(field, text) {
  const start = field.selectionStart ?? field.value.length;
  const end = field.selectionEnd ?? field.value.length;
  field.value = field.value.slice(0, start) + text + field.value.slice(end);
  field.focus();
  field.selectionStart = field.selectionEnd = start + text.length;
  updateCounter();
}

function updateCounter() {
  const text = $('#text').value;
  $('#text-count').textContent = `${text.length} chars`;
}

/* ────────────────────────── reference audio ────────────────────────── */

async function setReference(file) {
  if (!file) return;
  if (!file.type.startsWith('audio/') && !/\.(wav|mp3|flac|ogg|m4a|opus|webm)$/i.test(file.name)) {
    showError('That file does not look like audio.');
    return;
  }
  state.refFile = file;
  $('#ref-name').textContent = file.name;
  $('#ref-loaded').hidden = false;
  const preview = $('#ref-preview');
  preview.hidden = false;
  preview.src = URL.createObjectURL(file);
  try {
    const ctx = ensureCtx();
    const data = await file.arrayBuffer();
    const buffer = await ctx.decodeAudioData(data.slice(0));
    $('#ref-duration').textContent = `${buffer.duration.toFixed(2)}s · ${buffer.sampleRate} Hz`;
    drawReferenceWave(buffer);
  } catch (err) {
    $('#ref-duration').textContent = 'clip loaded';
  }
}

function clearReference() {
  state.refFile = null;
  $('#ref-input').value = '';
  $('#ref-loaded').hidden = true;
  $('#ref-preview').hidden = true;
  $('#ref-preview').src = '';
  $('#ref-duration').textContent = 'no file';
  $('#ref-wave').classList.remove('is-visible');
}

function drawReferenceWave(buffer) {
  const canvas = $('#ref-wave');
  const ratio = window.devicePixelRatio || 1;
  const width = canvas.clientWidth || 400;
  canvas.width = width * ratio;
  canvas.height = 48 * ratio;
  const ctx = canvas.getContext('2d');
  ctx.scale(ratio, ratio);
  const data = buffer.getChannelData(0);
  const bars = Math.max(40, Math.floor(width / 3));
  const step = Math.max(1, Math.floor(data.length / bars));
  const height = 48;
  ctx.clearRect(0, 0, width, height);
  for (let i = 0; i < bars; i++) {
    let peak = 0;
    const start = i * step;
    for (let j = start; j < start + step && j < data.length; j += 32) {
      peak = Math.max(peak, Math.abs(data[j]));
    }
    const h = Math.max(2, peak * height * 0.9);
    const x = (i / bars) * width;
    ctx.fillStyle = rgba(themeColor.acc, 0.35 + Math.min(0.6, peak));
    ctx.fillRect(x, (height - h) / 2, Math.max(1.5, width / bars - 1.5), h);
  }
  canvas.classList.add('is-visible');
}

/* ────────────────────────── visualizer ────────────────────────── */

const viz = { canvas: null, ctx: null, data: null };

function initViz() {
  viz.canvas = $('#viz');
  viz.ctx = viz.canvas.getContext('2d');
  resizeViz();
  window.addEventListener('resize', resizeViz);
  requestAnimationFrame(drawViz);
}

function resizeViz() {
  const ratio = window.devicePixelRatio || 1;
  viz.canvas.width = viz.canvas.clientWidth * ratio;
  viz.canvas.height = viz.canvas.clientHeight * ratio;
  viz.ctx.setTransform(ratio, 0, 0, ratio, 0, 0);
}

function drawViz() {
  requestAnimationFrame(drawViz);
  const ctx = viz.ctx;
  const width = viz.canvas.clientWidth;
  const height = viz.canvas.clientHeight;
  ctx.clearRect(0, 0, width, height);

  const active = engine.playing || (state.busy && state.streaming);
  let values = null;
  if (engine.analyser && active) {
    if (!viz.data || viz.data.length !== engine.analyser.frequencyBinCount) {
      viz.data = new Uint8Array(engine.analyser.frequencyBinCount);
    }
    engine.analyser.getByteFrequencyData(viz.data);
    values = viz.data;
  }

  const bars = 72;
  const gap = 3;
  const barWidth = Math.max(2, (width - gap * (bars - 1)) / bars);
  const baseY = height * 0.55;
  ctx.fillStyle = rgba(themeColor.acc, values ? 0.9 : 0.4);
  for (let i = 0; i < bars; i++) {
    let amplitude;
    if (values) {
      const index = Math.floor((i / bars) * values.length * 0.7);
      amplitude = (values[index] / 255) * height * 0.72;
    } else {
      const wave = Math.sin(i * 0.32 + performance.now() / 1400) * 0.5 + 0.5;
      amplitude = 3 + wave * 6;
    }
    amplitude = Math.max(2, amplitude);
    const x = i * (barWidth + gap);
    const radius = Math.min(barWidth / 2, 4);
    roundRect(ctx, x, baseY - amplitude, barWidth, amplitude * 2, radius);
  }
}

function roundRect(ctx, x, y, w, h, r) {
  ctx.beginPath();
  ctx.moveTo(x + r, y);
  ctx.arcTo(x + w, y, x + w, y + h, r);
  ctx.arcTo(x + w, y + h, x, y + h, r);
  ctx.arcTo(x, y + h, x, y, r);
  ctx.arcTo(x, y, x + w, y, r);
  ctx.closePath();
  ctx.fill();
}

/* ────────────────────────── transport UI ────────────────────────── */

function syncTransport() {
  const hasBuffer = Boolean(engine.buffer);
  const duration = hasBuffer ? engine.buffer.duration : 0;
  const pos = hasBuffer && !state.busy ? position() : 0;

  $('#play').disabled = !hasBuffer || state.busy;
  $('#play').textContent = engine.playing ? '❚❚' : '▶';
  $('#scrubber').disabled = !hasBuffer || state.busy;
  $('#download-wav').disabled = !state.audio;
  $('#download-pcm').disabled = !state.audio;

  $('#time-total').textContent = formatTime(duration);
  if (state.busy && state.streaming) {
    const live =
      engine.liveStart && engine.ctx ? Math.max(0, engine.ctx.currentTime - engine.liveStart) : 0;
    $('#time-current').textContent = formatTime(live);
    $('#scrubber').value = '0';
  } else {
    $('#time-current').textContent = formatTime(pos);
    if (duration > 0) {
      $('#scrubber').value = String(Math.round((pos / duration) * 1000));
    } else {
      $('#scrubber').value = '0';
    }
  }
  $('#stage-overlay').hidden = hasBuffer || state.busy;
}

function setStats({ ttfa, total, duration }) {
  $('#stat-ttfa').textContent = Number.isFinite(ttfa) ? `${Math.round(ttfa)} ms` : '—';
  $('#stat-total').textContent = Number.isFinite(total) ? `${(total / 1000).toFixed(2)} s` : '—';
  $('#stat-rtf').textContent =
    duration > 0 && Number.isFinite(total) ? `${(total / 1000 / duration).toFixed(2)}×` : '—';
  $('#stat-length').textContent = duration > 0 ? `${duration.toFixed(2)} s` : '—';
}

function showError(message) {
  const el = $('#error');
  el.textContent = message;
  el.hidden = false;
}

function clearError() {
  $('#error').hidden = true;
}

function setBusy(busy, streaming) {
  state.busy = busy;
  state.streaming = Boolean(streaming);
  $('#generate').disabled = busy;
  $('#spinner').hidden = !busy;
  $('#cancel').hidden = !busy;
  $('#generate-label').textContent = busy ? 'Generating…' : 'Generate speech';
  syncTransport();
}

/* ────────────────────────── generation ────────────────────────── */

function buildForm() {
  const form = new FormData();
  form.append('text', $('#text').value.trim());
  form.append('instruction', $('#instruction').value);
  form.append('cfg_scale', $('#cfg').value);
  form.append('seed', String(parseInt($('#seed').value, 10) || 42));
  form.append('ref_text', $('#ref-text').value);
  form.append('lang', state.lang);
  if (state.refFile) form.append('ref_audio', state.refFile, state.refFile.name);
  return form;
}

async function readError(response) {
  try {
    const payload = await response.json();
    return payload.detail || `Request failed (${response.status})`;
  } catch (err) {
    return `Request failed (${response.status})`;
  }
}

async function generate() {
  if (state.busy) return;
  clearError();

  const mode = modeById(state.mode);
  const text = $('#text').value.trim();
  if (!text) return showError('Enter some text to speak first.');
  if (mode.needs_reference && !state.refFile) {
    return showError(`${mode.name} needs a reference clip and its transcript.`);
  }
  if (mode.needs_reference && !$('#ref-text').value.trim()) {
    return showError('Add the exact transcript of the reference clip.');
  }
  if (mode.needs_instruction && !$('#instruction').value.trim() && mode.id === 'design') {
    return showError('Voice Design needs an instruction describing the voice.');
  }

  const streaming = $('#stream-toggle').checked;
  const form = buildForm();
  const controller = new AbortController();
  state.abort = controller;

  resetEngine();
  setBusy(true, streaming);
  $('#meta-demo').hidden = true;
  $('#meta-template').textContent = mode.template;
  setStats({ ttfa: NaN, total: NaN, duration: 0 });

  const startedAt = performance.now();
  try {
    const result = streaming
      ? await runStreaming(form, controller, startedAt)
      : await runBuffered(form, controller, startedAt);
    // The file is written by the server before the response ends, so the shelf
    // is complete by the time control returns here.
    if (result) await loadOutputs();
  } catch (err) {
    if (err.name === 'AbortError') {
      showError('Generation cancelled.');
    } else {
      showError(err.message || String(err));
    }
  } finally {
    state.abort = null;
    setBusy(false, false);
    refreshStatus();
  }
}

async function runStreaming(form, controller, startedAt) {
  const response = await fetch('/api/stream', {
    method: 'POST',
    body: form,
    signal: controller.signal,
  });
  if (!response.ok) throw new Error(await readError(response));

  const sampleRate = parseInt(response.headers.get('X-Sample-Rate') || '24000', 10) || 24000;
  const template = response.headers.get('X-Breeze-Template') || '—';
  const demo = response.headers.get('X-Breeze-Demo') === '1';
  const warning = response.headers.get('X-Breeze-Warning') || '';
  applyHeaders(template, demo, warning);

  ensureCtx(sampleRate);
  if (engine.ctx.state === 'suspended') await engine.ctx.resume();

  const reader = response.body.getReader();
  let leftover = new Uint8Array(0);
  let ttfa = null;
  const chunks = [];

  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    if (!value || !value.length) continue;
    const merged = new Uint8Array(leftover.length + value.length);
    merged.set(leftover, 0);
    merged.set(value, leftover.length);
    const even = merged.length - (merged.length % 2);
    leftover = merged.slice(even);
    if (even === 0) continue;
    const view = new DataView(merged.buffer, merged.byteOffset, even);
    const float32 = new Float32Array(even / 2);
    for (let i = 0; i < float32.length; i++) {
      float32[i] = view.getInt16(i * 2, true) / 32768;
    }
    if (ttfa === null) ttfa = performance.now() - startedAt;
    chunks.push(float32);
    pushLive(float32, sampleRate);
  }

  const total = performance.now() - startedAt;
  const merged = concatChunks(chunks);
  const duration = merged.length / sampleRate;
  state.audio = { float32: merged, sampleRate };
  engine.buffer = makeBuffer(merged, sampleRate);
  engine.pauseOffset = 0;
  engine.liveChunks = [];
  setStats({ ttfa: ttfa ?? total, total, duration });
  syncTransport();
  return { float32: merged, sampleRate, template, demo, duration, ttfa: ttfa ?? total, total };
}

async function runBuffered(form, controller, startedAt) {
  const response = await fetch('/api/generate', {
    method: 'POST',
    body: form,
    signal: controller.signal,
  });
  if (!response.ok) throw new Error(await readError(response));

  const sampleRate = parseInt(response.headers.get('X-Sample-Rate') || '24000', 10) || 24000;
  const template = response.headers.get('X-Breeze-Template') || '—';
  const demo = response.headers.get('X-Breeze-Demo') === '1';
  const warning = response.headers.get('X-Breeze-Warning') || '';
  applyHeaders(template, demo, warning);

  const raw = await response.arrayBuffer();
  const ctx = ensureCtx(sampleRate);
  const buffer = await ctx.decodeAudioData(raw.slice(0));
  const float32 = buffer.getChannelData(0).slice();
  const total = performance.now() - startedAt;
  const duration = buffer.duration;

  state.audio = { float32, sampleRate: buffer.sampleRate };
  engine.buffer = makeBuffer(float32, buffer.sampleRate);
  engine.pauseOffset = 0;
  setStats({ ttfa: total, total, duration });
  playBuffer(0);
  return {
    float32,
    sampleRate: buffer.sampleRate,
    template,
    demo,
    duration,
    ttfa: total,
    total,
  };
}

function concatChunks(chunks) {
  const total = chunks.reduce((sum, c) => sum + c.length, 0);
  const merged = new Float32Array(total);
  let offset = 0;
  for (const chunk of chunks) {
    merged.set(chunk, offset);
    offset += chunk.length;
  }
  return merged;
}

function applyHeaders(template, demo, warning) {
  $('#meta-template').textContent = template;
  $('#meta-demo').hidden = !demo;
  if (demo && warning) {
    $('#meta-demo').title = warning;
  } else if (demo) {
    $('#meta-demo').title = 'Served by the built-in demo synth because the model is offline.';
  }
}

/* ────────────────────────── output shelf ────────────────────────── */

// The shelf is a VIEW OF THE OUTPUT FOLDER, not a second copy of it. Every take
// the server finishes is written to disk with a JSON sidecar, so re-reading the
// list is both the refresh and the persistence -- no client-side bookkeeping to
// get out of step with what is actually on disk.

async function loadOutputs() {
  try {
    const data = await (await fetch('/api/outputs')).json();
    state.outputs = data.outputs || [];
    const dir = data.dir || 'outputs/';
    $('#outputs-dir').textContent = `saving to ${dir}`;
    $('#outputs-dir').title = dir;
  } catch (err) {
    state.outputs = [];
    $('#outputs-dir').textContent = 'output folder unavailable';
  }
  renderHistory();
}

function renderHistory() {
  const container = $('#history');
  container.innerHTML = '';
  if (!state.outputs.length) {
    container.innerHTML =
      '<p class="empty">Nothing saved yet — every take lands in the output folder and appears here.</p>';
    return;
  }
  for (const output of state.outputs) {
    const mode = state.catalog
      ? (state.catalog.modes.find((m) => m.id === output.mode) || {}).name
      : null;
    const el = document.createElement('div');
    el.className = 'take' + (state.activeOutput === output.name ? ' is-playing' : '');
    el.innerHTML = `
      <button class="take-play" type="button" aria-label="Play take">▶</button>
      <div class="take-info">
        <div class="take-text"></div>
        <div class="take-sub">
          <span>${mode || output.mode || 'take'}</span>
          <span>${output.template || ''}</span>
          <span>${Number(output.duration || 0).toFixed(2)}s</span>
          <span>ttfa ${Math.round(output.ttfa_ms || 0)}ms</span>
          <span>seed ${output.seed ?? '—'}</span>
          ${output.demo ? '<span class="tag demo">demo</span>' : ''}
        </div>
      </div>
      <div class="take-actions">
        <button class="ghost small act-download" type="button">WAV</button>
        <button class="ghost small act-reuse" type="button">Reuse</button>
        <button class="ghost small act-delete" type="button" title="delete this take">✕</button>
      </div>`;
    el.querySelector('.take-text').textContent = output.text || output.name;

    el.querySelector('.take-play').addEventListener('click', () => toggleTake(output));
    el.querySelector('.act-download').addEventListener('click', () =>
      download(output.url, output.name)
    );
    el.querySelector('.act-reuse').addEventListener('click', () => {
      $('#text').value = output.text || '';
      $('#instruction').value = output.instruction || '';
      updateCounter();
    });
    el.querySelector('.act-delete').addEventListener('click', () => deleteOutput(output));
    container.appendChild(el);
  }
}

function toggleTake(output) {
  const alreadyPlaying = shelfPlaying && shelfPlaying.name === output.name;
  if (shelfAudio) {
    shelfAudio.onended = null;
    shelfAudio.pause();
  }
  shelfAudio = null;
  shelfPlaying = null;
  state.activeOutput = null;
  if (alreadyPlaying) {
    renderHistory();
    return;
  }
  pauseBuffer();
  shelfAudio = new Audio(output.url);
  shelfPlaying = output;
  state.activeOutput = output.name;
  shelfAudio.onended = () => {
    shelfAudio = null;
    shelfPlaying = null;
    state.activeOutput = null;
    renderHistory();
  };
  shelfAudio.play();
  renderHistory();
}

async function deleteOutput(output) {
  if (!window.confirm(`Delete ${output.name} from the output folder?`)) return;
  try {
    await fetch(`/api/outputs/${encodeURIComponent(output.name)}`, { method: 'DELETE' });
  } catch (err) {
    showError(`Could not delete ${output.name}.`);
  }
  await loadOutputs();
}

/* ────────────────────────── status ────────────────────────── */

// The indicator is a bare dot, so the wording it stands for goes into the
// tooltip and the accessible name instead of onto the page.
function setStatus(label, detail) {
  const badge = $('#status-badge');
  $('#status-text').textContent = label;
  badge.title = detail;
  badge.setAttribute('aria-label', detail);
}

async function refreshStatus() {
  const badge = $('#status-badge');
  badge.className = 'badge compact is-loading';
  setStatus('checking the model server…', 'Checking the model server…');
  try {
    const response = await fetch('/api/status');
    const data = await response.json();
    const upstream = data.upstream || 'the model server';
    const latency = data.latency_ms ? `, ${data.latency_ms} ms` : '';
    if (data.connected) {
      badge.className = 'badge compact is-online';
      setStatus(
        'model online',
        `Model online at ${upstream} (${data.detail}${latency}). Click to re-check.`
      );
    } else if (data.fallback) {
      badge.className = 'badge compact is-demo';
      setStatus(
        'demo audio, model offline',
        `No model at ${upstream} (${data.detail}${latency}), so takes are demo audio, ` +
          'not the model. Click to re-check.'
      );
    } else {
      badge.className = 'badge compact is-offline';
      setStatus(
        'model offline',
        `No model at ${upstream} (${data.detail}${latency}). Generation will fail. ` +
          'Click to re-check.'
      );
    }
    $('#sample-rate-chip').textContent = `${(data.sample_rate / 1000).toFixed(1)} kHz · mono`;
  } catch (err) {
    badge.className = 'badge compact is-offline';
    setStatus('app server unreachable', 'The app server is unreachable. Click to re-check.');
  }
}

/* ────────────────────────── update ────────────────────────── */

// Pulls the latest source out of this app's own repository, then restarts the
// server so the new Python is actually loaded. The repository is public, so
// the update needs no credential -- the button just fetches.

function escapeHtml(s) {
  return String(s).replace(/[&<>]/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;' }[c]));
}

function updateMsg(html) {
  $('#update-msg').innerHTML = html;
}

function openUpdate() {
  updateMsg('');
  $('#update-go').disabled = false;
  $('#update-rev').textContent = 'checking the installed revision…';
  fetch('/api/update')
    .then((r) => r.json())
    .then((d) => {
      if (d.repo) {
        $('#update-src').textContent = d.repo;
        $('#update-src').href = 'https://github.com/' + d.repo;
      }
      $('#update-rev').textContent = d.rev
        ? `installed ${d.short || d.rev}`
        : 'installed revision unknown (this image carries no .tostai_rev)';
    })
    .catch(() => {
      $('#update-rev').textContent = '';
    });
  $('#update-dialog').showModal();
  $('#update-go').focus();
}

function closeUpdate() {
  $('#update-dialog').close();
}

// Wait until the NEW revision is the one answering. Waiting for "any response"
// is not enough: the re-exec is delayed so the update's own response can flush,
// so for the first second or so the OLD process still answers, and a naive
// readiness check would reload straight back into the old code.
async function waitForRev(rev) {
  for (let i = 0; i < 120; i++) {
    await new Promise((r) => setTimeout(r, 500));
    try {
      const r = await fetch('/api/update', { cache: 'no-store' });
      if (!r.ok) continue;
      const d = await r.json();
      if (d.rev === rev) return true;
    } catch (err) {
      /* still down */
    }
  }
  return false;
}

async function runUpdate() {
  $('#update-go').disabled = true;
  updateMsg('<div class="hint">Fetching the latest source…</div>');
  let d;
  try {
    const r = await fetch('/api/update', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({}),
    });
    // Read the body as text and parse it here, rather than calling r.json(). A
    // failure before the handler runs answers with a bare text/plain "Internal
    // Server Error", and r.json() then throws a parser complaint instead of the
    // actual cause.
    const raw = await r.text();
    try {
      d = JSON.parse(raw);
    } catch (err) {
      d = {
        ok: false,
        error: `HTTP ${r.status} from the server: ${raw.slice(0, 400)}`,
      };
    }
  } catch (err) {
    updateMsg(`<div class="err">${escapeHtml(String(err))}</div>`);
    $('#update-go').disabled = false;
    return;
  }
  if (!d.ok) {
    updateMsg(`<div class="err">${escapeHtml(d.error || 'update failed')}</div>`);
    $('#update-go').disabled = false;
    return;
  }
  if (!d.updated) {
    updateMsg(`<div class="okbox">Already up to date at ${escapeHtml(d.rev.slice(0, 10))}.</div>`);
    $('#update-go').disabled = false;
    return;
  }
  updateMsg(
    `<div class="hint">Updated to ${escapeHtml(d.rev.slice(0, 10))} (${d.files} files). Restarting…</div>`
  );
  if (!(await waitForRev(d.rev))) {
    updateMsg(
      '<div class="err">The server did not come back on the new revision.\n' +
        'Check it with:  docker logs <container></div>'
    );
    $('#update-go').disabled = false;
    return;
  }
  location.reload();
}

/* ────────────────────────── wiring ────────────────────────── */

function wireEvents() {
  $('#generate').addEventListener('click', generate);
  $('#cancel').addEventListener('click', () => state.abort && state.abort.abort());
  $('#reset').addEventListener('click', () => {
    clearError();
    $('#text').value = '';
    $('#instruction').value = '';
    $('#ref-text').value = '';
    $('#seed').value = '42';
    clearReference();
    updateCounter();
    selectMode(state.mode);
  });

  $('#text').addEventListener('input', updateCounter);
  $('#text').addEventListener('keydown', (event) => {
    if ((event.metaKey || event.ctrlKey) && event.key === 'Enter') generate();
  });

  $$('.lang-btn').forEach((btn) =>
    btn.addEventListener('click', () => selectLang(btn.dataset.lang))
  );

  $('#cfg').addEventListener('input', () => {
    $('#cfg-value').textContent = Number($('#cfg').value).toFixed(1);
  });
  $('#seed-random').addEventListener('click', () => {
    $('#seed').value = String(Math.floor(Math.random() * 1000000));
  });

  const drop = $('#ref-drop');
  const input = $('#ref-input');
  drop.addEventListener('click', () => input.click());
  drop.addEventListener('keydown', (event) => {
    if (event.key === 'Enter' || event.key === ' ') input.click();
  });
  input.addEventListener('change', () => setReference(input.files[0]));
  ['dragenter', 'dragover'].forEach((type) =>
    drop.addEventListener(type, (event) => {
      event.preventDefault();
      drop.classList.add('is-over');
    })
  );
  ['dragleave', 'drop'].forEach((type) =>
    drop.addEventListener(type, (event) => {
      event.preventDefault();
      drop.classList.remove('is-over');
    })
  );
  drop.addEventListener('drop', (event) => setReference(event.dataTransfer.files[0]));
  $('#ref-clear').addEventListener('click', (event) => {
    event.stopPropagation();
    clearReference();
  });

  $('#play').addEventListener('click', () => {
    if (engine.playing) pauseBuffer();
    else playBuffer(engine.pauseOffset || 0);
  });
  $('#scrubber').addEventListener('input', (event) => {
    if (!engine.buffer || state.busy) return;
    const ratio = Number(event.target.value) / 1000;
    const target = ratio * engine.buffer.duration;
    if (engine.playing) playBuffer(target);
    else {
      engine.pauseOffset = target;
      syncTransport();
    }
  });

  $('#download-wav').addEventListener('click', () => {
    if (state.audio) {
      download(encodeWav(state.audio.float32, state.audio.sampleRate), 'breeze-take.wav');
    }
  });
  $('#download-pcm').addEventListener('click', () => {
    if (state.audio) download(encodePcm(state.audio.float32), 'breeze-take.pcm');
  });

  $('#refresh-outputs').addEventListener('click', loadOutputs);

  $('#status-badge').addEventListener('click', refreshStatus);
  $('#theme-toggle').addEventListener('click', toggleTheme);
  $('#facts-button').addEventListener('click', () => $('#facts-dialog').showModal());
  $('#facts-close').addEventListener('click', () => $('#facts-dialog').close());
  $('#update-button').addEventListener('click', openUpdate);
  $('#update-close').addEventListener('click', closeUpdate);
  $('#update-cancel').addEventListener('click', closeUpdate);
  $('#update-go').addEventListener('click', runUpdate);

  document.addEventListener('keydown', (event) => {
    if (event.key === 'Escape' && engine.playing) pauseBuffer();
  });

  setInterval(() => {
    if (engine.playing && !state.busy) syncTransport();
  }, 250);
  setInterval(() => {
    if (!state.busy) refreshStatus();
  }, 20000);
}

/* ────────────────────────── boot ────────────────────────── */

async function boot() {
  setTheme(document.documentElement.dataset.theme || 'light');
  initViz();
  wireEvents();
  try {
    state.catalog = await (await fetch('/api/catalog')).json();
  } catch (err) {
    showError('Could not load the Breeze catalog from the app server.');
    return;
  }
  renderModes();
  renderFastStages();
  renderFacts();
  selectMode('design');
  selectLang('en');
  updateCounter();
  syncTransport();
  await loadOutputs();
  refreshStatus();
}

boot();
