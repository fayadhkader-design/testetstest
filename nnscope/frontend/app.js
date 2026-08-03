/* Dashboard state, transport and wiring. */

import { LineChart, ScatterPlot, cssVar, formatValue, seriesColor } from "./charts.js";
import { appendFrame, indexForStep, isSameRun } from "./history.js";

const MAX_COLOURED_CLASSES = 8;
const MAX_HIGHLIGHTS = 3;
const TABLE_ROWS = 40;
const SETTLE_FRAMES = 40;

const el = (id) => document.getElementById(id);

const state = {
  run: null,
  frames: [],
  capacity: 600,
  /** null means "follow the newest frame"; otherwise the step being viewed. */
  viewStep: null,
  controls: { paused: false, stepBudget: 0 },
  lr: null,
  step: 0,
  metrics: [],
  classes: [],
  selected: new Set(),
};

const view = {
  scatter: null,
  charts: new Map(),
  dirty: true,
  settle: 0,
};

/* ---- derived state ------------------------------------------------------ */

const isLive = () => state.viewStep === null;

function currentIndex() {
  return indexForStep(state.frames, state.viewStep);
}

const currentFrame = () => state.frames[currentIndex()] ?? null;

function classColors() {
  const colors = new Map();
  if (state.classes.length <= MAX_COLOURED_CLASSES) {
    state.classes.forEach((label, i) => colors.set(label, seriesColor(i)));
    return colors;
  }
  // Past eight, a generated hue is indistinguishable under colour-vision
  // deficiency, so hue is spent only on what the user highlights. The first
  // three slots are the ones that clear the all-pairs floors.
  [...state.selected].slice(0, MAX_HIGHLIGHTS).forEach((label, i) => {
    colors.set(label, seriesColor(i));
  });
  return colors;
}

function metricColor(name) {
  const index = state.metrics.indexOf(name);
  return seriesColor(index % MAX_COLOURED_CLASSES) ?? cssVar("--series-1");
}

/** Accuracy-style metrics read far better as percentages. Deliberately
 *  narrow: only these exact names, and only within [0, 1]. */
function formatMetric(name, value) {
  const asPercent = ["acc", "accuracy", "val_acc", "val_accuracy"].includes(name);
  if (asPercent && Number.isFinite(value) && value >= 0 && value <= 1) {
    return `${(value * 100).toFixed(1)}%`;
  }
  return formatValue(value);
}

/* ---- ingest ------------------------------------------------------------- */

function ingest(frame) {
  // Replay from a reconnect is dropped here; see history.js.
  if (!appendFrame(state.frames, frame, state.capacity)) return;

  for (const name of Object.keys(frame.metrics ?? {})) {
    if (!state.metrics.includes(name) && state.metrics.length < MAX_COLOURED_CLASSES) {
      state.metrics.push(name);
      buildMetricCard(name);
    }
  }

  const labels = frame.embedding?.labels;
  if (labels) {
    let added = false;
    for (const label of labels) {
      if (!state.classes.includes(label)) {
        state.classes.push(label);
        added = true;
      }
    }
    if (added) {
      state.classes.sort((a, b) => a - b);
      buildLegend();
    }
  }

  view.settle = SETTLE_FRAMES;
  view.dirty = true;
}

/** Tear everything derived from a run back down, so a different run starts
 *  from a clean slate rather than inheriting the previous one's charts. */
function resetRun() {
  state.frames = [];
  state.metrics = [];
  state.classes = [];
  state.selected = new Set();
  state.viewStep = null;

  view.charts.clear();
  const empty = el("metrics-empty");
  empty.hidden = false;
  el("metrics").replaceChildren(empty);

  el("legend").replaceChildren();
  el("legend-note").hidden = true;

  view.scatter.setFrame(null);
  view.scatter.setSelected(state.selected);
  view.scatter.snap();
  view.dirty = true;
}

/* ---- transport ---------------------------------------------------------- */

let socket = null;
let retryDelay = 500;

function connect() {
  const url = `${location.protocol === "https:" ? "wss" : "ws"}://${location.host}/ws`;
  socket = new WebSocket(url);

  socket.addEventListener("open", () => {
    retryDelay = 500;
    setStatus("live");
  });

  socket.addEventListener("close", () => {
    setStatus("closed");
    setTimeout(connect, retryDelay);
    retryDelay = Math.min(retryDelay * 2, 8000);
  });

  socket.addEventListener("message", (event) => {
    const message = JSON.parse(event.data);
    if (message.type === "hello") {
      // A tab left open across a restart reconnects to a different run on the
      // same port. Splicing the new run's frames onto the old one's would
      // produce a chart that never happened.
      if (state.run && !isSameRun(state.run, message.run)) resetRun();
      state.run = message.run;
      state.capacity = message.run.capacity ?? state.capacity;
      applyRunMeta(message.run);
    } else if (message.type === "backfill") {
      message.frames.forEach(ingest);
    } else if (message.type === "frame") {
      ingest(message.frame);
    } else if (message.type === "state") {
      state.controls = message.controls;
      state.lr = message.lr;
      state.step = message.step;
      applyControlState();
    } else if (message.type === "error") {
      console.warn("nnscope:", message.detail);
    }
  });
}

function send(message) {
  if (socket?.readyState === WebSocket.OPEN) socket.send(JSON.stringify(message));
}

function setStatus(kind) {
  const node = el("status");
  node.dataset.state = kind;
  node.textContent = { live: "live", closed: "reconnecting", connecting: "connecting" }[kind];
}

/* ---- chrome ------------------------------------------------------------- */

function applyRunMeta(run) {
  el("meta-model").textContent = run.model ?? "—";
  el("meta-layer").textContent = run.layer ?? "—";
  el("meta-device").textContent = run.device ?? "—";
}

function applyControlState() {
  const paused = state.controls.paused;
  el("pause").textContent = paused ? "Resume" : "Pause";
  el("stepone").disabled = !paused;

  const lrInput = el("lr");
  if (document.activeElement !== lrInput && state.lr !== null) {
    lrInput.value = String(state.lr);
  }
}

function buildMetricCard(name) {
  el("metrics-empty").hidden = true;

  const card = document.createElement("section");
  card.className = "card metric";
  card.innerHTML = `
    <div class="card__head"><h2>${name}</h2></div>
    <p class="metric__value" id="value-${name}">—</p>
    <div class="plot"><canvas></canvas></div>
    <p class="metric__foot"><span id="min-${name}">—</span><span id="max-${name}">—</span></p>`;
  el("metrics").appendChild(card);

  const chart = new LineChart(card.querySelector("canvas"), { color: metricColor(name) });
  view.charts.set(name, chart);
  attachLineHover(card.querySelector("canvas"), chart, name);
}

function buildLegend() {
  const legend = el("legend");
  legend.replaceChildren();

  const overflowing = state.classes.length > MAX_COLOURED_CLASSES;
  const note = el("legend-note");
  note.hidden = !overflowing;
  if (overflowing) {
    note.textContent =
      `${state.classes.length} classes — click one to light it up. Beyond eight, ` +
      `colour stops being reliably distinguishable, so it is spent only on the ` +
      `classes you select (up to ${MAX_HIGHLIGHTS}); position carries the rest.`;
  }

  const colors = classColors();
  for (const label of state.classes) {
    const button = document.createElement("button");
    button.type = "button";
    button.className = "legend__item";
    button.setAttribute("aria-pressed", String(state.selected.has(label)));
    button.dataset.dimmed = String(state.selected.size > 0 && !state.selected.has(label));
    button.style.setProperty("--swatch", colors.get(label) ?? cssVar("--ink-muted"));
    button.innerHTML = `<span class="legend__swatch" aria-hidden="true"></span>${label}`;
    button.addEventListener("click", () => toggleClass(label));
    legend.appendChild(button);
  }
}

function toggleClass(label) {
  if (state.selected.has(label)) state.selected.delete(label);
  else state.selected.add(label);
  buildLegend();
  view.dirty = true;
}

/* ---- rendering ---------------------------------------------------------- */

function render() {
  const frame = currentFrame();
  const index = currentIndex();

  el("timeline").max = String(Math.max(0, state.frames.length - 1));
  if (isLive()) el("timeline").value = String(Math.max(0, state.frames.length - 1));

  el("meta-step").textContent = frame ? frame.step.toLocaleString() : "0";
  el("scrub-pos").textContent = frame
    ? `${isLive() ? "live" : "rewound"} · step ${frame.step.toLocaleString()} · ${frame.t ?? 0}s`
    : "—";

  const embedding = frame?.embedding ?? null;
  el("scatter-empty").hidden = Boolean(embedding);
  el("explained").textContent = embedding?.explained
    ? `${Math.round(embedding.explained * 100)}%`
    : "—";

  view.scatter.setClassColors(classColors());
  view.scatter.setSelected(state.selected);
  view.scatter.setFrame(embedding);
  view.scatter.draw();

  for (const [name, chart] of view.charts) {
    // The full history sets the axes; only the rewound prefix gets drawn.
    const points = state.frames.map((f) => ({ x: f.step, y: f.metrics?.[name] ?? null }));
    chart.setData(points, index + 1);
    chart.draw();

    el(`value-${name}`).textContent = formatMetric(name, frame?.metrics?.[name]);
    const [min, max] = chart.range;
    el(`min-${name}`).textContent = Number.isFinite(min) ? `min ${formatValue(min)}` : "—";
    el(`max-${name}`).textContent = Number.isFinite(max) ? `max ${formatValue(max)}` : "—";
  }

  if (!el("table-view").hidden) renderTable();
}

function renderTable() {
  const head = el("frame-table-head");
  head.replaceChildren();
  for (const column of ["step", "t", ...state.metrics]) {
    const th = document.createElement("th");
    th.textContent = column;
    head.appendChild(th);
  }

  const body = el("frame-table-body");
  body.replaceChildren();
  const upTo = currentIndex() + 1;
  for (const frame of state.frames.slice(Math.max(0, upTo - TABLE_ROWS), upTo).reverse()) {
    const row = document.createElement("tr");
    const cells = [
      frame.step.toLocaleString(),
      `${frame.t ?? 0}s`,
      ...state.metrics.map((name) => formatValue(frame.metrics?.[name])),
    ];
    for (const value of cells) {
      const td = document.createElement("td");
      td.textContent = value;
      row.appendChild(td);
    }
    body.appendChild(row);
  }
}

function loop() {
  if (view.dirty || view.settle > 0) {
    view.settle = Math.max(0, view.settle - 1);
    view.dirty = false;
    render();
  }
  requestAnimationFrame(loop);
}

/* ---- tooltips ----------------------------------------------------------- */

function showTooltip(event, html) {
  const tip = el("tooltip");
  tip.innerHTML = html;
  tip.hidden = false;
  const box = tip.getBoundingClientRect();
  const x = Math.min(event.clientX + 14, window.innerWidth - box.width - 8);
  const y = Math.max(8, event.clientY - box.height - 12);
  tip.style.left = `${x}px`;
  tip.style.top = `${y}px`;
}

const hideTooltip = () => {
  el("tooltip").hidden = true;
};

function attachLineHover(canvas, chart, name) {
  canvas.addEventListener("pointermove", (event) => {
    const rect = canvas.getBoundingClientRect();
    const index = chart.pick(event.clientX - rect.left);
    chart.setHover(index);
    view.dirty = true;

    if (index === null || !chart.points[index]) return hideTooltip();
    const point = chart.points[index];
    showTooltip(
      event,
      `<b>${formatMetric(name, point.y)}</b><br><span>step ${point.x.toLocaleString()}</span>`
    );
  });
  canvas.addEventListener("pointerleave", () => {
    chart.setHover(null);
    hideTooltip();
    view.dirty = true;
  });
}

function attachScatterHover(canvas) {
  canvas.addEventListener("pointermove", (event) => {
    const rect = canvas.getBoundingClientRect();
    const index = view.scatter.pick(event.clientX - rect.left, event.clientY - rect.top);
    view.scatter.setHover(index);
    view.dirty = true;

    const frame = currentFrame()?.embedding;
    if (index === null || !frame) return hideTooltip();
    const label = frame.labels ? frame.labels[index] : null;
    showTooltip(
      event,
      label === null
        ? `<span>sample ${index}</span>`
        : `<b>class ${label}</b><br><span>sample ${index}</span>`
    );
  });
  canvas.addEventListener("pointerleave", () => {
    view.scatter.setHover(null);
    hideTooltip();
    view.dirty = true;
  });
}

/* ---- controls ----------------------------------------------------------- */

function goLive() {
  state.viewStep = null;
  el("live").classList.add("is-live");
  el("live").setAttribute("aria-pressed", "true");
  view.scatter.snap();
  view.dirty = true;
}

function scrubTo(index) {
  const frame = state.frames[index];
  if (!frame) return;
  const atEnd = index >= state.frames.length - 1;
  state.viewStep = atEnd ? null : frame.step;
  el("live").classList.toggle("is-live", atEnd);
  el("live").setAttribute("aria-pressed", String(atEnd));
  view.scatter.snap();
  view.dirty = true;
}

function wireControls() {
  el("timeline").addEventListener("input", (event) => scrubTo(Number(event.target.value)));
  el("live").addEventListener("click", goLive);

  el("pause").addEventListener("click", () => {
    send({ type: state.controls.paused ? "resume" : "pause" });
  });
  el("stepone").addEventListener("click", () => send({ type: "step", count: 1 }));
  el("shock").addEventListener("click", () => send({ type: "shock", magnitude: 0.5 }));

  el("lr").addEventListener("change", (event) => {
    const value = Number(event.target.value);
    if (Number.isFinite(value) && value >= 0) send({ type: "lr", value });
  });

  el("toggle-table").addEventListener("click", (event) => {
    const table = el("table-view");
    table.hidden = !table.hidden;
    event.currentTarget.setAttribute("aria-pressed", String(!table.hidden));
    view.dirty = true;
  });

  el("toggle-theme").addEventListener("click", () => {
    const next = document.documentElement.dataset.theme === "dark" ? "light" : "dark";
    document.documentElement.dataset.theme = next;
    view.dirty = true;
  });

  window.addEventListener("resize", () => {
    view.dirty = true;
  });

  document.addEventListener("keydown", (event) => {
    if (event.target.tagName === "INPUT") return;
    if (event.code === "Space") {
      event.preventDefault();
      send({ type: state.controls.paused ? "resume" : "pause" });
    } else if (event.code === "ArrowRight" && state.controls.paused) {
      send({ type: "step", count: 1 });
    }
  });
}

/* ---- boot --------------------------------------------------------------- */

view.scatter = new ScatterPlot(el("scatter"));
attachScatterHover(el("scatter"));
wireControls();
connect();
requestAnimationFrame(loop);
