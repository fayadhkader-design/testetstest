/* Dashboard state, transport and wiring. */

import {
  LineChart,
  ScatterPlot,
  barFraction,
  cssVar,
  formatValue,
  seriesColor,
} from "./charts.js";
import {
  appendFrame,
  gradientRange,
  indexForStep,
  isSameRun,
} from "./history.js";

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
  gradientRows: new Map(),
  gradientSignature: null,
  dirty: true,
  settle: 0,
  /** Set once, so a repeating render fault logs once instead of per frame. */
  renderFailed: false,
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
  view.gradientRows.clear();
  view.gradientSignature = null;
  el("gradients-card").hidden = true;

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

/**
 * Build one metric card.
 *
 * Every part is constructed as an element and the pieces are kept as direct
 * references, rather than being interpolated into markup and looked up by a
 * generated id. Metric names come from **kwargs, and Python does not require
 * those keys to be identifiers, so a name is arbitrary text.
 *
 * Interpolating it was doubly wrong: it injected the name as live markup, and
 * a name containing a quote truncated the id attribute so two cards ended up
 * sharing ids. The second lookup then returned null, render() threw, and the
 * dashboard froze for good.
 */
function buildMetricCard(name) {
  el("metrics-empty").hidden = true;

  const card = document.createElement("section");
  card.className = "card metric";

  const head = document.createElement("div");
  head.className = "card__head";
  const title = document.createElement("h2");
  title.textContent = name;
  head.appendChild(title);

  const value = document.createElement("p");
  value.className = "metric__value";
  value.textContent = "—";

  const plot = document.createElement("div");
  plot.className = "plot";
  const canvas = document.createElement("canvas");
  plot.appendChild(canvas);

  const foot = document.createElement("p");
  foot.className = "metric__foot";
  const low = document.createElement("span");
  low.textContent = "—";
  const high = document.createElement("span");
  high.textContent = "—";
  foot.append(low, high);

  card.append(head, value, plot, foot);
  el("metrics").appendChild(card);

  const chart = new LineChart(canvas, { color: metricColor(name) });
  view.charts.set(name, { chart, value, low, high });
  attachLineHover(canvas, chart, name);
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
    // Labels are ints server-side, so this one was never exploitable -- but
    // keeping the rule uniform is what stops the next one from being.
    const swatch = document.createElement("span");
    swatch.className = "legend__swatch";
    swatch.setAttribute("aria-hidden", "true");
    button.append(swatch, document.createTextNode(String(label)));
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

  renderGradients(frame);

  for (const [name, card] of view.charts) {
    const { chart, value, low, high } = card;

    // The full history sets the axes; only the rewound prefix gets drawn.
    const points = state.frames.map((f) => ({ x: f.step, y: f.metrics?.[name] ?? null }));
    chart.setData(points, index + 1);
    chart.draw();

    value.textContent = formatMetric(name, frame?.metrics?.[name]);
    const [min, max] = chart.range;
    low.textContent = Number.isFinite(min) ? `min ${formatValue(min)}` : "—";
    high.textContent = Number.isFinite(max) ? `max ${formatValue(max)}` : "—";
  }

  if (!el("table-view").hidden) renderTable();
}

/**
 * Per-layer gradient bars.
 *
 * Rows are built once per layer set and then mutated in place. Rebuilding
 * forty rows twenty times a second would churn the DOM for no reason, and it
 * would also throw away any text the user was mid-selection on.
 *
 * The value column is deliberate: it makes the panel its own table view, so
 * no reading here depends on being able to compare bar lengths by eye.
 */
function renderGradients(frame) {
  const card = el("gradients-card");
  const gradients = frame?.gradients;
  card.hidden = !gradients;
  if (!gradients) return;

  const { layers, norms } = gradients;
  const signature = layers.join(" ");
  if (view.gradientSignature !== signature) {
    buildGradientRows(layers);
    view.gradientSignature = signature;
  }

  const largest = Math.max(...norms.filter((n) => Number.isFinite(n) && n > 0));
  const ranges = gradientRange(state.frames, currentIndex() + 1);

  layers.forEach((layer, index) => {
    const row = view.gradientRows.get(layer);
    if (!row) return;

    const norm = norms[index];
    row.bar.style.width = `${barFraction(norm, largest) * 100}%`;
    row.value.textContent = formatValue(norm);
    row.root.dataset.zero = String(norm === 0);

    const seen = ranges.get(layer);
    if (!seen) {
      row.range.hidden = true;
      return;
    }
    // Both ends clamp inside barFraction, so a historical peak above the
    // current frame's maximum flattens against the right edge rather than
    // overflowing the track.
    const low = barFraction(seen.min, largest);
    const high = barFraction(seen.max, largest);
    row.range.hidden = high - low < 0.005;
    row.range.style.left = `${low * 100}%`;
    row.range.style.width = `${(high - low) * 100}%`;
  });
}

function buildGradientRows(layers) {
  const container = el("grads");
  container.replaceChildren();
  view.gradientRows = new Map();

  for (const layer of layers) {
    const root = document.createElement("div");
    root.className = "grad";
    root.title = layer;

    // Layer names come from the user's own module attributes, but they still
    // go in as text rather than markup -- there is no reason for a model to
    // be able to write HTML into the dashboard.
    const name = document.createElement("span");
    name.className = "grad__name";
    name.textContent = layer;

    const track = document.createElement("span");
    track.className = "grad__track";
    const range = document.createElement("i");
    range.className = "grad__range";
    range.hidden = true;
    const bar = document.createElement("i");
    bar.className = "grad__bar";
    track.append(range, bar);

    const value = document.createElement("span");
    value.className = "grad__value num";
    value.textContent = "—";

    root.append(name, track, value);
    container.appendChild(root);
    view.gradientRows.set(layer, { root, bar, value, range });
  }
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
  // The rescheduling lives in `finally` deliberately. Before this, a throw
  // anywhere in render() skipped the next requestAnimationFrame, so a single
  // bad frame stopped the dashboard permanently -- and silently, since the
  // page carried on looking exactly like a run that had gone quiet.
  //
  // One broken frame should cost one frame.
  try {
    if (view.dirty || view.settle > 0) {
      view.settle = Math.max(0, view.settle - 1);
      view.dirty = false;
      render();
    }
  } catch (error) {
    if (!view.renderFailed) {
      view.renderFailed = true;
      console.error("nnscope: render failed; continuing", error);
    }
  } finally {
    requestAnimationFrame(loop);
  }
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
  // Snap the thumb here rather than leaving it to the next render, so the
  // control never disagrees with the state it represents.
  el("timeline").value = String(Math.max(0, state.frames.length - 1));
  view.scatter.snap();
  view.dirty = true;
}

/** Move the view by whole frames, from wherever it currently sits. */
function scrubBy(delta) {
  const newest = Number(el("timeline").max);
  const from = isLive() ? newest : currentIndex();
  scrubTo(Math.min(newest, Math.max(0, from + delta)));
}

function scrubTo(index) {
  const frame = state.frames[index];
  if (!frame) return;

  // The thumb has to follow the view wherever the move came from -- dragging,
  // an arrow key, or Home. Assigning `value` fires no input event, so this
  // cannot loop back through the drag handler that calls it.
  el("timeline").value = String(index);

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
    // The scrubber is an <input type=range>, so when it has focus the browser
    // already moves it with the arrow keys. Bailing on inputs keeps this from
    // running as well and moving it twice per press.
    if (event.target.tagName === "INPUT") return;

    // Space is matched on `code` as well as `key`: `code` names the physical
    // key, so it survives layouts and IMEs where `key` is not a plain space.
    if (event.code === "Space" || event.key === " ") {
      event.preventDefault();
      send({ type: state.controls.paused ? "resume" : "pause" });
      return;
    }

    switch (event.key) {
      case "ArrowLeft":
      case "ArrowRight":
        event.preventDefault();
        scrubBy((event.key === "ArrowRight" ? 1 : -1) * (event.shiftKey ? 10 : 1));
        break;

      case "Home":
        event.preventDefault();
        scrubTo(0);
        break;

      case "End":
        event.preventDefault();
        goLive();
        break;

      // Advancing training one step used to be ArrowRight, which now belongs
      // to the scrubber. A separate key keeps "move the view" and "move the
      // run" from being the same gesture.
      case ".":
        if (state.controls.paused) send({ type: "step", count: 1 });
        break;
    }
  });
}

/* ---- boot --------------------------------------------------------------- */

view.scatter = new ScatterPlot(el("scatter"));
attachScatterHover(el("scatter"));
wireControls();
connect();
requestAnimationFrame(loop);
