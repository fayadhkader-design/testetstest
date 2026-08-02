/* Canvas renderers for the dashboard.
 *
 * Everything reads its colours from CSS custom properties at draw time rather
 * than caching hex values, so switching theme is just a re-render.
 *
 * Deliberately no charting library: the two forms here are a single-series
 * line and a scatter, both a few dozen lines of canvas, and a dependency-free
 * frontend means the repo stays readable and has no build step.
 */

const MAX_SERIES_SLOTS = 8;

/** Categorical slot for an entity index. Never cycles: past the eighth slot
 *  callers must fall back to neutral ink plus legend-driven highlighting,
 *  because a ninth generated hue is indistinguishable under colour-vision
 *  deficiency. */
export function seriesColor(index) {
  if (index < 0 || index >= MAX_SERIES_SLOTS) return null;
  return cssVar(`--series-${index + 1}`);
}

export function cssVar(name) {
  return getComputedStyle(document.documentElement).getPropertyValue(name).trim();
}

export function formatValue(value) {
  if (value === null || value === undefined || !Number.isFinite(value)) return "—";
  const magnitude = Math.abs(value);
  if (magnitude === 0) return "0";
  if (magnitude < 1e-3 || magnitude >= 1e5) return value.toExponential(2);
  if (magnitude < 1) return value.toFixed(4);
  if (magnitude < 100) return value.toFixed(3);
  return value.toFixed(1);
}

/** Handles device-pixel-ratio scaling so lines land on whole pixels. */
class Layer {
  constructor(canvas) {
    this.canvas = canvas;
    this.ctx = canvas.getContext("2d");
    this.width = 0;
    this.height = 0;
  }

  measure() {
    const rect = this.canvas.getBoundingClientRect();
    if (rect.width < 2 || rect.height < 2) return false;

    const dpr = window.devicePixelRatio || 1;
    const width = Math.round(rect.width * dpr);
    const height = Math.round(rect.height * dpr);
    if (this.canvas.width !== width || this.canvas.height !== height) {
      this.canvas.width = width;
      this.canvas.height = height;
    }
    this.ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    this.ctx.clearRect(0, 0, rect.width, rect.height);
    this.width = rect.width;
    this.height = rect.height;
    return true;
  }
}

/* ---- line chart -------------------------------------------------------- */

const LINE_PAD = { top: 8, right: 8, bottom: 6, left: 8 };

/**
 * A single-series line over training steps.
 *
 * One series per chart by construction. Two measures on one plot would need
 * two y-scales, which invents a correlation that isn't in the data; separate
 * cards keep every comparison honest.
 */
export class LineChart {
  constructor(canvas, { color }) {
    this.layer = new Layer(canvas);
    this.color = color;
    this.points = [];
    this.log = false;
    this.hover = null;
    this.range = [0, 1];
  }

  setData(points) {
    this.points = points;
    // A loss falling from 2.3 to 0.02 is unreadable on a linear axis; switch
    // once the span is wide enough for the detail to be lost.
    const values = points.map((p) => p.y).filter(Number.isFinite);
    const min = Math.min(...values);
    const max = Math.max(...values);
    this.log = values.length > 1 && min > 0 && max / min > 25;
    this.range = [min, max];
  }

  setHover(index) {
    this.hover = index;
  }

  /** Nearest sample to a canvas-space x, or null when out of range. */
  pick(x) {
    if (!this.points.length || !this.layer.width) return null;
    const { left, right } = this.plotBox();
    const ratio = (x - left) / Math.max(1, right - left);
    if (ratio < -0.05 || ratio > 1.05) return null;
    const index = Math.round(ratio * (this.points.length - 1));
    return Math.min(this.points.length - 1, Math.max(0, index));
  }

  plotBox() {
    return {
      left: LINE_PAD.left,
      right: this.layer.width - LINE_PAD.right,
      top: LINE_PAD.top,
      bottom: this.layer.height - LINE_PAD.bottom,
    };
  }

  project(index) {
    const box = this.plotBox();
    const [min, max] = this.paddedRange();
    const point = this.points[index];
    const scale = (value) => (this.log ? Math.log10(Math.max(value, 1e-12)) : value);
    const lo = scale(min);
    const hi = scale(max);
    const t = hi === lo ? 0.5 : (scale(point.y) - lo) / (hi - lo);
    const span = Math.max(1, this.points.length - 1);
    return {
      x: box.left + (index / span) * (box.right - box.left),
      y: box.bottom - t * (box.bottom - box.top),
    };
  }

  paddedRange() {
    let [min, max] = this.range;
    if (!Number.isFinite(min) || !Number.isFinite(max)) return [0, 1];
    if (min === max) return [min - 0.5, max + 0.5];
    if (this.log) return [min, max];
    const pad = (max - min) * 0.12;
    return [min - pad, max + pad];
  }

  draw() {
    if (!this.layer.measure()) return;
    const { ctx } = this.layer;
    const box = this.plotBox();

    // Recessive solid hairlines. Dashes read as "threshold" when it's a grid.
    ctx.strokeStyle = cssVar("--grid");
    ctx.lineWidth = 1;
    for (let i = 0; i <= 2; i++) {
      const y = Math.round(box.top + (i / 2) * (box.bottom - box.top)) + 0.5;
      ctx.beginPath();
      ctx.moveTo(box.left, y);
      ctx.lineTo(box.right, y);
      ctx.stroke();
    }

    const usable = this.points.filter((p) => Number.isFinite(p.y));
    if (usable.length < 2) return;

    ctx.strokeStyle = this.color;
    ctx.lineWidth = 2;
    ctx.lineJoin = "round";
    ctx.lineCap = "round";
    ctx.beginPath();
    let started = false;
    this.points.forEach((point, index) => {
      if (!Number.isFinite(point.y)) return;
      const at = this.project(index);
      if (started) ctx.lineTo(at.x, at.y);
      else {
        ctx.moveTo(at.x, at.y);
        started = true;
      }
    });
    ctx.stroke();

    const lastIndex = this.lastFiniteIndex();
    if (lastIndex !== null) this.drawMarker(this.project(lastIndex), 3.5);

    if (this.hover !== null && this.hover < this.points.length) {
      if (!Number.isFinite(this.points[this.hover].y)) return;
      const at = this.project(this.hover);
      ctx.strokeStyle = cssVar("--axis");
      ctx.lineWidth = 1;
      ctx.beginPath();
      ctx.moveTo(Math.round(at.x) + 0.5, box.top);
      ctx.lineTo(Math.round(at.x) + 0.5, box.bottom);
      ctx.stroke();
      this.drawMarker(at, 4);
    }
  }

  lastFiniteIndex() {
    for (let i = this.points.length - 1; i >= 0; i--) {
      if (Number.isFinite(this.points[i].y)) return i;
    }
    return null;
  }

  /** A 2px surface ring separates the marker from the line beneath it. */
  drawMarker(at, radius) {
    const { ctx } = this.layer;
    ctx.beginPath();
    ctx.arc(at.x, at.y, radius + 2, 0, Math.PI * 2);
    ctx.fillStyle = cssVar("--surface");
    ctx.fill();
    ctx.beginPath();
    ctx.arc(at.x, at.y, radius, 0, Math.PI * 2);
    ctx.fillStyle = this.color;
    ctx.fill();
  }
}

/* ---- embedding scatter -------------------------------------------------- */

const DOT_RADIUS = 3;
const VIEWPORT_MOMENTUM = 0.12;
const PICK_RADIUS = 14;

/**
 * The embedding scatter.
 *
 * The viewport eases toward the data's bounds rather than snapping to them.
 * Embeddings genuinely spread apart as classes separate, and a per-frame
 * refit would cancel exactly that motion out, leaving a plot that looks
 * static while the network is doing its most interesting work.
 */
export class ScatterPlot {
  constructor(canvas) {
    this.layer = new Layer(canvas);
    this.frame = null;
    this.viewport = null;
    this.selected = new Set();
    this.classColors = new Map();
    this.hover = null;
  }

  setFrame(embedding) {
    this.frame = embedding;
    if (embedding) this.updateViewport(embedding);
  }

  setClassColors(colors) {
    this.classColors = colors;
  }

  setSelected(selected) {
    this.selected = selected;
  }

  /** Jump the viewport instead of easing - used when the user scrubs. */
  snap() {
    this.viewport = null;
    if (this.frame) this.updateViewport(this.frame);
  }

  updateViewport(embedding) {
    const { x, y } = embedding;
    if (!x.length) return;

    let radius = 0;
    for (let i = 0; i < x.length; i++) {
      radius = Math.max(radius, Math.abs(x[i]), Math.abs(y[i]));
    }
    const target = Math.max(radius * 1.15, 1e-3);
    this.viewport =
      this.viewport === null
        ? target
        : this.viewport + (target - this.viewport) * VIEWPORT_MOMENTUM;
  }

  project(index) {
    const size = Math.min(this.layer.width, this.layer.height);
    const cx = this.layer.width / 2;
    const cy = this.layer.height / 2;
    const scale = (size / 2 - DOT_RADIUS - 4) / (this.viewport || 1);
    return {
      x: cx + this.frame.x[index] * scale,
      y: cy - this.frame.y[index] * scale,
    };
  }

  pick(px, py) {
    if (!this.frame || !this.viewport) return null;
    let best = null;
    let bestDistance = PICK_RADIUS * PICK_RADIUS;
    for (let i = 0; i < this.frame.x.length; i++) {
      const at = this.project(i);
      const distance = (at.x - px) ** 2 + (at.y - py) ** 2;
      if (distance < bestDistance) {
        bestDistance = distance;
        best = i;
      }
    }
    return best;
  }

  setHover(index) {
    this.hover = index;
  }

  colorFor(label) {
    return this.classColors.get(label) ?? cssVar("--ink-muted");
  }

  isDimmed(label) {
    return this.selected.size > 0 && !this.selected.has(label);
  }

  draw() {
    if (!this.layer.measure()) return;
    const { ctx } = this.layer;

    this.drawAxes();
    if (!this.frame || !this.frame.x.length || !this.viewport) return;

    const { x, labels } = this.frame;
    const surface = cssVar("--surface");

    // Dimmed points first so the selected class always sits on top.
    for (const pass of ["dim", "bright"]) {
      for (let i = 0; i < x.length; i++) {
        const label = labels ? labels[i] : null;
        const dimmed = label !== null && this.isDimmed(label);
        if ((pass === "dim") !== dimmed) continue;

        const at = this.project(i);
        ctx.globalAlpha = dimmed ? 0.18 : 0.92;

        ctx.beginPath();
        ctx.arc(at.x, at.y, DOT_RADIUS + 1, 0, Math.PI * 2);
        ctx.fillStyle = surface;
        ctx.fill();

        ctx.beginPath();
        ctx.arc(at.x, at.y, DOT_RADIUS, 0, Math.PI * 2);
        ctx.fillStyle = label === null ? cssVar("--series-1") : this.colorFor(label);
        ctx.fill();
      }
    }
    ctx.globalAlpha = 1;

    if (this.hover !== null && this.hover < x.length) {
      const at = this.project(this.hover);
      const label = labels ? labels[this.hover] : null;
      ctx.beginPath();
      ctx.arc(at.x, at.y, DOT_RADIUS + 4, 0, Math.PI * 2);
      ctx.strokeStyle = label === null ? cssVar("--series-1") : this.colorFor(label);
      ctx.lineWidth = 2;
      ctx.stroke();
    }
  }

  /** A faint origin cross: the batch mean, and the only fixed reference the
   *  projection has. No numeric ticks, since PCA units are not meaningful. */
  drawAxes() {
    const { ctx } = this.layer;
    const cx = Math.round(this.layer.width / 2) + 0.5;
    const cy = Math.round(this.layer.height / 2) + 0.5;

    ctx.strokeStyle = cssVar("--grid");
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(cx, 0);
    ctx.lineTo(cx, this.layer.height);
    ctx.moveTo(0, cy);
    ctx.lineTo(this.layer.width, cy);
    ctx.stroke();

    ctx.fillStyle = cssVar("--ink-muted");
    ctx.font = "10px system-ui, sans-serif";
    ctx.textAlign = "right";
    ctx.fillText("PC 1", this.layer.width - 6, cy - 6);
    ctx.textAlign = "left";
    ctx.fillText("PC 2", cx + 6, 12);
  }
}
