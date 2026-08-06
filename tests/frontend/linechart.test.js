import assert from "node:assert/strict";
import { describe, it } from "node:test";

import { LineChart } from "../../nnscope/frontend/charts.js";

/**
 * The chart's geometry and scale decisions are plain arithmetic and need no
 * browser -- only `new Layer(canvas)` touches the canvas, and only to grab a
 * 2d context it never uses until draw(). Stubbing that much lets the maths be
 * tested directly; draw() itself still needs a real browser, since it reads
 * theme colours off the document.
 */
function chart({ width = 300, height = 80 } = {}) {
  const canvas = { getContext: () => ({}) };
  const instance = new LineChart(canvas, { color: "#000" });
  instance.layer.width = width;
  instance.layer.height = height;
  return instance;
}

const series = (...values) => values.map((y, x) => ({ x, y }));

describe("setData", () => {
  it("takes the range from every point, not just the drawn ones", () => {
    // The axes must stay put while rewinding, so the range spans the whole
    // history even when only a prefix is rendered.
    const c = chart();
    c.setData(series(1, 2, 3, 400), 2);

    assert.deepEqual(c.range, [1, 400]);
    assert.equal(c.visible, 2);
  });

  it("defaults to drawing everything", () => {
    const c = chart();
    c.setData(series(1, 2, 3));
    assert.equal(c.visible, 3);
  });

  it("clamps visible into the available points", () => {
    const c = chart();
    c.setData(series(1, 2), 99);
    assert.equal(c.visible, 2);

    c.setData(series(1, 2), -5);
    assert.equal(c.visible, 0);
  });

  it("survives a series with nothing finite in it", () => {
    const c = chart();
    c.setData(series(null, NaN));

    assert.deepEqual(c.range, [0, 1]);
    assert.equal(c.log, false);
  });

  it("ignores nulls when computing the range", () => {
    const c = chart();
    c.setData(series(5, null, 2, null, 9));
    assert.deepEqual(c.range, [2, 9]);
  });
});

describe("log scale selection", () => {
  it("switches on once the span is wide enough to hide detail", () => {
    const c = chart();
    c.setData(series(1, 1000));
    assert.equal(c.log, true);
  });

  it("stays linear for an ordinary range", () => {
    const c = chart();
    c.setData(series(1, 2, 3));
    assert.equal(c.log, false);
  });

  it("refuses a log scale when anything is zero or negative", () => {
    // log10(0) is -Infinity; a loss that touches zero must not blank the plot.
    const c = chart();
    c.setData(series(0, 1000));
    assert.equal(c.log, false);

    c.setData(series(-5, 1000));
    assert.equal(c.log, false);
  });

  it("needs more than one point to decide", () => {
    const c = chart();
    c.setData(series(42));
    assert.equal(c.log, false);
  });
});

describe("paddedRange", () => {
  it("pads a linear range so the extremes are not on the edge", () => {
    const c = chart();
    c.setData(series(0, 10));
    const [low, high] = c.paddedRange();

    assert.ok(low < 0 && high > 10);
  });

  it("does not pad a log range", () => {
    const c = chart();
    c.setData(series(1, 1000));
    assert.deepEqual(c.paddedRange(), [1, 1000]);
  });

  it("opens out a flat series so it has somewhere to draw", () => {
    const c = chart();
    c.setData(series(7, 7, 7));
    const [low, high] = c.paddedRange();

    assert.ok(high > low, "a constant series still needs a non-zero span");
  });
});

describe("lastFiniteIndex", () => {
  it("skips trailing gaps", () => {
    const c = chart();
    c.setData(series(1, 2, null, null));
    assert.equal(c.lastFiniteIndex(), 1);
  });

  it("respects the rewound cut-off", () => {
    const c = chart();
    c.setData(series(1, 2, 3, 4), 2);
    assert.equal(c.lastFiniteIndex(), 1);
  });

  it("is null when nothing is drawable", () => {
    const c = chart();
    c.setData(series(null, null));
    assert.equal(c.lastFiniteIndex(), null);
  });
});

describe("pick", () => {
  it("is null before there is anything to pick", () => {
    assert.equal(chart().pick(50), null);
  });

  it("finds the nearest sample to an x position", () => {
    const c = chart({ width: 300 });
    c.setData(series(0, 1, 2, 3, 4));
    const box = c.plotBox();

    assert.equal(c.pick(box.left), 0);
    assert.equal(c.pick(box.right), 4);
    assert.equal(c.pick((box.left + box.right) / 2), 2);
  });

  it("returns null well outside the plot", () => {
    const c = chart({ width: 300 });
    c.setData(series(0, 1, 2));

    assert.equal(c.pick(-500), null);
    assert.equal(c.pick(5000), null);
  });

  it("never points past the rewound cut-off", () => {
    const c = chart({ width: 300 });
    c.setData(series(0, 1, 2, 3, 4), 2);

    assert.equal(c.pick(c.plotBox().right), 1);
  });
});

describe("project", () => {
  it("puts the lowest value at the bottom and the highest at the top", () => {
    const c = chart({ width: 300, height: 80 });
    c.setData(series(1, 10));

    assert.ok(c.project(0).y > c.project(1).y, "canvas y grows downward");
  });

  it("spreads points left to right across the full history", () => {
    const c = chart({ width: 300 });
    c.setData(series(1, 2, 3));

    assert.ok(c.project(0).x < c.project(1).x);
    assert.ok(c.project(1).x < c.project(2).x);
  });

  it("keeps x positions fixed when only the drawn prefix shrinks", () => {
    // Rewinding walks the endpoint back along a fixed axis rather than
    // restretching the curve, so a point's x must not depend on `visible`.
    const c = chart({ width: 300 });
    c.setData(series(1, 2, 3, 4));
    const full = c.project(1).x;

    c.setData(series(1, 2, 3, 4), 2);

    assert.equal(c.project(1).x, full);
  });
});
