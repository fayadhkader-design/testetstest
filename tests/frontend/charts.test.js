import assert from "node:assert/strict";
import { describe, it } from "node:test";

import {
  GRADIENT_DECADES,
  barFraction,
  formatValue,
} from "../../nnscope/frontend/charts.js";

describe("barFraction", () => {
  it("gives the largest norm a full bar", () => {
    assert.equal(barFraction(5, 5), 1);
  });

  it("places each decade down at an even step", () => {
    // Six decades across the scale, so one decade is a sixth of the width.
    assert.equal(barFraction(1e-1, 1), 1 - 1 / GRADIENT_DECADES);
    assert.equal(barFraction(1e-3, 1), 1 - 3 / GRADIENT_DECADES);
  });

  it("bottoms out past the end of the scale", () => {
    assert.equal(barFraction(1e-9, 1), 0);
  });

  it("is scale invariant: only the ratio matters", () => {
    assert.equal(barFraction(1e-2, 1), barFraction(1e4, 1e6));
  });

  it("treats zero as no bar rather than as the minimum", () => {
    // A zero gradient is a real reading, not a very small one.
    assert.equal(barFraction(0, 1), 0);
  });

  it("refuses non-finite and negative input", () => {
    assert.equal(barFraction(NaN, 1), 0);
    assert.equal(barFraction(Infinity, 1), 0);
    assert.equal(barFraction(-1, 1), 0);
    assert.equal(barFraction(1, 0), 0);
    assert.equal(barFraction(1, NaN), 0);
  });

  it("honours a custom decade span", () => {
    assert.equal(barFraction(1e-1, 1, 2), 0.5);
  });
});

describe("formatValue", () => {
  it("renders integers plainly", () => {
    assert.equal(formatValue(2), "2");
    assert.equal(formatValue(0), "0");
  });

  it("uses exponential notation at the extremes", () => {
    assert.equal(formatValue(1.2345e-9), "1.23e-9");
    assert.equal(formatValue(9.87e12), "9.87e+12");
  });

  it("keeps precision on small fractions", () => {
    assert.equal(formatValue(0.123456), "0.1235");
  });

  it("renders a dash for anything unreadable", () => {
    for (const bad of [null, undefined, NaN, Infinity]) {
      assert.equal(formatValue(bad), "—");
    }
  });
});
