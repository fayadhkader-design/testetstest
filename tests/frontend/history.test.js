import assert from "node:assert/strict";
import { describe, it } from "node:test";

import {
  appendFrame,
  gradientRange,
  indexForStep,
  isNewFrame,
  isSameRun,
} from "../../nnscope/frontend/history.js";

const frame = (step) => ({ step });

const withGradients = (step, layers, norms) => ({
  step,
  gradients: { layers, norms },
});

describe("isSameRun", () => {
  it("matches on run id", () => {
    assert.equal(isSameRun({ id: "abc" }, { id: "abc" }), true);
    assert.equal(isSameRun({ id: "abc" }, { id: "xyz" }), false);
  });

  it("treats a missing id as the same run", () => {
    // An older server sends no id; resetting on every reconnect would be
    // worse than assuming continuity.
    assert.equal(isSameRun({ model: "Net" }, { model: "Net" }), true);
  });

  it("is false when either side is absent", () => {
    assert.equal(isSameRun(null, { id: "abc" }), false);
    assert.equal(isSameRun({ id: "abc" }, null), false);
  });
});

describe("isNewFrame", () => {
  it("accepts anything into an empty history", () => {
    assert.equal(isNewFrame([], frame(0)), true);
  });

  it("accepts strictly increasing steps", () => {
    assert.equal(isNewFrame([frame(4)], frame(5)), true);
  });

  it("rejects a replayed step", () => {
    assert.equal(isNewFrame([frame(5)], frame(5)), false);
    assert.equal(isNewFrame([frame(5)], frame(2)), false);
  });

  it("rejects malformed frames", () => {
    assert.equal(isNewFrame([], null), false);
    assert.equal(isNewFrame([], {}), false);
  });
});

describe("appendFrame", () => {
  it("reports whether the frame was taken", () => {
    const frames = [];
    assert.equal(appendFrame(frames, frame(1), 10), true);
    assert.equal(appendFrame(frames, frame(1), 10), false);
    assert.deepEqual(frames.map((f) => f.step), [1]);
  });

  it("trims to capacity, keeping the newest", () => {
    const frames = [];
    for (let step = 0; step < 10; step++) appendFrame(frames, frame(step), 3);

    assert.deepEqual(frames.map((f) => f.step), [7, 8, 9]);
  });

  it("drops a whole replayed buffer without corrupting history", () => {
    // The exact reconnect case: the server resends everything it holds.
    const frames = [];
    const served = [1, 2, 3, 4, 5].map(frame);
    served.forEach((f) => appendFrame(frames, f, 100));

    served.forEach((f) => appendFrame(frames, f, 100));

    assert.deepEqual(frames.map((f) => f.step), [1, 2, 3, 4, 5]);
  });

  it("still accepts frames past the replayed tail", () => {
    const frames = [];
    [1, 2, 3].map(frame).forEach((f) => appendFrame(frames, f, 100));

    [2, 3, 4].map(frame).forEach((f) => appendFrame(frames, f, 100));

    assert.deepEqual(frames.map((f) => f.step), [1, 2, 3, 4]);
  });
});

describe("gradientRange", () => {
  it("is empty when nothing carries gradients", () => {
    assert.equal(gradientRange([frame(1), frame(2)]).size, 0);
    assert.equal(gradientRange([]).size, 0);
  });

  it("tracks the low and high water marks per layer", () => {
    const frames = [
      withGradients(1, ["a", "b"], [1.0, 5.0]),
      withGradients(2, ["a", "b"], [0.1, 7.0]),
      withGradients(3, ["a", "b"], [0.5, 6.0]),
    ];

    assert.deepEqual(gradientRange(frames).get("a"), { min: 0.1, max: 1.0 });
    assert.deepEqual(gradientRange(frames).get("b"), { min: 5.0, max: 7.0 });
  });

  it("remembers an excursion the current frame has recovered from", () => {
    // The whole point: at step 3 the layer looks healthy again, but it
    // collapsed six decades at step 2 and that has to remain visible.
    const frames = [
      withGradients(1, ["early"], [1e-2]),
      withGradients(2, ["early"], [1e-8]),
      withGradients(3, ["early"], [1e-2]),
    ];

    assert.equal(gradientRange(frames).get("early").min, 1e-8);
  });

  it("stops at upTo so a rewound view cannot see the future", () => {
    const frames = [
      withGradients(1, ["a"], [1.0]),
      withGradients(2, ["a"], [0.5]),
      withGradients(3, ["a"], [0.001]),
    ];

    assert.deepEqual(gradientRange(frames, 2).get("a"), { min: 0.5, max: 1.0 });
  });

  it("skips non-finite readings rather than poisoning the range", () => {
    const frames = [
      withGradients(1, ["a"], [1.0]),
      withGradients(2, ["a"], [null]),
      withGradients(3, ["a"], [2.0]),
    ];

    assert.deepEqual(gradientRange(frames).get("a"), { min: 1.0, max: 2.0 });
  });

  it("copes with layers appearing partway through", () => {
    const frames = [
      withGradients(1, ["a"], [1.0]),
      withGradients(2, ["a", "b"], [2.0, 9.0]),
    ];

    assert.deepEqual(gradientRange(frames).get("b"), { min: 9.0, max: 9.0 });
  });

  it("tolerates frames with no gradients mixed in", () => {
    const frames = [withGradients(1, ["a"], [1.0]), frame(2), withGradients(3, ["a"], [3.0])];

    assert.deepEqual(gradientRange(frames).get("a"), { min: 1.0, max: 3.0 });
  });
});

describe("indexForStep", () => {
  const frames = [10, 20, 30, 40].map(frame);

  it("returns -1 for an empty history", () => {
    assert.equal(indexForStep([], null), -1);
    assert.equal(indexForStep([], 5), -1);
  });

  it("returns the newest frame when live", () => {
    assert.equal(indexForStep(frames, null), 3);
    assert.equal(indexForStep(frames, undefined), 3);
  });

  it("finds an exact step", () => {
    assert.equal(indexForStep(frames, 20), 1);
  });

  it("rounds up to the next retained step", () => {
    // Frames are throttled, so the exact step being sought may never have
    // been sent; the next one held is the honest answer.
    assert.equal(indexForStep(frames, 25), 2);
  });

  it("clamps past the end to the newest frame", () => {
    assert.equal(indexForStep(frames, 999), 3);
  });

  it("clamps before the start to the oldest retained frame", () => {
    assert.equal(indexForStep(frames, 1), 0);
  });
});
