import assert from "node:assert/strict";
import { describe, it } from "node:test";

import {
  appendFrame,
  indexForStep,
  isNewFrame,
  isSameRun,
} from "../../nnscope/frontend/history.js";

const frame = (step) => ({ step });

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
