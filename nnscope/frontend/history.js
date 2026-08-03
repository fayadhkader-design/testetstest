/* The client's mirror of the server's frame window.
 *
 * Split out from app.js because this is the only part of the dashboard with
 * logic worth testing in isolation -- everything else there is DOM wiring.
 * Nothing here touches the document, so it runs under `node --test`.
 *
 * The rules it enforces exist because the socket reconnects on its own and
 * every reconnect replays the server's entire retained buffer.
 */

/** Do two hello payloads describe the same run?
 *
 * A server older than run ids sends no id at all; treating that as "same run"
 * keeps the dashboard working against it rather than resetting on every
 * reconnect. */
export function isSameRun(previous, next) {
  if (!previous || !next) return false;
  if (previous.id === undefined || next.id === undefined) return true;
  return previous.id === next.id;
}

/** Would this frame add anything to the history?
 *
 * Steps are strictly increasing within a run, so anything at or behind the
 * newest frame we hold is replay from a reconnect. */
export function isNewFrame(frames, frame) {
  if (!frame || typeof frame.step !== "number") return false;
  const newest = frames[frames.length - 1];
  return !newest || frame.step > newest.step;
}

/**
 * Append a frame if it is new, trimming to capacity. Returns whether it was
 * taken, so the caller knows if there is anything fresh to render.
 */
export function appendFrame(frames, frame, capacity) {
  if (!isNewFrame(frames, frame)) return false;

  frames.push(frame);
  while (frames.length > capacity) frames.shift();
  return true;
}

/**
 * Index of the frame to display for a given step, or the newest frame when
 * `step` is null (live).
 *
 * Steps are looked up by value rather than by position because the window
 * slides: the frame at index 40 is a different frame a second later, so a
 * held position would drift backwards through history on its own.
 */
export function indexForStep(frames, step) {
  if (!frames.length) return -1;
  if (step === null || step === undefined) return frames.length - 1;

  const found = frames.findIndex((frame) => frame.step >= step);
  return found === -1 ? frames.length - 1 : found;
}
