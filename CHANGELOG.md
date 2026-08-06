# Changelog

Notable changes to nnscope. Format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/);
versions follow [semantic versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Fixed

- **A metric name containing a quote froze the dashboard permanently.** Metric
  names arrive through `**kwargs`, which Python does not require to be
  identifiers, so a name is arbitrary text. It was interpolated into markup and
  into generated element ids; a quote truncated the id attribute, two cards ended
  up sharing ids, the second lookup returned `null`, `render()` threw — and since
  the loop rescheduled itself only on the success path, the dashboard stopped and
  never recovered. Silently: the page looked exactly like a run that had gone
  quiet. Cards are now built as elements with `textContent` and held by direct
  reference, so no ids are generated at all.
- Metric names are no longer injected as live markup.
- A render failure now costs one frame instead of the whole dashboard: the
  animation loop reschedules in a `finally`, and logs the fault once.

- A closed `Scope` no longer pins the model it was watching. `close()` left its
  `atexit` registration in place, and that registry holds the bound method, which
  holds the Scope, which holds the model — so every model ever watched stayed
  resident, with its device memory allocated, for the whole process. Sweeps,
  cross-validation loops and re-run notebook cells paid for this permanently.
- Removed a NUL byte from `app.js`. It ran correctly but made the file binary to
  `file`, `grep` and GitHub's diff view, so searching the frontend silently
  returned nothing.

### Added

- **Per-layer gradient norms.** The panel a loss curve cannot replace: a flat loss
  looks identical whether a network converged, an early layer's gradients vanished,
  or a late one is exploding. Log scaled over six decades below the strongest layer.
  Computed only on emitted frames, where the cost measured within noise (−0.3%);
  every step would have cost about 15%. Off with `watch(..., gradients=False)`.
- `examples/vanishing.py` — eight sigmoid layers under plain SGD that sit at chance
  forever, with the gradient panel showing why. `--activation relu` fixes it,
  `--optimizer adam` shows the pathology being papered over, and
  `--activation relu --lr 1.0` fails the opposite way.
- Frontend tests for the bar scale and value formatting.
- Each gradient bar now carries a shaded band showing the range that layer has
  covered over the retained window. The panel previously showed only the instant
  you were looking at, so a layer that collapsed at step 800 and recovered by 900
  left no trace unless you happened to scrub onto exactly that stretch. The band
  stops at the frame in view, so a rewound reading cannot see the future.
- A test that the shipped frontend assets are valid UTF-8 with no control bytes.
- Tests for `LineChart`'s geometry: range, log-scale selection, rewind clamping
  and hit-testing.

### Fixed

- Reconnecting no longer duplicates history. Every reconnect replays the server's
  retained buffer, and the client appended it wholesale on top of what it already
  held — one blip doubled the timeline and folded the loss curve back on itself.
- A dashboard left open across a restart no longer splices two runs together. Runs
  now carry an id, and a changed id tears down the charts, legend and history first.
- `Home` moved the view but left the timeline thumb where it was; the thumb now
  follows the view wherever the move came from.
- A port collision reports which port, that it is nnscope, and how to get past it,
  instead of a bare `[Errno 48]` raised from inside the websockets bind.

### Added

- Keyboard control for the timeline: `←`/`→` to scrub (`Shift` for ten), `Home` for
  the oldest retained frame, `End` for live. Single-stepping training moved from
  `→` to `.`, so moving the view and moving the run are no longer the same gesture.
- Frontend tests, on node's built-in runner. Still no dependencies and no build step.
- CI jobs for the frontend tests and strict type checking.
- Dependabot for GitHub Actions.

### Changed

- Minimum Python is now 3.10. The previous 3.9 floor had never been run; the suite
  is now exercised on 3.10 through 3.13 in CI.
- Annotations use PEP 585 built-in generics throughout.

### Added

- `py.typed` marker, so downstream type checkers stop ignoring the annotations.
- CI running tests on 3.10–3.13 plus a lint job.
- Ruff configuration with an explicit rule set.

## [0.1.0]

First working version.

### Added

- `nnscope.watch(model, optimizer)` and `scope.log(**metrics, labels=...)` — the
  whole integration surface.
- Live embedding scatter of the penultimate representation, projected to 2D with a
  basis that persists across frames via orthogonal Procrustes alignment, so the plot
  shows the embedding moving rather than PCA re-deciding which way is up.
- Automatic discovery of the layer to capture, by execution order rather than
  definition order.
- Rewind: a bounded frame window with a timeline scrubber that snaps every panel
  back to any retained step.
- Live controls — pause, single-step, learning rate, and a weight shock that
  perturbs each tensor by a fraction of its own standard deviation.
- Metric charts built from whatever the training loop reports, one series per chart.
- Dashboard and websocket served from a single port, with no build step.
- Light and dark themes, both validated for contrast and colour-vision separation
  against the surface each actually renders on.
- Table view of every value the charts draw.
- `examples/spirals.py` (no dataset needed) and `examples/mnist.py`.

[Unreleased]: https://github.com/fayadhkader-design/testetstest/compare/main...HEAD
