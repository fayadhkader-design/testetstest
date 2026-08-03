# Changelog

Notable changes to nnscope. Format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/);
versions follow [semantic versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

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
