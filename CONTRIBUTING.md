# Contributing

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev,examples]"
pytest -q
ruff check nnscope tests examples
```

Then run `python examples/spirals.py` — if the dashboard opens and three clusters
pull apart, your checkout is working.

## What this project is trying to be

A live view into one training run that costs nothing to add and nothing to run.
Two constraints follow from that, and most review comments trace back to one of them:

1. **Adding nnscope must not change how a script trains.** No holding the autograd
   graph, no work proportional to model size in the hot path, no exceptions escaping
   into someone's training loop. A visualizer that kills a twelve-hour run is worse
   than no visualizer.
2. **What's on screen must be true.** The projection is stabilized rather than
   re-fit precisely so the motion you see is the embedding moving. Anything that
   makes a plot prettier by making it less faithful is the wrong trade.

## Where things live

| | |
|---|---|
| `nnscope/session.py` | `watch()` / `log()` — the public API |
| `nnscope/instrument.py` | forward-hook capture, execution-order head discovery |
| `nnscope/projection.py` | rotation-stable PCA |
| `nnscope/buffer.py` | bounded frame window backing rewind |
| `nnscope/control.py` | pause / step / lr / shock, across threads |
| `nnscope/protocol.py` | wire format and client-command validation |
| `nnscope/server.py` | websocket + static host |
| `nnscope/frontend/` | vanilla JS, canvas, no build step |

The frontend has no build step on purpose. Please don't add one — being able to
read `app.js` in the browser's own devtools is worth more here than any bundler.

## Tests

Every module has a matching `tests/test_*.py`. New behaviour needs a test, and the
useful ones assert on a property rather than a golden value — `test_projection.py`
drifts a point cloud over 60 frames and requires consecutive projections stay
correlated, which is the actual contract and would catch a regression that a
hardcoded array comparison would sail past.

Threading and socket code is tested against real threads and real sockets, not
mocks. Two bugs so far only existed in the interaction between them.

## Style

Ruff enforces the rest (`line-length = 92`). Two things it can't check:

- **Comments say why, not what.** If a line needs explaining, explain the reason
  it's written that way, especially when the obvious alternative is wrong.
- **Commit messages explain the reasoning.** The subject says what changed; the
  body says why it needed to.

## Good first issues

- A gradient-norm panel. The instrumentation layer is built to take more probes
  than it currently ships; per-layer gradient norms are the most requested thing
  a loss curve can't tell you.
- Framework adapters. The core data layer is plain numpy and never imports torch —
  a TensorFlow or JAX capture layer would slot in beside `instrument.py`.
- Keyboard navigation for the timeline scrubber.
- A texture or shape channel for the scatter, so class identity survives print
  and forced-colors mode.
