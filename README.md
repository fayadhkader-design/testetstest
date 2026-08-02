# nnscope

Watch a neural network learn, live — and rewind it.

Two lines in any PyTorch training loop opens a dashboard that shows the network's
internal representation reorganizing itself in real time, with a timeline you can
drag backwards through.

```python
import nnscope

scope = nnscope.watch(model, optimizer)

for x, y in loader:
    loss = criterion(model(x), y)
    loss.backward(); optimizer.step(); optimizer.zero_grad()
    scope.log(loss=loss.item(), labels=y)
```

That's the whole integration. No config file, no logdir, no account.

---

## Why this instead of a loss curve

A loss curve tells you *that* a network is learning. It doesn't show you **what** it
learned.

nnscope captures the activations feeding the final layer — the representation the
network actually built — and projects them to 2D. At the start of training you see one
undifferentiated blob. Over the next few hundred steps it pulls itself apart into
clean per-class lobes. That separation *is* the learning, and watching it happen is a
very different experience from watching a number go down.

The projection is the hard part, and it's where most naive versions of this fall over
— see [How it works](#how-it-works).

## Install

```bash
pip install -e .
```

Requires Python 3.9+. `numpy` and `websockets` come with it; bring your own `torch`.
For the examples:

```bash
pip install -e ".[examples]"
```

## Try it

```bash
python examples/spirals.py
```

Three interleaved spirals, which are not linearly separable, so the network has no
choice but to build a representation that untangles them. No dataset download — the
data is generated. The dashboard opens automatically.

```bash
python examples/mnist.py
```

A small CNN on MNIST. Downloads ~11 MB on first run.

## What you can do while it's running

| | |
|---|---|
| **Rewind** | Drag the timeline. Every panel snaps back to that step. Release at the end to go live again. |
| **Pause / Step** | Freeze between steps and advance one at a time. `Space` and `→` work as shortcuts. |
| **Learning rate** | Type a new one. It reaches the optimizer on the next step. |
| **Shock** | Perturb every weight tensor by a fraction of its own standard deviation, then watch the clusters collapse and re-form. The fastest way to build intuition that training is a dynamical system and not a monotonic march downhill. |
| **Isolate a class** | Click a legend chip to dim everything else. |
| **Table** | Every value the charts draw, as numbers. |

## API

`nnscope.watch(model, optimizer=None, **kwargs)` returns a `Scope`.

| Argument | Default | |
|---|---|---|
| `optimizer` | `None` | Enables the learning-rate readout and live adjustment. |
| `embedding` | auto | Layer whose *input* to capture. A module or a dotted name. |
| `port` | `8420` | `0` picks a free one. |
| `capacity` | `600` | Frames retained for rewind. Bounds memory. |
| `every` | `1` | Emit at most one frame per N steps. |
| `min_interval` | `0.05` | Seconds between frames, so a fast loop can't flood the socket. |
| `max_points` | `512` | Points sampled per frame. |
| `momentum` | `0.15` | How fast the projection basis tracks new data. |
| `open_browser` | `True` | |
| `start_paused` | `False` | Useful when you want to be watching before step 1. |

`scope.log(labels=None, **metrics)` — call once per step, after the optimizer step.
Any keyword becomes a chart; `labels` colors the scatter. `Scope` is a context manager
and `scope.close()` is idempotent.

## How it works

**Finding the representation.** nnscope attaches to the input of the last
parameterized layer that *executes* — not the last one defined. A custom `forward()`
can run submodules in any order it likes, and models that declare the head early are
common enough that definition-order discovery silently captures the wrong tensor. One
forward pass is spent learning the real order; only layer names are recorded during
it, so no activations get pinned in memory.

**Keeping the projection stable.** This is the part that makes or breaks the whole
idea. Running PCA fresh on every frame produces an unusable animation: PCA's basis is
sign- and rotation-ambiguous, so consecutive frames come back mirrored or spun and the
plot strobes violently. Real structure is completely buried under basis churn.

nnscope treats the basis as something that persists. Each frame's new basis is rotated
onto the previous one via orthogonal Procrustes, then blended in with momentum, with
QR column signs pinned so the smoothing step can't reintroduce the flipping the
alignment just removed. What moves on screen is the embedding moving.

Scale is deliberately *not* normalized per frame. Embeddings genuinely spread apart as
classes separate, and a per-frame refit would cancel out exactly the effect worth
watching. The viewport eases instead.

**Staying out of the way.** Adding nnscope must not change how a script trains.
Frames are throttled by both step stride and wall clock, projection runs on a few
hundred sampled points, captured tensors are detached immediately, and a malformed
frame is logged and dropped rather than raised into your training loop. Slow browser
clients have frames dropped rather than buffered — stale frames are worthless in a
live view, and unbounded queues would let a throttled background tab grow your
training process's memory.

## Notes and limits

- **Color caps at eight classes.** Past that, a generated ninth hue isn't reliably
  distinguishable under color-vision deficiency, so nnscope stops assigning colors and
  spends them only on classes you highlight from the legend. Position carries the rest.
  The palette is validated for contrast and CVD separation against both light and dark
  surfaces.
- **Auto-discovery costs one frame.** The first step is spent learning execution order.
  Pass `embedding=` explicitly to skip it.
- **The scatter shows one batch**, not the whole dataset — the sample you're watching
  changes every step. Hold a batch fixed if you want strictly frame-to-frame motion.
- **Two measures never share a plot.** Each metric gets its own card with its own
  scale. Dual axes invent correlations that aren't in the data.
- Single run, single process. No multi-run comparison, no distributed training.

## Layout

```
nnscope/
  session.py       watch() / log() - the public API
  instrument.py    forward-hook capture, execution-order head discovery
  projection.py    rotation-stable PCA
  buffer.py        bounded frame window backing rewind
  control.py       pause / step / lr / shock, across threads
  protocol.py      wire format, client-command validation
  server.py        websocket + static host on one port
  frontend/        vanilla JS, canvas, no build step
examples/
tests/
```

Run the tests with `pytest`.

## License

MIT.
