"""The public API: attach a scope to a model, then log once per step.

    scope = nnscope.watch(model, optimizer)
    ...
    loss.backward(); optimizer.step()
    scope.log(loss=loss.item(), labels=y)

Everything else -- hooks, projection, the server, the browser -- is arranged
by those two calls. The design constraint throughout is that adding nnscope
to a script must never change how that script trains. Frames are throttled
rather than emitted per step, the projection runs on at most a few hundred
sampled points, and nothing touches the graph or the optimizer unless the
user asks for it from the dashboard.
"""

from __future__ import annotations

import atexit
import logging
import time
import webbrowser
from typing import Any, Dict, Iterable, Sequence

import numpy as np
import torch
from torch import nn

from . import protocol
from .buffer import FrameBuffer
from .control import Controls
from .instrument import EmbeddingCapture
from .projection import StableProjector
from .server import DashboardServer

logger = logging.getLogger("nnscope")


class Scope:
    """A live view of one training run.

    Args:
        model: The model to instrument.
        optimizer: Optional. Supplying it enables the learning-rate readout
            and live adjustment from the dashboard.
        embedding: Layer to capture the input of. Auto-discovered when omitted.
        port: Dashboard port. Pass 0 to let the OS choose.
        capacity: Frames retained for rewind. Memory is bounded by this.
        every: Emit at most one frame per N steps.
        min_interval: Seconds between emitted frames. A loop running at
            thousands of steps per second would otherwise spend real time
            serializing frames no one can see.
        open_browser: Open the dashboard on start.
    """

    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer | None = None,
        embedding: nn.Module | str | None = None,
        port: int = 8420,
        host: str = "127.0.0.1",
        capacity: int = 600,
        every: int = 1,
        min_interval: float = 0.05,
        max_points: int = 512,
        momentum: float = 0.15,
        open_browser: bool = True,
        name: str | None = None,
        start_paused: bool = False,
    ) -> None:
        if every < 1:
            raise ValueError(f"every must be >= 1, got {every}")

        self._model = model
        self._optimizer = optimizer
        self._name = name or type(model).__name__
        self._every = every
        self._min_interval = min_interval

        self._capture = EmbeddingCapture(model, module=embedding, max_points=max_points)
        self._projector = StableProjector(momentum=momentum)
        self._buffer = FrameBuffer(capacity=capacity)
        self._controls = Controls(paused=start_paused)

        self._step = 0
        self._started = time.monotonic()
        self._last_emit = 0.0
        self._closed = False

        self._server = DashboardServer(
            buffer=self._buffer,
            controls=self._controls,
            run_info=self._run_info,
            status=self._status,
            host=host,
            port=port,
        )
        self.url = self._server.start()
        atexit.register(self.close)

        print(f"nnscope: watching {self._name} at {self.url}", flush=True)
        if open_browser:
            webbrowser.open(self.url)

    # -- properties --------------------------------------------------------

    @property
    def step(self) -> int:
        return self._step

    @property
    def paused(self) -> bool:
        return self._controls.paused

    @property
    def layer_name(self) -> str | None:
        """Which layer is being captured, once discovery has settled."""
        return self._capture.layer_name

    # -- the one call a training loop makes --------------------------------

    def log(self, labels: Iterable[int] | torch.Tensor | None = None, **metrics: float) -> None:
        """Record one training step.

        Call once per step, after the optimizer step. Blocks while the run is
        paused from the dashboard, which is what makes pausing work at all.
        """
        if self._closed:
            return

        self._step += 1
        embeddings = self._capture.take()

        if self._should_emit():
            self._emit(embeddings, labels, metrics)

        self._apply_pending()
        self._controls.gate()

    def close(self) -> None:
        """Detach hooks and stop the server. Idempotent."""
        if self._closed:
            return
        self._closed = True
        self._controls.release()
        self._capture.close()
        self._server.stop()

    def __enter__(self) -> "Scope":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    # -- internals ---------------------------------------------------------

    def _should_emit(self) -> bool:
        if self._step % self._every != 0:
            return False
        now = time.monotonic()
        if now - self._last_emit < self._min_interval:
            return False
        self._last_emit = now
        return True

    def _emit(
        self,
        embeddings: np.ndarray | None,
        labels: Iterable[int] | torch.Tensor | None,
        metrics: Dict[str, Any],
    ) -> None:
        coords = None
        explained = None
        rotation = None
        label_list: Sequence[int] | None = None

        if embeddings is not None and len(embeddings) > 0:
            projection = self._projector.fit_transform(embeddings)
            coords = projection.coords
            explained = projection.explained_variance
            rotation = projection.rotation
            label_list = self._normalize_labels(labels, len(coords))

        try:
            frame = protocol.build_frame(
                step=self._step,
                elapsed=time.monotonic() - self._started,
                metrics={k: float(v) for k, v in metrics.items()},
                learning_rate=self._learning_rate(),
                coords=coords,
                labels=label_list,
                explained_variance=explained,
                rotation=rotation,
            )
        except (TypeError, ValueError):
            # A visualizer must never be the reason a training run dies.
            logger.exception("nnscope: dropped a malformed frame")
            return

        self._buffer.append(frame)
        self._server.publish(protocol.frame_message(frame))

    @staticmethod
    def _normalize_labels(
        labels: Iterable[int] | torch.Tensor | None, count: int
    ) -> Sequence[int] | None:
        """Trim labels to the sampled rows the capture actually kept."""
        if labels is None:
            return None
        if isinstance(labels, torch.Tensor):
            values = labels.detach().reshape(-1).cpu().tolist()
        else:
            values = list(labels)
        if len(values) < count:
            return None  # mismatched batch; better no colour than wrong colour
        return [int(value) for value in values[:count]]

    def _learning_rate(self) -> float | None:
        if self._optimizer is None or not self._optimizer.param_groups:
            return None
        return float(self._optimizer.param_groups[0]["lr"])

    def _apply_pending(self) -> None:
        requested_lr = self._controls.take_learning_rate()
        if requested_lr is not None and self._optimizer is not None:
            for group in self._optimizer.param_groups:
                group["lr"] = requested_lr
            logger.info("nnscope: learning rate set to %g", requested_lr)

        magnitude = self._controls.take_shock()
        if magnitude is not None:
            self._shock(magnitude)
            logger.info("nnscope: shocked weights at %g sigma", magnitude)

    @torch.no_grad()
    def _shock(self, magnitude: float) -> None:
        """Perturb each weight tensor by a fraction of its own spread.

        Scaling per-tensor rather than globally keeps the damage proportional:
        a uniform sigma would obliterate small-scale layers and barely graze
        large ones.
        """
        for param in self._model.parameters():
            if param.numel() < 2:
                continue
            spread = param.detach().float().std()
            if not torch.isfinite(spread) or spread == 0:
                continue
            param.add_(torch.randn_like(param) * (spread * magnitude))

    def _device(self) -> str:
        for param in self._model.parameters():
            return str(param.device)
        return "cpu"

    def _run_info(self) -> Dict[str, Any]:
        return {
            "model": self._name,
            "layer": self._capture.layer_name or "discovering…",
            "device": self._device(),
            "capacity": self._buffer.capacity,
        }

    def _status(self) -> Dict[str, Any]:
        return {
            "controls": self._controls.snapshot(),
            "lr": self._learning_rate(),
            "step": self._step,
        }


def watch(model: nn.Module, optimizer: torch.optim.Optimizer | None = None, **kwargs: Any) -> Scope:
    """Attach a scope to a model and open the dashboard. See :class:`Scope`."""
    return Scope(model, optimizer=optimizer, **kwargs)
