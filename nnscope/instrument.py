"""Capturing a model's internal representation without touching its code.

The representation worth watching is the one feeding the final classification
layer: by then the network has done all its work, and everything after is a
linear readout. Rather than asking the user to name that layer, we find it by
watching which parameterized layer runs *last* during a real forward pass and
attaching to its input.

Execution order is used deliberately instead of definition order. A custom
``forward()`` is free to run submodules in any sequence it likes, and plenty
of real models declare the head early or reuse layers; only the actual call
order tells the truth about which layer is the head.
"""

from __future__ import annotations

import threading
from typing import List, Tuple

import numpy as np
import torch
from torch import nn


class EmbeddingNotFound(RuntimeError):
    """Raised when no parameterized layer could be identified to attach to."""


def parameterized_leaves(model: nn.Module) -> List[Tuple[str, nn.Module]]:
    """Leaf modules that own parameters, in definition order.

    Containers are skipped -- attaching to an ``nn.Sequential`` would capture
    the same tensor as its first child. Parameter-free leaves like ReLU and
    Dropout are skipped too, since they are not where representation lives.
    """
    leaves = []
    for name, module in model.named_modules():
        has_children = any(True for _ in module.children())
        owns_params = any(True for _ in module.parameters(recurse=False))
        if not has_children and owns_params:
            leaves.append((name or "<root>", module))
    return leaves


def _to_matrix(tensor: torch.Tensor, max_points: int) -> np.ndarray:
    """Detach a captured tensor into a plain (n, d) float32 array."""
    flat = tensor.detach()
    if flat.ndim == 1:
        flat = flat.unsqueeze(0)
    flat = flat.reshape(flat.shape[0], -1)
    if flat.shape[0] > max_points:
        flat = flat[:max_points]
    return flat.to(dtype=torch.float32, device="cpu").numpy()


class EmbeddingCapture:
    """Attaches to a model and keeps the most recent penultimate activation.

    Args:
        model: The module to instrument.
        module: Optional explicit layer to capture the *input* of. Accepts a
            module instance or a dotted name from ``named_modules()``. When
            omitted the head is discovered automatically.
        max_points: Cap on rows kept per frame. A scatter plot saturates well
            before a few hundred points, and the cap keeps the wire payload
            and the SVD both cheap.
        max_dims: Above this width the activation is squeezed through a fixed
            random projection. Johnson-Lindenstrauss keeps pairwise geometry
            close to intact, and it bounds SVD cost for very wide heads.

    Auto-discovery costs one frame: the first forward pass is spent learning
    the execution order, and capture begins on the second.
    """

    def __init__(
        self,
        model: nn.Module,
        module: nn.Module | str | None = None,
        max_points: int = 512,
        max_dims: int = 1024,
        seed: int = 0,
    ) -> None:
        self._model = model
        self._max_points = max_points
        self._max_dims = max_dims
        self._seed = seed

        self._lock = threading.Lock()
        self._latest: np.ndarray | None = None
        self._handles: List[torch.utils.hooks.RemovableHandle] = []
        self._squeeze: np.ndarray | None = None

        self._name: str | None = None
        self._calibrating = False
        self._order: List[str] = []

        candidates = parameterized_leaves(model)
        if not candidates:
            raise EmbeddingNotFound(
                "no parameterized leaf modules found; pass module= to choose "
                "the layer to capture explicitly"
            )

        if module is not None:
            name, resolved = self._resolve(module)
            self._attach(name, resolved)
        else:
            self._begin_calibration(candidates)

    @property
    def layer_name(self) -> str | None:
        """Dotted name of the layer being captured, once known."""
        return self._name

    @property
    def calibrating(self) -> bool:
        return self._calibrating

    def take(self) -> np.ndarray | None:
        """Consume the activation captured since the last call, if any."""
        with self._lock:
            latest, self._latest = self._latest, None

        if self._calibrating and self._order:
            self._finish_calibration()
        return latest

    def close(self) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles.clear()

    def __enter__(self) -> "EmbeddingCapture":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    # -- internals ---------------------------------------------------------

    def _resolve(self, module: nn.Module | str) -> Tuple[str, nn.Module]:
        if isinstance(module, str):
            for name, candidate in self._model.named_modules():
                if name == module:
                    return name, candidate
            raise EmbeddingNotFound(f"no module named {module!r} in this model")

        for name, candidate in self._model.named_modules():
            if candidate is module:
                return name, candidate
        raise EmbeddingNotFound("the given module is not part of this model")

    def _begin_calibration(self, candidates: List[Tuple[str, nn.Module]]) -> None:
        """Watch one forward pass to learn which layer runs last."""
        self._calibrating = True
        self._order = []

        def record(name: str):
            def hook(_module: nn.Module, _inputs: tuple) -> None:
                # Names only. Holding the tensors would pin every candidate
                # layer's activations in memory for the whole pass.
                self._order.append(name)

            return hook

        for name, module in candidates:
            self._handles.append(module.register_forward_pre_hook(record(name)))

    def _finish_calibration(self) -> None:
        head_name = self._order[-1]
        head = dict(self._model.named_modules())[head_name]

        self.close()
        self._calibrating = False
        self._order = []
        self._attach(head_name, head)

    def _attach(self, name: str, module: nn.Module) -> None:
        self._name = name

        def hook(_module: nn.Module, inputs: tuple) -> None:
            if not inputs or not isinstance(inputs[0], torch.Tensor):
                return
            matrix = _to_matrix(inputs[0], self._max_points)
            matrix = self._maybe_squeeze(matrix)
            with self._lock:
                self._latest = matrix

        self._handles.append(module.register_forward_pre_hook(hook))

    def _maybe_squeeze(self, matrix: np.ndarray) -> np.ndarray:
        """Random-project very wide activations down to a workable width."""
        width = matrix.shape[1]
        if width <= self._max_dims:
            return matrix
        if self._squeeze is None or self._squeeze.shape[0] != width:
            rng = np.random.default_rng(self._seed)
            self._squeeze = rng.standard_normal(
                (width, self._max_dims)
            ).astype(np.float32) / np.sqrt(self._max_dims)
        return matrix @ self._squeeze
