"""nnscope - watch a neural network learn, live, and rewind it.

    import nnscope

    scope = nnscope.watch(model, optimizer)
    for x, y in loader:
        loss = criterion(model(x), y)
        loss.backward(); optimizer.step(); optimizer.zero_grad()
        scope.log(loss=loss.item(), labels=y)

Importing this package does not import torch. ``watch`` and ``Scope`` are
resolved lazily so the framework-agnostic pieces -- the frame buffer, the
projection, the wire protocol -- stay usable, and importable, on their own.
"""

from typing import TYPE_CHECKING, Any

__version__ = "0.1.0"

__all__ = ["Scope", "__version__", "watch"]

if TYPE_CHECKING:  # pragma: no cover
    from .session import Scope, watch


def __getattr__(name: str) -> Any:
    if name in ("Scope", "watch"):
        from . import session

        return getattr(session, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(__all__)
