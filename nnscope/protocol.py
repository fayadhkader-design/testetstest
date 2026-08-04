"""Wire format between the training process and the browser.

One JSON object per websocket message, discriminated by ``type``. Frames are
the high-volume message, so their shape is chosen for size: coordinates ship
as parallel flat arrays rather than a list of objects, and floats are rounded
to a precision well past what a scatter plot can resolve.

Everything arriving from a browser is treated as hostile input. A malformed
command must raise :class:`ProtocolError` and be reported back, never take
down a training run that may have been going for hours.
"""

from __future__ import annotations

import json
import math
from collections.abc import Iterable, Sequence
from typing import Any

import numpy as np

# Server -> client
HELLO = "hello"
FRAME = "frame"
BACKFILL = "backfill"
STATE = "state"
ERROR = "error"

# Client -> server
PAUSE = "pause"
RESUME = "resume"
STEP = "step"
SET_LR = "lr"
SHOCK = "shock"

COMMANDS = frozenset({PAUSE, RESUME, STEP, SET_LR, SHOCK})

COORD_PRECISION = 4
METRIC_PRECISION = 6


class ProtocolError(ValueError):
    """A message from a client could not be understood."""


def _finite(value: float) -> float | None:
    """JSON has no NaN or Infinity; diverged losses become null instead."""
    number = float(value)
    return number if math.isfinite(number) else None


def _round(value: float, places: int) -> float | None:
    number = _finite(value)
    return None if number is None else round(number, places)


def _signif(value: float, digits: int = 5) -> float | None:
    """Round to significant figures rather than decimal places.

    Gradient norms routinely span ten orders of magnitude in one model, and
    fixed decimal places would flatten every small one to zero -- which is
    exactly the reading the panel exists to distinguish from a dead layer.
    """
    number = _finite(value)
    if number is None or number == 0:
        return number
    exponent = math.floor(math.log10(abs(number)))
    return round(number, -exponent + (digits - 1))


def encode(message: dict[str, Any]) -> str:
    return json.dumps(message, separators=(",", ":"), allow_nan=False)


def hello(run: dict[str, Any]) -> dict[str, Any]:
    return {"type": HELLO, "run": run}


def state(
    controls: dict[str, Any], learning_rate: float | None, step: int
) -> dict[str, Any]:
    return {
        "type": STATE,
        "controls": controls,
        "lr": None if learning_rate is None else _finite(learning_rate),
        "step": step,
    }


def error(detail: str) -> dict[str, Any]:
    return {"type": ERROR, "detail": detail}


def frame_message(frame: dict[str, Any]) -> dict[str, Any]:
    return {"type": FRAME, "frame": frame}


def backfill(frames: Sequence[dict[str, Any]]) -> dict[str, Any]:
    return {"type": BACKFILL, "frames": list(frames)}


def build_frame(
    step: int,
    elapsed: float,
    metrics: dict[str, float],
    learning_rate: float | None = None,
    coords: np.ndarray | None = None,
    labels: Iterable[int] | None = None,
    explained_variance: float | None = None,
    rotation: float | None = None,
    gradients: dict[str, float] | None = None,
) -> dict[str, Any]:
    """Assemble one JSON-safe frame.

    ``coords`` is an (n, 2) array. Labels, when present, must line up with it
    row for row; mismatched lengths are a bug worth surfacing immediately
    rather than rendering a silently wrong scatter plot.
    """
    frame: dict[str, Any] = {
        "step": int(step),
        "t": _round(elapsed, 3),
        "metrics": {
            str(key): _round(value, METRIC_PRECISION) for key, value in metrics.items()
        },
    }
    if learning_rate is not None:
        frame["lr"] = _finite(learning_rate)

    if coords is not None and len(coords) > 0:
        points = np.asarray(coords, dtype=np.float64)
        if points.ndim != 2 or points.shape[1] != 2:
            raise ValueError(f"coords must be (n, 2), got shape {points.shape}")

        label_list: list[int] | None = None
        if labels is not None:
            label_list = [int(value) for value in labels]
            if len(label_list) != len(points):
                raise ValueError(
                    f"got {len(label_list)} labels for {len(points)} points"
                )

        rounded = np.round(np.nan_to_num(points, posinf=0.0, neginf=0.0), COORD_PRECISION)
        embedding: dict[str, Any] = {
            "x": rounded[:, 0].tolist(),
            "y": rounded[:, 1].tolist(),
        }
        if label_list is not None:
            embedding["labels"] = label_list
        if explained_variance is not None:
            embedding["explained"] = _round(explained_variance, 4)
        if rotation is not None:
            embedding["rotation"] = _round(rotation, 5)
        frame["embedding"] = embedding

    if gradients:
        # Parallel arrays, like the embedding: names repeat every frame, but
        # each frame staying self-contained is what makes rewind a lookup
        # rather than a reconstruction.
        frame["gradients"] = {
            "layers": list(gradients),
            "norms": [_signif(value) for value in gradients.values()],
        }

    return frame


def parse_command(raw: str | bytes) -> dict[str, Any]:
    """Validate one client message, returning a normalized command dict."""
    try:
        message = json.loads(raw)
    except (TypeError, ValueError) as exc:
        raise ProtocolError(f"not valid JSON: {exc}") from exc

    if not isinstance(message, dict):
        raise ProtocolError("expected a JSON object")

    kind = message.get("type")
    if kind not in COMMANDS:
        raise ProtocolError(f"unknown command type {kind!r}")

    if kind == STEP:
        count = message.get("count", 1)
        if not isinstance(count, int) or isinstance(count, bool) or count < 1:
            raise ProtocolError(f"step count must be a positive integer, got {count!r}")
        return {"type": STEP, "count": count}

    if kind == SET_LR:
        return {"type": SET_LR, "value": _positive_number(message.get("value"), "lr")}

    if kind == SHOCK:
        magnitude = message.get("magnitude", 0.5)
        return {"type": SHOCK, "magnitude": _positive_number(magnitude, "magnitude")}

    return {"type": kind}


def _positive_number(value: Any, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ProtocolError(f"{field} must be a number, got {value!r}")
    number = float(value)
    if not math.isfinite(number) or number < 0:
        raise ProtocolError(f"{field} must be finite and >= 0, got {value!r}")
    return number
