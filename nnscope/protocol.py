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
from typing import Any, Dict, Iterable, List, Sequence

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


def encode(message: Dict[str, Any]) -> str:
    return json.dumps(message, separators=(",", ":"), allow_nan=False)


def hello(run: Dict[str, Any]) -> Dict[str, Any]:
    return {"type": HELLO, "run": run}


def state(controls: Dict[str, Any], learning_rate: float | None, step: int) -> Dict[str, Any]:
    return {
        "type": STATE,
        "controls": controls,
        "lr": None if learning_rate is None else _finite(learning_rate),
        "step": step,
    }


def error(detail: str) -> Dict[str, Any]:
    return {"type": ERROR, "detail": detail}


def frame_message(frame: Dict[str, Any]) -> Dict[str, Any]:
    return {"type": FRAME, "frame": frame}


def backfill(frames: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    return {"type": BACKFILL, "frames": list(frames)}


def build_frame(
    step: int,
    elapsed: float,
    metrics: Dict[str, float],
    learning_rate: float | None = None,
    coords: np.ndarray | None = None,
    labels: Iterable[int] | None = None,
    explained_variance: float | None = None,
    rotation: float | None = None,
) -> Dict[str, Any]:
    """Assemble one JSON-safe frame.

    ``coords`` is an (n, 2) array. Labels, when present, must line up with it
    row for row; mismatched lengths are a bug worth surfacing immediately
    rather than rendering a silently wrong scatter plot.
    """
    frame: Dict[str, Any] = {
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

        label_list: List[int] | None = None
        if labels is not None:
            label_list = [int(value) for value in labels]
            if len(label_list) != len(points):
                raise ValueError(
                    f"got {len(label_list)} labels for {len(points)} points"
                )

        rounded = np.round(np.nan_to_num(points, posinf=0.0, neginf=0.0), COORD_PRECISION)
        embedding: Dict[str, Any] = {
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

    return frame


def parse_command(raw: str | bytes) -> Dict[str, Any]:
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
