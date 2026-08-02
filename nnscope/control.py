"""Live control of a running training loop from the browser.

The training loop runs on the main thread; the websocket server runs on
another. This module is the only place they rendezvous, so all of the
threading care lives here rather than being smeared across the codebase.

Pausing works by having the training thread call :meth:`Controls.gate` once
per step, which blocks until the UI lets it through. That is deliberate: it
freezes training exactly between steps, so a paused run holds a coherent
state rather than one caught mid-backward.
"""

from __future__ import annotations

import threading
from typing import Any


class Controls:
    """Thread-safe pause/step/learning-rate/shock state shared with the UI."""

    def __init__(self, paused: bool = False) -> None:
        self._cond = threading.Condition()
        self._paused = paused
        self._budget = 0
        self._pending_lr: float | None = None
        self._pending_shock: float | None = None
        self._released = False
        self._resumed_at_least_once = threading.Event()

    # -- state ------------------------------------------------------------

    @property
    def paused(self) -> bool:
        with self._cond:
            return self._paused

    @property
    def released(self) -> bool:
        with self._cond:
            return self._released

    def snapshot(self) -> dict[str, Any]:
        """A JSON-safe view of control state, for broadcasting to clients."""
        with self._cond:
            return {"paused": self._paused, "stepBudget": self._budget}

    # -- commands from the UI thread --------------------------------------

    def pause(self) -> None:
        with self._cond:
            self._paused = True
            self._budget = 0
            self._cond.notify_all()

    def resume(self) -> None:
        with self._cond:
            self._paused = False
            self._budget = 0
            self._cond.notify_all()
        self._resumed_at_least_once.set()

    def step(self, count: int = 1) -> None:
        """Pause, but let ``count`` further steps through first."""
        if count < 1:
            raise ValueError(f"count must be >= 1, got {count}")
        with self._cond:
            self._paused = True
            self._budget += count
            self._cond.notify_all()

    def set_learning_rate(self, value: float) -> None:
        if value < 0:
            raise ValueError(f"learning rate must be >= 0, got {value}")
        with self._cond:
            self._pending_lr = float(value)

    def shock(self, magnitude: float = 0.5) -> None:
        """Queue a weight perturbation, scaled per-tensor by its own std.

        Watching a network absorb a shock and re-separate its classes is the
        clearest demonstration that training is a dynamical system and not a
        monotonic march downhill.
        """
        if magnitude < 0:
            raise ValueError(f"magnitude must be >= 0, got {magnitude}")
        with self._cond:
            self._pending_shock = float(magnitude)

    def release(self) -> None:
        """Permanently un-block the training thread.

        Called when the server shuts down. Without it, a run paused from a
        browser tab that then closes would hang forever.
        """
        with self._cond:
            self._released = True
            self._paused = False
            self._cond.notify_all()

    # -- consumed by the training thread -----------------------------------

    def gate(self, timeout: float | None = None) -> bool:
        """Block until training may proceed. Returns False if it timed out."""
        with self._cond:
            if self._released:
                return True
            if not self._paused:
                return True
            if self._budget > 0:
                self._budget -= 1
                return True

            def ready() -> bool:
                return self._released or not self._paused or self._budget > 0

            if not self._cond.wait_for(ready, timeout=timeout):
                return False

            if not self._released and self._paused and self._budget > 0:
                self._budget -= 1
            return True

    def take_learning_rate(self) -> float | None:
        with self._cond:
            value, self._pending_lr = self._pending_lr, None
            return value

    def take_shock(self) -> float | None:
        with self._cond:
            value, self._pending_shock = self._pending_shock, None
            return value
