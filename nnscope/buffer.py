"""Bounded storage for training frames.

Rewind is the reason this module exists. A training run can emit hundreds of
thousands of frames, so we keep a fixed-size window of the most recent ones
rather than the whole history. Memory stays flat no matter how long the run
goes, and the scrubber in the UI is backed by exactly what is retained here.
"""

from __future__ import annotations

from collections import deque
from typing import Any, Dict, Iterator, List

Frame = Dict[str, Any]


class FrameBuffer:
    """A fixed-capacity FIFO of frames, oldest first.

    Indexing is positional within the retained window: ``buf[0]`` is the
    oldest frame still held, not the oldest frame ever appended. Callers that
    need absolute positions should read ``frame["step"]``.
    """

    def __init__(self, capacity: int = 600) -> None:
        if capacity < 1:
            raise ValueError(f"capacity must be >= 1, got {capacity}")
        self._frames: deque[Frame] = deque(maxlen=capacity)
        self._capacity = capacity
        self._appended = 0

    @property
    def capacity(self) -> int:
        return self._capacity

    @property
    def appended(self) -> int:
        """Total frames ever appended, including those since evicted."""
        return self._appended

    @property
    def dropped(self) -> int:
        """How many frames have aged out of the window."""
        return self._appended - len(self._frames)

    def append(self, frame: Frame) -> None:
        self._frames.append(frame)
        self._appended += 1

    def extend(self, frames: List[Frame]) -> None:
        for frame in frames:
            self.append(frame)

    def latest(self) -> Frame | None:
        return self._frames[-1] if self._frames else None

    def frames(self) -> List[Frame]:
        """A snapshot copy, oldest first. Safe to hand to another thread."""
        return list(self._frames)

    def clear(self) -> None:
        self._frames.clear()
        self._appended = 0

    def __len__(self) -> int:
        return len(self._frames)

    def __getitem__(self, index: int) -> Frame:
        return self._frames[index]

    def __iter__(self) -> Iterator[Frame]:
        return iter(self._frames)

    def __repr__(self) -> str:
        return (
            f"FrameBuffer(len={len(self._frames)}, capacity={self._capacity}, "
            f"dropped={self.dropped})"
        )
