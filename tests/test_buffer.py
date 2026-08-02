import pytest

from nnscope.buffer import FrameBuffer


def frame(step: int) -> dict:
    return {"step": step}


def test_rejects_nonpositive_capacity():
    with pytest.raises(ValueError):
        FrameBuffer(capacity=0)


def test_retains_up_to_capacity():
    buf = FrameBuffer(capacity=3)
    buf.extend([frame(i) for i in range(3)])

    assert len(buf) == 3
    assert [f["step"] for f in buf.frames()] == [0, 1, 2]
    assert buf.dropped == 0


def test_evicts_oldest_beyond_capacity():
    buf = FrameBuffer(capacity=3)
    buf.extend([frame(i) for i in range(10)])

    assert len(buf) == 3
    assert [f["step"] for f in buf.frames()] == [7, 8, 9]
    assert buf.appended == 10
    assert buf.dropped == 7


def test_index_zero_is_oldest_retained_not_oldest_ever():
    buf = FrameBuffer(capacity=2)
    buf.extend([frame(i) for i in range(5)])

    assert buf[0]["step"] == 3
    assert buf[-1]["step"] == 4


def test_latest_returns_none_when_empty():
    assert FrameBuffer(capacity=4).latest() is None


def test_latest_tracks_most_recent_append():
    buf = FrameBuffer(capacity=4)
    buf.append(frame(1))
    assert buf.latest()["step"] == 1
    buf.append(frame(2))
    assert buf.latest()["step"] == 2


def test_frames_returns_a_copy():
    buf = FrameBuffer(capacity=4)
    buf.append(frame(0))

    snapshot = buf.frames()
    buf.append(frame(1))

    assert len(snapshot) == 1, "snapshot must not observe later appends"


def test_clear_resets_counters():
    buf = FrameBuffer(capacity=2)
    buf.extend([frame(i) for i in range(5)])
    buf.clear()

    assert len(buf) == 0
    assert buf.appended == 0
    assert buf.dropped == 0
    assert buf.latest() is None
