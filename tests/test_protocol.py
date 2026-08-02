import json

import numpy as np
import pytest

from nnscope import protocol
from nnscope.protocol import ProtocolError, build_frame, encode, parse_command


def test_encode_is_compact():
    assert encode({"a": 1, "b": 2}) == '{"a":1,"b":2}'


def test_encode_refuses_nan():
    with pytest.raises(ValueError):
        encode({"loss": float("nan")})


def test_frame_carries_step_and_metrics():
    frame = build_frame(step=7, elapsed=1.25, metrics={"loss": 0.5})

    assert frame["step"] == 7
    assert frame["t"] == 1.25
    assert frame["metrics"]["loss"] == 0.5
    assert "embedding" not in frame


def test_diverged_metrics_become_null_rather_than_breaking_json():
    frame = build_frame(step=1, elapsed=0.0, metrics={"loss": float("inf")})

    assert frame["metrics"]["loss"] is None
    json.loads(encode(frame))


def test_metrics_are_rounded():
    frame = build_frame(step=1, elapsed=0.0, metrics={"loss": 0.123456789})

    assert frame["metrics"]["loss"] == 0.123457


def test_embedding_ships_as_parallel_arrays():
    coords = np.array([[1.0, 2.0], [3.0, 4.0]])
    frame = build_frame(1, 0.0, {}, coords=coords, labels=[0, 1])

    assert frame["embedding"]["x"] == [1.0, 3.0]
    assert frame["embedding"]["y"] == [2.0, 4.0]
    assert frame["embedding"]["labels"] == [0, 1]


def test_coordinates_are_rounded_to_plot_precision():
    frame = build_frame(1, 0.0, {}, coords=np.array([[0.123456789, 0.0]]))

    assert frame["embedding"]["x"] == [0.1235]


def test_non_finite_coordinates_are_zeroed():
    coords = np.array([[float("nan"), float("inf")]])
    frame = build_frame(1, 0.0, {}, coords=coords)

    json.loads(encode(frame))
    assert frame["embedding"]["x"] == [0.0]


def test_label_count_must_match_point_count():
    with pytest.raises(ValueError, match="2 labels for 1 points"):
        build_frame(1, 0.0, {}, coords=np.array([[0.0, 0.0]]), labels=[0, 1])


def test_wrong_coordinate_shape_is_rejected():
    with pytest.raises(ValueError, match=r"coords must be \(n, 2\)"):
        build_frame(1, 0.0, {}, coords=np.zeros((4, 3)))


def test_empty_coordinates_omit_the_embedding():
    assert "embedding" not in build_frame(1, 0.0, {}, coords=np.zeros((0, 2)))


def test_labels_are_optional():
    frame = build_frame(1, 0.0, {}, coords=np.array([[0.0, 1.0]]))
    assert "labels" not in frame["embedding"]


def test_variance_and_rotation_are_included_when_given():
    frame = build_frame(
        1, 0.0, {}, coords=np.array([[0.0, 1.0]]),
        explained_variance=0.6789123, rotation=0.0001234,
    )

    assert frame["embedding"]["explained"] == 0.6789
    assert frame["embedding"]["rotation"] == 0.00012


def test_parse_rejects_malformed_json():
    with pytest.raises(ProtocolError, match="not valid JSON"):
        parse_command("{oops")


def test_parse_rejects_non_objects():
    with pytest.raises(ProtocolError, match="expected a JSON object"):
        parse_command("[1, 2, 3]")


def test_parse_rejects_unknown_types():
    with pytest.raises(ProtocolError, match="unknown command"):
        parse_command('{"type":"drop_tables"}')


def test_parse_simple_commands():
    assert parse_command('{"type":"pause"}') == {"type": protocol.PAUSE}
    assert parse_command('{"type":"resume"}') == {"type": protocol.RESUME}


def test_step_defaults_to_one():
    assert parse_command('{"type":"step"}') == {"type": protocol.STEP, "count": 1}


def test_step_accepts_an_explicit_count():
    assert parse_command('{"type":"step","count":5}')["count"] == 5


@pytest.mark.parametrize("bad", ["0", "-3", '"4"', "true", "1.5"])
def test_step_rejects_bad_counts(bad):
    with pytest.raises(ProtocolError, match="positive integer"):
        parse_command('{"type":"step","count":%s}' % bad)


def test_learning_rate_command():
    assert parse_command('{"type":"lr","value":0.01}')["value"] == pytest.approx(0.01)


@pytest.mark.parametrize("bad", ['"fast"', "-1", "null", "true"])
def test_learning_rate_rejects_bad_values(bad):
    with pytest.raises(ProtocolError):
        parse_command('{"type":"lr","value":%s}' % bad)


def test_shock_defaults_and_overrides():
    assert parse_command('{"type":"shock"}')["magnitude"] == pytest.approx(0.5)
    assert parse_command('{"type":"shock","magnitude":2}')["magnitude"] == pytest.approx(2.0)


def test_state_message_tolerates_missing_learning_rate():
    message = protocol.state({"paused": False}, None, 42)

    assert message["lr"] is None
    assert message["step"] == 42
