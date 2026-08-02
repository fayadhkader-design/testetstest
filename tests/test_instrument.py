import numpy as np
import pytest
import torch
from torch import nn

from nnscope.instrument import (
    EmbeddingCapture,
    EmbeddingNotFound,
    parameterized_leaves,
)


def sequential_model():
    return nn.Sequential(
        nn.Linear(6, 16),
        nn.ReLU(),
        nn.Dropout(0.1),
        nn.Linear(16, 4),
    )


class DefinedBackwards(nn.Module):
    """Head declared first, executed last -- definition order is a lie here."""

    def __init__(self):
        super().__init__()
        self.head = nn.Linear(16, 4)
        self.body = nn.Linear(6, 16)

    def forward(self, x):
        return self.head(torch.relu(self.body(x)))


def run(capture, model, batch=8, features=6, passes=1):
    """Drive forward passes, returning whatever the last take() produced."""
    result = None
    for _ in range(passes):
        with torch.no_grad():
            model(torch.randn(batch, features))
        result = capture.take()
    return result


def test_parameterized_leaves_skips_containers_and_activations():
    names = [name for name, _ in parameterized_leaves(sequential_model())]
    assert names == ["0", "3"]


def test_raises_when_model_has_no_parameters():
    with pytest.raises(EmbeddingNotFound):
        EmbeddingCapture(nn.Sequential(nn.ReLU()))


def test_calibration_consumes_exactly_one_frame():
    model = sequential_model()
    with EmbeddingCapture(model) as capture:
        assert capture.calibrating

        assert run(capture, model) is None, "first frame is spent calibrating"

        assert not capture.calibrating
        assert capture.layer_name == "3"
        assert run(capture, model) is not None


def test_captures_input_to_the_head_not_its_output():
    model = sequential_model()
    with EmbeddingCapture(model) as capture:
        embeddings = run(capture, model, passes=2)

    # The head is Linear(16, 4); its input is the 16-wide representation.
    assert embeddings.shape == (8, 16)


def test_execution_order_beats_definition_order():
    model = DefinedBackwards()
    with EmbeddingCapture(model) as capture:
        embeddings = run(capture, model, passes=2)

        assert capture.layer_name == "head"
    assert embeddings.shape == (8, 16)


def test_explicit_module_instance_skips_calibration():
    model = sequential_model()
    with EmbeddingCapture(model, module=model[0]) as capture:
        assert not capture.calibrating
        assert capture.layer_name == "0"

        embeddings = run(capture, model)

    assert embeddings.shape == (8, 6), "input to the first layer"


def test_explicit_module_by_dotted_name():
    model = DefinedBackwards()
    with EmbeddingCapture(model, module="body") as capture:
        assert capture.layer_name == "body"
        assert run(capture, model).shape == (8, 6)


def test_unknown_module_name_raises():
    with pytest.raises(EmbeddingNotFound):
        EmbeddingCapture(sequential_model(), module="nope")


def test_foreign_module_raises():
    with pytest.raises(EmbeddingNotFound):
        EmbeddingCapture(sequential_model(), module=nn.Linear(2, 2))


def test_take_is_consuming():
    model = sequential_model()
    with EmbeddingCapture(model, module=model[3]) as capture:
        run(capture, model)
        assert capture.take() is None, "a frame should only be handed out once"


def test_batch_is_capped_at_max_points():
    model = sequential_model()
    with EmbeddingCapture(model, module=model[3], max_points=5) as capture:
        assert run(capture, model, batch=64).shape == (5, 16)


def test_conv_activations_are_flattened():
    model = nn.Sequential(
        nn.Conv2d(1, 4, 3, padding=1),
        nn.Flatten(),
        nn.Linear(4 * 8 * 8, 3),
    )
    with EmbeddingCapture(model) as capture:
        for _ in range(2):
            with torch.no_grad():
                model(torch.randn(6, 1, 8, 8))
            embeddings = capture.take()

    assert embeddings.shape == (6, 4 * 8 * 8)


def test_wide_activations_are_randomly_projected():
    model = nn.Sequential(nn.Linear(6, 4096), nn.Linear(4096, 2))
    with EmbeddingCapture(model, module=model[1], max_dims=128) as capture:
        assert run(capture, model).shape == (8, 128)


def test_random_projection_is_deterministic_across_frames():
    model = nn.Sequential(nn.Linear(6, 512), nn.Linear(512, 2))
    fixed = torch.randn(8, 6)

    with EmbeddingCapture(model, module=model[1], max_dims=32) as capture:
        with torch.no_grad():
            model(fixed)
        first = capture.take()
        with torch.no_grad():
            model(fixed)
        second = capture.take()

    np.testing.assert_allclose(first, second, atol=1e-6)


def test_close_detaches_all_hooks():
    model = sequential_model()
    capture = EmbeddingCapture(model, module=model[3])
    capture.close()

    run(capture, model)

    assert capture.take() is None


def test_capture_does_not_hold_the_graph():
    model = sequential_model()
    with EmbeddingCapture(model, module=model[3]) as capture:
        model(torch.randn(8, 6)).sum().backward()
        embeddings = capture.take()

    assert isinstance(embeddings, np.ndarray)
    assert embeddings.dtype == np.float32
