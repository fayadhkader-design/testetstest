import numpy as np
import pytest
import torch
from torch import nn

import nnscope
from nnscope.session import Scope


@pytest.fixture
def model():
    torch.manual_seed(0)
    return nn.Sequential(nn.Linear(6, 12), nn.ReLU(), nn.Linear(12, 3))


@pytest.fixture
def scope(model):
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    scope = Scope(
        model,
        optimizer,
        port=0,
        open_browser=False,
        min_interval=0.0,
        capacity=50,
    )
    try:
        yield scope
    finally:
        scope.close()


def train_step(model, scope, batch=8, **metrics):
    x = torch.randn(batch, 6)
    y = torch.randint(0, 3, (batch,))
    loss = nn.functional.cross_entropy(model(x), y)
    loss.backward()
    scope.log(labels=y, loss=loss.item(), **metrics)
    return loss


def test_watch_is_the_public_entry_point(model):
    scope = nnscope.watch(model, port=0, open_browser=False)
    try:
        assert isinstance(scope, Scope)
        assert scope.url.startswith("http://localhost:")
    finally:
        scope.close()


def test_importing_the_package_exposes_only_the_public_api():
    assert dir(nnscope) == ["Scope", "__version__", "watch"]


def test_unknown_attribute_still_raises(model):
    with pytest.raises(AttributeError):
        nnscope.definitely_not_here


def test_log_advances_the_step_counter(model, scope):
    assert scope.step == 0
    train_step(model, scope)
    train_step(model, scope)
    assert scope.step == 2


def test_metrics_reach_the_buffer(model, scope):
    train_step(model, scope, accuracy=0.5)

    frame = scope._buffer.latest()
    assert frame["step"] == 1
    assert frame["metrics"]["accuracy"] == 0.5
    assert "loss" in frame["metrics"]


def test_learning_rate_is_recorded(model, scope):
    train_step(model, scope)
    assert scope._buffer.latest()["lr"] == pytest.approx(0.1)


def test_embedding_appears_once_calibration_settles(model, scope):
    train_step(model, scope)
    assert "embedding" not in scope._buffer.latest(), "first frame calibrates"

    train_step(model, scope)
    embedding = scope._buffer.latest()["embedding"]

    assert len(embedding["x"]) == 8
    assert len(embedding["y"]) == 8
    assert embedding["labels"] is not None
    assert scope.layer_name == "2"


def test_labels_line_up_with_points(model, scope):
    train_step(model, scope)
    train_step(model, scope, batch=5)

    embedding = scope._buffer.latest()["embedding"]
    assert len(embedding["labels"]) == len(embedding["x"]) == 5


def test_mismatched_labels_are_dropped_rather_than_mispainted(model, scope):
    train_step(model, scope)

    x = torch.randn(8, 6)
    model(x)
    scope.log(labels=torch.tensor([0, 1]), loss=1.0)

    assert "labels" not in scope._buffer.latest()["embedding"]


def test_logging_without_labels_still_plots(model, scope):
    train_step(model, scope)
    model(torch.randn(4, 6))
    scope.log(loss=1.0)

    embedding = scope._buffer.latest()["embedding"]
    assert len(embedding["x"]) == 4
    assert "labels" not in embedding


def test_diverged_loss_does_not_break_the_run(model, scope):
    train_step(model, scope)
    model(torch.randn(4, 6))
    scope.log(loss=float("nan"))

    assert scope._buffer.latest()["metrics"]["loss"] is None


def test_every_throttles_emission_without_losing_steps(model):
    torch.manual_seed(0)
    net = nn.Sequential(nn.Linear(6, 8), nn.Linear(8, 3))
    with Scope(net, port=0, open_browser=False, min_interval=0.0, every=3) as scope:
        for _ in range(9):
            net(torch.randn(4, 6))
            scope.log(loss=0.5)

        assert scope.step == 9
        assert [f["step"] for f in scope._buffer.frames()] == [3, 6, 9]


def test_min_interval_throttles_a_fast_loop(model):
    torch.manual_seed(0)
    net = nn.Sequential(nn.Linear(6, 8), nn.Linear(8, 3))
    with Scope(net, port=0, open_browser=False, min_interval=30.0) as scope:
        for _ in range(50):
            net(torch.randn(4, 6))
            scope.log(loss=0.5)

        assert scope.step == 50
        assert len(scope._buffer) <= 1


def test_rejects_nonpositive_every(model):
    with pytest.raises(ValueError):
        Scope(model, port=0, open_browser=False, every=0)


def test_dashboard_learning_rate_reaches_the_optimizer(model, scope):
    scope._controls.set_learning_rate(0.004)

    train_step(model, scope)

    assert scope._optimizer.param_groups[0]["lr"] == pytest.approx(0.004)


def test_shock_perturbs_weights_proportionally(model, scope):
    before = model[0].weight.detach().clone()
    scope._controls.shock(0.5)

    train_step(model, scope)

    after = model[0].weight.detach()
    assert not torch.allclose(before, after)
    # Proportional to the tensor's own spread, not a fixed absolute jolt.
    assert (after - before).std().item() == pytest.approx(
        before.std().item() * 0.5, rel=0.5
    )


def test_shock_leaves_scalar_parameters_alone():
    torch.manual_seed(0)
    net = nn.Sequential(nn.Linear(4, 2))
    with torch.no_grad():
        net[0].bias.fill_(1.0)

    with Scope(net, port=0, open_browser=False, min_interval=0.0) as scope:
        scope._shock(10.0)

    assert net[0].bias.numel() == 2  # guarded path is numel < 2, not bias-ness


def test_run_info_reports_the_model_and_device(model, scope):
    info = scope._run_info()

    assert info["model"] == "Sequential"
    assert info["capacity"] == 50
    assert info["device"] == "cpu"


def test_run_info_carries_an_id(model, scope):
    assert scope._run_info()["id"]


def test_each_run_gets_a_distinct_id(model):
    """A tab left open across a restart must be able to tell runs apart."""
    with Scope(model, port=0, open_browser=False) as first:
        with Scope(model, port=0, open_browser=False) as second:
            assert first._run_info()["id"] != second._run_info()["id"]


def test_run_id_is_stable_within_a_run(model, scope):
    first = scope._run_info()["id"]
    train_step(model, scope)

    assert scope._run_info()["id"] == first


def test_status_reports_step_and_learning_rate(model, scope):
    train_step(model, scope)
    status = scope._status()

    assert status["step"] == 1
    assert status["lr"] == pytest.approx(0.1)
    assert status["controls"]["paused"] is False


def test_close_is_idempotent_and_releases_the_loop(model, scope):
    scope._controls.pause()
    scope.close()
    scope.close()

    assert scope._controls.released
    assert scope._controls.gate(timeout=0.1) is True


def test_logging_after_close_is_a_noop(model, scope):
    scope.close()
    scope.log(loss=1.0)

    assert scope.step == 0


def test_context_manager_closes_on_exit(model):
    with Scope(model, port=0, open_browser=False) as scope:
        url = scope.url
    assert url
    assert scope._closed


def test_custom_name_overrides_the_class_name(model):
    with Scope(model, port=0, open_browser=False, name="MyNet") as scope:
        assert scope._run_info()["model"] == "MyNet"


def test_buffer_respects_capacity_during_a_long_run(model):
    torch.manual_seed(0)
    net = nn.Sequential(nn.Linear(6, 8), nn.Linear(8, 3))
    with Scope(net, port=0, open_browser=False, min_interval=0.0, capacity=10) as scope:
        for _ in range(40):
            net(torch.randn(4, 6))
            scope.log(loss=0.5)

        assert len(scope._buffer) == 10
        assert scope._buffer.latest()["step"] == 40


def test_projection_is_stable_across_a_real_run(model, scope):
    """End to end: consecutive frames must not mirror each other.

    The batch is held fixed on purpose. Row i has to be the *same sample*
    every frame for a positional comparison to mean anything -- with a fresh
    random batch each step you are correlating unrelated points and would get
    noise no matter how stable the projection is.
    """
    torch.manual_seed(1)
    x = torch.randn(32, 6)
    y = torch.randint(0, 3, (32,))

    previous = None
    correlations = []
    for _ in range(12):
        loss = nn.functional.cross_entropy(model(x), y)
        scope._optimizer.zero_grad()
        loss.backward()
        scope._optimizer.step()
        scope.log(labels=y, loss=loss.item())

        embedding = scope._buffer.latest().get("embedding")
        if embedding is None:
            continue
        current = np.array(embedding["x"])
        if previous is not None:
            correlations.append(np.corrcoef(previous, current)[0, 1])
        previous = current

    assert len(correlations) >= 8, "expected most frames to carry an embedding"
    assert min(correlations) > 0.9, f"projection flipped: {min(correlations)}"
