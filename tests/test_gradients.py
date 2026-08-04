import math

import pytest
import torch
from torch import nn

from nnscope.gradients import layer_gradient_norms


@pytest.fixture
def model():
    torch.manual_seed(0)
    return nn.Sequential(nn.Linear(6, 12), nn.ReLU(), nn.Linear(12, 3))


def backward_once(model, batch=8):
    x = torch.randn(batch, 6)
    y = torch.randint(0, 3, (batch,))
    nn.functional.cross_entropy(model(x), y).backward()


def test_empty_before_any_backward(model):
    assert layer_gradient_norms(model) == {}


def test_empty_after_grads_are_cleared(model):
    backward_once(model)
    model.zero_grad(set_to_none=True)

    assert layer_gradient_norms(model) == {}


def test_reports_one_entry_per_parameterized_layer(model):
    backward_once(model)

    # Activations own no parameters, so they are not layers here.
    assert list(layer_gradient_norms(model)) == ["0", "2"]


def test_preserves_definition_order():
    torch.manual_seed(0)
    net = nn.Sequential(nn.Linear(4, 8), nn.Linear(8, 8), nn.Linear(8, 2))
    net(torch.randn(5, 4)).sum().backward()

    assert list(layer_gradient_norms(net)) == ["0", "1", "2"]


def test_norm_matches_a_hand_computed_l2():
    layer = nn.Linear(3, 2)
    with torch.no_grad():
        layer.weight.fill_(0.0)
        layer.bias.fill_(0.0)
    layer.weight.grad = torch.full_like(layer.weight, 2.0)  # 6 entries
    layer.bias.grad = torch.full_like(layer.bias, 1.0)  # 2 entries

    # sqrt(6 * 2^2 + 2 * 1^2) = sqrt(26)
    norm = layer_gradient_norms(nn.Sequential(layer))["0"]

    assert norm == pytest.approx(math.sqrt(26))


def test_weight_and_bias_collapse_into_one_layer():
    layer = nn.Linear(3, 2)
    layer.weight.grad = torch.ones_like(layer.weight)
    layer.bias.grad = torch.ones_like(layer.bias)

    assert list(layer_gradient_norms(nn.Sequential(layer))) == ["0"]


def test_partially_populated_layer_uses_what_exists():
    layer = nn.Linear(3, 2)
    layer.weight.grad = torch.full_like(layer.weight, 3.0)  # 6 entries, bias None

    assert layer_gradient_norms(nn.Sequential(layer))["0"] == pytest.approx(
        math.sqrt(6 * 9)
    )


def test_a_layer_without_gradients_is_omitted_not_zeroed():
    """Absent and zero are different diagnoses; only one is a dead layer."""
    net = nn.Sequential(nn.Linear(3, 3), nn.Linear(3, 3))
    net[1].weight.grad = torch.ones_like(net[1].weight)

    norms = layer_gradient_norms(net)

    assert "0" not in norms
    assert "1" in norms


def test_genuinely_zero_gradients_are_reported_as_zero():
    layer = nn.Linear(3, 2)
    layer.weight.grad = torch.zeros_like(layer.weight)
    layer.bias.grad = torch.zeros_like(layer.bias)

    assert layer_gradient_norms(nn.Sequential(layer))["0"] == 0.0


def test_uses_dotted_names_for_nested_modules():
    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.body = nn.Sequential(nn.Linear(4, 4))
            self.head = nn.Linear(4, 2)

        def forward(self, x):
            return self.head(self.body(x))

    net = Net()
    net(torch.randn(3, 4)).sum().backward()

    assert list(layer_gradient_norms(net)) == ["body.0", "head"]


def test_values_are_plain_floats():
    model = nn.Sequential(nn.Linear(3, 2))
    model(torch.randn(4, 3)).sum().backward()

    (value,) = layer_gradient_norms(model).values()
    assert isinstance(value, float)


def test_does_not_disturb_the_gradients_it_reads(model):
    backward_once(model)
    before = model[0].weight.grad.clone()

    layer_gradient_norms(model)

    torch.testing.assert_close(model[0].weight.grad, before)


def test_vanishing_profile_is_visible(model):
    """The diagnostic this exists for: deep layers dominating shallow ones."""
    torch.manual_seed(0)
    deep = nn.Sequential(
        nn.Linear(4, 16), nn.Sigmoid(),
        nn.Linear(16, 16), nn.Sigmoid(),
        nn.Linear(16, 16), nn.Sigmoid(),
        nn.Linear(16, 2),
    )
    # Saturate the sigmoids so the chain of derivatives collapses.
    with torch.no_grad():
        for layer in deep:
            if isinstance(layer, nn.Linear):
                layer.weight.mul_(12.0)

    deep(torch.randn(32, 4)).pow(2).mean().backward()
    norms = list(layer_gradient_norms(deep).values())

    assert norms[0] < norms[-1], "expected the earliest layer to be the faintest"
