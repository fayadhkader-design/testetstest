"""Watch gradients vanish, then fix it.

Eight sigmoid layers on the spirals data. Each sigmoid's derivative peaks at
0.25, so the chain rule multiplies eight numbers below a quarter together on
the way back to the first layer, and almost nothing survives the trip.

    python examples/vanishing.py                     # vanishing: fails at chance
    python examples/vanishing.py --activation relu   # the fix
    python examples/vanishing.py --optimizer adam    # the problem, hidden
    python examples/vanishing.py --activation relu --lr 1.0   # the opposite failure

The numbers below are measured on this exact setup (SGD, depth 8, 3000 steps,
seed 0), comparing the last layer's gradient norm against the first:

| run                        | accuracy | last / first  |
|----------------------------|----------|---------------|
| sigmoid (default)          | ~34%     | ~12,000,000x  |
| `--activation relu`        | 100%     | ~1x           |
| `--optimizer adam`         | 100%     | ~1x           |
| `--activation relu --lr 1` | ~34%     | ~1e28x        |

Three classes, so ~33% is chance: the default run does not learn at all. Watch
the Gradient norms panel and you can see why, which is the entire point --
`body.0` sits seven decades below `head`, so the front of the network is
receiving no signal to learn from. The representation plot agrees from the
other side: the classes never separate.

Switch to ReLU and the bars level out, because ReLU's derivative is exactly 1
wherever the unit is active, so nothing decays on the way back.

The Adam run is worth doing too. It reaches 100% with sigmoid, because Adam
divides by a running estimate of gradient magnitude and so rescales the tiny
early-layer gradients back up. The pathology is still there in the raw
gradients; the optimizer is papering over it. That is a good thing to have
seen once, because it is also why a working Adam run can hide a network that
is badly conditioned underneath.

And `--activation relu --lr 1.0` fails the other way: gradients explode
instead of vanishing, and the same panel reads it just as clearly.

This is the thing a loss curve cannot tell you. All four runs above have a
loss curve. Only the panel says which of them is actually training.
"""

from __future__ import annotations

import argparse

import torch
from spirals import CLASSES, make_spirals, pick_device
from torch import nn

import nnscope

ACTIVATIONS = {"sigmoid": nn.Sigmoid, "relu": nn.ReLU, "tanh": nn.Tanh}


class DeepNet(nn.Module):
    """Deliberately deeper than the problem needs, which is the point."""

    def __init__(self, activation: str, depth: int = 8, width: int = 32):
        super().__init__()
        make = ACTIVATIONS[activation]

        layers: list[nn.Module] = [nn.Linear(2, width), make()]
        for _ in range(depth - 1):
            layers += [nn.Linear(width, width), make()]

        self.body = nn.Sequential(*layers)
        self.head = nn.Linear(width, CLASSES)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.body(x))


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--activation", choices=sorted(ACTIVATIONS), default="sigmoid")
    # Plain SGD on purpose. Adam rescales by gradient magnitude and so hides
    # the very effect this example exists to show -- see --optimizer adam.
    parser.add_argument("--optimizer", choices=("sgd", "adam"), default="sgd")
    parser.add_argument("--depth", type=int, default=8)
    parser.add_argument("--steps", type=int, default=3000)
    parser.add_argument("--batch", type=int, default=256)
    parser.add_argument("--lr", type=float, default=0.1)
    parser.add_argument("--port", type=int, default=8420)
    parser.add_argument("--no-browser", action="store_true")
    args = parser.parse_args()

    device = pick_device()
    x, y = make_spirals()
    x, y = x.to(device), y.to(device)

    torch.manual_seed(0)
    model = DeepNet(args.activation, depth=args.depth).to(device)
    build = torch.optim.Adam if args.optimizer == "adam" else torch.optim.SGD
    optimizer = build(model.parameters(), lr=args.lr)

    scope = nnscope.watch(
        model,
        optimizer,
        port=args.port,
        name=f"DeepNet({args.activation}, {args.optimizer})",
        open_browser=not args.no_browser,
    )

    with scope:
        for _step in range(args.steps):
            batch = torch.randint(0, len(x), (args.batch,), device=device)
            inputs, targets = x[batch], y[batch]

            logits = model(inputs)
            loss = nn.functional.cross_entropy(logits, targets)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            accuracy = (logits.argmax(1) == targets).float().mean()
            scope.log(labels=targets, loss=loss.item(), accuracy=accuracy.item())

        print(f"done: {args.steps} steps. The dashboard stays up until you quit.")
        try:
            input("press enter to exit\n")
        except (EOFError, KeyboardInterrupt):
            pass


if __name__ == "__main__":
    main()
