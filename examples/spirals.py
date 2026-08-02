"""Three interleaved spirals, learned by a small MLP.

The best demo of what nnscope shows, and it needs no dataset: three spiral
arms are not linearly separable, so the network has no choice but to build a
representation that untangles them. Watch the scatter and you see exactly
that happen -- one mixed blob that pulls itself into three clean lobes.

    python examples/spirals.py

Then try the transport controls: pause mid-training, drag the timeline back
to step 50, and hit Shock to scatter the weights and watch the clusters
re-form.
"""

from __future__ import annotations

import argparse
import time

import torch
from torch import nn

import nnscope

CLASSES = 3


def make_spirals(per_class: int = 500, noise: float = 0.18, seed: int = 0):
    """Classic interleaved spiral arms, one per class."""
    generator = torch.Generator().manual_seed(seed)
    xs, ys = [], []

    for label in range(CLASSES):
        radius = torch.linspace(0.05, 1.0, per_class)
        angle = (
            torch.linspace(0.0, 3.4, per_class)
            + label * (2 * torch.pi / CLASSES)
            + torch.randn(per_class, generator=generator) * noise
        )
        xs.append(torch.stack([radius * torch.sin(angle), radius * torch.cos(angle)], 1))
        ys.append(torch.full((per_class,), label, dtype=torch.long))

    return torch.cat(xs), torch.cat(ys)


class SpiralNet(nn.Module):
    """Deliberately narrow at the end: the 32-wide layer feeding the head is
    what nnscope captures, and a tight bottleneck makes the separation vivid."""

    def __init__(self, width: int = 96, embedding: int = 32):
        super().__init__()
        self.body = nn.Sequential(
            nn.Linear(2, width), nn.Tanh(),
            nn.Linear(width, width), nn.Tanh(),
            nn.Linear(width, embedding), nn.Tanh(),
        )
        self.head = nn.Linear(embedding, CLASSES)

    def forward(self, x):
        return self.head(self.body(x))


def pick_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=4000)
    parser.add_argument("--batch", type=int, default=256)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--port", type=int, default=8420)
    parser.add_argument(
        "--delay",
        type=float,
        default=0.0,
        help="seconds to pause per step; the separation happens fast, so "
        "try 0.01 if you want to watch it unfold slowly",
    )
    parser.add_argument(
        "--no-browser",
        action="store_true",
        help="print the URL instead of opening it, for headless or remote boxes",
    )
    args = parser.parse_args()

    device = pick_device()
    x, y = make_spirals()
    x, y = x.to(device), y.to(device)

    torch.manual_seed(0)
    model = SpiralNet().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    scope = nnscope.watch(
        model,
        optimizer,
        port=args.port,
        name="SpiralNet",
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

            if args.delay:
                time.sleep(args.delay)

        print(f"done: {args.steps} steps. The dashboard stays up until you quit.")
        try:
            input("press enter to exit\n")
        except (EOFError, KeyboardInterrupt):
            # No tty (piped, or run under a supervisor): just exit cleanly.
            pass


if __name__ == "__main__":
    main()
