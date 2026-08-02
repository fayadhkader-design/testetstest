"""A small CNN on MNIST.

Ten classes, which is past the point where colour can carry identity on its
own -- so the dashboard stops assigning hues and lets you light up classes
from the legend instead. Click 4 and 9, or 3 and 5, and watch how long those
pairs stay entangled after everything else has separated. That confusion is
visible in the representation well before it shows up in the accuracy number.

    pip install -e ".[examples]"
    python examples/mnist.py

Downloads ~11 MB into ./data on first run.
"""

from __future__ import annotations

import argparse

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

import nnscope


class SmallCNN(nn.Module):
    """Two conv blocks into a 64-wide representation, then a linear readout.

    nnscope attaches to the input of `head`, so the 64-wide layer is what you
    watch. Everything before it is the network's actual work; the head is
    just a linear boundary drawn through the result.
    """

    def __init__(self, embedding: int = 64):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(32 * 7 * 7, embedding), nn.ReLU(),
        )
        self.head = nn.Linear(embedding, 10)

    def forward(self, x):
        return self.head(self.features(x))


def pick_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--port", type=int, default=8420)
    parser.add_argument("--data", default="./data")
    parser.add_argument("--no-browser", action="store_true")
    args = parser.parse_args()

    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
    )
    train = datasets.MNIST(args.data, train=True, download=True, transform=transform)
    # num_workers=0 keeps this a single process, which avoids the macOS
    # spawn-vs-MPS pitfalls and costs nothing at this size.
    loader = DataLoader(train, batch_size=args.batch, shuffle=True, num_workers=0)

    device = pick_device()
    torch.manual_seed(0)
    model = SmallCNN().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    scope = nnscope.watch(
        model,
        optimizer,
        port=args.port,
        name="SmallCNN",
        open_browser=not args.no_browser,
    )

    with scope:
        for epoch in range(args.epochs):
            for inputs, targets in loader:
                inputs, targets = inputs.to(device), targets.to(device)

                logits = model(inputs)
                loss = nn.functional.cross_entropy(logits, targets)

                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()

                accuracy = (logits.argmax(1) == targets).float().mean()
                scope.log(
                    labels=targets,
                    loss=loss.item(),
                    accuracy=accuracy.item(),
                    epoch=epoch,
                )

        print(f"done: {args.epochs} epochs. The dashboard stays up until you quit.")
        try:
            input("press enter to exit\n")
        except (EOFError, KeyboardInterrupt):
            pass


if __name__ == "__main__":
    main()
