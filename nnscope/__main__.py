"""Environment diagnostics: ``python -m nnscope``.

Most nnscope bug reports come down to one of four things -- which torch, which
accelerator, whether the frontend assets made it into the install, and whether
the port is already taken. This prints all four in a form that can be pasted
into an issue.

Nothing here imports torch at module level; the package is usable without it
and the diagnostic has to survive a broken or absent install rather than
traceback on the very command someone runs to diagnose it.
"""

from __future__ import annotations

import argparse
import platform
import socket
import sys
from pathlib import Path

from . import __version__

DEFAULT_PORT = 8420


def _row(label: str, value: str) -> str:
    return f"  {label:<12} {value}"


def _version_of(module_name: str) -> str:
    try:
        module = __import__(module_name)
    except Exception as exc:
        # Catching broadly is the point: a half-installed dependency raises
        # things other than ImportError, and "what went wrong" is the answer
        # this command exists to report.
        return f"not available ({type(exc).__name__})"
    return getattr(module, "__version__", "unknown version")


def _torch_report() -> list[str]:
    try:
        import torch
    except Exception as exc:
        # A broken torch install is a far more common report than a missing
        # one, and it surfaces here as something other than ImportError.
        return [_row("torch", f"not installed ({type(exc).__name__})")]

    accelerators = []
    if torch.backends.mps.is_available():
        accelerators.append("mps")
    if torch.cuda.is_available():
        accelerators.append(f"cuda ({torch.cuda.get_device_name(0)})")
    available = ", ".join(accelerators) if accelerators else "cpu only"

    return [_row("torch", torch.__version__), _row("accelerator", available)]


def _frontend_report() -> str:
    root = Path(__file__).parent / "frontend"
    if not root.is_dir():
        return "MISSING - the install did not include the dashboard assets"

    assets = sorted(p for p in root.iterdir() if p.is_file())
    if not assets:
        return "MISSING - frontend directory is empty"

    total = sum(p.stat().st_size for p in assets)
    names = ", ".join(p.name for p in assets)
    return f"{len(assets)} files, {total / 1024:.1f} KB ({names})"


def _port_report(port: int) -> str:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        probe.settimeout(0.25)
        if probe.connect_ex(("127.0.0.1", port)) == 0:
            return f"{port} is already in use - pass port= to pick another"
    return f"{port} is free"


def report(port: int = DEFAULT_PORT) -> str:
    lines = [
        f"nnscope {__version__}",
        "",
        _row("python", f"{platform.python_version()} ({sys.executable})"),
        _row("platform", platform.platform()),
        _row("numpy", _version_of("numpy")),
        _row("websockets", _version_of("websockets")),
        *_torch_report(),
        _row("frontend", _frontend_report()),
        _row("port", _port_report(port)),
    ]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m nnscope",
        description="Print environment diagnostics for a bug report.",
    )
    parser.add_argument("--version", action="store_true", help="print the version only")
    parser.add_argument(
        "--port",
        type=int,
        default=DEFAULT_PORT,
        help=f"port to check availability of (default {DEFAULT_PORT})",
    )
    args = parser.parse_args(argv)

    if args.version:
        print(__version__)
        return 0

    print(report(args.port))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
