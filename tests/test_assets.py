"""Guards on the files that get shipped inside the package.

These exist because a stray NUL byte once made it into ``app.js``. It ran
fine -- the browser did not care -- but it made the file binary to ``file``,
``grep`` and GitHub's diff view, so searching the frontend silently returned
nothing. A defect that breaks tooling while leaving behaviour intact is
exactly the kind that survives review, so it gets a test rather than a
promise to be careful.
"""

from pathlib import Path

import pytest

ASSETS = Path(__file__).resolve().parent.parent / "nnscope" / "frontend"

EXPECTED = {"index.html", "style.css", "app.js", "charts.js", "history.js"}

# Everything below 0x20 except tab, newline and carriage return. A source file
# has no business containing any of them.
FORBIDDEN = set(range(0, 9)) | set(range(14, 32))


def asset_files():
    return sorted(path for path in ASSETS.iterdir() if path.is_file())


def test_the_expected_assets_are_present():
    """Also a packaging check: these have to survive an install to be served."""
    assert {path.name for path in asset_files()} >= EXPECTED


@pytest.mark.parametrize("path", asset_files(), ids=lambda p: p.name)
def test_asset_is_valid_utf8(path):
    path.read_bytes().decode("utf-8")


@pytest.mark.parametrize("path", asset_files(), ids=lambda p: p.name)
def test_asset_has_no_control_characters(path):
    data = path.read_bytes()
    found = [(index, byte) for index, byte in enumerate(data) if byte in FORBIDDEN]

    if found:
        index, byte = found[0]
        line = data[:index].count(b"\n") + 1
        pytest.fail(
            f"{path.name} line {line} contains a 0x{byte:02x} control byte; "
            f"this makes the file binary to grep, file(1) and diff viewers"
        )


@pytest.mark.parametrize("path", asset_files(), ids=lambda p: p.name)
def test_asset_is_not_empty(path):
    assert path.stat().st_size > 0


@pytest.mark.parametrize("path", asset_files(), ids=lambda p: p.name)
def test_asset_ends_with_a_newline(path):
    assert path.read_bytes().endswith(b"\n"), "missing trailing newline"
