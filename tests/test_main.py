import socket

import pytest

from nnscope import __version__
from nnscope.__main__ import _frontend_report, _port_report, _version_of, main, report


def test_report_leads_with_the_version():
    assert report().splitlines()[0] == f"nnscope {__version__}"


def test_report_covers_the_things_bug_reports_need():
    text = report()
    for field in ("python", "platform", "numpy", "websockets", "frontend", "port"):
        assert field in text


def test_frontend_report_finds_the_shipped_assets():
    text = _frontend_report()

    assert "MISSING" not in text
    for asset in ("index.html", "app.js", "charts.js", "style.css"):
        assert asset in text


def test_version_of_reports_missing_modules_without_raising():
    assert "not available" in _version_of("a_module_that_does_not_exist")


def test_version_of_reads_real_modules():
    assert _version_of("numpy")[0].isdigit()


def test_port_report_detects_a_free_port():
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        free = probe.getsockname()[1]
    # The socket is closed, so nothing is listening on that port any more.
    assert "is free" in _port_report(free)


def test_port_report_detects_a_busy_port():
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        listener.listen()
        busy = listener.getsockname()[1]

        assert "already in use" in _port_report(busy)


def test_main_prints_the_full_report(capsys):
    assert main([]) == 0
    assert f"nnscope {__version__}" in capsys.readouterr().out


def test_version_flag_prints_only_the_version(capsys):
    assert main(["--version"]) == 0

    assert capsys.readouterr().out.strip() == __version__


def test_port_flag_is_honoured(capsys):
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        listener.listen()
        busy = listener.getsockname()[1]

        main(["--port", str(busy)])

    assert "already in use" in capsys.readouterr().out


def test_bad_arguments_exit_nonzero():
    with pytest.raises(SystemExit):
        main(["--port", "not-a-number"])
