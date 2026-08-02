import asyncio
import json
import urllib.error
import urllib.request

import pytest
from websockets.asyncio.client import connect

from nnscope import protocol
from nnscope.buffer import FrameBuffer
from nnscope.control import Controls
from nnscope.server import DashboardServer


@pytest.fixture
def dashboard(tmp_path):
    (tmp_path / "index.html").write_text("<h1>nnscope</h1>")
    (tmp_path / "app.js").write_text("// app")
    (tmp_path / "secret").mkdir()

    buffer = FrameBuffer(capacity=8)
    controls = Controls()
    server = DashboardServer(
        buffer=buffer,
        controls=controls,
        run_info=lambda: {"model": "Test", "layer": "head"},
        status=lambda: {"controls": controls.snapshot(), "lr": 0.01, "step": 3},
        port=0,
        static_root=tmp_path,
    )
    server.start()
    try:
        yield server, buffer, controls
    finally:
        server.stop()


def get(server, path):
    return urllib.request.urlopen(f"{server.url}{path}", timeout=5)


def ws_url(server):
    return f"ws://127.0.0.1:{server.port}/ws"


async def _handshake(server, extra=0):
    """Connect and read the three opening messages, plus `extra` more."""
    async with connect(ws_url(server)) as socket:
        return [json.loads(await socket.recv()) for _ in range(3 + extra)]


def test_start_assigns_a_real_port(dashboard):
    server, _, _ = dashboard
    assert server.port > 0
    assert server.url.startswith("http://localhost:")


def test_double_start_is_rejected(dashboard):
    server, _, _ = dashboard
    with pytest.raises(RuntimeError, match="already started"):
        server.start()


def test_serves_index_at_root(dashboard):
    server, _, _ = dashboard
    response = get(server, "/")

    assert response.status == 200
    assert b"nnscope" in response.read()
    assert response.headers["Content-Type"] == "text/html"


def test_serves_named_assets_with_content_type(dashboard):
    server, _, _ = dashboard
    response = get(server, "/app.js")

    assert response.status == 200
    assert "javascript" in response.headers["Content-Type"]


def test_missing_asset_is_404(dashboard):
    server, _, _ = dashboard
    with pytest.raises(urllib.error.HTTPError) as excinfo:
        get(server, "/nope.js")
    assert excinfo.value.code == 404


def test_directories_are_not_served(dashboard):
    server, _, _ = dashboard
    assert server._resolve_static("/secret") is None


@pytest.mark.parametrize(
    "path",
    ["/../pyproject.toml", "/../../etc/passwd", "/a/../../setup.py"],
)
def test_path_traversal_is_refused(dashboard, path):
    server, _, _ = dashboard
    assert server._resolve_static(path) is None


def test_new_client_receives_hello_backfill_and_state(dashboard):
    server, buffer, _ = dashboard
    buffer.append(protocol.build_frame(1, 0.0, {"loss": 2.0}))
    buffer.append(protocol.build_frame(2, 0.1, {"loss": 1.5}))

    messages = asyncio.run(_handshake(server))

    assert messages[0]["type"] == protocol.HELLO
    assert messages[0]["run"]["model"] == "Test"
    assert messages[1]["type"] == protocol.BACKFILL
    assert [f["step"] for f in messages[1]["frames"]] == [1, 2]
    assert messages[2]["type"] == protocol.STATE
    assert messages[2]["step"] == 3


def test_published_frames_reach_a_connected_client(dashboard):
    server, _, _ = dashboard

    async def scenario():
        async with connect(ws_url(server)) as socket:
            for _ in range(3):
                await socket.recv()
            server.publish(protocol.frame_message(protocol.build_frame(9, 1.0, {})))
            return json.loads(await asyncio.wait_for(socket.recv(), timeout=5))

    message = asyncio.run(scenario())

    assert message["type"] == protocol.FRAME
    assert message["frame"]["step"] == 9


def test_publish_without_clients_is_a_noop(dashboard):
    server, _, _ = dashboard
    server.publish(protocol.frame_message(protocol.build_frame(1, 0.0, {})))
    assert server.client_count == 0


def test_pause_command_reaches_the_controls(dashboard):
    server, _, controls = dashboard

    async def scenario():
        async with connect(ws_url(server)) as socket:
            for _ in range(3):
                await socket.recv()
            await socket.send(json.dumps({"type": "pause"}))
            return json.loads(await asyncio.wait_for(socket.recv(), timeout=5))

    echoed = asyncio.run(scenario())

    assert controls.paused
    assert echoed["type"] == protocol.STATE
    assert echoed["controls"]["paused"] is True


def test_learning_rate_command_is_queued(dashboard):
    server, _, controls = dashboard

    async def scenario():
        async with connect(ws_url(server)) as socket:
            for _ in range(3):
                await socket.recv()
            await socket.send(json.dumps({"type": "lr", "value": 0.005}))
            await asyncio.wait_for(socket.recv(), timeout=5)

    asyncio.run(scenario())

    assert controls.take_learning_rate() == pytest.approx(0.005)


def test_malformed_command_reports_an_error_without_dropping_the_socket(dashboard):
    server, _, controls = dashboard

    async def scenario():
        async with connect(ws_url(server)) as socket:
            for _ in range(3):
                await socket.recv()
            await socket.send("not json at all")
            first = json.loads(await asyncio.wait_for(socket.recv(), timeout=5))

            # The connection must still work afterwards.
            await socket.send(json.dumps({"type": "pause"}))
            second = json.loads(await asyncio.wait_for(socket.recv(), timeout=5))
            return first, second

    error, state = asyncio.run(scenario())

    assert error["type"] == protocol.ERROR
    assert state["type"] == protocol.STATE
    assert controls.paused


def test_client_count_tracks_connections(dashboard):
    server, _, _ = dashboard

    async def scenario():
        async with connect(ws_url(server)) as socket:
            for _ in range(3):
                await socket.recv()
            return server.client_count

    assert asyncio.run(scenario()) == 1


def test_slow_clients_drop_stale_frames_instead_of_growing(dashboard):
    """A throttled tab must not be able to grow the training process."""
    server, _, _ = dashboard
    queue = asyncio.Queue(maxsize=4)
    server._clients.add(queue)
    try:
        for step in range(50):
            server._fanout(f'{{"step":{step}}}')

        assert queue.qsize() == 4
        newest = json.loads(queue._queue[-1])
        assert newest["step"] == 49, "the freshest frame must survive"
    finally:
        server._clients.discard(queue)


def test_stop_is_idempotent(dashboard):
    server, _, _ = dashboard
    server.stop()
    server.stop()
