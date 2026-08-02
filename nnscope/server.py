"""Websocket server and static host for the dashboard.

Runs an asyncio loop on a daemon thread so the training loop keeps the main
thread and stays a perfectly ordinary Python program. The two threads only
meet through :meth:`DashboardServer.publish` and the :class:`Controls` object.

The dashboard and the socket share a single port, which matters more than it
sounds: one URL to print, no CORS, and nothing to configure.

Slow clients get frames dropped rather than buffered. A backlog of stale
frames is worthless in a live view, and unbounded buffering would let a
throttled background browser tab grow the training process's memory without
limit.
"""

from __future__ import annotations

import asyncio
import logging
import mimetypes
import threading
from pathlib import Path
from typing import Any, Callable, Dict, Set

from websockets.asyncio.server import ServerConnection, serve
from websockets.datastructures import Headers
from websockets.exceptions import ConnectionClosed
from websockets.http11 import Request, Response

from . import protocol
from .buffer import FrameBuffer
from .control import Controls

logger = logging.getLogger("nnscope")

WS_PATH = "/ws"
STATIC_ROOT = Path(__file__).parent / "frontend"


class DashboardServer:
    """Serves the dashboard and streams frames to every connected browser."""

    def __init__(
        self,
        buffer: FrameBuffer,
        controls: Controls,
        run_info: Callable[[], Dict[str, Any]],
        status: Callable[[], Dict[str, Any]],
        host: str = "127.0.0.1",
        port: int = 8420,
        static_root: Path = STATIC_ROOT,
        queue_size: int = 64,
    ) -> None:
        self._buffer = buffer
        self._controls = controls
        self._run_info = run_info
        self._status = status
        self._host = host
        self._port = port
        self._static_root = Path(static_root)
        self._queue_size = queue_size

        self._thread: threading.Thread | None = None
        self._loop: asyncio.AbstractEventLoop | None = None
        self._shutdown: asyncio.Event | None = None
        self._clients: Set[asyncio.Queue] = set()
        self._started = threading.Event()
        self._failure: BaseException | None = None

    @property
    def port(self) -> int:
        return self._port

    @property
    def url(self) -> str:
        host = "localhost" if self._host in ("0.0.0.0", "127.0.0.1", "") else self._host
        return f"http://{host}:{self._port}"

    @property
    def client_count(self) -> int:
        return len(self._clients)

    # -- lifecycle ---------------------------------------------------------

    def start(self, timeout: float = 10.0) -> str:
        """Start the server thread and block until it is accepting."""
        if self._thread is not None:
            raise RuntimeError("server already started")

        self._thread = threading.Thread(
            target=lambda: asyncio.run(self._run()),
            name="nnscope-server",
            daemon=True,
        )
        self._thread.start()

        if not self._started.wait(timeout):
            raise RuntimeError(f"server did not start within {timeout}s")
        if self._failure is not None:
            raise self._failure
        return self.url

    def stop(self, timeout: float = 5.0) -> None:
        """Shut the server down. Safe to call more than once, and safe to call
        from an atexit hook after the loop has already torn itself down."""
        loop, self._loop = self._loop, None
        shutdown, self._shutdown = self._shutdown, None

        if loop is not None and shutdown is not None and not loop.is_closed():
            try:
                loop.call_soon_threadsafe(shutdown.set)
            except RuntimeError:
                pass  # loop closed between the check and the call

        if self._thread is not None:
            self._thread.join(timeout=timeout)
            self._thread = None

        self._clients.clear()
        self._started.clear()
        self._failure = None

    async def _run(self) -> None:
        self._loop = asyncio.get_running_loop()
        self._shutdown = asyncio.Event()
        try:
            async with serve(
                self._handle,
                self._host,
                self._port,
                process_request=self._process_request,
            ) as server:
                # Resolve the real port so callers can pass port=0.
                self._port = server.sockets[0].getsockname()[1]
                self._started.set()
                await self._shutdown.wait()
        except BaseException as exc:  # noqa: BLE001 - reported to start()
            self._failure = exc
            self._started.set()

    # -- publishing --------------------------------------------------------

    def publish(self, message: Dict[str, Any]) -> None:
        """Queue a message for every connected client. Safe from any thread."""
        if self._loop is None or not self._clients:
            return
        try:
            payload = protocol.encode(message)
        except ValueError:
            logger.exception("nnscope: refusing to send unencodable message")
            return
        try:
            self._loop.call_soon_threadsafe(self._fanout, payload)
        except RuntimeError:
            pass  # loop already closed; the run is shutting down

    def _fanout(self, payload: str) -> None:
        for queue in list(self._clients):
            if queue.full():
                try:
                    queue.get_nowait()  # drop the stalest frame
                except asyncio.QueueEmpty:
                    pass
            try:
                queue.put_nowait(payload)
            except asyncio.QueueFull:
                pass

    def broadcast_state(self) -> None:
        status = self._status()
        self.publish(
            protocol.state(status["controls"], status.get("lr"), status.get("step", 0))
        )

    # -- connection handling -----------------------------------------------

    async def _handle(self, connection: ServerConnection) -> None:
        queue: asyncio.Queue = asyncio.Queue(maxsize=self._queue_size)
        self._clients.add(queue)
        try:
            await connection.send(protocol.encode(protocol.hello(self._run_info())))
            await connection.send(
                protocol.encode(protocol.backfill(self._buffer.frames()))
            )
            status = self._status()
            await connection.send(
                protocol.encode(
                    protocol.state(
                        status["controls"], status.get("lr"), status.get("step", 0)
                    )
                )
            )

            reader = asyncio.create_task(self._receive(connection))
            writer = asyncio.create_task(self._send(connection, queue))
            _, pending = await asyncio.wait(
                {reader, writer}, return_when=asyncio.FIRST_COMPLETED
            )
            for task in pending:
                task.cancel()
        except ConnectionClosed:
            pass
        finally:
            self._clients.discard(queue)

    async def _send(self, connection: ServerConnection, queue: asyncio.Queue) -> None:
        while True:
            payload = await queue.get()
            await connection.send(payload)

    async def _receive(self, connection: ServerConnection) -> None:
        async for raw in connection:
            try:
                command = protocol.parse_command(raw)
            except protocol.ProtocolError as exc:
                await connection.send(protocol.encode(protocol.error(str(exc))))
                continue
            self._apply(command)
            self.broadcast_state()

    def _apply(self, command: Dict[str, Any]) -> None:
        kind = command["type"]
        if kind == protocol.PAUSE:
            self._controls.pause()
        elif kind == protocol.RESUME:
            self._controls.resume()
        elif kind == protocol.STEP:
            self._controls.step(command["count"])
        elif kind == protocol.SET_LR:
            self._controls.set_learning_rate(command["value"])
        elif kind == protocol.SHOCK:
            self._controls.shock(command["magnitude"])

    # -- static files ------------------------------------------------------

    def _process_request(
        self, connection: ServerConnection, request: Request
    ) -> Response | None:
        path = request.path.split("?", 1)[0]
        if path == WS_PATH:
            return None  # let the websocket handshake proceed

        target = self._resolve_static(path)
        if target is None:
            return connection.respond(404, "not found\n")

        body = target.read_bytes()
        content_type = mimetypes.guess_type(target.name)[0] or "application/octet-stream"
        headers = Headers(
            {
                "Content-Type": content_type,
                "Content-Length": str(len(body)),
                "Cache-Control": "no-store",
            }
        )
        return Response(200, "OK", headers, body)

    def _resolve_static(self, path: str) -> Path | None:
        """Map a URL path to a file, refusing anything outside the static root."""
        relative = path.lstrip("/") or "index.html"
        root = self._static_root.resolve()
        try:
            candidate = (root / relative).resolve()
        except OSError:
            return None

        # `..` in the URL must not escape the package directory.
        if candidate != root and root not in candidate.parents:
            return None
        return candidate if candidate.is_file() else None
