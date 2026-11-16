"""WebSocket server that receives frames from the Android streamer in a background thread."""
from __future__ import annotations

import asyncio
import logging
import queue
import threading
import time

from websockets.asyncio.server import serve

from .protocol import PhoneFrame, decode

log = logging.getLogger(__name__)


class PhoneServer:
    """Usage:
        with PhoneServer(port=8765) as srv:
            frame = srv.latest(timeout=5)       # newest frame (for live display / policy)
            for f in srv.drain(): ...           # every frame since the last drain (for recording)
    """

    def __init__(self, host: str = "0.0.0.0", port: int = 8765, buffer: int = 600):
        self.host, self.port = host, port
        self._queue: queue.Queue[PhoneFrame] = queue.Queue(maxsize=buffer)
        self._latest: PhoneFrame | None = None
        self._cond = threading.Condition()
        self._loop: asyncio.AbstractEventLoop | None = None
        self._stop: asyncio.Event | None = None
        self._thread: threading.Thread | None = None
        self._ready = threading.Event()
        self.clients = 0
        self.dropped = 0

    # ------------------------------------------------------------------ lifecycle
    def start(self) -> "PhoneServer":
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        if not self._ready.wait(5):
            raise RuntimeError("phone server failed to start")
        return self

    def stop(self) -> None:
        if self._loop and self._stop:
            self._loop.call_soon_threadsafe(self._stop.set)
        if self._thread:
            self._thread.join(timeout=5)

    def __enter__(self):
        return self.start()

    def __exit__(self, *exc):
        self.stop()

    def _run(self) -> None:
        asyncio.run(self._main())

    async def _main(self) -> None:
        self._loop = asyncio.get_running_loop()
        self._stop = asyncio.Event()
        async with serve(self._handler, self.host, self.port, max_size=None, compression=None):
            log.info("phone server listening on ws://%s:%d", self.host, self.port)
            self._ready.set()
            await self._stop.wait()

    async def _handler(self, ws) -> None:
        self.clients += 1
        log.info("phone connected: %s", ws.remote_address)
        try:
            async for msg in ws:
                if isinstance(msg, str):
                    continue
                self._push(decode(msg, recv_time=time.time()))
        except Exception as e:  # connection drops are normal
            log.info("phone connection ended: %s", e)
        finally:
            self.clients -= 1

    def _push(self, frame: PhoneFrame) -> None:
        try:
            self._queue.put_nowait(frame)
        except queue.Full:
            self._queue.get_nowait()
            self._queue.put_nowait(frame)
            self.dropped += 1
        with self._cond:
            self._latest = frame
            self._cond.notify_all()

    # ------------------------------------------------------------------ consumers
    def latest(self, timeout: float | None = None, newer_than: int | None = None) -> PhoneFrame | None:
        """Newest frame; optionally wait until its seq is > newer_than."""
        deadline = None if timeout is None else time.monotonic() + timeout
        with self._cond:
            while self._latest is None or (newer_than is not None and self._latest.seq <= newer_than):
                remaining = None if deadline is None else deadline - time.monotonic()
                if remaining is not None and remaining <= 0:
                    return None
                self._cond.wait(remaining)
            return self._latest

    def drain(self) -> list[PhoneFrame]:
        out = []
        while True:
            try:
                out.append(self._queue.get_nowait())
            except queue.Empty:
                return out
