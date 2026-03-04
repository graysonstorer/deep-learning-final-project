# ui/ws_server.py
# WebSocket server that pipes bot events to the live browser dashboard.
# Run this in a thread alongside bot.py when --ui is passed.
#
# Clients (dashboard.html) connect and receive real-time JSON messages.

import asyncio
import json
import logging
import threading
import queue
import webbrowser
import os
from pathlib import Path

logger = logging.getLogger(__name__)

try:
    import websockets
    WEBSOCKETS_AVAILABLE = True
except ImportError:
    WEBSOCKETS_AVAILABLE = False
    logger.warning("websockets not installed. UI disabled. Run: pip install websockets")


class UIBroadcaster:
    """
    Thread-safe event broadcaster.
    Bot code calls .emit(event_type, **data) to push events.
    WS server picks them up and broadcasts to all connected clients.
    """

    def __init__(self):
        self._queue: queue.Queue = queue.Queue()
        self._clients: set = set()
        self._loop: asyncio.AbstractEventLoop = None
        self._thread: threading.Thread = None

    def emit(self, event_type: str, **data):
        """Called from bot thread to queue a UI event."""
        msg = {"type": event_type, **data}
        self._queue.put(msg)

    def start(self, port: int = 8765):
        """Start the WebSocket server in a background thread."""
        if not WEBSOCKETS_AVAILABLE:
            logger.warning("websockets not available, skipping UI server")
            return

        def run():
            self._loop = asyncio.new_event_loop()
            asyncio.set_event_loop(self._loop)
            self._loop.run_until_complete(self._serve(port))

        self._thread = threading.Thread(target=run, daemon=True)
        self._thread.start()

        # Open the dashboard in browser
        dashboard_path = Path(__file__).parent / "dashboard.html"
        if dashboard_path.exists():
            webbrowser.open(f"file://{dashboard_path.absolute()}")
            logger.info(f"Dashboard opened: {dashboard_path}")

    async def _serve(self, port: int):
        async def handler(websocket, path=None):
            self._clients.add(websocket)
            logger.info(f"Dashboard client connected. Total: {len(self._clients)}")
            try:
                # Keep connection alive
                async for _ in websocket:
                    pass
            except Exception:
                pass
            finally:
                self._clients.discard(websocket)

        async def broadcaster():
            while True:
                try:
                    msg = self._queue.get_nowait()
                    if self._clients:
                        payload = json.dumps(msg)
                        dead = set()
                        for client in self._clients.copy():
                            try:
                                await client.send(payload)
                            except Exception:
                                dead.add(client)
                        self._clients -= dead
                except queue.Empty:
                    await asyncio.sleep(0.05)

        async with websockets.serve(handler, "localhost", port):
            logger.info(f"UI WebSocket server running on ws://localhost:{port}")
            await broadcaster()


# Singleton broadcaster instance
broadcaster = UIBroadcaster()
