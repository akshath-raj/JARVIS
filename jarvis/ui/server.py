"""Local HUD web server + launcher.

A tiny Starlette app (static page + a JSON polling endpoint) run by uvicorn on a
daemon thread, so it lives alongside the LiveKit voice loop without blocking it.
`open_dashboard()` launches the page in a chromeless Chrome "app" window for the
full-screen Iron Man feel (falls back to the default browser).
"""
from __future__ import annotations

import asyncio
import logging
import subprocess
import threading
import time
from pathlib import Path

import uvicorn
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import JSONResponse
from starlette.routing import Mount, Route, WebSocketRoute
from starlette.staticfiles import StaticFiles
from starlette.websockets import WebSocket, WebSocketDisconnect

from jarvis.ui.controller import UIController

logger = logging.getLogger("jarvis.ui")

_STATIC = Path(__file__).parent / "static"
# Refresh passive data (profile / memories / conversations) this often. Events
# (reveal / explanation / navigate) are pushed instantly and don't wait for this.
_SNAPSHOT_EVERY = 3.0


class UIServer:
    def __init__(self, controller: UIController, *, port: int) -> None:
        self._controller = controller
        self._port = port
        self._thread: threading.Thread | None = None
        self._started = False
        self._loop: asyncio.AbstractEventLoop | None = None
        self._clients: set[WebSocket] = set()

    # ── WebSocket push ──────────────────────────────────────────────────────
    async def _send(self, ws: WebSocket, msg: dict) -> None:
        try:
            await ws.send_json(msg)
        except Exception:
            self._clients.discard(ws)

    async def _broadcast(self, msg: dict) -> None:
        for ws in list(self._clients):
            await self._send(ws, msg)

    def _notify(self, event: dict) -> None:
        """Called from the agent thread — schedule an instant push on the server
        loop (the two run on different event loops)."""
        loop = self._loop
        if loop is None:
            return
        try:
            asyncio.run_coroutine_threadsafe(
                self._broadcast({"type": "event", "event": event}), loop
            )
        except Exception:
            pass

    async def _snapshot_loop(self) -> None:
        while True:
            await asyncio.sleep(_SNAPSHOT_EVERY)
            if self._clients:
                await self._broadcast({"type": "snapshot", "state": self._controller.state(0)})

    def _app(self) -> Starlette:
        controller = self._controller
        server = self

        async def index(request: Request):
            from starlette.responses import FileResponse

            return FileResponse(_STATIC / "index.html")

        async def state(request: Request):
            try:
                since = int(request.query_params.get("since", "0"))
            except ValueError:
                since = 0
            return JSONResponse(controller.state(since))

        async def set_name(request: Request):
            try:
                data = await request.json()
            except Exception:
                data = {}
            name = controller.set_user_name(str((data or {}).get("name", "")))
            return JSONResponse({"ok": bool(name), "user": name})

        async def memory_add(request: Request):
            try:
                data = await request.json()
            except Exception:
                data = {}
            ok = controller.add_memory((data or {}).get("text", ""))
            return JSONResponse(
                {"ok": ok, "memories": controller.state(0)["memories"]},
                status_code=200 if ok else 400,
            )

        async def memory_delete(request: Request):
            try:
                data = await request.json()
            except Exception:
                data = {}
            ok = controller.delete_memory((data or {}).get("text", ""))
            return JSONResponse(
                {"ok": ok, "memories": controller.state(0)["memories"]},
                status_code=200 if ok else 404,
            )

        async def todo_delete(request: Request):
            try:
                data = await request.json()
            except Exception:
                data = {}
            data = data or {}
            ok = controller.remove_todo(str(data.get("id", "")), str(data.get("text", "")))
            st = controller.state(0)
            return JSONResponse(
                {"ok": ok, "todos": st["todos"], "agenda": st["agenda"]},
                status_code=200 if ok else 404,
            )

        async def agenda_delete(request: Request):
            try:
                data = await request.json()
            except Exception:
                data = {}
            data = data or {}
            ok = controller.remove_agenda_item(
                str(data.get("id", "")), str(data.get("kind", "")), str(data.get("text", ""))
            )
            st = controller.state(0)
            return JSONResponse(
                {"ok": ok, "todos": st["todos"], "agenda": st["agenda"]},
                status_code=200 if ok else 404,
            )

        async def favicon(request: Request):
            from starlette.responses import Response

            return Response(status_code=204)

        async def ws_endpoint(ws: WebSocket):
            await ws.accept()
            server._clients.add(ws)
            # send a full snapshot immediately so a fresh/late client renders + can
            # replay the current reveal state and last explanation.
            await server._send(ws, {"type": "snapshot", "state": controller.state(0)})
            try:
                while True:
                    raw = await ws.receive_text()
                    # the browser sends commands back (e.g. the STOP-alarm button)
                    import json as _json

                    try:
                        controller.handle_command(_json.loads(raw))
                    except Exception:
                        pass
            except WebSocketDisconnect:
                pass
            finally:
                server._clients.discard(ws)

        from contextlib import asynccontextmanager

        @asynccontextmanager
        async def lifespan(app):
            server._loop = asyncio.get_running_loop()
            controller.set_notifier(server._notify)
            task = asyncio.create_task(server._snapshot_loop())
            try:
                yield
            finally:
                task.cancel()

        routes = [
            Route("/", index),
            Route("/api/state", state),
            Route("/api/user/name", set_name, methods=["POST"]),
            Route("/api/memory/add", memory_add, methods=["POST"]),
            Route("/api/memory/delete", memory_delete, methods=["POST"]),
            Route("/api/todo/delete", todo_delete, methods=["POST"]),
            Route("/api/agenda/delete", agenda_delete, methods=["POST"]),
            Route("/favicon.ico", favicon),
            WebSocketRoute("/ws", ws_endpoint),
            Mount("/static", app=StaticFiles(directory=str(_STATIC)), name="static"),
        ]
        return Starlette(routes=routes, lifespan=lifespan)

    def start(self) -> None:
        if self._started:
            return
        self._started = True
        config = uvicorn.Config(
            self._app(), host="127.0.0.1", port=self._port, log_level="warning"
        )
        server = uvicorn.Server(config)

        def _run() -> None:
            try:
                server.run()
            except Exception as e:  # never take the assistant down with the HUD
                logger.warning("HUD server stopped: %s", e)

        self._thread = threading.Thread(target=_run, daemon=True, name="jarvis-ui")
        self._thread.start()
        logger.info("HUD server on http://127.0.0.1:%d", self._port)

    @property
    def url(self) -> str:
        return f"http://127.0.0.1:{self._port}"


def open_dashboard(url: str, *, chrome_path: str = "") -> None:
    """Open the HUD in a chromeless Chrome app window (best-effort), else the
    default browser. Runs detached so it never blocks the voice loop."""
    chrome = chrome_path or "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"
    try:
        if Path(chrome).exists():
            subprocess.Popen(
                [
                    chrome,
                    f"--app={url}",
                    "--new-window",
                    "--window-size=1440,900",
                ],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            return
    except Exception:
        pass
    try:
        import webbrowser

        webbrowser.open(url)
    except Exception as e:
        logger.warning("couldn't open the dashboard: %s", e)


def wait_until_up(url: str, timeout: float = 3.0) -> bool:
    """Best-effort readiness check before opening the browser."""
    import urllib.request

    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            urllib.request.urlopen(url + "/api/state?since=0", timeout=0.5)
            return True
        except Exception:
            time.sleep(0.1)
    return False
