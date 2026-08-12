"""Local HUD web server + launcher.

A tiny Starlette app (static page + a JSON polling endpoint) run by uvicorn on a
daemon thread, so it lives alongside the LiveKit voice loop without blocking it.
`open_dashboard()` launches the page in a chromeless Chrome "app" window for the
full-screen Iron Man feel (falls back to the default browser).
"""
from __future__ import annotations

import logging
import subprocess
import threading
import time
from pathlib import Path

import uvicorn
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import JSONResponse
from starlette.routing import Mount, Route
from starlette.staticfiles import StaticFiles

from jarvis.ui.controller import UIController

logger = logging.getLogger("jarvis.ui")

_STATIC = Path(__file__).parent / "static"


class UIServer:
    def __init__(self, controller: UIController, *, port: int) -> None:
        self._controller = controller
        self._port = port
        self._thread: threading.Thread | None = None
        self._started = False

    def _app(self) -> Starlette:
        controller = self._controller

        async def index(request: Request):
            from starlette.responses import FileResponse

            return FileResponse(_STATIC / "index.html")

        async def state(request: Request):
            try:
                since = int(request.query_params.get("since", "0"))
            except ValueError:
                since = 0
            return JSONResponse(controller.state(since))

        async def favicon(request: Request):
            from starlette.responses import Response

            return Response(status_code=204)

        routes = [
            Route("/", index),
            Route("/api/state", state),
            Route("/favicon.ico", favicon),
            Mount("/static", app=StaticFiles(directory=str(_STATIC)), name="static"),
        ]
        return Starlette(routes=routes)

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
