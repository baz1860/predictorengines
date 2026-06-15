"""Sports Predictor — desktop entry point.

Starts the FastAPI backend on a background thread, then opens a native macOS
window (PyWebView / WKWebView) pointing at it. The window has no browser chrome,
shows up in the dock as "Sports Predictor", and reuses the Python engines
in-process.

Run:
    python3 -m app.main          # from the "Soccer Prediction" folder

Headless / dev (no window, just the API for curl or a browser):
    uvicorn app.server:app --port 8765
"""
from __future__ import annotations

import socket
import threading
import time

import uvicorn

from .server import app

APP_NAME = "Sports Predictor"


def _free_port() -> int:
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


def _serve(port: int) -> None:
    uvicorn.run(app, host="127.0.0.1", port=port, log_level="warning")


def main() -> None:
    port = _free_port()
    threading.Thread(target=_serve, args=(port,), daemon=True).start()

    # Wait for the server to accept connections before opening the window.
    url = f"http://127.0.0.1:{port}"
    for _ in range(100):
        try:
            with socket.create_connection(("127.0.0.1", port), timeout=0.2):
                break
        except OSError:
            time.sleep(0.05)

    import webview  # imported here so headless use doesn't require pywebview

    webview.create_window(APP_NAME, url, width=1100, height=760, min_size=(900, 600))
    webview.start()


if __name__ == "__main__":
    main()
