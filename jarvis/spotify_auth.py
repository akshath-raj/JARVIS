"""One-time Spotify user authorization (for playlist read/modify).

Run once in your terminal:

    python -m jarvis.spotify_auth

It opens your browser, you approve, and a refresh token is saved to
~/.jarvis/spotify_tokens.json. JARVIS then refreshes access tokens automatically
— no further logins. Playing/pausing a song does NOT need this; only playlist
features do.

Prerequisite: add the redirect URI to your app at
https://developer.spotify.com/dashboard  ->  your app  ->  Settings  ->
Redirect URIs:   http://127.0.0.1:8080/callback
"""
from __future__ import annotations

import base64
import json
import sys
import time
import urllib.parse
import webbrowser
from http.server import BaseHTTPRequestHandler, HTTPServer

import requests

from jarvis.config import config
from jarvis.tools.spotify import OAUTH_SCOPES, TOKENS_PATH

_AUTH_URL = "https://accounts.spotify.com/authorize"
_TOKEN_URL = "https://accounts.spotify.com/api/token"
_REDIRECT = config.spotify_redirect_uri
_HOST, _PORT = "127.0.0.1", 8080


class _Handler(BaseHTTPRequestHandler):
    code: str | None = None

    def do_GET(self):
        q = urllib.parse.urlparse(self.path).query
        params = urllib.parse.parse_qs(q)
        _Handler.code = (params.get("code") or [None])[0]
        self.send_response(200)
        self.send_header("Content-Type", "text/html")
        self.end_headers()
        msg = "JARVIS is authorized. You can close this tab." if _Handler.code else "Authorization failed."
        self.wfile.write(f"<html><body><h2>{msg}</h2></body></html>".encode())

    def log_message(self, *a):  # silence
        pass


def main() -> int:
    if not (config.spotify_client_id and config.spotify_client_secret):
        print("Set SPOTIFY_CLIENT_ID and SPOTIFY_CLIENT_SECRET in .env first.")
        return 1

    # Remove any stale token so we always request a fresh grant with all scopes.
    try:
        TOKENS_PATH.unlink()
    except FileNotFoundError:
        pass
    params = urllib.parse.urlencode({
        "client_id": config.spotify_client_id,
        "response_type": "code",
        "redirect_uri": _REDIRECT,
        "scope": OAUTH_SCOPES,
        "show_dialog": "true",  # force the consent screen so all scopes are granted
    })
    print(f"Requesting scopes: {OAUTH_SCOPES}")
    url = f"{_AUTH_URL}?{params}"
    print(f"Opening browser to authorize...\nIf it doesn't open, visit:\n{url}\n")
    webbrowser.open(url)

    server = HTTPServer((_HOST, _PORT), _Handler)
    print(f"Waiting for redirect on {_REDIRECT} ...")
    while _Handler.code is None:
        server.handle_request()

    code = _Handler.code
    creds = f"{config.spotify_client_id}:{config.spotify_client_secret}".encode()
    resp = requests.post(
        _TOKEN_URL,
        data={"grant_type": "authorization_code", "code": code, "redirect_uri": _REDIRECT},
        headers={"Authorization": "Basic " + base64.b64encode(creds).decode()},
        timeout=10,
    )
    if resp.status_code != 200:
        print(f"Token exchange failed ({resp.status_code}): {resp.text}")
        return 1
    p = resp.json()
    TOKENS_PATH.parent.mkdir(parents=True, exist_ok=True)
    TOKENS_PATH.write_text(json.dumps({
        "access_token": p["access_token"],
        "refresh_token": p["refresh_token"],
        "expires_at": time.time() + p.get("expires_in", 3600),
    }))
    print(f"Authorized. Tokens saved to {TOKENS_PATH}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
