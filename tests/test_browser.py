"""Tests for the browser/web tools (site resolution, YouTube scrape, Tavily).

Network (requests) and the `open` command (subprocess) are mocked, so these run
anywhere and never actually launch Chrome or hit the internet.
"""
from __future__ import annotations

from unittest import mock

import pytest

from jarvis.tools.browser import BrowserController, BrowserError
from jarvis.tools.web import TavilyClient, WebError


# ── site resolution ───────────────────────────────────────────────────────
@pytest.mark.parametrize(
    "name,expected",
    [
        ("youtube", "https://www.youtube.com"),
        ("YouTube", "https://www.youtube.com"),
        ("insta", "https://www.instagram.com"),
        ("reels", "https://www.instagram.com/reels/"),
        ("github", "https://github.com"),
        ("espn.com", "https://espn.com"),
        ("news.ycombinator.com/newest", "https://news.ycombinator.com/newest"),
        ("https://example.com/x", "https://example.com/x"),
    ],
)
def test_resolve_site_known_and_domains(name, expected):
    assert BrowserController().resolve_site(name) == expected


def test_resolve_site_unknown_falls_back_to_google():
    got = BrowserController().resolve_site("how tall is everest")
    assert got == "https://www.google.com/search?q=how%20tall%20is%20everest"


# ── open_url (subprocess mocked) ───────────────────────────────────────────
def test_open_url_invokes_open_with_app(monkeypatch):
    calls = {}

    def fake_run(args, **kw):
        calls["args"] = args
        return mock.Mock(returncode=0, stderr="")

    monkeypatch.setattr("jarvis.tools.browser.subprocess.run", fake_run)
    url = BrowserController("Google Chrome").open_url("https://www.youtube.com")
    assert url == "https://www.youtube.com"
    assert calls["args"] == ["open", "-a", "Google Chrome", "https://www.youtube.com"]


def test_open_url_rejects_non_web_scheme(monkeypatch):
    ran = []
    monkeypatch.setattr("jarvis.tools.browser.subprocess.run", lambda *a, **k: ran.append(a))
    for bad in ["file:///etc/passwd", "javascript:alert(1)", "; rm -rf ~", "chrome://settings"]:
        with pytest.raises(BrowserError):
            BrowserController().open_url(bad)
    assert ran == []  # nothing was ever opened


def test_open_site_resolves_then_opens(monkeypatch):
    opened = {}
    monkeypatch.setattr(BrowserController, "open_url", lambda self, url: opened.setdefault("url", url) or url)
    BrowserController().open_site("instagram")
    assert opened["url"] == "https://www.instagram.com"


# ── YouTube play / channel-latest (requests mocked) ────────────────────────
_RESULTS_HTML = 'xx{"videoRenderer":{"videoId":"abc123DEFgh","title":...}} more {"videoId":"zzzzzzzzzzz"}'


def test_play_youtube_scrapes_first_id_and_opens(monkeypatch):
    monkeypatch.setattr(BrowserController, "_get", lambda self, url: _RESULTS_HTML)
    opened = {}
    monkeypatch.setattr(BrowserController, "open_url", lambda self, url: opened.setdefault("url", url) or url)
    msg = BrowserController().play_youtube("never gonna give you up")
    assert opened["url"] == "https://www.youtube.com/watch?v=abc123DEFgh"
    assert "youtube" in msg.lower()


def test_play_youtube_empty_query_errors():
    with pytest.raises(BrowserError):
        BrowserController().play_youtube("   ")


def test_latest_channel_video_uses_handle(monkeypatch):
    seen = {}

    def fake_get(self, url):
        seen["url"] = url
        return _RESULTS_HTML

    monkeypatch.setattr(BrowserController, "_get", fake_get)
    monkeypatch.setattr(BrowserController, "open_url", lambda self, url: url)
    msg = BrowserController().latest_channel_video("Mr Beast")
    assert seen["url"] == "https://www.youtube.com/@MrBeast/videos"
    assert "MrBeast".lower() in msg.lower().replace(" ", "")


def test_first_video_id_missing_raises(monkeypatch):
    monkeypatch.setattr(BrowserController, "_get", lambda self, url: "no ids here")
    with pytest.raises(BrowserError):
        BrowserController().play_youtube("something")


def test_open_reels_platform_switch(monkeypatch):
    opened = []
    monkeypatch.setattr(BrowserController, "open_url", lambda self, url: opened.append(url) or url)
    bc = BrowserController()
    bc.open_reels("instagram")
    bc.open_reels("youtube")
    bc.open_shorts()
    assert opened == [
        "https://www.instagram.com/reels/",
        "https://www.youtube.com/shorts",
        "https://www.youtube.com/shorts",
    ]


# ── Tavily web search ──────────────────────────────────────────────────────
def _resp(payload, status=200):
    m = mock.Mock()
    m.status_code = status
    m.json.return_value = payload
    m.text = str(payload)
    return m


def test_tavily_returns_answer(monkeypatch):
    captured = {}

    def fake_post(url, json=None, timeout=None):
        captured["url"] = url
        captured["json"] = json
        return _resp({"answer": "The Fed held rates steady.", "results": []})

    monkeypatch.setattr("jarvis.tools.web.requests.post", fake_post)
    out = TavilyClient("key").search("fed decision", topic="news")
    assert out == "The Fed held rates steady."
    assert captured["json"]["topic"] == "news"
    assert captured["json"]["api_key"] == "key"
    assert captured["json"]["include_answer"] is True


def test_tavily_falls_back_to_top_result(monkeypatch):
    monkeypatch.setattr(
        "jarvis.tools.web.requests.post",
        lambda *a, **k: _resp({"answer": "", "results": [{"title": "T", "content": "snippet"}]}),
    )
    out = TavilyClient("key").search("something")
    assert "T" in out and "snippet" in out


def test_tavily_requires_key():
    with pytest.raises(WebError):
        TavilyClient("").search("anything")


def test_tavily_empty_query():
    with pytest.raises(WebError):
        TavilyClient("key").search("   ")


def test_tavily_bad_key_status(monkeypatch):
    monkeypatch.setattr("jarvis.tools.web.requests.post", lambda *a, **k: _resp({}, status=401))
    with pytest.raises(WebError, match="401"):
        TavilyClient("key").search("x")
