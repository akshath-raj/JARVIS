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


def test_open_url_dedupes_immediate_double_open(monkeypatch):
    # A single "open Netflix" can reach open_url twice (duplicate tool call / turn);
    # the second identical open within the window is skipped so it never opens twice.
    calls = []
    monkeypatch.setattr("jarvis.tools.browser.subprocess.run",
                        lambda args, **kw: calls.append(args) or mock.Mock(returncode=0, stderr=""))
    bc = BrowserController("Google Chrome")
    bc.open_url("https://www.netflix.com/watch/80057281")
    bc.open_url("https://www.netflix.com/watch/80057281")   # instant duplicate → skipped
    assert len(calls) == 1
    # a DIFFERENT url still opens, and so does the same url after the window lapses
    bc.open_url("https://www.youtube.com")
    assert len(calls) == 2
    bc._last_open_at -= 10  # simulate the dedup window having elapsed
    bc.open_url("https://www.youtube.com")
    assert len(calls) == 3


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


# ── Netflix play (title id resolved from web-search URLs) ──────────────────
def test_netflix_id_from_urls_finds_first_and_decodes():
    # title/ and watch/ both carry the same id; encoded redirect URLs still resolve
    urls = [
        "https://www.google.com/",
        "https://www.netflix.com/title/81234567",
        "https://www.netflix.com/title/70000000",
    ]
    assert BrowserController.netflix_id_from_urls(urls) == "81234567"
    encoded = ["https://r.example/l/?u=https%3A%2F%2Fwww.netflix.com%2Fwatch%2F81111111"]
    assert BrowserController.netflix_id_from_urls(encoded) == "81111111"
    assert BrowserController.netflix_id_from_urls(["https://imdb.com/x"]) == ""


def test_play_netflix_opens_watch_url_for_resolved_id(monkeypatch):
    opened = {}
    monkeypatch.setattr(BrowserController, "open_url",
                        lambda self, url: opened.setdefault("url", url) or url)
    msg = BrowserController().play_netflix("Wednesday", title_id="81234567")
    assert opened["url"] == "https://www.netflix.com/watch/81234567"  # /watch → plays
    assert "netflix" in msg.lower()


def test_play_netflix_falls_back_to_search_without_id(monkeypatch):
    opened = {}
    monkeypatch.setattr(BrowserController, "open_url",
                        lambda self, url: opened.setdefault("url", url) or url)
    msg = BrowserController().play_netflix("an obscure title")   # no title_id resolved
    assert opened["url"] == "https://www.netflix.com/search?q=an%20obscure%20title"
    assert "search" in msg.lower()


def test_play_netflix_rejects_non_numeric_id(monkeypatch):
    # a junk id must not build a bogus /watch URL — fall back to search instead
    opened = {}
    monkeypatch.setattr(BrowserController, "open_url",
                        lambda self, url: opened.setdefault("url", url) or url)
    BrowserController().play_netflix("x", title_id="not-a-number")
    assert opened["url"].startswith("https://www.netflix.com/search?q=")


def test_play_netflix_empty_query_errors():
    with pytest.raises(BrowserError):
        BrowserController().play_netflix("  ")


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


def test_tavily_search_urls_returns_result_urls(monkeypatch):
    payload = {"results": [
        {"url": "https://www.netflix.com/title/81234567", "title": "Wednesday"},
        {"url": "https://en.wikipedia.org/wiki/Wednesday", "title": "wiki"},
        {"title": "no url here"},
    ]}
    captured = {}

    def fake_post(url, json=None, timeout=None):
        captured["json"] = json
        return _resp(payload)

    monkeypatch.setattr("jarvis.tools.web.requests.post", fake_post)
    urls = TavilyClient("key").search_urls("Wednesday netflix")
    assert urls == ["https://www.netflix.com/title/81234567",
                    "https://en.wikipedia.org/wiki/Wednesday"]
    assert captured["json"]["include_answer"] is False   # URLs mode skips the synthesis
