"""Tests for the SAFE web-image fetcher — the guardrails are the whole point, so
each one gets an explicit test. The network (`requests.get`) is stubbed, so these
run offline and never touch the internet or disk outside tmp_path.
"""
from __future__ import annotations

import pytest

import jarvis.tools.images as im
from jarvis.tools.images import ImageError, SafeImageFetcher

PNG = b"\x89PNG\r\n\x1a\n" + b"0" * 64        # valid PNG magic bytes
HOSTS = ("upload.wikimedia.org", "commons.wikimedia.org")


def _fetcher(tmp_path, **kw):
    return SafeImageFetcher(
        download_dir=str(tmp_path), allowed_hosts=HOSTS, max_bytes=1024, **kw
    )


class _Resp:
    """Minimal stand-in for a streamed requests.Response."""
    def __init__(self, *, url, body=b"", ctype="image/png", history_urls=(),
                 json_data=None, status_ok=True, status_code=200):
        self.url = url
        self.status_code = status_code
        self._body = body
        self.headers = {"Content-Type": ctype}
        self.history = [type("H", (), {"url": u})() for u in history_urls]
        self._json = json_data
        self._ok = status_ok

    def raise_for_status(self):
        if not self._ok:
            import requests
            raise requests.HTTPError("bad status")

    def iter_content(self, n):
        for i in range(0, len(self._body), n):
            yield self._body[i:i + n]

    def json(self):
        return self._json

    def __enter__(self): return self
    def __exit__(self, *a): return False


def _patch_get(monkeypatch, resp):
    monkeypatch.setattr(im.requests, "get", lambda *a, **k: resp)


# ── allowlist / scheme ────────────────────────────────────────────────────
def test_rejects_non_https(tmp_path):
    with pytest.raises(ImageError):
        _fetcher(tmp_path).download("http://upload.wikimedia.org/a.png")


def test_rejects_host_not_on_allowlist(tmp_path):
    with pytest.raises(ImageError, match="not an allowed"):
        _fetcher(tmp_path).download("https://evil.example.com/a.png")


def test_subdomain_of_allowed_host_is_ok(tmp_path):
    f = _fetcher(tmp_path)
    assert f._host_ok("https://upload.wikimedia.org/x.png")
    assert f._host_ok("https://s3.upload.wikimedia.org/x.png")  # subdomain allowed
    assert not f._host_ok("https://wikimedia.org.evil.com/x.png")


# ── redirect must not leave the allowlist ─────────────────────────────────
def test_rejects_redirect_off_allowlist(tmp_path, monkeypatch):
    # final URL is allowlisted, but a redirect hop went through an untrusted host
    resp = _Resp(url="https://upload.wikimedia.org/final.png", body=PNG,
                 history_urls=["https://tracker.evil.com/redirect"])
    _patch_get(monkeypatch, resp)
    with pytest.raises(ImageError, match="untrusted host"):
        _fetcher(tmp_path).download("https://upload.wikimedia.org/a.png")


# ── content-type ──────────────────────────────────────────────────────────
def test_rejects_non_image_content_type(tmp_path, monkeypatch):
    _patch_get(monkeypatch, _Resp(url="https://upload.wikimedia.org/a.png",
                                  body=b"<html>", ctype="text/html"))
    with pytest.raises(ImageError, match="not a raster image"):
        _fetcher(tmp_path).download("https://upload.wikimedia.org/a.png")


def test_rejects_svg(tmp_path, monkeypatch):
    _patch_get(monkeypatch, _Resp(url="https://upload.wikimedia.org/a.svg",
                                  body=b"<svg>", ctype="image/svg+xml"))
    with pytest.raises(ImageError):
        _fetcher(tmp_path).download("https://upload.wikimedia.org/a.svg")


# ── size cap ──────────────────────────────────────────────────────────────
def test_rejects_oversize(tmp_path, monkeypatch):
    big = b"\x89PNG\r\n\x1a\n" + b"0" * 5000    # exceeds max_bytes=1024
    _patch_get(monkeypatch, _Resp(url="https://upload.wikimedia.org/a.png", body=big))
    with pytest.raises(ImageError, match="exceeds"):
        _fetcher(tmp_path).download("https://upload.wikimedia.org/a.png")


# ── magic-byte verification ───────────────────────────────────────────────
def test_rejects_html_masquerading_as_png(tmp_path, monkeypatch):
    # server SAYS image/png but the bytes are HTML → magic-byte check catches it
    _patch_get(monkeypatch, _Resp(url="https://upload.wikimedia.org/a.png",
                                  body=b"<!DOCTYPE html><html>gotcha", ctype="image/png"))
    with pytest.raises(ImageError, match="not a valid image"):
        _fetcher(tmp_path).download("https://upload.wikimedia.org/a.png")


# ── happy path ────────────────────────────────────────────────────────────
def test_accepts_and_saves_valid_png(tmp_path, monkeypatch):
    _patch_get(monkeypatch, _Resp(url="https://upload.wikimedia.org/Deepfake.png",
                                  body=PNG, ctype="image/png"))
    path = _fetcher(tmp_path).download("https://upload.wikimedia.org/Deepfake.png",
                                       name_hint="deepfake example")
    assert path.endswith(".png")
    import os
    assert os.path.getsize(path) == len(PNG)


def test_never_overwrites(tmp_path, monkeypatch):
    _patch_get(monkeypatch, _Resp(url="https://upload.wikimedia.org/a.png", body=PNG))
    f = _fetcher(tmp_path)
    p1 = f.download("https://upload.wikimedia.org/a.png", name_hint="pic")
    p2 = f.download("https://upload.wikimedia.org/a.png", name_hint="pic")
    assert p1 != p2  # second save is auto-renamed, first untouched


# ── search filters to allowlisted, raster results ─────────────────────────
def test_search_returns_only_safe_raster_results(tmp_path, monkeypatch):
    api = {"query": {"pages": {
        "1": {"title": "File:Deepfake.png", "imageinfo": [
            {"thumburl": "https://upload.wikimedia.org/thumb/Deepfake.png",
             "mime": "image/png", "descriptionurl": "https://commons.wikimedia.org/wiki/File:Deepfake.png",
             "thumbwidth": 800, "thumbheight": 600}]},
        "2": {"title": "File:Diagram.svg", "imageinfo": [          # svg → dropped
            {"thumburl": "https://upload.wikimedia.org/Diagram.svg", "mime": "image/svg+xml"}]},
        "3": {"title": "File:Bad.png", "imageinfo": [              # off-allowlist → dropped
            {"thumburl": "https://cdn.evil.com/Bad.png", "mime": "image/png"}]},
    }}}
    _patch_get(monkeypatch, _Resp(url=im._COMMONS_API, json_data=api))
    results = _fetcher(tmp_path).search("deepfake", count=6)
    assert [r.title for r in results] == ["Deepfake.png"]
    assert results[0].url.startswith("https://upload.wikimedia.org/")
    assert results[0].source_page.endswith("File:Deepfake.png")


def test_search_rejects_empty_query(tmp_path):
    with pytest.raises(ImageError):
        _fetcher(tmp_path).search("   ")


# ── fetch_one tries candidates until one downloads safely ─────────────────
def test_fetch_one_skips_a_bad_candidate(tmp_path, monkeypatch):
    f = _fetcher(tmp_path)
    from jarvis.tools.images import ImageResult
    monkeypatch.setattr(f, "search", lambda q, count=6: [
        ImageResult("Bad", "https://upload.wikimedia.org/bad.png", "src1"),
        ImageResult("Good", "https://upload.wikimedia.org/good.png", "src2"),
    ])

    def fake_download(url, *, name_hint=""):
        if "bad" in url:
            raise ImageError("corrupt")
        return str(tmp_path / "good.png")

    monkeypatch.setattr(f, "download", fake_download)
    path, res = f.fetch_one("deepfake")
    assert res.title == "Good" and path.endswith("good.png")


def test_fetch_one_raises_when_nothing_found(tmp_path, monkeypatch):
    f = _fetcher(tmp_path)
    monkeypatch.setattr(f, "search", lambda q, count=6: [])
    with pytest.raises(ImageError, match="no safe image"):
        f.fetch_one("nonexistent-topic-xyz")


def test_fetch_one_accept_gate_skips_rejected(tmp_path, monkeypatch):
    """An `accept` callback (the vetter) rejects the first image, so fetch_one moves on
    to the next candidate and returns the accepted one."""
    from jarvis.tools.images import ImageResult
    f = _fetcher(tmp_path)
    monkeypatch.setattr(f, "search", lambda q, count=6: [
        ImageResult("Bad", "https://upload.wikimedia.org/bad.png", "s1"),
        ImageResult("Good", "https://upload.wikimedia.org/good.png", "s2"),
    ])
    monkeypatch.setattr(f, "download", lambda url, name_hint="": url)
    accept = lambda path, res: res.title == "Good"      # vet rejects "Bad"
    path, res = f.fetch_one("deepfake", accept=accept)
    assert res.title == "Good"


def test_fetch_one_falls_back_to_best_safe_when_all_rejected(tmp_path, monkeypatch):
    """If the vetter accepts none, we still return the first SAFE image (never fail the
    document just because nothing was judged a perfect fit)."""
    from jarvis.tools.images import ImageResult
    f = _fetcher(tmp_path)
    monkeypatch.setattr(f, "search", lambda q, count=6: [
        ImageResult("A", "https://upload.wikimedia.org/a.png", "s1"),
        ImageResult("B", "https://upload.wikimedia.org/b.png", "s2"),
    ])
    monkeypatch.setattr(f, "download", lambda url, name_hint="": url)
    path, res = f.fetch_one("deepfake", accept=lambda p, r: False)
    assert res.title == "A"   # first safe candidate used as fallback
