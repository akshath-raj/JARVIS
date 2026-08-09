"""Tests for the Spotify controller (search parsing + play-path routing).

Network (requests) and AppleScript (subprocess) are mocked, so these run
anywhere. A live end-to-end test is included but skipped unless
RUN_SPOTIFY_LIVE=1 (it actually plays music on this Mac).
"""
from __future__ import annotations

import os
from unittest import mock

import pytest

from jarvis.tools.spotify import PlayResult, SpotifyController, SpotifyError, Track

ITEM = {
    "uri": "spotify:track:abc",
    "name": "Bohemian Rhapsody",
    "artists": [{"name": "Queen"}],
}


def _resp(payload, status=200):
    m = mock.Mock()
    m.status_code = status
    m.json.return_value = payload
    return m


def _token():
    return _resp({"access_token": "tok", "expires_in": 3600})


# ── Search (Web API) ──────────────────────────────────────────────────────
def test_search_track_parses_top_hit():
    sc = SpotifyController("id", "secret")
    with mock.patch("jarvis.tools.spotify.requests.post", return_value=_token()), mock.patch(
        "jarvis.tools.spotify.requests.get", return_value=_resp({"tracks": {"items": [ITEM]}})
    ):
        t = sc.search_track("bohemian rhapsody queen")
    assert t == Track("spotify:track:abc", "Bohemian Rhapsody", "Queen")
    assert t.label == "Bohemian Rhapsody by Queen"


def test_search_track_no_results_returns_none():
    sc = SpotifyController("id", "secret")
    with mock.patch("jarvis.tools.spotify.requests.post", return_value=_token()), mock.patch(
        "jarvis.tools.spotify.requests.get", return_value=_resp({"tracks": {"items": []}})
    ):
        assert sc.search_track("zzz nonsense") is None


def test_search_requires_credentials():
    sc = SpotifyController()  # no keys
    with pytest.raises(SpotifyError):
        sc.search_track("anything")


# ── play_query routing (app first, web fallback) ──────────────────────────
def test_web_mode_searches_then_plays_uri(monkeypatch):
    sc = SpotifyController("id", "secret")
    monkeypatch.setattr(sc, "search_track", lambda q: Track("spotify:track:abc", "Bohemian Rhapsody", "Queen"))
    played = {}
    monkeypatch.setattr(sc, "ensure_running", lambda: played.__setitem__("ran", True))
    monkeypatch.setattr(sc, "play_uri", lambda uri: played.__setitem__("uri", uri))

    res = sc.play_query("bohemian rhapsody", mode="web")
    assert res == PlayResult(label="Bohemian Rhapsody by Queen", source="web")
    assert played == {"ran": True, "uri": "spotify:track:abc"}


def test_auto_falls_back_to_web_when_app_fails(monkeypatch):
    sc = SpotifyController("id", "secret")
    monkeypatch.setattr(sc, "_try_play_via_app", lambda q: False)  # app couldn't play
    monkeypatch.setattr(sc, "search_track", lambda q: Track("spotify:track:abc", "Bohemian Rhapsody", "Queen"))
    monkeypatch.setattr(sc, "ensure_running", lambda: None)
    played = {}
    monkeypatch.setattr(sc, "play_uri", lambda uri: played.__setitem__("uri", uri))

    res = sc.play_query("bohemian rhapsody", mode="auto")
    assert res.source == "web"
    assert played["uri"] == "spotify:track:abc"


def test_auto_uses_app_when_it_succeeds(monkeypatch):
    sc = SpotifyController()  # no web keys at all
    monkeypatch.setattr(sc, "_try_play_via_app", lambda q: True)
    monkeypatch.setattr(sc, "current_track", lambda: "Bohemian Rhapsody by Queen")

    res = sc.play_query("bohemian rhapsody", mode="auto")
    assert res == PlayResult(label="Bohemian Rhapsody by Queen", source="app")


def test_auto_errors_when_app_fails_and_no_key(monkeypatch):
    sc = SpotifyController()  # no keys
    monkeypatch.setattr(sc, "_try_play_via_app", lambda q: False)
    with pytest.raises(SpotifyError):
        sc.play_query("x", mode="auto")


def test_volume_clamped(monkeypatch):
    sc = SpotifyController()
    scripts = []
    monkeypatch.setattr(sc, "_osa", lambda s: scripts.append(s) or "")
    sc.set_volume(250)
    assert "set sound volume to 100" in scripts[0]


def test_play_uri_rejects_injection(monkeypatch):
    """The one templated AppleScript value must reject anything but a clean URI."""
    sc = SpotifyController()
    scripts = []
    monkeypatch.setattr(sc, "_osa", lambda s: scripts.append(s) or "")
    monkeypatch.setattr("jarvis.tools.spotify.time.sleep", lambda *_: None)
    sc.play_uri("spotify:track:2JiDi0qAXsPwhPqA2qaKGt")     # valid track
    sc.play_uri("spotify:playlist:37i9dQZF1DXcBWIGoYBM5M")  # valid playlist context
    plays = [s for s in scripts if "play track" in s]
    assert len(plays) == 2
    for bad in ['spotify:track:x" \n do shell script "rm -rf ~"', "spotify:episode:x", "; ls", "not a uri"]:
        with pytest.raises(SpotifyError):
            sc.play_uri(bad)
    # no extra 'play track' AppleScript executed for malformed inputs
    assert len([s for s in scripts if "play track" in s]) == 2


def test_play_query_loop_sets_repeat(monkeypatch):
    sc = SpotifyController("id", "secret")
    monkeypatch.setattr(sc, "search_track", lambda q: Track("spotify:track:abc", "X", "Y"))
    monkeypatch.setattr(sc, "ensure_running", lambda: None)
    monkeypatch.setattr(sc, "play_uri", lambda uri: None)
    repeats = []
    monkeypatch.setattr(sc, "set_repeat", lambda on: repeats.append(on))
    sc.play_query("x", mode="web", loop=True)
    assert repeats == [True]


def test_add_current_to_playlist(monkeypatch):
    sc = SpotifyController("id", "secret")
    monkeypatch.setattr(sc, "current_track_uri", lambda: "spotify:track:abc")
    monkeypatch.setattr(sc, "current_track", lambda: "X by Y")
    monkeypatch.setattr(sc, "find_playlist", lambda name: ("spotify:playlist:pl123", "Favourites"))
    monkeypatch.setattr(sc, "_user_token", lambda: "utok")
    posted = {}

    def fake_post(url, headers=None, json=None, timeout=None):
        posted["url"] = url
        posted["uris"] = json["uris"]
        m = mock.Mock()
        m.status_code = 201
        return m

    monkeypatch.setattr("jarvis.tools.spotify.requests.post", fake_post)
    track, pl = sc.add_current_to_playlist("favourites")
    assert (track, pl) == ("X by Y", "Favourites")
    assert posted["uris"] == ["spotify:track:abc"]
    assert posted["url"].endswith("/playlists/pl123/tracks")


def test_list_playlists_handles_null_counts(monkeypatch):
    sc = SpotifyController("id", "secret")
    monkeypatch.setattr(sc, "_user_token", lambda: "utok")
    calls = {"n": 0}

    def fake_get(url, headers=None, timeout=None):
        calls["n"] += 1
        m = mock.Mock()
        m.status_code = 200
        if "/me/playlists" in url:
            m.json.return_value = {
                "items": [
                    {"id": "a", "name": "Focus", "tracks": {"total": 12}},
                    {"id": "b", "name": "Workout", "tracks": {"total": None}},
                ],
                "next": None,
            }
        else:  # per-playlist fetch for the null one -> still null (403-like)
            m.json.return_value = {"tracks": {"total": None}}
        return m

    monkeypatch.setattr("jarvis.tools.spotify.requests.get", fake_get)
    out = sc.list_playlists()
    assert out == [("Focus", 12), ("Workout", None)]


def test_top_tracks_and_liked_parse(monkeypatch):
    sc = SpotifyController("id", "secret")

    def fake_user_get(path):
        if "/me/top/tracks" in path:
            return {"items": [{"uri": "spotify:track:a", "name": "Song A", "artists": [{"name": "X"}]}]}
        if "/me/tracks" in path:
            return {"items": [{"track": {"uri": "spotify:track:b", "name": "Song B", "artists": [{"name": "Y"}]}}]}
        if "/me/top/artists" in path:
            return {"items": [{"name": "Artist Z"}]}
        return {"items": []}

    monkeypatch.setattr(sc, "_user_get", fake_user_get)
    assert sc.top_tracks()[0] == Track("spotify:track:a", "Song A", "X")
    assert sc.liked_songs()[0] == Track("spotify:track:b", "Song B", "Y")
    assert sc.top_artists() == ["Artist Z"]


def test_user_get_scope_error(monkeypatch):
    sc = SpotifyController("id", "secret")
    monkeypatch.setattr(sc, "_user_token", lambda: "t")
    m = mock.Mock()
    m.status_code = 403
    m.text = '{"error":{"status":403,"message":"Insufficient client scope"}}'
    monkeypatch.setattr("jarvis.tools.spotify.requests.get", lambda *a, **k: m)
    with pytest.raises(SpotifyError, match="spotify_auth"):
        sc.top_tracks()


def test_playlist_features_need_login(monkeypatch, tmp_path):
    monkeypatch.setattr("jarvis.tools.spotify.TOKENS_PATH", tmp_path / "nope.json")
    sc = SpotifyController("id", "secret")
    with pytest.raises(SpotifyError, match="spotify_auth"):
        sc._user_token()


# ── Live (opt-in) ─────────────────────────────────────────────────────────
@pytest.mark.skipif(os.getenv("RUN_SPOTIFY_LIVE") != "1", reason="live test; set RUN_SPOTIFY_LIVE=1")
def test_live_play():
    from jarvis.config import config

    sc = SpotifyController(config.spotify_client_id, config.spotify_client_secret)
    res = sc.play_query("bohemian rhapsody queen", mode=config.spotify_search_mode)
    assert res.label
    assert sc.player_state() == "playing"
