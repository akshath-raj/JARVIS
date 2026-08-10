"""LangChain tools for the JARVIS graph — thin wrappers over the existing
controllers (Spotify / Chrome / Tavily) plus the memory store.

Every tool is async and pushes blocking work (AppleScript, HTTP, embeddings) onto
a worker thread via `asyncio.to_thread`, so playback/search never blocks the voice
loop. Action tools also log to long-term memory so JARVIS learns the user's tastes.
The controllers themselves are unchanged — this only re-exposes their behaviour to
LangGraph, so music keeps working exactly as before.
"""
from __future__ import annotations

import asyncio
import random

from langchain_core.tools import tool

from jarvis.browser_agent.client import run_browser_task
from jarvis.config import config
from jarvis.graph.memory import MemoryStore
from jarvis.tools.browser import BrowserController, BrowserError
from jarvis.tools.media import MediaController, MediaError
from jarvis.tools.spotify import SpotifyController, SpotifyError
from jarvis.tools.web import TavilyClient, WebError


def build_tools(
    *,
    spotify: SpotifyController,
    browser: BrowserController,
    tavily: TavilyClient,
    memory: MemoryStore,
    search_mode: str = "auto",
    announce=None,
    workspace=None,
    media: MediaController | None = None,
) -> list:
    # ── Music (Spotify) ───────────────────────────────────────────────────
    @tool
    async def play_song(query: str, loop: bool = False) -> str:
        """Search for a song and play it on Spotify. Set loop=true to repeat it.
        Pass exactly what the user asked for (title/artist/genre/mood)."""
        def _do():
            res = spotify.play_query(query, search_mode, loop)
            memory.log_activity(f"played music: {query}")
            return f"now playing {res.label}" + (" on repeat" if loop else "")
        try:
            return await asyncio.to_thread(_do)
        except SpotifyError as e:
            return f"error: {e}"

    @tool
    async def play_playlist(name: str, loop: bool = False) -> str:
        """Play one of the user's own Spotify playlists by name."""
        def _do():
            pl = spotify.play_playlist(name, loop)
            memory.log_activity(f"played playlist: {pl}")
            return f"playing your {pl} playlist"
        try:
            return await asyncio.to_thread(_do)
        except SpotifyError as e:
            return f"error: {e}"

    @tool
    async def list_playlists() -> str:
        """List the user's Spotify playlists and how many songs each has."""
        try:
            pls = await asyncio.to_thread(spotify.list_playlists)
        except SpotifyError as e:
            return f"error: {e}"
        if not pls:
            return "you have no playlists"
        if any(c is not None for _, c in pls):
            return f"{len(pls)} playlists: " + ", ".join(
                f"{n} ({c} songs)" if c is not None else n for n, c in pls
            )
        return f"{len(pls)} playlists: " + ", ".join(n for n, _ in pls)

    @tool
    async def add_current_song_to_playlist(playlist: str) -> str:
        """Add the currently playing song to one of the user's playlists."""
        try:
            track, pl = await asyncio.to_thread(spotify.add_current_to_playlist, playlist)
        except SpotifyError as e:
            return f"error: {e}"
        return f"added {track} to {pl}"

    @tool
    async def set_loop(enabled: bool) -> str:
        """Loop the currently playing song on repeat (enabled=true) or stop looping."""
        try:
            if enabled:
                label = await asyncio.to_thread(spotify.loop_current)
                return f"looping {label}"
            await asyncio.to_thread(spotify.set_repeat, False)
            return "looping off"
        except SpotifyError as e:
            return f"error: {e}"

    @tool
    async def pause_music() -> str:
        """Pause Spotify playback."""
        try:
            await asyncio.to_thread(spotify.pause)
            return "paused"
        except SpotifyError as e:
            return f"error: {e}"

    @tool
    async def resume_music() -> str:
        """Resume Spotify playback."""
        try:
            await asyncio.to_thread(spotify.resume)
            return "resumed"
        except SpotifyError as e:
            return f"error: {e}"

    @tool
    async def next_song() -> str:
        """Skip to the next track on Spotify."""
        try:
            await asyncio.to_thread(spotify.next_track)
            return "skipped"
        except SpotifyError as e:
            return f"error: {e}"

    @tool
    async def set_music_volume(level: int) -> str:
        """Set Spotify volume to an EXACT level (0-100). Use only when the user gives
        a number, e.g. 'set volume to 40'."""
        try:
            await asyncio.to_thread(spotify.set_volume, level)
            return f"volume set to {level}"
        except SpotifyError as e:
            return f"error: {e}"

    @tool
    async def change_volume(direction: str) -> str:
        """Turn the music volume UP or DOWN relative to now. Use for 'increase/turn up/
        louder' (direction='up') or 'decrease/turn down/quieter' (direction='down') when
        the user gives no exact number."""
        up = any(w in direction.lower() for w in ("up", "loud", "increase", "higher", "raise", "more"))
        try:
            level = await asyncio.to_thread(spotify.nudge_volume, up)
            return f"volume {'up' if up else 'down'} to {level}"
        except SpotifyError as e:
            return f"error: {e}"

    @tool
    async def whats_playing() -> str:
        """Report the currently playing Spotify track."""
        try:
            return await asyncio.to_thread(spotify.current_track)
        except SpotifyError as e:
            return f"error: {e}"

    @tool
    async def list_top_tracks() -> str:
        """List the user's most-played (top) songs."""
        try:
            tracks = await asyncio.to_thread(spotify.top_tracks, 10)
        except SpotifyError as e:
            return f"error: {e}"
        return "your top tracks: " + ", ".join(t.label for t in tracks) if tracks else "no top tracks yet"

    @tool
    async def play_most_played() -> str:
        """Play the user's single most-played song."""
        def _do():
            tracks = spotify.top_tracks(5)
            if not tracks:
                return "no top tracks yet"
            label = spotify.play_track(tracks[0])
            memory.log_activity(f"played most-played: {label}")
            return f"now playing your most played, {label}"
        try:
            return await asyncio.to_thread(_do)
        except SpotifyError as e:
            return f"error: {e}"

    @tool
    async def list_liked_songs() -> str:
        """List the user's liked/favourite songs."""
        try:
            tracks = await asyncio.to_thread(spotify.liked_songs, 15)
        except SpotifyError as e:
            return f"error: {e}"
        return f"{len(tracks)} liked songs: " + ", ".join(t.label for t in tracks) if tracks else "no liked songs yet"

    @tool
    async def play_favorite_song() -> str:
        """Play one of the user's liked/favourite songs."""
        def _do():
            tracks = spotify.liked_songs(50)
            if not tracks:
                return "no liked songs yet"
            label = spotify.play_track(random.choice(tracks))
            memory.log_activity(f"played a favourite: {label}")
            return f"now playing one of your favourites, {label}"
        try:
            return await asyncio.to_thread(_do)
        except SpotifyError as e:
            return f"error: {e}"

    @tool
    async def list_top_artists() -> str:
        """List the user's most-listened-to (top) artists."""
        try:
            artists = await asyncio.to_thread(spotify.top_artists, 10)
        except SpotifyError as e:
            return f"error: {e}"
        return "your top artists: " + ", ".join(artists) if artists else "no top artists yet"

    @tool
    async def recently_played() -> str:
        """List songs the user played recently."""
        try:
            tracks = await asyncio.to_thread(spotify.recently_played, 10)
        except SpotifyError as e:
            return f"error: {e}"
        return "recently played: " + ", ".join(t.label for t in tracks) if tracks else "nothing played recently"

    # ── Browser (Chrome) ──────────────────────────────────────────────────
    @tool
    async def open_site(name: str) -> str:
        """Open a website or web app in Chrome by name (youtube, instagram, gmail,
        github, netflix, …) or a URL. Unknown names are searched on Google."""
        def _do():
            res = browser.open_site(name)
            memory.log_activity(f"opened in browser: {name}")
            return res
        try:
            return await asyncio.to_thread(_do)
        except BrowserError as e:
            return f"error: {e}"

    @tool
    async def play_youtube(query: str) -> str:
        """Search YouTube for a song/video and open the top result (it autoplays).
        Use for 'play <x> on youtube' or 'play the <x> video'."""
        def _do():
            res = browser.play_youtube(query)
            memory.log_activity(f"watched on YouTube: {query}")
            return res
        try:
            return await asyncio.to_thread(_do)
        except BrowserError as e:
            return f"error: {e}"

    @tool
    async def latest_channel_video(channel: str) -> str:
        """Open the newest upload from a YouTube channel. Use for 'open a new
        <creator> video' or 'latest <creator> video'."""
        def _do():
            res = browser.latest_channel_video(channel)
            memory.log_activity(f"watched latest video from: {channel}")
            return res
        try:
            return await asyncio.to_thread(_do)
        except BrowserError as e:
            return f"error: {e}"

    @tool
    async def open_reels(platform: str = "instagram") -> str:
        """Open a scrolling short-form video feed. platform: 'instagram' (default)
        or 'youtube' for Shorts."""
        try:
            return await asyncio.to_thread(browser.open_reels, platform)
        except BrowserError as e:
            return f"error: {e}"

    @tool
    async def open_shorts() -> str:
        """Open the YouTube Shorts feed."""
        try:
            return await asyncio.to_thread(browser.open_shorts)
        except BrowserError as e:
            return f"error: {e}"

    # ── Web search (Tavily) ───────────────────────────────────────────────
    @tool
    async def web_search(query: str) -> str:
        """Look up current or recent information on the web — news, live prices,
        recent events, anything that changes over time. Returns a short answer."""
        def _do():
            ans = tavily.search(query)
            memory.log_activity(f"searched the web: {query}")
            return ans
        try:
            return await asyncio.to_thread(_do)
        except WebError as e:
            return f"error: {e}"

    # ── Memory (explicit control) ─────────────────────────────────────────
    @tool
    async def remember(text: str) -> str:
        """Save a durable fact or preference about the user to long-term memory.
        Use when the user says 'remember that I …' or states a lasting preference."""
        return await asyncio.to_thread(memory.remember, text, "explicit")

    @tool
    async def forget(query: str) -> str:
        """Forget a stored memory that best matches the description."""
        return await asyncio.to_thread(memory.forget, query)

    @tool
    async def recall_about_me() -> str:
        """Recall what JARVIS knows about the user. Use for 'what do you know about
        me?' or 'what have you remembered?'."""
        return await asyncio.to_thread(memory.recall_text)

    # ── Browser agent (multi-step, logged-in web workflows) ───────────────
    @tool
    async def browser_task(instruction: str) -> str:
        """Perform a multi-step task on a website that requires logging in,
        navigating, searching, downloading files, or reading specific account
        info. Use for things like "download the OS study materials by Professor X
        from VTOP", "check the balance on my AWS account", or "find my latest
        order on Amazon". Do NOT use this to merely open a website (use open_site).

        Args:
            instruction: The full task in the user's words, with any subject,
                teacher, site, or account details they mentioned.
        """
        return await run_browser_task(instruction, announce=announce)

    # ── Documents / assignments ───────────────────────────────────────────
    @tool
    async def download_and_explain(what: str) -> str:
        """Download a document/assignment from a website, then read and EXPLAIN it
        aloud. Use for "download the latest assignment from VTOP and explain it",
        "get my <thing> and tell me what's in it".

        Args:
            what: What to download, in the user's words (site + which document).
        """
        if workspace is None:
            return "documents aren't set up, sir"
        from jarvis.documents import flow
        asyncio.create_task(flow.download_and_explain(what, workspace, announce))
        return "On it, sir — downloading and reading it now."

    @tool
    async def do_assignment(instructions: str) -> str:
        """Complete the currently loaded assignment with an AI and save the answer
        file. Use for "finish the assignment", "do it and make it a jupyter
        notebook / word doc / powerpoint / python script". Pass along any format or
        extra instructions the user gave.

        Args:
            instructions: The user's instructions incl. desired output format.
        """
        if workspace is None:
            return "documents aren't set up, sir"
        from jarvis.documents import flow
        asyncio.create_task(flow.do_assignment(instructions, workspace, announce))
        return "On it, sir — working on the answer now."

    @tool
    async def open_answer() -> str:
        """Open the finished answer document for review in the right app (Preview,
        Word, PowerPoint, or VS Code). Use when the user says yes to reviewing, or
        asks to open the answer/it/the document."""
        if workspace is None:
            return "there's nothing to open, sir"
        from jarvis.documents import flow
        return flow.open_answer(workspace)

    # ── Live video / tab control (the user's open Chrome tab) ─────────────
    @tool
    async def control_video(action: str, value: str = "") -> str:
        """Control the video the user is watching in Chrome (YouTube/Netflix/etc).
        ALWAYS call this for such requests. Map the request to `action`:
        'subtitles'/'captions'/'CC' -> captions; 'speed up'/'faster' -> faster;
        'slow down'/'slower' -> slower; 'play at 1.5x'/'set speed to 2' -> set_speed
        (value=the number); 'louder'/'turn it up' -> volume_up; 'quieter' ->
        volume_down; 'set volume to 40' -> set_volume (value=0-100); 'mute';
        'brighter'/'darker'; 'pause'; 'continue'/'resume'/'keep playing'/'play' ->
        play; 'start over'/'from the beginning' -> restart; 'skip ahead' -> forward;
        'go back'/'rewind' -> back; 'fullscreen'. Use play (not restart) to continue
        where it left off."""
        if media is None:
            return "video control isn't set up, sir"
        try:
            return await asyncio.to_thread(media.control, action, value)
        except MediaError as e:
            return f"error: {e}"

    @tool
    async def close_tab() -> str:
        """Close the current/active browser tab."""
        if media is None:
            return "I can't control the browser, sir"
        try:
            return await asyncio.to_thread(media.close_tab)
        except MediaError as e:
            return f"error: {e}"

    tools = [
        play_song, play_playlist, list_playlists, add_current_song_to_playlist,
        set_loop, pause_music, resume_music, next_song, set_music_volume, change_volume,
        whats_playing, list_top_tracks, play_most_played, list_liked_songs,
        play_favorite_song, list_top_artists, recently_played,
        open_site, play_youtube, latest_channel_video, open_reels, open_shorts,
        web_search, remember, forget, recall_about_me,
    ]
    if config.browser_agent_enabled:
        tools.append(browser_task)
    if workspace is not None:
        tools += [download_and_explain, do_assignment, open_answer]
    if media is not None:
        tools += [control_video, close_tab]
    return tools
