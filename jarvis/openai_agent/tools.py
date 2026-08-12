"""OpenAI Agents SDK tools — thin wrappers over the SAME controllers the LangGraph
brain uses (Spotify / Chrome / media / Tavily / memory / browser-agent / documents)
plus the new screen-vision tool.

Each builder returns a list of `function_tool`s scoped to one specialist agent, so
the behaviour is identical to the local brain — only the agent runtime differs.
Blocking work (AppleScript, HTTP, screenshots) is pushed to a worker thread.
"""
from __future__ import annotations

import asyncio
import random

from agents import function_tool

from jarvis.browser_agent.client import run_browser_task
from jarvis.tools.browser import BrowserController, BrowserError
from jarvis.tools.media import MediaController, MediaError
from jarvis.tools.screen import ScreenController, ScreenError
from jarvis.tools.spotify import SpotifyController, SpotifyError
from jarvis.tools.web import TavilyClient, WebError


# ── Music (Spotify) ───────────────────────────────────────────────────────
def build_music_tools(*, spotify: SpotifyController, memory, search_mode: str = "auto") -> list:
    @function_tool
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

    @function_tool
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

    @function_tool
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

    @function_tool
    async def add_current_song_to_playlist(playlist: str) -> str:
        """Add the currently playing song to one of the user's playlists."""
        try:
            track, pl = await asyncio.to_thread(spotify.add_current_to_playlist, playlist)
        except SpotifyError as e:
            return f"error: {e}"
        return f"added {track} to {pl}"

    @function_tool
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

    @function_tool
    async def pause_music() -> str:
        """Pause Spotify playback."""
        try:
            await asyncio.to_thread(spotify.pause)
            return "paused"
        except SpotifyError as e:
            return f"error: {e}"

    @function_tool
    async def resume_music() -> str:
        """Resume Spotify playback."""
        try:
            await asyncio.to_thread(spotify.resume)
            return "resumed"
        except SpotifyError as e:
            return f"error: {e}"

    @function_tool
    async def next_song() -> str:
        """Skip to the next track on Spotify."""
        try:
            await asyncio.to_thread(spotify.next_track)
            return "skipped"
        except SpotifyError as e:
            return f"error: {e}"

    @function_tool
    async def set_music_volume(level: int) -> str:
        """Set Spotify volume to an EXACT level (0-100). Use only when the user gives
        a number, e.g. 'set volume to 40'."""
        try:
            await asyncio.to_thread(spotify.set_volume, level)
            return f"volume set to {level}"
        except SpotifyError as e:
            return f"error: {e}"

    @function_tool
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

    @function_tool
    async def whats_playing() -> str:
        """Report the currently playing Spotify track."""
        try:
            return await asyncio.to_thread(spotify.current_track)
        except SpotifyError as e:
            return f"error: {e}"

    @function_tool
    async def list_top_tracks() -> str:
        """List the user's most-played (top) songs."""
        try:
            tracks = await asyncio.to_thread(spotify.top_tracks, 10)
        except SpotifyError as e:
            return f"error: {e}"
        return "your top tracks: " + ", ".join(t.label for t in tracks) if tracks else "no top tracks yet"

    @function_tool
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

    @function_tool
    async def list_liked_songs() -> str:
        """List the user's liked/favourite songs."""
        try:
            tracks = await asyncio.to_thread(spotify.liked_songs, 15)
        except SpotifyError as e:
            return f"error: {e}"
        return f"{len(tracks)} liked songs: " + ", ".join(t.label for t in tracks) if tracks else "no liked songs yet"

    @function_tool
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

    @function_tool
    async def list_top_artists() -> str:
        """List the user's most-listened-to (top) artists."""
        try:
            artists = await asyncio.to_thread(spotify.top_artists, 10)
        except SpotifyError as e:
            return f"error: {e}"
        return "your top artists: " + ", ".join(artists) if artists else "no top artists yet"

    @function_tool
    async def recently_played() -> str:
        """List songs the user played recently."""
        try:
            tracks = await asyncio.to_thread(spotify.recently_played, 10)
        except SpotifyError as e:
            return f"error: {e}"
        return "recently played: " + ", ".join(t.label for t in tracks) if tracks else "nothing played recently"

    return [
        play_song, play_playlist, list_playlists, add_current_song_to_playlist,
        set_loop, pause_music, resume_music, next_song, set_music_volume, change_volume,
        whats_playing, list_top_tracks, play_most_played, list_liked_songs,
        play_favorite_song, list_top_artists, recently_played,
    ]


# ── Browser / media / tasks / documents ────────────────────────────────────
def build_browser_tools(
    *, browser: BrowserController, memory, media: MediaController | None = None,
    announce=None, workspace=None, browser_agent_enabled: bool = True,
) -> list:
    @function_tool
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

    @function_tool
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

    @function_tool
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

    @function_tool
    async def open_reels(platform: str = "instagram") -> str:
        """Open a scrolling short-form video feed. platform: 'instagram' (default)
        or 'youtube' for Shorts."""
        try:
            return await asyncio.to_thread(browser.open_reels, platform)
        except BrowserError as e:
            return f"error: {e}"

    @function_tool
    async def open_shorts() -> str:
        """Open the YouTube Shorts feed."""
        try:
            return await asyncio.to_thread(browser.open_shorts)
        except BrowserError as e:
            return f"error: {e}"

    @function_tool
    async def browser_task(instruction: str) -> str:
        """Perform a multi-step task on a website that requires logging in,
        navigating, searching, downloading files, or reading specific account
        info (e.g. "download the OS study materials by Professor X from VTOP",
        "check the balance on my AWS account", "find my latest Amazon order"). Do
        NOT use this to merely open a website (use open_site)."""
        return await run_browser_task(instruction, announce=announce)

    @function_tool
    async def download_and_explain(what: str) -> str:
        """Download a document/assignment from a website, then read and EXPLAIN it
        aloud. Use for "download the latest assignment from VTOP and explain it"."""
        if workspace is None:
            return "documents aren't set up, sir"
        from jarvis.documents import flow
        asyncio.create_task(flow.download_and_explain(what, workspace, announce))
        return "On it, sir — downloading and reading it now."

    @function_tool
    async def do_assignment(instructions: str) -> str:
        """Complete the currently loaded assignment with an AI and save the answer
        file. Use for "finish the assignment", "do it and make it a jupyter
        notebook / word doc / powerpoint / python script"."""
        if workspace is None:
            return "documents aren't set up, sir"
        from jarvis.documents import flow
        asyncio.create_task(flow.do_assignment(instructions, workspace, announce))
        return "On it, sir — working on the answer now."

    @function_tool
    async def open_answer() -> str:
        """Open the finished answer document for review in the right app (Preview,
        Word, PowerPoint, or VS Code)."""
        if workspace is None:
            return "there's nothing to open, sir"
        from jarvis.documents import flow
        return flow.open_answer(workspace)

    @function_tool
    async def control_video(action: str, value: str = "") -> str:
        """Control the video the user is watching in Chrome (YouTube/Netflix/etc).
        Map the request to `action`: 'subtitles'/'captions'/'CC' -> captions;
        'speed up'/'faster' -> faster; 'slow down'/'slower' -> slower; 'play at
        1.5x'/'set speed to 2' -> set_speed (value=the number); 'louder' ->
        volume_up; 'quieter' -> volume_down; 'set volume to 40' -> set_volume
        (value=0-100); 'mute'; 'brighter'/'darker'; 'pause'; 'continue'/'resume'/
        'keep playing'/'play' -> play; 'start over' -> restart; 'skip ahead' ->
        forward; 'go back'/'rewind' -> back; 'fullscreen'. Use play (not restart)
        to continue where it left off."""
        if media is None:
            return "video control isn't set up, sir"
        try:
            return await asyncio.to_thread(media.control, action, value)
        except MediaError as e:
            return f"error: {e}"

    @function_tool
    async def close_tab() -> str:
        """Close the current/active browser tab."""
        if media is None:
            return "I can't control the browser, sir"
        try:
            return await asyncio.to_thread(media.close_tab)
        except MediaError as e:
            return f"error: {e}"

    tools = [open_site, play_youtube, latest_channel_video, open_reels, open_shorts]
    if browser_agent_enabled:
        tools.append(browser_task)
    if workspace is not None:
        tools += [download_and_explain, do_assignment, open_answer]
    if media is not None:
        tools += [control_video, close_tab]
    return tools


# ── Screen vision ──────────────────────────────────────────────────────────
def build_screen_tools(*, screen: ScreenController, memory, ui=None) -> list:
    @function_tool
    async def explain_screen(question: str = "", show_in_ui: bool = False) -> str:
        """Take a screenshot of the user's screen and explain what's on it, in
        detail. Use whenever the user asks about what is CURRENTLY displayed /
        open / shown on their screen (e.g. "what's on my screen", "explain in
        detail what I'm looking at", "read this for me", "explain this formula",
        "what does this error mean"). Pass the user's specific question as
        `question` (empty = describe everything). Set show_in_ui=true when the user
        asks to SHOW / OPEN / DISPLAY the explanation on the dashboard/UI/screen —
        it will render the captured screen and the full analysis on the HUD. Return
        the explanation to the user in full — do not shorten it."""
        def _do():
            if show_in_ui and ui is not None:
                ans, img = screen.analyse(question)
                ui.show_explanation(
                    title=question.strip()[:60] or "Screen analysis",
                    body=ans, image_b64=img, source="screen vision · gpt-4o",
                )
            else:
                ans = screen.explain(question)
            memory.log_activity("explained the user's screen")
            return ans
        try:
            return await asyncio.to_thread(_do)
        except ScreenError as e:
            return f"error: {e}"

    @function_tool
    async def take_screenshot() -> str:
        """Take a screenshot of the screen and save it, without analysing it. Use
        only when the user just wants a screenshot captured."""
        try:
            path = await asyncio.to_thread(screen.capture)
            return f"screenshot saved to {path}"
        except ScreenError as e:
            return f"error: {e}"

    return [explain_screen, take_screenshot]


# ── HUD dashboard (hidden UI, voice-summoned) ───────────────────────────────
def build_ui_tools(*, ui, open_cb=None) -> list:
    @function_tool
    async def show_dashboard() -> str:
        """Reveal the JARVIS dashboard / HUD on screen. Use when the user asks to
        "open the dashboard", "show me the interface", "bring up the UI", "show
        yourself", or similar. It boots up with a welcome animation and displays
        the user's profile, memories, and past conversations."""
        ui.reveal()
        if open_cb is not None:
            await asyncio.to_thread(open_cb)
        return "Bringing up the interface now, sir."

    @function_tool
    async def hide_dashboard() -> str:
        """Hide / dismiss the JARVIS dashboard. Use for "hide the dashboard",
        "close the interface", "dismiss the UI"."""
        ui.hide()
        return "Dashboard dismissed, sir."

    @function_tool
    async def open_dashboard_section(section: str) -> str:
        """Direct the user to a section of the dashboard and highlight it. section
        must be one of: 'about' (profile), 'memories', 'conversations', 'analysis'.
        Use for "show me what you know about me", "pull up my memories", "show my
        past conversations"."""
        s = section.strip().lower()
        if s not in ("about", "memories", "conversations", "analysis"):
            s = "about"
        ui.navigate(s)
        return f"Here are your {s}, sir."

    @function_tool
    async def display_on_dashboard(title: str, content: str) -> str:
        """Show a piece of text/explanation on the dashboard's main panel. Use when
        the user asks to SHOW or DISPLAY an answer/explanation on the UI/screen that
        is NOT about their screen contents (for a screen analysis use
        explain_screen with show_in_ui=true instead). Pass a short `title` and the
        full `content` to render."""
        ui.show_explanation(title=title or "Analysis", body=content or "", source="JARVIS")
        return "Displayed on your dashboard, sir."

    return [show_dashboard, hide_dashboard, open_dashboard_section, display_on_dashboard]


# ── Web + memory (live on the triage agent) ────────────────────────────────
def build_triage_tools(*, tavily: TavilyClient, memory) -> list:
    @function_tool
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

    @function_tool
    async def remember(text: str) -> str:
        """Save a durable fact or preference about the user to long-term memory.
        Use when the user says 'remember that I …' or states a lasting preference."""
        return await asyncio.to_thread(memory.remember, text, "explicit")

    @function_tool
    async def forget(query: str) -> str:
        """Forget a stored memory that best matches the description."""
        return await asyncio.to_thread(memory.forget, query)

    @function_tool
    async def recall_about_me() -> str:
        """Recall what JARVIS knows about the user. Use for 'what do you know about
        me?' or 'what have you remembered?'."""
        return await asyncio.to_thread(memory.recall_text)

    return [web_search, remember, forget, recall_about_me]
