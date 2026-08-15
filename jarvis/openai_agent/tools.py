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
from jarvis.tools.images import ImageError, SafeImageFetcher
from jarvis.tools.media import MediaController, MediaError
from jarvis.tools.pages import PagesController, PagesError
from jarvis.tools.pages_author import AuthorError, PagesAuthor
from jarvis.tools.reading_mode import ReadingMode
from jarvis.tools.screen import ScreenController, ScreenError
from jarvis.tools.spotify import SpotifyController, SpotifyError
from jarvis.tools.system_settings import SettingsError, SystemSettings
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
        """Set the SPOTIFY MUSIC volume to an EXACT level (0-100). Use ONLY when the
        user explicitly means the music/song/Spotify ('set the music to 40', 'Spotify
        volume to 50'). For a plain 'set the volume to 40' use set_system_volume."""
        try:
            await asyncio.to_thread(spotify.set_volume, level)
            return f"music volume set to {level}"
        except SpotifyError as e:
            return f"error: {e}"

    @function_tool
    async def change_volume(direction: str) -> str:
        """Turn the SPOTIFY MUSIC volume UP or DOWN. Use ONLY when the user explicitly
        means the music/song/Spotify ('turn the music up', 'make the song quieter').
        For a plain 'louder'/'turn it down'/'volume up' use change_system_volume."""
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
    announce=None, workspace=None, browser_agent_enabled: bool = True, focus=None,
) -> list:
    def _blocked(target: str) -> str | None:
        """If focus mode is on and `target` is a distraction, the refusal line."""
        if focus is not None and focus.active and focus.is_distraction(target):
            return (f"That's blocked in focus mode, sir — {target} stays shut. "
                    "Say end focus mode to lift it.")
        return None

    @function_tool
    async def open_site(name: str) -> str:
        """Open a website or web app in Chrome by name (youtube, instagram, gmail,
        github, netflix, …) or a URL. Unknown names are searched on Google."""
        if (msg := _blocked(name)):
            return msg
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
        if (msg := _blocked("youtube")):
            return msg
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
        if (msg := _blocked("youtube")):
            return msg
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
        if (msg := _blocked(platform)) or (msg := _blocked("reels")):
            return msg
        try:
            return await asyncio.to_thread(browser.open_reels, platform)
        except BrowserError as e:
            return f"error: {e}"

    @function_tool
    async def open_shorts() -> str:
        """Open the YouTube Shorts feed."""
        if (msg := _blocked("shorts")):
            return msg
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
def build_screen_tools(*, screen: ScreenController, memory, ui=None, rag=None) -> list:
    @function_tool
    async def explain_screen(question: str = "", show_in_ui: bool = False) -> str:
        """Take a screenshot of the user's screen and explain what's on it, in
        detail. This is the ONE tool for any request to look at / read / describe /
        explain the screen — including "take a screenshot AND tell me what's there",
        "screenshot this and explain it", "what's on my screen", "read this for me",
        "explain this formula", "what does this error mean". It captures the screen
        AND analyses it with the vision model, then returns the real description —
        so NEVER answer such a request from your own guess, and never use
        take_screenshot for it. Pass the user's specific question as `question`
        (empty = describe everything). Set show_in_ui=true when the user asks to
        SHOW / OPEN / DISPLAY the explanation on the dashboard/UI/screen — it will
        render the captured screen and the full analysis on the HUD. Return the
        explanation to the user in full — do not shorten it."""
        def _do():
            if show_in_ui and ui is not None:
                ans, img = screen.analyse(question)
                ui.show_explanation(
                    title=question.strip()[:60] or "Screen analysis",
                    body=ans, image_b64=img, source=f"screen vision · {screen.model}",
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
        """Save a raw screenshot to a file WITHOUT looking at it. Use ONLY when the
        user explicitly wants a screenshot file saved and nothing explained (e.g.
        "save a screenshot", "capture my screen to a file", "grab a screenshot for
        later"). This tool does NOT look at the screen, so NEVER use it to answer,
        describe, read, or explain what is shown — use explain_screen for anything
        that involves understanding the screen."""
        try:
            path = await asyncio.to_thread(screen.capture)
            return f"screenshot saved to {path}"
        except ScreenError as e:
            return f"error: {e}"

    tools = [explain_screen, take_screenshot]

    if rag is not None:
        @function_tool
        async def explain_this(question: str = "", show_in_ui: bool = False) -> str:
            """Explain the topic in the document the user is CURRENTLY looking at on
            screen. Use for "explain this topic I don't understand", "explain this
            slide", "what does this section mean", "I'm stuck on this page" — when the
            user refers to something in a document open in front of them rather than
            asking a standalone question. JARVIS finds which of the open documents is
            live on screen, works out the page/slide and subtopic, then reads the
            pages ABOVE and BELOW that subtopic from the indexed file (not just what's
            visible) and explains it. Falls back to plain screen vision if the file
            isn't indexed. Set show_in_ui=true to also render it on the HUD."""
            def _do():
                # 1) Look at the screen: which document, page/slide, subtopic, snippet.
                ctx = screen.read_screen_context(question)
                name_hint = ctx.get("document") or ctx.get("window_title") or ""
                # 2) Explain from the indexed document, expanding across pages.
                result = rag.explain_topic(
                    question,
                    document=name_hint,
                    locator_text=ctx.get("snippet", ""),
                    page=ctx.get("page", 0) or 0,
                )
                if result is None:
                    # not indexed / couldn't locate → fall back to pure screen vision
                    ans, img = screen.analyse(question or "Explain the topic shown here.")
                    cite = f"screen vision · {screen.model}"
                    title = (question.strip()[:60] or "On-screen explanation")
                else:
                    ans = result["answer"]
                    img = ctx.get("image_b64", "")
                    cite = f"document · {result.get('cite') or result.get('document','')}"
                    title = (question.strip()[:60] or result.get("subtopic")
                             or f"{result.get('document','Document')} explained")
                if show_in_ui and ui is not None:
                    ui.show_explanation(title=title, body=ans, image_b64=img, source=cite)
                memory.log_activity("explained a topic in the on-screen document")
                return ans
            try:
                return await asyncio.to_thread(_do)
            except ScreenError as e:
                return f"error: {e}"

        tools.append(explain_this)

    return tools


# ── Planner: calendar / to-dos / reminders + alarms ─────────────────────────
def build_planner_tools(*, scheduler, memory=None) -> list:
    from jarvis.scheduler.scheduler import _fmt as fmt_time

    store = scheduler.store

    @function_tool
    async def add_todo(task: str) -> str:
        """Add an item to the user's to-do list. Use for "add X to my to-do list",
        "I need to do X", "put X on my list" (with NO specific time — if they give a
        time to be reminded, use add_reminder instead)."""
        store.add_todo(task)
        return f"Added to your to-do list: {task}."

    @function_tool
    async def list_todos() -> str:
        """List the user's outstanding to-do items."""
        todos = store.list_todos()
        if not todos:
            return "Your to-do list is empty, sir."
        return f"You have {len(todos)} to-do(s): " + "; ".join(t["text"] for t in todos)

    @function_tool
    async def complete_todo(task: str) -> str:
        """Mark a to-do as done / completed. Use when the user says "I'm done with
        X", "I finished X", "cross off X", "mark X complete". Also stops any alarm
        still ringing for that item."""
        done = store.complete_todo(task)
        # silence a ringing alarm for this item and cancel any pending reminder so
        # it won't fire later.
        for a in scheduler.active_alarms:
            if task.lower() in a.get("text", "").lower() or a.get("text", "").lower() in task.lower():
                scheduler.dismiss(a["id"])
        store.cancel_reminder(task)
        if not done:
            return f"I couldn't find '{task}' on your list, sir."
        return f"Done — I've crossed '{done['text']}' off your list, sir."

    @function_tool
    async def remove_todo(task: str) -> str:
        """Remove/delete a to-do item without completing it."""
        rem = store.remove_todo(task)
        return f"Removed '{rem['text']}' from your list." if rem else f"'{task}' isn't on your list, sir."

    @function_tool
    async def add_reminder(task: str, when: str) -> str:
        """Set a time-based reminder that RINGS AN ALARM at the given time. Use for
        "remind me to X at 9pm", "remind me to X in an hour", "remind me in 30
        minutes to X". Pass the thing to be reminded of as `task`, and the timing as
        `when`. `when` may be an absolute time you've computed from the current time
        as an ISO 8601 datetime (preferred, e.g. '2026-08-12T21:00:00'), or a natural
        phrase like 'at 9pm today' / 'in 1 hour' / 'tomorrow at 8am'."""
        res = await asyncio.to_thread(scheduler.add_reminder, task, when)
        if memory is not None and res.startswith("Reminder set"):
            memory.log_activity(f"set a reminder: {task}")
        return res

    @function_tool
    async def list_reminders() -> str:
        """List the user's upcoming (not-yet-fired) reminders."""
        rs = store.list_reminders()
        if not rs:
            return "You have no upcoming reminders, sir."
        return f"{len(rs)} reminder(s): " + "; ".join(f"{r['text']} at {fmt_time(r['due'])}" for r in rs)

    @function_tool
    async def cancel_reminder(task: str) -> str:
        """Cancel an upcoming reminder before it fires."""
        r = store.cancel_reminder(task)
        return f"Cancelled the reminder for '{r['text']}'." if r else f"No pending reminder matches '{task}', sir."

    @function_tool
    async def add_calendar_event(title: str, when: str) -> str:
        """Add an event to the user's calendar/agenda. Use for "add X to my
        calendar", "schedule X for tomorrow at 3pm", "I have a meeting at ...".
        `when` is an ISO 8601 datetime you computed, or a natural phrase."""
        return await asyncio.to_thread(scheduler.add_event, title, when)

    @function_tool
    async def list_calendar_events() -> str:
        """List the user's upcoming calendar events."""
        evs = store.list_events()
        if not evs:
            return "Your calendar is clear, sir."
        return f"{len(evs)} event(s): " + "; ".join(f"{e['title']} at {fmt_time(e['start'])}" for e in evs)

    @function_tool
    async def cancel_calendar_event(title: str) -> str:
        """Remove/cancel a calendar event."""
        e = store.remove_event(title)
        return f"Removed '{e['title']}' from your calendar." if e else f"No event matches '{title}', sir."

    @function_tool
    async def stop_alarm() -> str:
        """Stop / silence / dismiss a currently ringing reminder alarm. Use when the
        user says "stop the alarm", "turn it off", "dismiss", "I'm done" while an
        alarm is ringing."""
        if not scheduler.active_alarms:
            return "There's no alarm ringing, sir."
        dismissed = scheduler.dismiss()
        return "Alarm dismissed, sir." + (f" Shall I mark '{dismissed['text']}' as done?" if dismissed else "")

    @function_tool
    async def current_time() -> str:
        """Tell the current date and time. Use for "what time is it", "what's the
        date"."""
        import datetime
        now = datetime.datetime.now()
        return now.strftime("It's %-I:%M %p on %A, %-d %B %Y, sir.")

    return [
        add_todo, list_todos, complete_todo, remove_todo,
        add_reminder, list_reminders, cancel_reminder,
        add_calendar_event, list_calendar_events, cancel_calendar_event,
        stop_alarm, current_time,
    ]


# ── Files & documents: RAG Q&A + sandboxed organise / move / copy / open ────
def build_files_tools(*, rag=None, organizer=None, memory=None) -> list:
    from jarvis.files.organizer import FileError
    from jarvis.files.sandbox import SandboxError

    tools = []

    if rag is not None:
        @function_tool
        async def ask_documents(question: str, document: str = "", page: int = 0) -> str:
            """Answer ANY question from the user's indexed documents (PDFs, Word,
            PowerPoint, notes) — "what does my report say about X", "according to my
            notes …", "explain the formula on page 4", "what's on slide 3". You do
            NOT need the exact filename: set `document` to a free-text description of
            the file ("my machine learning assignment", "the OS notes") and it is
            resolved automatically. Set `page` to a page/slide number to focus on
            that page. Leave both blank to search across everything."""
            def _do():
                ans = rag.answer(question, document=document, page=page)
                if memory is not None:
                    memory.log_activity(f"answered from documents: {question}")
                return ans
            return await asyncio.to_thread(_do)

        @function_tool
        async def summarize_document(document: str, page: int = 0) -> str:
            """Summarise a whole document (or one page/slide). `document` is a
            free-text description of the file ("my thermodynamics notes"); no exact
            filename needed. Set `page` to summarise just that page/slide. Use for
            "summarise my <doc>", "give me the gist of X", "what's slide 5 about"."""
            def _do():
                return rag.summarize(document, page=page)
            return await asyncio.to_thread(_do)

        @function_tool
        async def reindex_documents() -> str:
            """Rescan the user's folders and update the document index (add new
            files, refresh changed ones, drop deleted ones). Use for "reindex my
            documents", "scan my files", or if a document seems missing."""
            s = await asyncio.to_thread(rag.sync)
            return (f"Index updated, sir: {s['added']} added, {s['updated']} refreshed, "
                    f"{s['removed']} removed ({s['unchanged']} unchanged).")

        tools += [ask_documents, summarize_document, reindex_documents]

    # opening/locating a document by DESCRIPTION needs both the index (to resolve
    # the right file) and the organiser (to open it in its default app).
    if rag is not None and organizer is not None:
        @function_tool
        async def open_document(description: str) -> str:
            """Open a document by DESCRIPTION — no exact filename needed. Resolves
            the best-matching indexed file from a phrase like "open my machine
            learning assignment", "open the OS notes I downloaded", "pull up the
            resume I was editing", then opens it in its default app. If several files
            match closely it will read back the top few and ASK which one — the user
            can then reply with a name or a bit more detail and you call this again."""
            def _do():
                ranked = rag.rank_documents(description)
                if not ranked:
                    return f"I couldn't find a document matching “{description}”, sir."
                top = ranked[0]
                # ambiguous when a close runner-up exists → ask instead of guessing.
                rival = ranked[1] if len(ranked) > 1 else None
                unsure = rival is not None and rival["score"] >= 0.75 * max(top["score"], 1e-6)
                if unsure:
                    names = [d["filename"] for d in ranked[:3]]
                    listing = "; ".join(f"{i+1}) {n}" for i, n in enumerate(names))
                    return (f"I found a few that match, sir: {listing}. "
                            "Which one — you can say the name or a bit more detail?")
                organizer.open_paths([top["source"]])
                return f"Opening {top['filename']}, sir."
            try:
                return await asyncio.to_thread(_do)
            except (SandboxError, OSError) as e:
                return f"I couldn't open that, sir: {e}"

        @function_tool
        async def find_document(description: str) -> str:
            """Locate a document by DESCRIPTION and report where it is and when it
            was downloaded / last opened — WITHOUT opening it. Use for "where's my
            X", "when did I download the Y report", "which folder is Z in"."""
            def _do():
                doc = rag.resolve_document(description)
                if not doc:
                    return f"I couldn't find a document matching “{description}”, sir."
                bits = [doc["filename"]]
                if doc.get("location"):
                    bits.append(f"in {doc['location']}")
                if doc.get("created"):
                    bits.append(f"downloaded {doc['created']}")
                if doc.get("modified") and doc.get("modified") != doc.get("created"):
                    bits.append(f"last modified {doc['modified']}")
                opened = organizer.last_opened(doc["source"])
                if opened:
                    bits.append(f"last opened {opened}")
                return ", ".join(bits) + "."
            return await asyncio.to_thread(_do)

        tools += [open_document, find_document]

    if organizer is not None:
        @function_tool
        async def organize_folder(folder: str = "~/Downloads", mode: str = "move") -> str:
            """Reorganise a cluttered folder, sorting files into category subfolders
            (Documents, Images, Code, …) and setting exact duplicates aside in a
            _Duplicates folder (nothing is ever deleted). Use for "organise my
            downloads", "tidy up folder X", "sort out my files". mode is 'move'
            (default) or 'copy'."""
            def _do():
                return organizer.organize(folder, mode=mode)
            try:
                s = await asyncio.to_thread(_do)
            except (FileError, SandboxError) as e:
                return f"I couldn't do that, sir: {e}"
            cats = ", ".join(f"{n} {c}" for c, n in s["by_category"].items()) or "nothing"
            return (f"Organised {s['moved']} file(s) into {cats}"
                    + (f"; set aside {s['duplicates']} duplicate(s)" if s['duplicates'] else "")
                    + ", sir.")

        @function_tool
        async def move_file(source: str, destination: str) -> str:
            """Move a file or folder to another location (within your files). Use for
            "move X to Y". Never overwrites (auto-renames on collision)."""
            try:
                return await asyncio.to_thread(organizer.move, source, destination)
            except (FileError, SandboxError, OSError) as e:
                return f"I couldn't move that, sir: {e}"

        @function_tool
        async def copy_file(source: str, destination: str) -> str:
            """Copy a file or folder to another location. Use for "copy X to Y",
            "duplicate X into Y". Never overwrites."""
            try:
                return await asyncio.to_thread(organizer.copy, source, destination)
            except (FileError, SandboxError, OSError) as e:
                return f"I couldn't copy that, sir: {e}"

        @function_tool
        async def open_file(path: str) -> str:
            """Open a specific file or folder in its default app. Use for "open X"."""
            try:
                return await asyncio.to_thread(organizer.open_paths, [path])
            except (SandboxError, OSError) as e:
                return f"I couldn't open that, sir: {e}"

        @function_tool
        async def open_recent(folder: str = "~/Downloads", count: int = 3) -> str:
            """Open the most recently added/modified files in a folder. Use for "open
            the file I just downloaded", "open my recent downloads", "open the last 3
            files in X"."""
            def _do():
                recents = organizer.recent_files(folder, n=count)
                return organizer.open_paths([str(p) for p in recents])
            try:
                return await asyncio.to_thread(_do)
            except (SandboxError, OSError) as e:
                return f"I couldn't do that, sir: {e}"

        @function_tool
        async def open_folder_files(folder: str) -> str:
            """Open all the files inside a folder. Use for "open all files under X",
            "open everything in my <folder> folder"."""
            def _do():
                files = [str(p) for p in organizer.list_dir(folder) if p.is_file()]
                return organizer.open_paths(files)
            try:
                return await asyncio.to_thread(_do)
            except (SandboxError, OSError) as e:
                return f"I couldn't do that, sir: {e}"

        @function_tool
        async def list_folder(folder: str = "~/Downloads") -> str:
            """List what's in a folder. Use for "what's in my downloads", "list folder X"."""
            def _do():
                items = organizer.list_dir(folder)
                names = [p.name + ("/" if p.is_dir() else "") for p in items]
                return names
            try:
                names = await asyncio.to_thread(_do)
            except (SandboxError, OSError) as e:
                return f"I couldn't list that, sir: {e}"
            if not names:
                return "That folder is empty, sir."
            head = ", ".join(names[:25])
            return f"{len(names)} item(s): {head}" + (" …" if len(names) > 25 else "")

        tools += [organize_folder, move_file, copy_file, open_file,
                  open_recent, open_folder_files, list_folder]

    return tools


# ── Focus assist (close distractions, block reopening, timed regime) ────────
def build_focus_tools(*, focus, memory=None) -> list:
    from jarvis.focus.controller import TECHNIQUES

    @function_tool
    async def start_focus_mode(technique: str = "") -> str:
        """Turn ON focus mode. Immediately closes every open distraction
        (Instagram, YouTube, Netflix, TikTok, Reddit, X, …), starts a timed regime,
        and keeps closing anything distracting the user reopens — until they say to
        end focus mode. Use for "focus mode", "help me focus", "start a pomodoro",
        "block distractions", "deep work session". `technique` is optional: pomodoro
        (default, 25/5), classic (50/10), 52-17, 90-20, or flowtime."""
        # async: the blocking tab-sweep runs off-loop, but the timer/watch tasks are
        # created on THIS event loop (running them via to_thread would lose the loop).
        msg = await focus.start(technique)
        if memory is not None:
            memory.log_activity(f"focus mode on ({focus.status()['technique']})")
        return msg

    @function_tool
    async def stop_focus_mode() -> str:
        """Turn OFF focus mode and lift all blocks. Use ONLY when the user clearly
        asks to stop — "end focus mode", "stop focus mode", "I'm done focusing",
        "turn off focus". Reports how many focus sprints they completed."""
        msg = focus.stop()
        if memory is not None:
            memory.log_activity("focus mode off")
        return msg

    @function_tool
    async def focus_status() -> str:
        """Report the current focus session: whether it's on, the technique, the
        current work/break phase, minutes left, and sprints done. Use for "how's my
        focus session", "how long left", "am I in focus mode"."""
        s = focus.status()
        if not s["active"]:
            return "Focus mode is off, sir."
        return (f"Focus mode is on — {s['technique']}, currently in a {s['phase']} phase "
                f"with about {s['minutes_left_in_phase']} minute(s) left; "
                f"{s['cycles']} sprint(s) done so far.")

    return [start_focus_mode, stop_focus_mode, focus_status]


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


# ── System settings (the Mac's own brightness & sound) ────────────────────
def build_system_tools(*, system: SystemSettings, memory=None) -> list:
    def _log(msg: str) -> None:
        if memory is not None:
            memory.log_activity(msg)

    @function_tool
    async def set_brightness(percent: int) -> str:
        """Set the LAPTOP SCREEN brightness to a percentage, 0–100 ('brightness to
        40%', 'dim the screen to 20', 'brightness max'). This is the Mac's DISPLAY
        brightness, not a video's."""
        def _do():
            frac = system.set_brightness(max(0, min(100, percent)) / 100.0)
            _log(f"set screen brightness to {percent}%")
            return f"screen brightness set to {round(frac * 100)}%"
        try:
            return await asyncio.to_thread(_do)
        except SettingsError as e:
            return f"error: {e}"

    @function_tool
    async def change_brightness(up: bool) -> str:
        """Nudge the LAPTOP SCREEN brightness up or down a step ('brighter',
        'dimmer', 'turn the screen up/down')."""
        def _do():
            frac = system.nudge_brightness(up)
            _log(f"turned screen brightness {'up' if up else 'down'}")
            if frac is None:
                return f"screen a bit {'brighter' if up else 'dimmer'}"
            return f"screen brightness {'up' if up else 'down'} to {round(frac * 100)}%"
        try:
            return await asyncio.to_thread(_do)
        except SettingsError as e:
            return f"error: {e}"

    @function_tool
    async def set_system_volume(percent: int) -> str:
        """Set the LAPTOP's system output volume to a percentage, 0–100 ('volume to
        30', 'set the sound to 50%'). This is the Mac's overall sound, NOT the
        Spotify app's own volume."""
        def _do():
            level = system.set_volume(max(0, min(100, percent)))
            _log(f"set system volume to {level}%")
            return f"system volume set to {level}%"
        try:
            return await asyncio.to_thread(_do)
        except SettingsError as e:
            return f"error: {e}"

    @function_tool
    async def change_system_volume(up: bool) -> str:
        """Nudge the LAPTOP's system output volume up or down ('turn it up/down',
        'louder'/'quieter' when they mean the whole computer, not just music)."""
        def _do():
            level = system.nudge_volume(up)
            _log(f"turned system volume {'up' if up else 'down'}")
            return f"system volume {'up' if up else 'down'} to {level}%"
        try:
            return await asyncio.to_thread(_do)
        except SettingsError as e:
            return f"error: {e}"

    @function_tool
    async def mute_system(muted: bool) -> str:
        """Mute (muted=true) or unmute (muted=false) the LAPTOP's sound entirely."""
        def _do():
            system.set_muted(muted)
            _log(f"{'muted' if muted else 'unmuted'} the system")
            return "muted" if muted else "unmuted"
        try:
            return await asyncio.to_thread(_do)
        except SettingsError as e:
            return f"error: {e}"

    @function_tool
    async def set_dark_mode(on: bool) -> str:
        """Turn the Mac's Dark Mode on (on=true) or off (on=false) — 'dark mode',
        'light mode', 'switch to dark/light appearance'."""
        def _do():
            system.set_dark_mode(on)
            _log(f"set dark mode {'on' if on else 'off'}")
            return f"dark mode {'on' if on else 'off'}"
        try:
            return await asyncio.to_thread(_do)
        except SettingsError as e:
            return f"error: {e}"

    return [set_brightness, change_brightness, set_system_volume,
            change_system_volume, mute_system, set_dark_mode]


# ── Reading mode (an ergonomic reading environment) ───────────────────────
def build_reading_tools(*, reading: ReadingMode, memory=None) -> list:
    @function_tool
    async def start_reading_mode(music: str = "") -> str:
        """Turn on READING MODE: a comfortable reading setup — warm tone + dark,
        low-glare background + softer brightness, with soft instrumental music on
        Spotify in the background. Use for 'reading mode', 'set up for reading',
        'I'm going to read'. Pass `music` only if the user asks for specific
        background music; otherwise leave it empty for the default calm mix."""
        def _do():
            msg = reading.start(music)
            if memory is not None:
                memory.log_activity("started reading mode")
            return msg
        return await asyncio.to_thread(_do)

    @function_tool
    async def stop_reading_mode() -> str:
        """Turn OFF reading mode and restore the previous brightness, volume,
        appearance and Night Shift. Use for 'stop reading mode', 'exit reading
        mode', 'I'm done reading'."""
        def _do():
            msg = reading.stop()
            if memory is not None:
                memory.log_activity("stopped reading mode")
            return msg
        return await asyncio.to_thread(_do)

    return [start_reading_mode, stop_reading_mode]


# ── Pages (Apple's word processor) + clipboard ────────────────────────────
def build_pages_tools(*, pages: PagesController, memory=None) -> list:
    """Full control of the Pages app: create documents, insert/select/remove text,
    headings & titles, tables and images — plus copy/paste and the clipboard."""

    def _log(msg: str) -> None:
        if memory is not None:
            memory.log_activity(msg)

    async def _run(fn, activity: str = "") -> str:
        try:
            res = await asyncio.to_thread(fn)
            if activity:
                _log(activity)
            return res
        except PagesError as e:
            return f"error: {e}"

    @function_tool
    async def pages_new_document() -> str:
        """Open the Pages app and create a NEW blank document. Use for 'open Pages',
        'start a new Pages document', 'new document', 'make a new page'."""
        return await _run(pages.new_document, "opened a new Pages document")

    @function_tool
    async def pages_add_title(text: str) -> str:
        """Insert a large document TITLE at the cursor in Pages, then a new line. Use
        for 'add a title', 'title it X', 'the heading at the top should be X'."""
        return await _run(lambda: pages.add_title(text), f"added a Pages title: {text[:40]}")

    @function_tool
    async def pages_add_heading(text: str, level: int = 1) -> str:
        """Insert a HEADING at the cursor in Pages (level 1 = bigger section heading,
        level 2 = smaller sub-heading), then a new line. Use for 'add a heading X',
        'put a section called X'."""
        return await _run(lambda: pages.add_heading(text, level), f"added a Pages heading: {text[:40]}")

    @function_tool
    async def pages_insert_text(text: str) -> str:
        """Insert body/paragraph TEXT at the cursor in the front Pages document,
        keeping everything already there. Use for 'write X', 'add a paragraph
        saying X', 'type X', 'insert this text'."""
        return await _run(lambda: pages.insert_text(text), "inserted text in Pages")

    @function_tool
    async def pages_append_text(text: str) -> str:
        """Append text to the END of the Pages document (after all existing content).
        Use for 'add X at the end', 'append X'."""
        return await _run(lambda: (pages.append_text(text), "appended the text")[1],
                          "appended text in Pages")

    @function_tool
    async def pages_get_text() -> str:
        """Read back ALL the text currently in the front Pages document. Use for
        'what does the document say', 'read the page back to me'."""
        return await _run(pages.get_text)

    @function_tool
    async def pages_replace_all(text: str) -> str:
        """Replace the ENTIRE contents of the Pages document with new text (overwrite
        everything). Use for 'replace everything with X', 'rewrite the document as X'."""
        return await _run(lambda: (pages.set_text(text), "replaced the document text")[1],
                          "replaced Pages document text")

    @function_tool
    async def pages_clear() -> str:
        """Erase ALL the text in the front Pages document, leaving it blank. Use for
        'clear the document', 'delete everything', 'wipe the page'."""
        return await _run(pages.clear, "cleared the Pages document")

    @function_tool
    async def pages_save(name: str = "") -> str:
        """Save the front Pages document. For a brand-new document pass a `name` (a
        filename, saved to the user's Documents folder) so it saves without a dialog;
        for an already-saved document leave `name` empty. Use for 'save the document',
        'save it as deepfakes'."""
        def _do():
            path = ""
            if name.strip():
                from pathlib import Path
                path = str(Path.home() / "Documents" / name.strip())
            return pages.save(path)
        return await _run(lambda: _do(), "saved a Pages document")

    @function_tool
    async def pages_insert_table() -> str:
        """Insert a TABLE at the cursor in the Pages document. Use for 'add a table',
        'insert a table here'."""
        return await _run(pages.insert_table, "inserted a table in Pages")

    @function_tool
    async def pages_insert_image(path: str) -> str:
        """Insert an IMAGE file (by its file path) at the cursor in the Pages
        document. Use for 'add the image at ~/Pictures/x.png', 'insert this photo'."""
        return await _run(lambda: pages.insert_image(path), "inserted an image in Pages")

    @function_tool
    async def pages_select_all() -> str:
        """Select ALL content in the front Pages document (⌘A). Use before copying or
        deleting everything."""
        return await _run(pages.select_all)

    @function_tool
    async def pages_copy() -> str:
        """Copy the current selection in Pages to the clipboard (⌘C)."""
        return await _run(pages.copy_selection, "copied from Pages")

    @function_tool
    async def pages_cut() -> str:
        """Cut the current selection in Pages to the clipboard (⌘X)."""
        return await _run(pages.cut_selection, "cut from Pages")

    @function_tool
    async def pages_paste() -> str:
        """Paste the clipboard into the Pages document at the cursor (⌘V)."""
        return await _run(pages.paste, "pasted into Pages")

    @function_tool
    async def pages_delete_selection() -> str:
        """Delete the currently selected content in Pages (does NOT copy it). Use for
        'remove this', 'delete the selected part'."""
        return await _run(pages.delete_selection, "deleted a selection in Pages")

    @function_tool
    async def copy_to_clipboard(text: str) -> str:
        """Put text on the system clipboard so it can be pasted anywhere. Use for
        'copy X', 'copy this to the clipboard'."""
        def _do():
            pages.set_clipboard(text)
            return "copied to the clipboard"
        return await _run(_do, "set the clipboard")

    @function_tool
    async def read_clipboard() -> str:
        """Read the current text contents of the system clipboard. Use for 'what's on
        my clipboard', 'what did I copy'."""
        def _do():
            txt = pages.get_clipboard()
            return txt or "the clipboard is empty"
        return await _run(_do)

    return [
        pages_new_document, pages_add_title, pages_add_heading, pages_insert_text,
        pages_append_text, pages_get_text, pages_replace_all, pages_clear,
        pages_save, pages_insert_table, pages_insert_image, pages_select_all, pages_copy,
        pages_cut, pages_paste, pages_delete_selection,
        copy_to_clipboard, read_clipboard,
    ]


# ── Safe web images (search + guardrailed download, optional Pages insert) ──
def build_image_tools(*, images: SafeImageFetcher, pages: PagesController | None = None,
                      vetter=None, memory=None) -> list:
    """Fetch images from trusted web sources under strict safety guardrails, and
    (if Pages is available) drop them straight into the document. When a `vetter` is
    supplied, each candidate is looked at (vision) and judged for fit (reasoning), so
    JARVIS keeps searching until an image genuinely matches the request."""

    def _log(msg: str) -> None:
        if memory is not None:
            memory.log_activity(msg)

    def _accept_for(intent: str):
        return vetter.accept_for(intent) if vetter is not None else None

    @function_tool
    async def download_web_image(query: str) -> str:
        """Find and download a real image from the web for a topic (e.g. 'deepfake',
        'GAN architecture diagram', 'neural network flowchart') and save it locally.
        Downloads ONLY from trusted, freely-licensed sources (Wikimedia Commons /
        Wikipedia) with safety checks — it will refuse anything unsafe. Returns the
        saved file path. Use when the user wants an image/photo/diagram/flowchart
        pulled from the web."""
        def _do():
            path, res = images.fetch_one(query, accept=_accept_for(query))
            _log(f"downloaded a safe web image for: {query}")
            src = f" (source: {res.source_page})" if res.source_page else ""
            return f"saved image to {path}{src}"
        try:
            return await asyncio.to_thread(_do)
        except ImageError as e:
            return f"error: {e}"

    tools = [download_web_image]

    if pages is not None:
        @function_tool
        async def pages_insert_web_image(query: str) -> str:
            """Find a relevant image or flowchart/diagram on the web and insert it
            into the current Pages document — in ONE step. Use for 'add an image of
            deepfakes', 'insert a flowchart of how deepfakes are made', 'put a diagram
            here'. Downloads only from trusted sources with safety checks, then places
            it at the cursor. `query` describes the picture wanted."""
            def _do():
                path, res = images.fetch_one(query, accept=_accept_for(query))
                pages.insert_image(path)
                _log(f"inserted a web image into Pages: {query}")
                credit = res.title or query
                return f"inserted an image of {credit} into the document"
            try:
                return await asyncio.to_thread(_do)
            except (ImageError, PagesError) as e:
                return f"error: {e}"

        tools.append(pages_insert_web_image)

    return tools


# ── Editing the OPEN Pages document (targeted formatting changes) ───────────
def build_editor_tools(*, editor, memory=None) -> list:
    """Make targeted formatting changes to the Pages document the user is working on
    right now — change a heading's size, recolour text, reformat the whole thing."""
    from jarvis.tools.pages_editor import PagesEditError

    def _log(msg: str) -> None:
        if memory is not None:
            memory.log_activity(msg)

    @function_tool
    async def read_open_document() -> str:
        """Read the Pages document the user currently has open, returning its NAME and
        every paragraph with a 1-based number. Use this FIRST when the user asks to
        change something in 'this'/'the' document ('make the heading bigger', 'reformat
        this', 'change the second paragraph') so you can see which paragraph number is
        the title/heading/body, then call format_pages_paragraph with that number."""
        def _do():
            name = editor.active_document()
            if not name:
                return "there's no Pages document open, sir"
            paras = editor.read_paragraphs()
            listing = "\n".join(f"{i}. {p}" for i, p in enumerate(paras, 1) if p.strip())
            return f"Open document “{name}” has {len(paras)} paragraph(s):\n{listing}"
        try:
            return await asyncio.to_thread(_do)
        except PagesEditError as e:
            return f"error: {e}"

    @function_tool
    async def format_pages_paragraph(
        paragraph: int, size: int = 0, font: str = "", bold: bool = False,
        italic: bool = False, color: str = "",
    ) -> str:
        """Change the formatting of ONE paragraph (1-based) in the open Pages document.
        First call read_open_document to find the right paragraph number. `size` in
        points (0 = leave as is), `font` a family name (e.g. 'Georgia'), `bold`/`italic`
        to embolden/italicise, `color` a name or #hex ('dark blue', '#1a3c8c'). Use for
        'make the heading 32', 'increase the title size to 40', 'make paragraph 3 bold',
        'colour the heading dark blue'."""
        def _do():
            msg = editor.format_paragraph(
                paragraph, size=size or None, font=font or None,
                bold=bold, italic=italic, color=color or None,
            )
            _log(f"formatted Pages paragraph {paragraph}")
            return msg
        try:
            return await asyncio.to_thread(_do)
        except PagesEditError as e:
            return f"error: {e}"

    @function_tool
    async def format_all_pages_text(size: int = 0, font: str = "", color: str = "") -> str:
        """Apply a font/size/colour to the ENTIRE open Pages document at once. Use for
        'make everything 12 point', 'change the whole document to Georgia', 'set all
        the text to black'."""
        def _do():
            msg = editor.format_all(size=size or None, font=font or None, color=color or None)
            _log("formatted the whole Pages document")
            return msg
        try:
            return await asyncio.to_thread(_do)
        except PagesEditError as e:
            return f"error: {e}"

    @function_tool
    async def reformat_pages_document(
        font: str = "Helvetica Neue", title_size: int = 26,
        heading_size: int = 17, body_size: int = 12,
    ) -> str:
        """Tidy the OPEN Pages document into a clean, consistent look — the first line
        becomes a styled title, short lines become headings, and the rest becomes evenly
        sized body text. Use for 'reformat this', 'clean up the formatting', 'make this
        look consistent'. Optionally tune the font and the title/heading/body sizes."""
        def _do():
            msg = editor.reformat(font=font, title_size=title_size,
                                  heading_size=heading_size, body_size=body_size)
            _log("reformatted the Pages document")
            return msg
        try:
            return await asyncio.to_thread(_do)
        except PagesEditError as e:
            return f"error: {e}"

    return [read_open_document, format_pages_paragraph,
            format_all_pages_text, reformat_pages_document]


# ── Authoring a whole styled Pages document (real Title/Heading styles) ─────
def build_authoring_tools(*, author: PagesAuthor, images: SafeImageFetcher | None = None,
                          vetter=None, memory=None) -> list:
    """Create a COMPLETE, properly-formatted Pages document in one shot, using Pages'
    real built-in paragraph styles (Title / Subtitle / Heading / Body). This is the
    right tool for 'make me a document/report/essay/notes about X' — far better
    formatting than inserting text piece by piece."""
    import json
    from pathlib import Path

    @function_tool
    async def create_pages_document(
        title: str, outline_json: str, subtitle: str = "", filename: str = "",
    ) -> str:
        """Create a NEW, well-formatted Pages document with REAL Pages styles (Title,
        Subtitle, Heading, Body) — use this for 'make/write me a document/report/essay/
        notes/article about X', especially when it needs sections, images, or a table.
        Much better looking than adding text piece by piece.

        `title` is the document title. `subtitle` is optional. `outline_json` is a JSON
        array of content blocks, in order, each an object:
          {"type":"heading","text":"Section name","level":1}   (level 1 or 2)
          {"type":"body","text":"A paragraph. Use \\n\\n between paragraphs."}
          {"type":"image","query":"deepfake"}      → safely downloads a web image
          {"type":"image","path":"/abs/file.png"}  → an image already on disk
          {"type":"table","rows":[["Header A","Header B"],["a1","b1"]]}
        Compose the full body text yourself (real sentences/paragraphs), then pass the
        blocks. Example outline_json:
          [{"type":"heading","text":"Overview","level":1},
           {"type":"body","text":"Deepfakes are synthetic media..."},
           {"type":"image","query":"deepfake example"}]
        `filename` optionally names the saved file (defaults to the title)."""
        def _do():
            try:
                raw = json.loads(outline_json) if outline_json.strip() else []
            except (json.JSONDecodeError, ValueError) as e:
                return f"error: the outline wasn't valid JSON ({e})"
            if not isinstance(raw, list):
                return "error: the outline must be a JSON array of blocks"

            blocks: list[dict] = [{"type": "title", "text": title}]
            if subtitle.strip():
                blocks.append({"type": "subtitle", "text": subtitle})

            skipped = 0
            for b in raw:
                if not isinstance(b, dict):
                    continue
                if b.get("type") == "image" and b.get("query") and images is not None:
                    try:
                        intent = str(b["query"])
                        accept = vetter.accept_for(intent) if vetter is not None else None
                        path, _res = images.fetch_one(intent, accept=accept)
                        blocks.append({"type": "image", "path": path,
                                       "caption": b.get("caption", "")})
                    except ImageError:
                        skipped += 1  # a missing image never fails the whole document
                    continue
                blocks.append(b)

            out = str(Path.home() / "Documents" / (filename.strip() or title or "Document"))
            docx_path = author.create_document(blocks, out)
            author.open_in_pages(docx_path)
            if memory is not None:
                memory.log_activity(f"authored a Pages document: {title}")
            note = f" ({skipped} image(s) couldn't be fetched safely)" if skipped else ""
            return f"created and opened “{title}” in Pages with proper styles{note}"

        try:
            return await asyncio.to_thread(_do)
        except (AuthorError, OSError) as e:
            return f"error: {e}"

    return [create_pages_document]
