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
                    cite = "screen vision · gpt-4o"
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
