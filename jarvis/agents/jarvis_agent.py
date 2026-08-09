"""JarvisAgent — a single, reliable agent that answers questions AND controls
Spotify.

We consolidated the earlier multi-agent mesh into one agent: with local models,
agent-to-agent handoffs proved unreliable (the model emitted tool calls as plain
text after a handoff instead of executing them). A single agent with all tools
executes reliably. All Spotify calls run in a worker thread so playback never
blocks the voice loop or steals focus.
"""
from __future__ import annotations

import asyncio

from livekit.agents import ChatContext, RunContext, function_tool

from jarvis.agents.base import BaseJarvisAgent
from jarvis.context import JarvisContext
from jarvis.tools.spotify import SpotifyError

# Keep this LEAN and behavioral. Enumerating tools/JSON here makes local models
# narrate the tool call as text instead of actually calling it — the tool
# docstrings already describe usage.
INSTRUCTIONS = (
    "You are JARVIS, a concise, witty British butler. For general questions, "
    "answer directly from your own knowledge in one short spoken sentence — you "
    "do not need any tool or the internet for facts. For music, ALWAYS call the "
    "matching Spotify tool to actually perform the action; never merely say you "
    "did it. Pass exactly what the user asked for as the query and don't "
    "substitute. If they ask for music but name no song, ask what to play. Only "
    "loop when they say repeat. Address the user as 'sir' occasionally. /no_think"
)


class JarvisAgent(BaseJarvisAgent):
    def __init__(self, chat_ctx: ChatContext | None = None) -> None:
        super().__init__(instructions=INSTRUCTIONS, chat_ctx=chat_ctx)

    async def on_enter(self) -> None:
        if not self.session.userdata.greeted:
            self.session.userdata.greeted = True
            await self.session.generate_reply(
                instructions="Introduce yourself in one short sentence as JARVIS, "
                "at the user's service. Do not ask a question."
            )

    # ---- Spotify tools ---------------------------------------------------
    @function_tool()
    async def play_song(self, context: RunContext[JarvisContext], query: str, loop: bool = False) -> str:
        """Search for a song and play it. Set loop=true to repeat it.

        Args:
            query: What to play — a song title and/or artist, genre, or mood.
            loop: If true, put it on repeat.
        """
        ud = context.userdata
        try:
            result = await asyncio.to_thread(ud.spotify.play_query, query, ud.search_mode, loop)
        except SpotifyError as e:
            return f"error: {e}"
        return f"now playing {result.label}" + (" on repeat" if loop else "")

    @function_tool()
    async def play_playlist(self, context: RunContext[JarvisContext], name: str, loop: bool = False) -> str:
        """Play one of the user's own playlists by name.

        Args:
            name: The playlist name, e.g. "anime" or "Focus".
            loop: If true, loop the playlist.
        """
        try:
            pl = await asyncio.to_thread(context.userdata.spotify.play_playlist, name, loop)
        except SpotifyError as e:
            return f"error: {e}"
        return f"playing your {pl} playlist"

    @function_tool()
    async def list_playlists(self, context: RunContext[JarvisContext]) -> str:
        """List the user's playlists and how many songs each has."""
        try:
            pls = await asyncio.to_thread(context.userdata.spotify.list_playlists)
        except SpotifyError as e:
            return f"error: {e}"
        if not pls:
            return "you have no playlists"
        have = any(c is not None for _, c in pls)
        if have:
            listing = ", ".join(f"{n} ({c} songs)" if c is not None else n for n, c in pls)
            note = ""
        else:
            listing = ", ".join(n for n, _ in pls)
            note = " (Spotify isn't sharing song counts for this account)"
        return f"{len(pls)} playlists: {listing}{note}"

    @function_tool()
    async def add_current_song_to_playlist(self, context: RunContext[JarvisContext], playlist: str) -> str:
        """Add the currently playing song to one of the user's playlists.

        Args:
            playlist: Target playlist name.
        """
        try:
            track, pl = await asyncio.to_thread(context.userdata.spotify.add_current_to_playlist, playlist)
        except SpotifyError as e:
            return f"error: {e}"
        return f"added {track} to {pl}"

    @function_tool()
    async def set_loop(self, context: RunContext[JarvisContext], enabled: bool) -> str:
        """Turn repeat/loop on or off."""
        try:
            await asyncio.to_thread(context.userdata.spotify.set_repeat, enabled)
        except SpotifyError as e:
            return f"error: {e}"
        return "looping on" if enabled else "looping off"

    @function_tool()
    async def pause_music(self, context: RunContext[JarvisContext]) -> str:
        """Pause Spotify playback."""
        try:
            await asyncio.to_thread(context.userdata.spotify.pause)
        except SpotifyError as e:
            return f"error: {e}"
        return "paused"

    @function_tool()
    async def resume_music(self, context: RunContext[JarvisContext]) -> str:
        """Resume Spotify playback."""
        try:
            await asyncio.to_thread(context.userdata.spotify.resume)
        except SpotifyError as e:
            return f"error: {e}"
        return "resumed"

    @function_tool()
    async def next_song(self, context: RunContext[JarvisContext]) -> str:
        """Skip to the next track."""
        try:
            await asyncio.to_thread(context.userdata.spotify.next_track)
        except SpotifyError as e:
            return f"error: {e}"
        return "skipped"

    @function_tool()
    async def set_music_volume(self, context: RunContext[JarvisContext], level: int) -> str:
        """Set Spotify volume (0-100)."""
        try:
            await asyncio.to_thread(context.userdata.spotify.set_volume, level)
        except SpotifyError as e:
            return f"error: {e}"
        return f"volume set to {level}"

    @function_tool()
    async def whats_playing(self, context: RunContext[JarvisContext]) -> str:
        """Report the currently playing track."""
        try:
            return await asyncio.to_thread(context.userdata.spotify.current_track)
        except SpotifyError as e:
            return f"error: {e}"
