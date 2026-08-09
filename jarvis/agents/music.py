"""MusicAgent — Spotify specialist. Owns all playback tools.

All Spotify calls run in a background thread (asyncio.to_thread) so launching the
app, searching, and starting playback never block the voice pipeline or steal
focus from whatever the user is doing.
"""
from __future__ import annotations

import asyncio

from livekit.agents import ChatContext, RunContext, function_tool

from jarvis.agents.base import VOICE_STYLE, BaseJarvisAgent
from jarvis.context import JarvisContext
from jarvis.tools.spotify import SpotifyError


class MusicAgent(BaseJarvisAgent):
    def __init__(self, chat_ctx: ChatContext | None = None, llm=None) -> None:
        super().__init__(
            llm=llm,
            instructions=(
                "You are the music specialist controlling Spotify in the background. "
                "Use your tools rather than describing what you would do. After an "
                "action, confirm in one short sentence (e.g. \"Now playing Bohemian "
                "Rhapsody by Queen\"). Set loop=true when the user asks to repeat or "
                "loop a song. If a tool returns an error, apologise briefly and say "
                "what went wrong. If the user switches to a non-music topic, call "
                "back_to_coordinator. " + VOICE_STYLE
            ),
            chat_ctx=chat_ctx,
        )

    async def on_enter(self) -> None:
        await self.session.generate_reply()

    # ---- Playback --------------------------------------------------------
    @function_tool()
    async def play_song(
        self, context: RunContext[JarvisContext], query: str, loop: bool = False
    ) -> str:
        """Search for a song and play it. Set loop=true to repeat it.

        Args:
            query: Song request, e.g. "Bohemian Rhapsody by Queen".
            loop: If true, put the song on repeat.
        """
        ud = context.userdata
        try:
            result = await asyncio.to_thread(
                ud.spotify.play_query, query, ud.search_mode, loop
            )
        except SpotifyError as e:
            return f"error: {e}"
        return f"now playing {result.label}" + (" on repeat" if loop else "")

    @function_tool()
    async def play_playlist(
        self, context: RunContext[JarvisContext], name: str, loop: bool = False
    ) -> str:
        """Play one of the user's own playlists by name. Set loop=true to repeat it.

        Args:
            name: The playlist name, e.g. "Focus" or "Workout".
            loop: If true, loop the playlist.
        """
        try:
            pl = await asyncio.to_thread(context.userdata.spotify.play_playlist, name, loop)
        except SpotifyError as e:
            return f"error: {e}"
        return f"playing your {pl} playlist"

    @function_tool()
    async def add_current_song_to_playlist(
        self, context: RunContext[JarvisContext], playlist: str
    ) -> str:
        """Add the currently playing song to one of the user's playlists.

        Args:
            playlist: Target playlist name, e.g. "Favourites".
        """
        try:
            track, pl = await asyncio.to_thread(
                context.userdata.spotify.add_current_to_playlist, playlist
            )
        except SpotifyError as e:
            return f"error: {e}"
        return f"added {track} to {pl}"

    @function_tool()
    async def set_loop(self, context: RunContext[JarvisContext], enabled: bool) -> str:
        """Turn repeat/loop on or off for the current playback."""
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

    @function_tool()
    async def back_to_coordinator(self, context: RunContext[JarvisContext]):
        """Hand control back to the coordinator for non-music requests."""
        from jarvis.agents.router import RouterAgent

        return RouterAgent(chat_ctx=self.chat_ctx.copy(exclude_instructions=True))
