"""MusicAgent — Spotify specialist. Owns all playback tools.

Reads the shared SpotifyController from session.userdata (set up once at session
start) rather than re-creating it on every handoff.
"""
from __future__ import annotations

from livekit.agents import ChatContext, RunContext, function_tool

from jarvis.agents.base import VOICE_STYLE, BaseJarvisAgent
from jarvis.context import JarvisContext
from jarvis.tools.spotify import SpotifyError


class MusicAgent(BaseJarvisAgent):
    def __init__(self, chat_ctx: ChatContext | None = None) -> None:
        super().__init__(
            instructions=(
                "You are the music specialist. Use your tools to control Spotify "
                "rather than describing what you would do. After an action, confirm "
                "briefly (e.g. \"Now playing Bohemian Rhapsody by Queen\"). If a tool "
                "returns an error, apologise briefly and say what went wrong. If the "
                "user switches to a non-music topic, call back_to_coordinator. "
                + VOICE_STYLE
            ),
            chat_ctx=chat_ctx,
        )

    async def on_enter(self) -> None:
        await self.session.generate_reply()

    # ---- Spotify tools ---------------------------------------------------
    @function_tool()
    async def play_song(self, context: RunContext[JarvisContext], query: str) -> str:
        """Search and immediately play the best matching song.

        Args:
            query: Natural-language song request, e.g. "Bohemian Rhapsody by Queen".
        """
        ud = context.userdata
        try:
            result = ud.spotify.play_query(query, mode=ud.search_mode)
        except SpotifyError as e:
            return f"error: {e}"
        return f"now playing {result.label}"

    @function_tool()
    async def pause_music(self, context: RunContext[JarvisContext]) -> str:
        """Pause Spotify playback."""
        try:
            context.userdata.spotify.pause()
        except SpotifyError as e:
            return f"error: {e}"
        return "paused"

    @function_tool()
    async def resume_music(self, context: RunContext[JarvisContext]) -> str:
        """Resume Spotify playback."""
        try:
            context.userdata.spotify.resume()
        except SpotifyError as e:
            return f"error: {e}"
        return "resumed"

    @function_tool()
    async def next_song(self, context: RunContext[JarvisContext]) -> str:
        """Skip to the next track."""
        try:
            context.userdata.spotify.next_track()
        except SpotifyError as e:
            return f"error: {e}"
        return "skipped"

    @function_tool()
    async def set_music_volume(self, context: RunContext[JarvisContext], level: int) -> str:
        """Set Spotify volume (0-100)."""
        try:
            context.userdata.spotify.set_volume(level)
        except SpotifyError as e:
            return f"error: {e}"
        return f"volume set to {level}"

    @function_tool()
    async def whats_playing(self, context: RunContext[JarvisContext]) -> str:
        """Report the currently playing track."""
        try:
            return context.userdata.spotify.current_track()
        except SpotifyError as e:
            return f"error: {e}"

    @function_tool()
    async def back_to_coordinator(self, context: RunContext[JarvisContext]):
        """Hand control back to the coordinator for non-music requests."""
        from jarvis.agents.router import RouterAgent

        return RouterAgent(chat_ctx=self.chat_ctx.copy(exclude_instructions=True))
