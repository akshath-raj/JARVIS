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
                "You control Spotify. Always call the tool that matches the "
                "request; never reply without acting. Map requests to tools:\n"
                "- play a song/artist/genre/mood -> play_song (query = the user's "
                "OWN words; strip 'play','some','a','any','random' and a trailing "
                "'on spotify'; pass unfamiliar titles through, never substitute; "
                "loop=true only if they say repeat/loop/on repeat)\n"
                "- pause/stop -> pause_music;  resume/continue/unpause/play again "
                "-> resume_music;  next/skip -> next_song;  volume -> "
                "set_music_volume;  turn repeat on/off -> set_loop\n"
                "- what's playing -> whats_playing\n"
                "- list/how many playlists or songs -> list_playlists;  play a "
                "named playlist -> play_playlist;  add the current song to a "
                "playlist -> add_current_song_to_playlist\n"
                "- only if the user names nothing to play (just 'play a song' / "
                "'play something') ask 'What would you like to hear, sir?'\n"
                "- non-music request -> back_to_coordinator\n" + VOICE_STYLE
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
    async def list_playlists(self, context: RunContext[JarvisContext]) -> str:
        """List the user's playlists with how many songs each has.

        Use for questions like "what playlists do I have" or "how many songs in
        each playlist".
        """
        try:
            pls = await asyncio.to_thread(context.userdata.spotify.list_playlists)
        except SpotifyError as e:
            return f"error: {e}"
        if not pls:
            return "you have no playlists"
        have_counts = any(c is not None for _, c in pls)
        if have_counts:
            listing = ", ".join(
                f"{name} ({c} songs)" if c is not None else name for name, c in pls
            )
        else:
            listing = ", ".join(name for name, _ in pls)
        note = "" if have_counts else " (Spotify isn't sharing song counts for this account)"
        return f"{len(pls)} playlists: {listing}{note}"

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
