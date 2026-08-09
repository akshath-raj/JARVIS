"""ChatAgent — general knowledge and conversation. No external tools."""
from __future__ import annotations

from livekit.agents import Agent, ChatContext, RunContext, function_tool

from jarvis.agents.base import VOICE_STYLE
from jarvis.context import JarvisContext


class ChatAgent(Agent):
    def __init__(self, chat_ctx: ChatContext | None = None) -> None:
        super().__init__(
            instructions=(
                "You handle general questions, facts, and casual conversation, "
                "answering directly from your own knowledge. If the user asks for "
                "music or Spotify control, call transfer_to_music instead of "
                "answering. " + VOICE_STYLE
            ),
            chat_ctx=chat_ctx,
        )

    async def on_enter(self) -> None:
        # Respond to whatever the user just asked (already in chat history).
        await self.session.generate_reply()

    @function_tool()
    async def transfer_to_music(self, context: RunContext[JarvisContext]):
        """Hand off to the music specialist for playing or controlling Spotify."""
        from jarvis.agents.music import MusicAgent

        return MusicAgent(chat_ctx=self.chat_ctx.copy(exclude_instructions=True))
