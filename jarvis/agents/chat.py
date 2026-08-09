"""ChatAgent — general knowledge and conversation. No external tools."""
from __future__ import annotations

from livekit.agents import ChatContext, RunContext, function_tool

from jarvis.agents.base import VOICE_STYLE, BaseJarvisAgent
from jarvis.context import JarvisContext


class ChatAgent(BaseJarvisAgent):
    def __init__(self, chat_ctx: ChatContext | None = None) -> None:
        super().__init__(
            instructions=(
                "You handle general questions, facts, and casual conversation, "
                "answering directly from your own knowledge. If the user asks for "
                "music, calendar, or files, call back_to_coordinator so it can be "
                "routed. " + VOICE_STYLE
            ),
            chat_ctx=chat_ctx,
        )

    async def on_enter(self) -> None:
        await self.session.generate_reply()

    @function_tool()
    async def back_to_coordinator(self, context: RunContext[JarvisContext]):
        """Hand control back to the coordinator to route a non-chat request."""
        from jarvis.agents.router import RouterAgent

        return RouterAgent(chat_ctx=self.chat_ctx.copy(exclude_instructions=True))
