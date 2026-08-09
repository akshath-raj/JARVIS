"""RouterAgent — the JARVIS coordinator.

Does no task work itself. It greets the user, figures out intent, and hands off
to the right specialist. This is the session's entry point.
"""
from __future__ import annotations

from livekit.agents import ChatContext, RunContext, function_tool

from jarvis.agents.base import VOICE_STYLE, BaseJarvisAgent
from jarvis.context import JarvisContext


class RouterAgent(BaseJarvisAgent):
    def __init__(self, chat_ctx: ChatContext | None = None) -> None:
        super().__init__(
            instructions=(
                "You are the coordinator. You never answer questions or perform "
                "tasks yourself — you only route. Decide what the user wants and "
                "call the matching tool:\n"
                "- Music or Spotify (play a song, pause, skip, volume, what's "
                "playing) -> transfer_to_music.\n"
                "- Calendar or scheduling (events, meetings, agenda, reminders) "
                "-> transfer_to_calendar.\n"
                "- Files or folders (find, open, list files) -> transfer_to_files.\n"
                "- General questions, facts, or chit-chat -> transfer_to_chat.\n"
                "Do not describe the routing; just call the tool. " + VOICE_STYLE
            ),
            chat_ctx=chat_ctx,
        )

    async def on_enter(self) -> None:
        ud = self.session.userdata
        if not ud.greeted:
            ud.greeted = True
            await self.session.generate_reply(
                instructions="Greet the user briefly as JARVIS and ask how you can help."
            )
        else:
            # Returned from a specialist: route the user's latest request.
            await self.session.generate_reply()

    @function_tool()
    async def transfer_to_music(self, context: RunContext[JarvisContext]):
        """Route to the music specialist for playing or controlling Spotify."""
        from jarvis.agents.music import MusicAgent

        return MusicAgent(
            chat_ctx=self.chat_ctx.copy(exclude_instructions=True),
            llm=context.userdata.local_llm,
        )

    @function_tool()
    async def transfer_to_calendar(self, context: RunContext[JarvisContext]):
        """Route to the calendar specialist for events, meetings, and scheduling."""
        from jarvis.agents.calendar import CalendarAgent

        return CalendarAgent(
            chat_ctx=self.chat_ctx.copy(exclude_instructions=True),
            llm=context.userdata.local_llm,
        )

    @function_tool()
    async def transfer_to_files(self, context: RunContext[JarvisContext]):
        """Route to the files specialist for finding, listing, or opening files."""
        from jarvis.agents.files import FileAgent

        return FileAgent(
            chat_ctx=self.chat_ctx.copy(exclude_instructions=True),
            llm=context.userdata.local_llm,
        )

    @function_tool()
    async def transfer_to_chat(self, context: RunContext[JarvisContext]):
        """Route to the conversation specialist for general questions and chit-chat."""
        from jarvis.agents.chat import ChatAgent

        return ChatAgent(chat_ctx=self.chat_ctx.copy(exclude_instructions=True))
