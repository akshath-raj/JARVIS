"""CalendarAgent — reads the local macOS Calendar."""
from __future__ import annotations

from livekit.agents import ChatContext, RunContext, function_tool

from jarvis.agents.base import VOICE_STYLE, BaseJarvisAgent
from jarvis.context import JarvisContext
from jarvis.tools import system


class CalendarAgent(BaseJarvisAgent):
    def __init__(self, chat_ctx: ChatContext | None = None) -> None:
        super().__init__(
            instructions=(
                "You are the calendar specialist. Use your tools to read the user's "
                "schedule and answer succinctly. If the user switches to a non-"
                "calendar topic, call back_to_coordinator. " + VOICE_STYLE
            ),
            chat_ctx=chat_ctx,
        )

    async def on_enter(self) -> None:
        await self.session.generate_reply()

    @function_tool()
    async def todays_events(self, context: RunContext[JarvisContext]) -> str:
        """List the user's calendar events for today."""
        try:
            events = system.calendar_today()
        except Exception as e:
            return f"error: {e}"
        if not events:
            return "no events today"
        return "; ".join(events)

    @function_tool()
    async def back_to_coordinator(self, context: RunContext[JarvisContext]):
        """Hand control back to the coordinator for non-calendar requests."""
        from jarvis.agents.router import RouterAgent

        return RouterAgent(chat_ctx=self.chat_ctx.copy(exclude_instructions=True))
