"""FileAgent — finds and opens local files via Spotlight."""
from __future__ import annotations

from livekit.agents import ChatContext, RunContext, function_tool

from jarvis.agents.base import VOICE_STYLE, BaseJarvisAgent
from jarvis.context import JarvisContext
from jarvis.tools import system


class FileAgent(BaseJarvisAgent):
    def __init__(self, chat_ctx: ChatContext | None = None, llm=None) -> None:
        super().__init__(
            llm=llm,
            instructions=(
                "You are the files specialist. Use your tools to find and open files "
                "on this Mac. When you find files, read out just the file names, not "
                "full paths, unless asked. If the user switches to a non-file topic, "
                "call back_to_coordinator. " + VOICE_STYLE
            ),
            chat_ctx=chat_ctx,
        )

    async def on_enter(self) -> None:
        await self.session.generate_reply()

    @function_tool()
    async def find_files(self, context: RunContext[JarvisContext], name: str) -> str:
        """Find files whose name matches a query, via Spotlight.

        Args:
            name: A word or phrase from the file name, e.g. "budget 2026".
        """
        try:
            hits = system.spotlight_find(name)
        except Exception as e:
            return f"error: {e}"
        if not hits:
            return f"no files matching '{name}'"
        return "; ".join(hits)

    @function_tool()
    async def open_file(self, context: RunContext[JarvisContext], path: str) -> str:
        """Open a file or folder by its full path.

        Args:
            path: Absolute path from a previous find_files result.
        """
        try:
            system.open_path(path)
        except Exception as e:
            return f"error: {e}"
        return f"opened {path}"

    @function_tool()
    async def back_to_coordinator(self, context: RunContext[JarvisContext]):
        """Hand control back to the coordinator for non-file requests."""
        from jarvis.agents.router import RouterAgent

        return RouterAgent(chat_ctx=self.chat_ctx.copy(exclude_instructions=True))
