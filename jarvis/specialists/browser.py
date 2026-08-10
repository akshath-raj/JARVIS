"""BrowserSpecialist — the "chrome / web" focused sub-agent.

Owns everything about the browser: opening sites, playing YouTube videos, short
-form feeds (Shorts/Reels), and looking things up live on the web via Tavily. The
supervisor delegates a whole browser/web request here with one `browser(request)`
call; this specialist runs its own scoped tool-calling loop to carry it out.

Deliberately separate from the music pipeline — different module, different
controller, opposite focus policy (browser comes to the front).
"""
from __future__ import annotations

from openai import OpenAI

from jarvis.specialists.base import Specialist, ToolSpec, no_params
from jarvis.tools.browser import BrowserController
from jarvis.tools.web import TavilyClient

# Lean + behavioral (same lesson as the music prompt): map each kind of request to
# its tool and CALL it; don't narrate. Do not enumerate JSON here.
INSTRUCTIONS = (
    "You control the user's Chrome browser and can look things up on the web. "
    "For any request, CALL the single matching tool — never just say you did it. "
    "To open an app or website use open_site with the site's name. To play a "
    "specific YouTube video or search-and-play, use play_youtube with the user's "
    "words. For the newest video from a named channel, use latest_channel_video. "
    "For a scrolling short-video feed use open_reels (Instagram) or open_shorts "
    "(YouTube). For questions about current events, news, prices, or live facts, "
    "use web_lookup. Pick exactly the tools needed and pass the user's own words."
)

_STR = lambda name, desc: {  # noqa: E731 - tiny schema helper
    "type": "object",
    "properties": {name: {"type": "string", "description": desc}},
    "required": [name],
}


class BrowserSpecialist(Specialist):
    name = "browser"
    instructions = INSTRUCTIONS

    def __init__(self, client: OpenAI, model: str, browser: BrowserController, tavily: TavilyClient):
        self._browser = browser
        self._tavily = tavily
        tools = [
            ToolSpec(
                "open_site",
                "Open a website or web app in Chrome by name (e.g. youtube, "
                "instagram, gmail, github, netflix) or a URL. Unknown names are "
                "searched on Google.",
                _STR("name", "The site name or URL to open, e.g. 'instagram'."),
                lambda name: self._browser.open_site(name),
            ),
            ToolSpec(
                "play_youtube",
                "Search YouTube for a song/video and open the top result, which "
                "starts playing. Use for 'play <x> on youtube' or 'play the <x> video'.",
                _STR("query", "What to search and play, e.g. 'lofi hip hop mix'."),
                lambda query: self._browser.play_youtube(query),
            ),
            ToolSpec(
                "latest_channel_video",
                "Open the newest upload from a YouTube channel. Use for 'open a new "
                "<creator> video' or 'latest <creator> video'.",
                _STR("channel", "The channel/creator name, e.g. 'MrBeast'."),
                lambda channel: self._browser.latest_channel_video(channel),
            ),
            ToolSpec(
                "open_reels",
                "Open a scrolling short-form video feed (reels). platform is "
                "'instagram' (default) or 'youtube' for Shorts.",
                {
                    "type": "object",
                    "properties": {
                        "platform": {
                            "type": "string",
                            "description": "'instagram' or 'youtube'",
                        }
                    },
                    "required": [],
                },
                lambda platform="instagram": self._browser.open_reels(platform),
            ),
            ToolSpec(
                "open_shorts",
                "Open the YouTube Shorts feed.",
                no_params(),
                lambda: self._browser.open_shorts(),
            ),
            ToolSpec(
                "web_lookup",
                "Look up current/live information on the web (news, prices, recent "
                "events, facts you need fresh) and return a short answer.",
                _STR("query", "What to look up, e.g. 'latest news on the Fed'."),
                lambda query: self._tavily.search(query),
            ),
        ]
        super().__init__(client, model, tools)
