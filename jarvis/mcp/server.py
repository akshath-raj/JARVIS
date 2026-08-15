"""JARVIS MCP server.

Exposes the JARVIS tool-belt (Spotify, browser, on-screen video/tab control, web
search, document RAG, files, and long-term memory) as Model Context Protocol tools
over streamable HTTP, so ANY LLM client can drive them — no STT, no wake word, no
agent reasoning, just the functions.

Design:
  * Each capability GROUP is built behind a try/except and only its tools are
    registered when its controller initialises and its keys/deps are present, so the
    SAME server file runs fully-featured on your Mac and as a slimmer web service on
    Koyeb/Render (where macOS-only tools are simply not offered).
  * Tools are thin async wrappers over the exact controllers the voice assistant
    uses, so behaviour is identical. Blocking work runs in a worker thread.
  * Transport is streamable HTTP on ``/mcp`` (host 0.0.0.0, ``$PORT``) — the shape
    Claude/OpenAI/Cursor expect for a remote MCP server.

Run locally:      python -m jarvis.mcp            (or: python -m jarvis.mcp.server)
Env that matters: PORT, and the usual JARVIS_* / *_API_KEY vars (see .env.example).
"""
from __future__ import annotations

import asyncio
import logging
import os
import platform
import sys

from mcp.server.mcpserver import MCPServer

from jarvis.config import config

logger = logging.getLogger("jarvis.mcp")

# macOS-only tools drive the LOCAL machine (open Chrome, AppleScript, screenshots),
# so they're meaningful only when the server runs on the user's own Mac. On a cloud
# host they'd act on the server, so they're hidden unless explicitly forced on.
_IS_MAC = platform.system() == "Darwin"


def _truthy(v: str) -> bool:
    return v.strip().lower() in ("1", "true", "yes", "on")


def _local_tools_enabled() -> bool:
    override = os.getenv("JARVIS_MCP_LOCAL_TOOLS", "")
    return _truthy(override) if override else _IS_MAC


# ── helpers ──────────────────────────────────────────────────────────────────
async def _run(fn, *args, **kwargs):
    """Run a blocking controller call off the event loop."""
    return await asyncio.to_thread(fn, *args, **kwargs)


def _err(prefix: str, e: Exception) -> str:
    return f"{prefix}: {e}"


# ── tool groups ──────────────────────────────────────────────────────────────
def register_spotify(server: MCPServer) -> bool:
    """Spotify control via the Web API (works anywhere the account can authorise)."""
    from jarvis.tools.spotify import SpotifyController, SpotifyError

    sp = SpotifyController(config.spotify_client_id, config.spotify_client_secret)
    mode = config.spotify_search_mode

    @server.tool()
    async def play_song(query: str, loop: bool = False) -> str:
        """Search Spotify and play a song, artist, genre or mood (e.g. "some jazz",
        "Daft Punk", "lo-fi beats"). Set loop=true to repeat it. This is for anything
        that isn't the user's named playlist."""
        try:
            res = await _run(sp.play_query, query, mode, loop)
            return f"Now playing {res.label}" + (" on repeat." if loop else ".")
        except SpotifyError as e:
            return _err("Spotify error", e)

    @server.tool()
    async def play_playlist(name: str, loop: bool = False) -> str:
        """Play one of the user's OWN Spotify playlists by name (e.g. "my Focus
        playlist"). For a genre/mood/song use play_song instead."""
        try:
            return f"Playing your {await _run(sp.play_playlist, name, loop)} playlist."
        except SpotifyError as e:
            return _err("Spotify error", e)

    @server.tool()
    async def list_playlists() -> str:
        """List the user's Spotify playlists and how many songs each has."""
        try:
            pls = await _run(sp.list_playlists)
        except SpotifyError as e:
            return _err("Spotify error", e)
        if not pls:
            return "You have no playlists."
        return "Playlists: " + ", ".join(
            f"{n} ({c} songs)" if c is not None else n for n, c in pls
        )

    @server.tool()
    async def pause_music() -> str:
        """Pause Spotify playback."""
        try:
            await _run(sp.pause)
            return "Paused."
        except SpotifyError as e:
            return _err("Spotify error", e)

    @server.tool()
    async def resume_music() -> str:
        """Resume Spotify playback."""
        try:
            await _run(sp.resume)
            return "Resumed."
        except SpotifyError as e:
            return _err("Spotify error", e)

    @server.tool()
    async def next_song() -> str:
        """Skip to the next track on Spotify."""
        try:
            await _run(sp.next_track)
            return "Skipped."
        except SpotifyError as e:
            return _err("Spotify error", e)

    @server.tool()
    async def set_music_volume(level: int) -> str:
        """Set Spotify volume to an exact level from 0 to 100."""
        try:
            await _run(sp.set_volume, level)
            return f"Volume set to {level}."
        except SpotifyError as e:
            return _err("Spotify error", e)

    @server.tool()
    async def change_volume(direction: str) -> str:
        """Nudge Spotify volume up or down. direction: "up" or "down"."""
        up = any(w in direction.lower() for w in ("up", "loud", "increase", "higher", "raise", "more"))
        try:
            level = await _run(sp.nudge_volume, up)
            return f"Volume {'up' if up else 'down'} to {level}."
        except SpotifyError as e:
            return _err("Spotify error", e)

    @server.tool()
    async def whats_playing() -> str:
        """Report the currently playing Spotify track."""
        try:
            return await _run(sp.current_track)
        except SpotifyError as e:
            return _err("Spotify error", e)

    @server.tool()
    async def recently_played() -> str:
        """List tracks the user played recently on Spotify."""
        try:
            tracks = await _run(sp.recently_played, 10)
        except SpotifyError as e:
            return _err("Spotify error", e)
        return "Recently played: " + ", ".join(t.label for t in tracks) if tracks else "Nothing recent."

    @server.tool()
    async def list_liked_songs() -> str:
        """List the user's liked/favourite Spotify songs."""
        try:
            tracks = await _run(sp.liked_songs, 15)
        except SpotifyError as e:
            return _err("Spotify error", e)
        return ", ".join(t.label for t in tracks) if tracks else "No liked songs."

    return True


def register_web(server: MCPServer) -> bool:
    """Live web search via Tavily (portable — just needs TAVILY_API_KEY)."""
    from jarvis.tools.web import TavilyClient, WebError

    tv = TavilyClient(config.tavily_api_key)
    if not tv.enabled:
        logger.info("web: TAVILY_API_KEY not set — skipping web search tools")
        return False

    @server.tool()
    async def web_search(query: str, topic: str = "general") -> str:
        """Search the live web and return a concise answer. Use for current events,
        prices, weather, scores, or any fact that changes over time. topic can be
        "general" or "news"."""
        try:
            return await _run(tv.search, query, topic)
        except WebError as e:
            return _err("Web search error", e)

    @server.tool()
    async def web_search_urls(query: str) -> str:
        """Search the web and return the top result URLs (one per line)."""
        try:
            urls = await _run(tv.search_urls, query)
        except WebError as e:
            return _err("Web search error", e)
        return "\n".join(urls) if urls else "No results."

    return True


def register_documents(server: MCPServer) -> bool:
    """Document RAG over the user's indexed files (answer / summarise / review)."""
    from jarvis.rag import build_rag

    rag = build_rag(config)

    @server.tool()
    async def ask_documents(question: str, document: str = "", page: int = 0) -> str:
        """Answer a question from the user's indexed documents. `document` is an
        optional free-text description to scope to one file (no exact filename
        needed); `page` optionally focuses on a page/slide. Leave both blank to search
        everything."""
        return await _run(rag.answer, question, document=document, page=page)

    @server.tool()
    async def summarize_document(document: str, page: int = 0) -> str:
        """Summarise a whole document (or one page/slide). `document` is a free-text
        description of the file."""
        return await _run(rag.summarize, document, page=page)

    @server.tool()
    async def review_document(document: str, focus: str = "") -> str:
        """Review/critique one of the user's documents and give concrete, actionable
        improvement suggestions (résumé, essay, report…). `focus` optionally narrows
        the critique (e.g. "clarity")."""
        return await _run(rag.review, document, focus=focus)

    @server.tool()
    async def find_document(description: str) -> str:
        """Locate a document by description and report its filename and where it is."""
        doc = await _run(rag.resolve_document, description)
        if not doc:
            return f"No document matches “{description}”."
        bits = [doc["filename"]]
        if doc.get("location"):
            bits.append(f"in {doc['location']}")
        return ", ".join(bits) + "."

    @server.tool()
    async def list_related_documents(topic: str) -> str:
        """List the indexed documents related to a topic (filename per line), best
        first — the same set 'open all my X docs' would open (max 10)."""
        docs = await _run(rag.related_documents, topic)
        return "\n".join(d["filename"] for d in docs) if docs else f"No documents on “{topic}”."

    @server.tool()
    async def reindex_documents() -> str:
        """Rescan the watched folders and update the document index."""
        s = await _run(rag.sync)
        return (f"Index updated: {s['added']} added, {s['updated']} refreshed, "
                f"{s['removed']} removed, {s['unchanged']} unchanged.")

    return True


def register_memory(server: MCPServer) -> bool:
    """Durable user profile / preferences (local file store)."""
    from jarvis.graph.memory import MemoryStore

    mem = MemoryStore(config.memory_dir)  # no summariser model needed for CRUD

    @server.tool()
    async def remember(fact: str) -> str:
        """Store a durable fact or preference about the user (e.g. "prefers window
        seats", "allergic to peanuts")."""
        return await _run(mem.remember, fact)

    @server.tool()
    async def forget(query: str) -> str:
        """Remove stored facts matching a description."""
        return await _run(mem.forget, query)

    @server.tool()
    async def recall_about_me() -> str:
        """Return everything stored about the user (their profile/preferences)."""
        text = await _run(mem.recall_text)
        return text or "I have nothing stored about you yet."

    return True


def register_files(server: MCPServer) -> bool:
    """Sandboxed file organiser — read/copy/move/open on the server host (never
    delete). Acts on the machine the server runs on."""
    from jarvis.files import FileOrganizer, Sandbox
    from jarvis.files.organizer import FileError
    from jarvis.files.sandbox import SandboxError

    org = FileOrganizer(Sandbox(config.files_sandbox))

    @server.tool()
    async def list_folder(folder: str = "~") -> str:
        """List the files and folders in a directory (within the sandbox)."""
        try:
            entries = await _run(org.list_dir, folder)
        except (FileError, SandboxError, OSError) as e:
            return _err("File error", e)
        return "\n".join(p.name for p in entries) if entries else "(empty)"

    @server.tool()
    async def move_file(source: str, destination: str) -> str:
        """Move a file or folder to another location (never overwrites)."""
        try:
            return await _run(org.move, source, destination)
        except (FileError, SandboxError, OSError) as e:
            return _err("File error", e)

    @server.tool()
    async def copy_file(source: str, destination: str) -> str:
        """Copy a file or folder to another location (never overwrites)."""
        try:
            return await _run(org.copy, source, destination)
        except (FileError, SandboxError, OSError) as e:
            return _err("File error", e)

    @server.tool()
    async def organize_folder(folder: str = "~/Downloads", mode: str = "move") -> str:
        """Sort a cluttered folder into category subfolders (Documents, Images,
        Code…), setting exact duplicates aside (nothing is deleted). mode: "move" or
        "copy"."""
        try:
            s = await _run(org.organize, folder, mode=mode)
        except (FileError, SandboxError, OSError) as e:
            return _err("File error", e)
        cats = ", ".join(f"{n} {c}" for c, n in s["by_category"].items()) or "nothing"
        return f"Organised {s['moved']} file(s) into {cats}."

    return True


def register_browser(server: MCPServer) -> bool:
    """Open sites / play videos in the HOST machine's Chrome (macOS `open`)."""
    from jarvis.tools.browser import BrowserController, BrowserError

    br = BrowserController(config.browser_app)

    @server.tool()
    async def open_site(name: str) -> str:
        """Open a website or web app in Chrome by name (youtube, gmail, github,
        netflix…) or a URL. Opens on the machine the server runs on."""
        try:
            await _run(br.open_site, name)
            return "Opened."
        except BrowserError as e:
            return _err("Browser error", e)

    @server.tool()
    async def play_youtube(query: str) -> str:
        """Search YouTube and open the top result (it autoplays)."""
        try:
            return await _run(br.play_youtube, query)
        except BrowserError as e:
            return _err("Browser error", e)

    @server.tool()
    async def play_netflix(title: str) -> str:
        """Play a specific show/movie on Netflix in the logged-in Chrome. Resolves the
        title id via web search when a Tavily key is set."""
        title_id = ""
        try:
            from jarvis.tools.web import TavilyClient
            tv = TavilyClient(config.tavily_api_key)
            if tv.enabled:
                title_id = br.netflix_id_from_urls(await _run(tv.search_urls, f"{title} netflix"))
        except Exception:
            title_id = ""
        try:
            return await _run(br.play_netflix, title, title_id=title_id)
        except BrowserError as e:
            return _err("Browser error", e)

    return True


def register_media(server: MCPServer) -> bool:
    """Control the video playing in the host's Chrome, and close tabs (AppleScript)."""
    from jarvis.tools.media import MediaController, MediaError

    md = MediaController(config.browser_app)

    @server.tool()
    async def control_video(action: str, value: str = "") -> str:
        """Control the video playing in Chrome. action: pause, play, faster, slower,
        set_speed (value=number), volume_up, volume_down, set_volume (value=0-100),
        mute, captions, forward, back, restart, fullscreen."""
        try:
            return await _run(md.control, action, value)
        except MediaError as e:
            return _err("Media error", e)

    @server.tool()
    async def close_tabs(match: str = "") -> str:
        """Close tabs in the host's Chrome. Pass `match` to close every tab whose URL
        or title contains it (e.g. "vtop", "youtube"); empty closes the active tab."""
        try:
            if (match or "").strip():
                return await _run(md.close_tabs_matching, match)
            return await _run(md.close_tab)
        except MediaError as e:
            return _err("Media error", e)

    return True


# name → (registrar, is_macos_only)
_GROUPS = {
    "spotify": (register_spotify, False),
    "web": (register_web, False),
    "documents": (register_documents, False),
    "memory": (register_memory, False),
    "files": (register_files, False),
    "browser": (register_browser, True),
    "media": (register_media, True),
}


def build_server() -> MCPServer:
    """Assemble the MCP server, registering every group that initialises cleanly."""
    server = MCPServer(
        name="jarvis",
        title="JARVIS Tools",
        version="0.1.0",
        instructions=(
            "JARVIS's tool-belt: control Spotify, search the live web, and query / "
            "summarise / review the user's own documents; plus long-term memory and "
            "(when self-hosted on the user's Mac) opening sites, controlling the "
            "on-screen video, and closing browser tabs. Call a tool to DO the thing — "
            "prefer play_song for genres/moods and play_playlist only for named "
            "playlists; use review_document (not ask_documents) to critique a file."
        ),
    )
    local_ok = _local_tools_enabled()
    enabled, skipped = [], []
    for name, (registrar, mac_only) in _GROUPS.items():
        if mac_only and not local_ok:
            skipped.append(f"{name} (host-only)")
            continue
        try:
            if registrar(server):
                enabled.append(name)
            else:
                skipped.append(f"{name} (missing key/deps)")
        except Exception as e:  # a broken group must never sink the whole server
            logger.warning("MCP group %r unavailable: %s", name, e)
            skipped.append(f"{name} (error)")
    logger.info("JARVIS MCP ready — enabled: %s | skipped: %s",
                ", ".join(enabled) or "none", ", ".join(skipped) or "none")
    return server


def main() -> None:
    logging.basicConfig(
        level=os.getenv("JARVIS_MCP_LOG", "INFO").upper(),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    server = build_server()
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", "8000"))
    logger.info("Serving MCP (streamable HTTP) on http://%s:%d/mcp", host, port)
    # Stateless HTTP so it scales cleanly behind Koyeb/Render's load balancer and any
    # number of clients can connect without sticky sessions.
    server.run(
        transport="streamable-http",
        host=host,
        port=port,
        streamable_http_path="/mcp",
        stateless_http=True,
    )


if __name__ == "__main__":
    sys.exit(main())
