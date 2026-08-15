# JARVIS MCP Server

Every JARVIS capability — Spotify, live web search, document Q&A / summarise /
review, long-term memory, and (when self-hosted on a Mac) opening sites, controlling
the on-screen video and closing browser tabs — exposed as **Model Context Protocol**
tools. No STT, no wake word, no agent reasoning: just the functions, so **any** MCP
client (Claude, OpenAI, Cursor, Zed, …) can call them.

Transport: **streamable HTTP** at `/mcp`. Nothing to learn per-client — it's the
standard remote-MCP shape.

---

## Tools

| Group | Tools | Works in the cloud? |
|-------|-------|---------------------|
| **Spotify** | `play_song`, `play_playlist`, `list_playlists`, `pause_music`, `resume_music`, `next_song`, `set_music_volume`, `change_volume`, `whats_playing`, `recently_played`, `list_liked_songs` | Search/metadata yes; playback needs an active device |
| **Web** | `web_search`, `web_search_urls` | ✅ (needs `TAVILY_API_KEY`) |
| **Documents (RAG)** | `ask_documents`, `summarize_document`, `review_document`, `find_document`, `list_related_documents`, `reindex_documents` | ✅ if the docs + index are on the host |
| **Memory** | `remember`, `forget`, `recall_about_me` | ✅ |
| **Files** | `list_folder`, `move_file`, `copy_file`, `organize_folder` | Acts on the **host** filesystem |
| **Browser** (macOS host) | `open_site`, `play_youtube`, `play_netflix` | Self-hosted Mac only |
| **Media** (macOS host) | `control_video`, `close_tabs` | Self-hosted Mac only |

The macOS-only groups drive the *host* machine (Chrome/AppleScript/screen), so they're
**auto-hidden** unless the server runs on a Mac. Force them on/off with
`JARVIS_MCP_LOCAL_TOOLS=1|0`.

---

## Run it

```bash
# from the repo root
pip install -r deploy/requirements-mcp.txt
python -m jarvis.mcp            # serves http://0.0.0.0:8000/mcp
```

Key env vars (see `.env.example` for the rest): `PORT`, `TAVILY_API_KEY`,
`OPENAI_API_KEY`, `SPOTIFY_CLIENT_ID`, `SPOTIFY_CLIENT_SECRET`, `JARVIS_RAG_DIRS`,
`JARVIS_MCP_LOCAL_TOOLS`.

---

## Connect a client

**Claude (Desktop / claude.ai → Settings → Connectors → Add custom connector):**
give it the URL `https://<your-host>/mcp`.

**Claude Code:**
```bash
claude mcp add --transport http jarvis https://<your-host>/mcp
```

**OpenAI (Responses API — hosted MCP tool):**
```python
client.responses.create(
    model="gpt-5.4-mini",
    tools=[{"type": "mcp", "server_label": "jarvis",
            "server_url": "https://<your-host>/mcp",
            "require_approval": "never"}],
    input="summarise my kalman filtering notes",
)
```

**Cursor / Zed / Windsurf:** add an MCP server of type `http` with URL
`https://<your-host>/mcp`.

**Local stdio** (some clients prefer launching a process): run
`python -m jarvis.mcp` — or adapt `main()` to `transport="stdio"`.

---

## Deploy

### Render
`New → Blueprint`, point at this repo (`deploy/render.yaml`). Set the secret env
vars in the dashboard. Endpoint: `https://<svc>.onrender.com/mcp`.

### Koyeb
`Create Service → Docker → Dockerfile = deploy/Dockerfile`, build context = repo root,
port `8000`. Add the same env vars. Endpoint: `https://<app>-<org>.koyeb.app/mcp`.

### Any Docker host
```bash
docker build -f deploy/Dockerfile -t jarvis-mcp .
docker run -p 8000:8000 -e TAVILY_API_KEY=… -e OPENAI_API_KEY=… jarvis-mcp
```

---

## Getting it to show up when people search MCPs in Claude

Two different things — don't conflate them:

1. **Anyone can already add it** as a *custom connector* by pasting the `/mcp` URL
   (above). That works today with zero approval.

2. **Appearing in Claude's built-in Connectors directory** (what users see when they
   *browse/search* connectors) requires **submitting it to Anthropic's directory for
   review** — it is not automatic. Anthropic's current bar for a directory listing:
   - a **remote** MCP server over streamable HTTP (this is that ✅),
   - **OAuth 2.1** auth so each user connects with their own account (this server is
     currently **open / unauthenticated** — see below; that's the main gap),
   - a public **privacy policy + terms**, a stable hosted URL, and a support contact,
   - the server must be safe, reliable, and do what it says.

   Submit via Anthropic's MCP **directory/connector submission form** (linked from the
   MCP docs at `modelcontextprotocol.io` and Anthropic's "Connectors" help pages).
   Listing is at Anthropic's discretion and can take time.

   **What to build next for eligibility:** add OAuth. The MCP SDK supports it directly
   — `MCPServer(..., auth_server_provider=…, token_verifier=…, auth=AuthSettings(…))`
   — so you'd front the tools with an OAuth provider (your own, or Auth0/Clerk/etc.)
   and scope Spotify/Drive-style access per user. Until then, keep the URL private or
   put it behind a gateway, because **anyone with the URL can call every tool.**

---

## Security note

This server currently has **no authentication** — every client with the URL can
invoke every enabled tool (including file moves and, on a Mac host, browser control).
For anything beyond local/private use, put it behind auth (OAuth as above, or at least
a reverse-proxy token) and only enable the groups you intend to expose.
