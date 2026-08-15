# JARVIS MCP server — slim, cloud-hostable image (Koyeb / Render / Fly / any Docker host).
# Build context is the REPO ROOT:  docker build -f deploy/Dockerfile -t jarvis-mcp .
FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PORT=8000

WORKDIR /app

# Install only the slim MCP dependency set first (better layer caching).
COPY deploy/requirements-mcp.txt /app/deploy/requirements-mcp.txt
RUN pip install --no-cache-dir -r deploy/requirements-mcp.txt

# Copy just the package (not the whole repo) so the image stays lean.
COPY jarvis /app/jarvis

# The macOS-only tools (open Chrome, AppleScript, screenshots) are auto-hidden on a
# non-Mac host; set JARVIS_MCP_LOCAL_TOOLS=1 only if you know the host has them.
ENV JARVIS_MCP_LOCAL_TOOLS=0

EXPOSE 8000
# Streamable-HTTP MCP endpoint is served at /mcp on $PORT.
CMD ["python", "-m", "jarvis.mcp"]
