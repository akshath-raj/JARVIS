"""Tavily web search — recent news and live information from the net.

This is the ONE place JARVIS reaches the open web for facts. Only the search
query leaves the machine (like the Spotify catalog search); the user's voice and
conversation never go out. Tavily returns a short synthesised `answer` that is
ideal to speak aloud, plus source snippets.

No SDK is added — a single `requests.post` keeps the local stack lightweight.
"""
from __future__ import annotations

import requests

_SEARCH_URL = "https://api.tavily.com/search"


class WebError(RuntimeError):
    pass


class TavilyClient:
    def __init__(self, api_key: str = ""):
        self._api_key = api_key

    @property
    def enabled(self) -> bool:
        return bool(self._api_key)

    def search(self, query: str, topic: str = "general", max_results: int = 5) -> str:
        """Return a concise, speakable answer for `query`.

        topic: "news" for current events / latest headlines, else "general".
        """
        if not (query or "").strip():
            raise WebError("empty search query")
        if not self.enabled:
            raise WebError(
                "Web search needs a Tavily API key — set TAVILY_API_KEY in your .env."
            )
        payload = {
            "api_key": self._api_key,
            "query": query,
            "topic": "news" if topic == "news" else "general",
            "search_depth": "basic",
            "include_answer": True,
            "max_results": max_results,
        }
        try:
            resp = requests.post(_SEARCH_URL, json=payload, timeout=20)
        except requests.RequestException as e:
            raise WebError(f"web search failed: {e}") from e
        if resp.status_code == 401:
            raise WebError("Tavily rejected the API key (401). Check TAVILY_API_KEY.")
        if resp.status_code != 200:
            raise WebError(f"web search failed ({resp.status_code}): {resp.text[:120]}")
        data = resp.json()
        answer = (data.get("answer") or "").strip()
        if answer:
            return answer
        # No synthesised answer — stitch the top result titles/snippets instead.
        results = data.get("results", []) or []
        if not results:
            return f"I found nothing on the web for '{query}'."
        top = results[0]
        title = (top.get("title") or "").strip()
        snippet = (top.get("content") or "").strip()
        return f"{title}: {snippet}"[:400] if title or snippet else (
            f"I found nothing useful on the web for '{query}'."
        )

    def search_urls(self, query: str, max_results: int = 6) -> list[str]:
        """Return the result URLs for `query`, best match first. Used to resolve a
        thing to its canonical page (e.g. a Netflix title's page). Raises WebError
        the same way `search` does."""
        if not (query or "").strip():
            raise WebError("empty search query")
        if not self.enabled:
            raise WebError("Web search needs a Tavily API key — set TAVILY_API_KEY in your .env.")
        payload = {
            "api_key": self._api_key,
            "query": query,
            "topic": "general",
            "search_depth": "basic",
            "include_answer": False,
            "max_results": max_results,
        }
        try:
            resp = requests.post(_SEARCH_URL, json=payload, timeout=20)
        except requests.RequestException as e:
            raise WebError(f"web search failed: {e}") from e
        if resp.status_code != 200:
            raise WebError(f"web search failed ({resp.status_code}): {resp.text[:120]}")
        return [r.get("url", "") for r in (resp.json().get("results") or []) if r.get("url")]
