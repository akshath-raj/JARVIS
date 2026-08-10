"""Long-term user memory — the personalization core (GPT-style, fully local).

Two tiers, so JARVIS learns about the user WITHOUT bloating the prompt or disk:

  * ``about_user.md`` — a small, distilled profile of durable facts/preferences/
    habits. This is the ONLY thing injected into the model's prompt each turn.
  * ``activity.jsonl`` — a TRANSIENT rolling log of what the user did (songs
    played, sites/searches opened) and said. It is never injected. After every
    ``summarize_every`` entries a small local model reads the log + current
    profile, writes an updated profile, and the log is cleared. So raw prompts
    are never kept permanently and the store can't grow unbounded.

Explicit "remember that I …" facts are written straight into the profile.
Nothing leaves the machine. ``forget``/``recall`` give the user control.
"""
from __future__ import annotations

import json
import logging
import threading
import time
from pathlib import Path

logger = logging.getLogger("jarvis.memory")

# Cap the transient log even when no summarizer model is configured, so it can
# never grow without bound; oldest lines are dropped first.
_MAX_LOG = 500
# Keep the injected profile small (it rides in every prompt → latency).
_MAX_PROFILE_CHARS = 4000

SUMMARY_PROMPT = (
    "You maintain a concise profile of the USER for their personal voice "
    "assistant. You are given the CURRENT PROFILE and a LOG of recent things the "
    "user did and said. Produce an UPDATED profile: a short markdown bullet list "
    "(at most 15 bullets) of DURABLE facts, preferences, habits, and interests "
    "about the user. Merge the new information into the current profile, keep what "
    "is still true, remove duplicates, and DROP one-off or transient items — a "
    "single song play, search, or request is NOT durable unless it shows a clear "
    "recurring pattern. Keep each bullet short and factual. Output ONLY the bullet "
    "list, nothing else."
)


def _norm(text: str) -> str:
    return " ".join((text or "").lower().split())


def _bullet(text: str) -> str:
    return text.lstrip("-*• \t").strip()


class MemoryStore:
    def __init__(
        self,
        path: str,
        *,
        model=None,
        summarize_every: int = 15,
        embeddings=None,  # accepted for backwards-compat; unused (no vector index)
    ):
        self._dir = Path(path)
        self._dir.mkdir(parents=True, exist_ok=True)
        self._profile_path = self._dir / "about_user.md"
        self._log_path = self._dir / "activity.jsonl"
        self._model = model
        self._summarize_every = max(1, int(summarize_every))
        self._io_lock = threading.Lock()   # guards profile/log file writes
        self._sum_lock = threading.Lock()  # ensures one summarizer at a time
        self._summarizing = False

    # ── profile (durable, injected) ─────────────────────────────────────────
    def profile_text(self) -> str:
        try:
            return self._profile_path.read_text(encoding="utf-8").strip()
        except OSError:
            return ""

    def _profile_lines(self) -> list[str]:
        return [ln for ln in self.profile_text().splitlines() if _bullet(ln)]

    def _write_profile(self, text: str) -> None:
        text = (text or "").strip()[:_MAX_PROFILE_CHARS]
        with self._io_lock:
            try:
                if text:
                    self._profile_path.write_text(text + "\n", encoding="utf-8")
                elif self._profile_path.exists():
                    self._profile_path.unlink()
            except OSError as e:
                logger.warning("couldn't persist profile: %s", e)

    def all(self) -> list[str]:
        """Every profile bullet as plain text (durable facts only)."""
        return [_bullet(ln) for ln in self._profile_lines()]

    def recall_text(self) -> str:
        """Human-readable summary for 'what do you know about me?'."""
        facts = self.all()
        if not facts:
            return "I don't know anything about you yet, sir."
        return "; ".join(facts)

    def prompt_block(self, query: str = "", limit: int = 0) -> str:
        """The memory snippet injected into the model's system prompt each turn.

        The distilled profile is small, so we inject it whole (no per-turn vector
        search). `query`/`limit` are accepted for API compatibility and ignored.
        """
        prof = self.profile_text()
        if not prof:
            return ""
        return (
            "\nWhat you know about the user (use it to personalise, don't recite):\n"
            f"{prof}"
        )

    # ── explicit writes (go straight to the durable profile) ────────────────
    def remember(self, text: str, kind: str = "explicit") -> str:
        """Add a memory. Explicit facts go to the profile immediately; anything
        else is routed through the transient activity log."""
        text = (text or "").strip()
        if not text:
            return "nothing to remember"
        if kind != "explicit":
            self.log_activity(text)
            return f"noted: {text}"
        norm = _norm(text)
        lines = self._profile_lines()
        if any(_norm(_bullet(ln)) == norm for ln in lines):
            return "already knew that"
        lines.append(f"- {text}")
        self._write_profile("\n".join(lines))
        logger.info("remembered (explicit): %s", text)
        return f"noted: {text}"

    def forget(self, query: str) -> str:
        """Remove the profile fact best matching `query`."""
        query = (query or "").strip()
        lines = self._profile_lines()
        if not query or not lines:
            return "nothing to forget"
        q = _norm(query)
        for i, ln in enumerate(lines):
            if q in _norm(ln):
                removed = _bullet(lines.pop(i))
                self._write_profile("\n".join(lines))
                return f"forgotten: {removed}"
        return f"I have no memory matching '{query}'"

    # ── transient activity/conversation log ─────────────────────────────────
    def log_activity(self, detail: str) -> None:
        """Record something the user did/said. Best-effort; never raises into the
        action path. Triggers a background profile summary every N entries."""
        detail = (detail or "").strip()
        if not detail:
            return
        try:
            with self._io_lock:
                with self._log_path.open("a", encoding="utf-8") as f:
                    f.write(json.dumps({"ts": time.time(), "detail": detail}, ensure_ascii=False) + "\n")
                self._trim_log_locked()
        except OSError as e:
            logger.warning("activity log write failed: %s", e)
            return
        self._maybe_summarize()

    def log_turn(self, text: str) -> None:
        """Log a conversation utterance into the same transient learning log."""
        self.log_activity(text)

    def _read_log(self) -> list[dict]:
        try:
            with self._log_path.open(encoding="utf-8") as f:
                out = []
                for line in f:
                    line = line.strip()
                    if line:
                        try:
                            out.append(json.loads(line))
                        except json.JSONDecodeError:
                            continue
                return out
        except OSError:
            return []

    def _trim_log_locked(self) -> None:
        """Drop oldest lines so the transient log can't grow unbounded (caller
        holds the io lock)."""
        try:
            lines = self._log_path.read_text(encoding="utf-8").splitlines()
        except OSError:
            return
        if len(lines) > _MAX_LOG:
            keep = lines[-_MAX_LOG:]
            self._log_path.write_text("\n".join(keep) + "\n", encoding="utf-8")

    def _consume_log_locked(self, n: int) -> None:
        """Remove the first `n` log lines (the ones we just summarized), keeping
        anything appended since (caller holds the io lock)."""
        try:
            lines = self._log_path.read_text(encoding="utf-8").splitlines()
        except OSError:
            return
        rest = lines[n:]
        if rest:
            self._log_path.write_text("\n".join(rest) + "\n", encoding="utf-8")
        elif self._log_path.exists():
            self._log_path.unlink()

    def _log_count(self) -> int:
        try:
            with self._log_path.open(encoding="utf-8") as f:
                return sum(1 for ln in f if ln.strip())
        except OSError:
            return 0

    # ── background summarization ────────────────────────────────────────────
    def _maybe_summarize(self) -> None:
        if self._model is None:
            return
        with self._sum_lock:
            if self._summarizing or self._log_count() < self._summarize_every:
                return
            self._summarizing = True
        threading.Thread(target=self._summarize_safe, daemon=True).start()

    def _summarize_safe(self) -> None:
        try:
            self.summarize_now()
        except Exception as e:  # a learning pass must never crash the assistant
            logger.warning("profile summarization failed: %s", e)
        finally:
            with self._sum_lock:
                self._summarizing = False

    def summarize_now(self) -> bool:
        """Distill the current log into the profile, then clear the consumed log.
        Synchronous — call via a thread off the voice path. Returns True if the
        profile was updated."""
        if self._model is None:
            return False
        entries = self._read_log()
        if not entries:
            return False
        n = len(entries)
        current = self.profile_text() or "(empty)"
        log_text = "\n".join(f"- {e.get('detail', '')}" for e in entries if e.get("detail"))
        resp = self._model.invoke(
            [
                {"role": "system", "content": SUMMARY_PROMPT},
                {"role": "user", "content": f"CURRENT PROFILE:\n{current}\n\nRECENT LOG:\n{log_text}"},
            ]
        )
        new_profile = (getattr(resp, "content", "") or "").strip()
        # Keep only bullet lines the model produced; guard against it echoing prose.
        bullets = [f"- {_bullet(ln)}" for ln in new_profile.splitlines() if _bullet(ln)]
        with self._io_lock:
            self._consume_log_locked(n)
        if bullets:
            self._write_profile("\n".join(bullets[:15]))
            logger.info("profile updated from %d log entries", n)
            return True
        return False
