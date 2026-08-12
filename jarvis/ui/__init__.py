"""JARVIS HUD dashboard — a hidden, voice-summoned Iron-Man-style interface.

Reveals on voice command, shows what JARVIS knows about you (profile, memories,
past conversations), and renders explanations (e.g. a screenshot analysis) on
screen. Local-only (served on 127.0.0.1).
"""
from __future__ import annotations

from jarvis.ui.controller import UIController
from jarvis.ui.conversations import ConversationLog
from jarvis.ui.server import UIServer, open_dashboard, wait_until_up

__all__ = ["UIController", "ConversationLog", "UIServer", "open_dashboard", "wait_until_up"]
