"""Calendar / to-do / reminder + alarm subsystem.

`TaskStore` persists to-dos, reminders, and events; `AlarmScheduler` fires
reminders at their due time (ringing alarm + spoken alert + HUD popup) and
handles dismissal. Time awareness comes from `timeparse.parse_when`.
"""
from __future__ import annotations

from jarvis.scheduler.scheduler import AlarmScheduler
from jarvis.scheduler.store import TaskStore
from jarvis.scheduler.timeparse import parse_when

__all__ = ["AlarmScheduler", "TaskStore", "parse_when"]
