"""Utilities for maintaining a lightweight running summary of the dialogue."""
from __future__ import annotations

from typing import List, Optional


def make_running_summary(history: List[str], previous_summary: Optional[str]) -> str:
    """Return a folded summary using the last turns and previous state."""

    tail = history[-10:]
    base = (previous_summary or "").strip()
    parts = []
    if base:
        parts.append(f"Сводка до этого: {base}")
    if tail:
        parts.append("Новые факты: " + " | ".join(tail))
    return "\n".join(parts)


__all__ = ["make_running_summary"]
