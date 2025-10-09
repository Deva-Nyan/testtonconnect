"""Post-processing helpers that enforce persona consistency."""
from __future__ import annotations

import re

from .persona import FORBIDDEN_KINSHIP


def enforce_consistency(reply: str) -> str:
    """Rewrite parts of the reply that contradict persona constraints."""

    text = reply.strip()
    if not text:
        return reply
    lowered = text.lower()
    if any(token in lowered for token in FORBIDDEN_KINSHIP):
        text = re.sub(
            r"\b(мама|папа|сын|дочь|отец|мать|брат|сестра)\b.*",
            "Не буду путать нас с семьёй. Я твоя вайфу ^_^",
            text,
            flags=re.IGNORECASE,
        )
    text = re.sub(r"\bты моя\b.*", "Ты мой любимый человек!", text, flags=re.IGNORECASE)
    return text


__all__ = ["enforce_consistency"]
