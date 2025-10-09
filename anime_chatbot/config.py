"""Configuration objects for the anime chatbot."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

from .persona import SYSTEM_PERSONA


@dataclass
class GenerationConfig:
    """Low-level generation settings for the language model."""

    max_new_tokens: int = 160
    temperature: float = 0.6
    top_p: float = 0.9
    top_k: int = 40
    repetition_penalty: float = 1.15
    no_repeat_ngram_size: int = 3
    stop_tokens: List[str] = field(
        default_factory=lambda: ["User:", "Bot:", "Пользователь:", "Бот:"]
    )


@dataclass
class ChatbotConfig:
    """High-level configuration for :class:`AnimeChatbot`."""

    model_name: str = "t-bank-ai/ruDialoGPT-medium"
    device: Optional[str] = None
    system_prompt: str = SYSTEM_PERSONA
    welcome_message: str = "Привет! Я Дэся, твоя вайфу-подружка. Чем займёмся?"
    history_tail: int = 6
    summary_interval: int = 8
    memory_snippets: int = 5
    fallback_reply: str = "Я чуть запуталась, повтори иначе, пожалуйста?"
    generation: GenerationConfig = field(default_factory=GenerationConfig)

    def as_dict(self) -> dict:
        """Return a serializable representation of the configuration."""

        return {
            "model_name": self.model_name,
            "device": self.device,
            "system_prompt": self.system_prompt,
            "welcome_message": self.welcome_message,
            "history_tail": self.history_tail,
            "summary_interval": self.summary_interval,
            "memory_snippets": self.memory_snippets,
            "fallback_reply": self.fallback_reply,
            "generation": self.generation.__dict__,
        }


__all__ = ["ChatbotConfig", "GenerationConfig"]
