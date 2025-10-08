"""Configuration objects for the anime chatbot."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class GenerationConfig:
    """Low-level generation settings for the language model."""

    max_new_tokens: int = 120
    min_new_tokens: int = 20
    temperature: float = 0.95
    top_p: float = 0.9
    repetition_penalty: float = 1.05
    stop_tokens: List[str] = field(default_factory=lambda: ["\nUser:"])


@dataclass
class ChatbotConfig:
    """High-level configuration for :class:`AnimeChatbot`."""

    model_name: str = "tinkoff-ai/ruDialoGPT-medium"
    device: Optional[str] = None
    system_prompt: str = (
        "Ты весёлая аниме-девочка по имени Нэко. Ты говоришь дружелюбно, "
        "часто используешь смайлики и японские междометия вроде 'nya~'. "
        "Отвечай кратко и с иронией, поддерживай неформальный разговор."
    )
    welcome_message: str = "Привет-привет! Нэко уже здесь, чем займёмся nya~?"
    history_turns: int = 6
    memory_snippets: int = 3
    fallback_reply: str = "Ня... пока не знаю, что ответить, давай попробуем ещё раз?"
    generation: GenerationConfig = field(default_factory=GenerationConfig)

    def as_dict(self) -> dict:
        """Return a serializable representation of the configuration."""

        return {
            "model_name": self.model_name,
            "device": self.device,
            "system_prompt": self.system_prompt,
            "welcome_message": self.welcome_message,
            "history_turns": self.history_turns,
            "memory_snippets": self.memory_snippets,
            "fallback_reply": self.fallback_reply,
            "generation": self.generation.__dict__,
        }
