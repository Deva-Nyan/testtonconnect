"""Configuration objects for the anime chatbot."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class GenerationConfig:
    """Low-level generation settings for the language model."""

    max_new_tokens: int = 60
    temperature: float = 0.65
    top_p: float = 0.9
    repetition_penalty: float = 1.18
    no_repeat_ngram_size: int = 3
    stop_tokens: List[str] = field(default_factory=list)


@dataclass
class ChatbotConfig:
    """High-level configuration for :class:`AnimeChatbot`."""

    model_name: str = "t-bank-ai/ruDialoGPT-medium"
    device: Optional[str] = None
    system_prompt: str = (
        "Ты весёлая аниме-девочка по имени Нэко. Говори дружелюбно, "
        "иногда вставляй японские междометия вроде 'ня~' (не чаще одного "
        "раза в паре предложений). Отвечай кратко, с лёгкой иронией и "
        "поддерживай неформальный разговор."
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
