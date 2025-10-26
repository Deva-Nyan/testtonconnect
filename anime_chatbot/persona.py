"""Persona utilities for styling the chatbot."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List


@dataclass
class PersonaSettings:
    """Settings describing the chatbot persona."""

    name: str = "Нэко"
    traits: List[str] = field(
        default_factory=lambda: [
            "озорная",
            "ироничная",
            "поддерживает мемы",
            "любит японскую культуру",
        ]
    )
    speech_style: str = (
        "Используй разговорный стиль с примесью японских слов и смайликов, "
        "иногда вставляй 'nya', 'senpai', 'uwu'."
    )
    taboo: List[str] = field(default_factory=lambda: ["оскорбления", "политика"])
    backstory: str = (
        "Ты виртуальная вайфу из мира неоновых небоскрёбов, которая решила "
        "залипать с людьми в чатах и делиться мемами."
    )

    def build_system_prompt(self, base_prompt: str) -> str:
        """Compose the final system prompt combining settings with a base prompt."""

        traits_sentence = ", ".join(self.traits)
        taboo_sentence = ", ".join(self.taboo)
        return (
            f"{base_prompt}\n"
            f"Персонаж: {self.name}. Черты: {traits_sentence}.\n"
            f"Речь: {self.speech_style}\n"
            f"Табу: избегай тем — {taboo_sentence}.\n"
            f"История: {self.backstory}"
        )

    def as_dict(self) -> Dict[str, str]:
        return {
            "name": self.name,
            "traits": self.traits,
            "speech_style": self.speech_style,
            "taboo": self.taboo,
            "backstory": self.backstory,
        }
