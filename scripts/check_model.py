"""Quick smoke-test for the anime chatbot model."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console

from anime_chatbot import AnimeChatbot, ChatbotConfig, DvachMemory, PersonaSettings

app = typer.Typer(help="Проверить, что модель отвечает на тестовый запрос")
console = Console()


def load_persona(path: Optional[Path]) -> Optional[PersonaSettings]:
    if not path:
        return None
    data = json.loads(path.read_text(encoding="utf-8"))
    return PersonaSettings(**data)


def load_memory(path: Optional[Path]) -> Optional[DvachMemory]:
    if not path:
        return None
    return DvachMemory.from_jsonl(path)


@app.command()
def main(
    message: str = typer.Option("Привет!", "--message", "-m", help="Сообщение для теста"),
    model_name: str = typer.Option("microsoft/DialoGPT-medium", help="Hugging Face модель"),
    device: Optional[str] = typer.Option(None, help="Устройство: cpu/cuda/индекс"),
    persona_file: Optional[Path] = typer.Option(None, exists=True, help="JSON с настройками персонажа"),
    memory_file: Optional[Path] = typer.Option(
        None,
        exists=True,
        help="JSONL с постами Двача для подстановки контекста",
    ),
) -> None:
    persona = load_persona(persona_file)
    memory = load_memory(memory_file)
    config = ChatbotConfig(model_name=model_name, device=device)
    bot = AnimeChatbot(config=config, persona=persona, memory=memory)
    console.print("[bold]Отправляем тестовое сообщение модели...")
    bot.preload_greeting()
    reply = bot.chat(message)
    console.print(f"[bold magenta]Ответ: {reply}")
    if reply == config.fallback_reply:
        console.print(
            "[yellow]Получен запасной ответ. Проверьте, что модель скачана и попробуйте "
            "изменить prompt или параметры генерации."
        )
        raise typer.Exit(code=2)


if __name__ == "__main__":
    app()
