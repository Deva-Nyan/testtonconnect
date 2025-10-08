"""Quick sanity check for the anime chatbot model."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console

from anime_chatbot import AnimeChatbot, ChatbotConfig, DvachMemory, PersonaSettings

app = typer.Typer(help="Отправить одиночный запрос модели и проверить ответ")
console = Console()


def load_persona(path: Optional[Path]) -> Optional[PersonaSettings]:
    if not path:
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise typer.BadParameter(f"Не удалось прочитать JSON персонажа: {exc}") from exc
    try:
        return PersonaSettings(**data)
    except TypeError as exc:
        raise typer.BadParameter(f"JSON персонажа имеет неверную структуру: {exc}") from exc


def load_memory(path: Optional[Path]) -> Optional[DvachMemory]:
    if not path:
        return None
    try:
        return DvachMemory.from_jsonl(path)
    except FileNotFoundError as exc:
        raise typer.BadParameter(f"Файл памяти не найден: {path}") from exc


@app.command()
def main(
    prompt: str = typer.Option("Привет! Как настроение?", help="Тестовое сообщение"),
    model_name: str = typer.Option("microsoft/DialoGPT-medium", help="Hugging Face модель"),
    device: Optional[str] = typer.Option(None, help="Устройство: cpu/cuda/индекс"),
    persona_file: Optional[Path] = typer.Option(None, exists=True, help="JSON с настройками персонажа"),
    memory_file: Optional[Path] = typer.Option(None, exists=True, help="JSONL с контекстом из тредов"),
) -> None:
    persona = load_persona(persona_file)
    memory = load_memory(memory_file)

    config = ChatbotConfig(model_name=model_name, device=device)
    chatbot = AnimeChatbot(config=config, persona=persona, memory=memory)
    chatbot.preload_greeting()
    console.print("Отправляем тестовое сообщение модели...")
    reply = chatbot.chat(prompt)
    console.print(f"Ответ: {reply}")
    if reply.strip() == config.fallback_reply.strip():
        console.print(
            "[yellow]Получен запасной ответ. Проверьте, что модель скачана, и попробуйте изменить prompt или параметры генерации."
        )
        raise typer.Exit(code=1)
    console.print("[green]Модель ответила без запасного сообщения — всё готово!")


if __name__ == "__main__":
    app()
