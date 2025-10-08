"""CLI for chatting with the anime bot."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.prompt import Prompt

from anime_chatbot import AnimeChatbot, ChatbotConfig, DvachMemory, PersonaSettings

app = typer.Typer(help="Интерактивный чат с аниме-ботом")
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
    model_name: str = typer.Option("tinkoff-ai/ruDialoGPT-medium", help="Hugging Face модель"),
    device: Optional[str] = typer.Option(None, help="Устройство: cpu/cuda/индекс"),
    persona_file: Optional[Path] = typer.Option(None, exists=True, help="JSON с настройками персонажа"),
    memory_file: Optional[Path] = typer.Option(
        None,
        exists=True,
        help="JSONL с постами Двача для подстановки контекста",
    ),
    test_prompt: Optional[str] = typer.Option(
        None,
        help="Отправить одиночный запрос и выйти (для быстрой проверки модели)",
    ),
) -> None:
    persona = load_persona(persona_file)
    memory = load_memory(memory_file)
    config = ChatbotConfig(model_name=model_name, device=device)
    chatbot = AnimeChatbot(config=config, persona=persona, memory=memory)
    role, greeting = chatbot.preload_greeting()
    console.print(f"[bold magenta]{greeting}")
    if test_prompt:
        console.print("Запускаем одиночный запрос для проверки генерации...")
        reply = chatbot.chat(test_prompt)
        console.print(f"[bold cyan]Ты:[/] {test_prompt}")
        console.print(f"[bold magenta]Нэко:[/] {reply}")
        console.print("[green]Готово! Запустите скрипт без --test-prompt для полноценного диалога.")
        return

    console.print("Введи текст или /exit для выхода, /reset для очистки истории.")

    while True:
        user_message = Prompt.ask("[bold cyan]Ты")
        if user_message.strip() in {"/exit", "/quit"}:
            console.print("[bold]Пока-пока nya~!")
            break
        if user_message.strip() == "/reset":
            chatbot.reset_history()
            console.print("История очищена, начинаем заново nya~")
            continue
        reply = chatbot.chat(user_message)
        console.print(f"[bold magenta]{reply}")


if __name__ == "__main__":
    app()
