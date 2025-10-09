"""CLI for chatting with the anime bot."""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.prompt import Prompt

from anime_chatbot import AnimeChatbot, ChatbotConfig, JsonlMemory
from anime_chatbot.persona import BOT_NAME

app = typer.Typer(help="Интерактивный чат с аниме-ботом")
console = Console()


def load_config(model_name: str, device: Optional[str]) -> ChatbotConfig:
    config = ChatbotConfig(model_name=model_name, device=device)
    return config


def load_memory(path: Optional[Path]) -> Optional[JsonlMemory]:
    if not path:
        return None
    return JsonlMemory(str(path))


@app.command()
def main(
    model_name: str = typer.Option("t-bank-ai/ruDialoGPT-medium", help="Hugging Face модель"),
    device: Optional[str] = typer.Option(None, help="Устройство: cpu/cuda/индекс"),
    memory_file: Optional[Path] = typer.Option(
        Path("data/memory.jsonl"),
        help="JSONL-файл с фактами для долговременной памяти",
    ),
    test_prompt: Optional[str] = typer.Option(
        None,
        help="Отправить одиночный запрос и выйти (для быстрой проверки модели)",
    ),
) -> None:
    config = load_config(model_name, device)
    memory = load_memory(memory_file)
    chatbot = AnimeChatbot(config=config, memory=memory)
    _, greeting = chatbot.preload_greeting()
    console.print(f"[bold magenta]{greeting}")
    if test_prompt:
        console.print("Запускаем одиночный запрос для проверки генерации...")
        reply = chatbot.chat(test_prompt)
        console.print(f"[bold cyan]Ты:[/] {test_prompt}")
        console.print(f"[bold magenta]{BOT_NAME}:[/] {reply}")
        console.print("[green]Готово! Запустите скрипт без --test-prompt для полноценного диалога.")
        return

    console.print("Введи текст или /exit для выхода, /reset для очистки истории.")

    while True:
        user_message = Prompt.ask("[bold cyan]Ты")
        if user_message.strip() in {"/exit", "/quit"}:
            console.print("[bold]Пока-пока ^_^")
            break
        if user_message.strip() == "/reset":
            chatbot.reset_history()
            console.print("История очищена, начинаем заново!")
            continue
        reply = chatbot.chat(user_message)
        console.print(f"[bold magenta]{BOT_NAME}:[/] {reply}")


if __name__ == "__main__":
    app()
