"""Quick health-check script for the anime chatbot model."""
from __future__ import annotations

import typer
from rich.console import Console

from typing import Optional

from anime_chatbot import AnimeChatbot, ChatbotConfig

app = typer.Typer(help="Проверить, что модель запускается и выдаёт ответ")
console = Console()


@app.command()
def main(
    prompt: str = typer.Option("Привет! Как твои дела?", "--prompt", "-p", help="Тестовое сообщение"),
    model_name: str = typer.Option("microsoft/DialoGPT-medium", help="Hugging Face модель"),
    device: Optional[str] = typer.Option(None, help="Устройство для генерации (cpu/cuda/индекс)"),
) -> None:
    config = ChatbotConfig(model_name=model_name, device=device)
    chatbot = AnimeChatbot(config=config)
    _, greeting = chatbot.preload_greeting()
    console.print(f"[bold magenta]{greeting}")
    console.print(f"[bold cyan]Тестовый запрос:[/] {prompt}")
    reply = chatbot.chat(prompt)
    console.print(f"[bold magenta]Ответ:[/] {reply}")
    if reply.strip() == config.fallback_reply.strip():
        console.print(
            "[red]Бот вернул только запасной ответ. Проверьте установку, подключение к Интернету"
            " или параметры модели."  # noqa: RUF001
        )
        raise typer.Exit(code=1)
    console.print("[green]Модель успешно сгенерировала ответ.")


if __name__ == "__main__":
    app()
