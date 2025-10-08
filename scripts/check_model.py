"""Quick sanity check for the anime chatbot model."""

from __future__ import annotations

import importlib
import json
from pathlib import Path
from typing import Iterable, Optional, Tuple

import typer
from rich.console import Console
from transformers import AutoConfig
from transformers.utils import TRANSFORMERS_CACHE

from anime_chatbot import AnimeChatbot, ChatbotConfig, DvachMemory, PersonaSettings

REQUIRED_PACKAGES: Tuple[str, ...] = ("torch", "transformers", "requests")

app = typer.Typer(help="Отправить одиночный запрос модели и проверить ответ")
console = Console()


def check_dependencies(packages: Iterable[str]) -> None:
    """Ensure the core dependencies for the chatbot are importable."""

    missing = []
    for name in packages:
        if importlib.util.find_spec(name) is None:
            missing.append(name)
            continue
        module = importlib.import_module(name)
        version = getattr(module, "__version__", "unknown")
        console.print(f"[green]✓ {name} {version}")
    if missing:
        console.print("[red]Не найдены обязательные пакеты: " + ", ".join(missing))
        console.print("[yellow]Запустите scripts/bootstrap_env.(sh|ps1) или установите зависимости вручную.")
        raise typer.Exit(code=1)


def check_model_files(model_name: str) -> None:
    """Verify that the configured Hugging Face model is available."""

    console.print(f"Проверяем модель: [bold]{model_name}[/]")
    cache_path = Path(TRANSFORMERS_CACHE).expanduser()
    console.print(f"Каталог кэша transformers: {cache_path}")
    try:
        config = AutoConfig.from_pretrained(model_name, local_files_only=True)
        console.print(
            f"[green]Найдены локальные файлы модели. Архитектура: {config.model_type}"
        )
        return
    except Exception:
        console.print(
            "[yellow]Локальные файлы не обнаружены. Попробуем проверить доступность модели онлайн..."
        )
    try:
        config = AutoConfig.from_pretrained(model_name)
    except Exception as exc:  # pragma: no cover - network issues are environment dependent
        console.print("[red]Не удалось получить конфигурацию модели: {exc}")
        console.print(
            "[yellow]Убедитесь, что есть доступ к Hugging Face Hub или скачайте модель вручную "
            "(https://huggingface.co/models) и распакуйте её в каталог кэша."
        )
        raise typer.Exit(code=1) from exc
    console.print(
        "[green]Конфигурация модели доступна онлайн. При первом запуске веса будут скачаны автоматически." 
        f" (архитектура: {config.model_type})"
    )


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
    console.print("[bold]Проверяем зависимости...[/]")
    check_dependencies(REQUIRED_PACKAGES)
    check_model_files(model_name)

    persona = load_persona(persona_file)
    memory = load_memory(memory_file)

    config = ChatbotConfig(model_name=model_name, device=device)
    chatbot = AnimeChatbot(config=config, persona=persona, memory=memory)
    chatbot.preload_greeting()
    console.print("Отправляем тестовое сообщение модели...")
    prompt_text = chatbot._build_prompt(prompt)  # type: ignore[attr-defined]
    raw_reply = chatbot._generate(prompt_text)  # type: ignore[attr-defined]
    reply = chatbot._extract_reply(raw_reply)  # type: ignore[attr-defined]
    console.print(f"Сырый ответ модели: {raw_reply!r}")
    console.print(f"Обработанный ответ: {reply}")
    if reply.strip():
        chatbot.history.append((prompt, reply))
        chatbot._trim_history()  # type: ignore[attr-defined]
    if reply.strip() == config.fallback_reply.strip():
        console.print(
            "[yellow]Получен запасной ответ. Проверьте, что модель скачана, и попробуйте изменить prompt или параметры генерации."
        )
        raise typer.Exit(code=1)
    console.print("[green]Модель ответила без запасного сообщения — всё готово!")


if __name__ == "__main__":
    app()
