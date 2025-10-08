"""CLI tool for ingesting dvach threads."""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console

from anime_chatbot.data_ingestion import (
    DvachAPIError,
    DvachClient,
    append_jsonl,
    DVACH_API,
    export_jsonl,
    load_thread_from_html,
    load_thread_from_json,
)

app = typer.Typer(help="Скачать тред 2ch.hk и сохранить в JSONL")
console = Console()


def configure_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(level=level, format="[%(levelname)s] %(message)s")


def ensure_single_source(html_file: Optional[Path], json_file: Optional[Path]) -> None:
    if html_file and json_file:
        raise typer.BadParameter("Используйте либо --html-file, либо --json-file, но не оба сразу")


@app.command()
def main(
    board: str = typer.Option(..., prompt=True, help="Доска, например b"),
    thread: str = typer.Option(..., prompt=True, help="Номер треда"),
    out: Path = typer.Option(Path("data/dvach.jsonl"), help="Файл для записи"),
    append: bool = typer.Option(False, help="Добавить к существующему файлу"),
    html_file: Optional[Path] = typer.Option(
        None,
        exists=True,
        help="HTML-файл, сохранённый из браузера (альтернатива API)",
    ),
    json_file: Optional[Path] = typer.Option(
        None,
        exists=True,
        help="JSON с данными Makaba, сохранённый вручную",
    ),
    cookie: Optional[str] = typer.Option(
        None,
        help="Значение Cookie для обхода капчи (скопируйте из браузера)",
    ),
    user_agent: Optional[str] = typer.Option(
        None,
        help="Переопределить User-Agent при запросе Makaba",
    ),
    base_url: Optional[str] = typer.Option(
        None,
        help="Альтернативный URL Makaba (например https://2ch.org/makaba/makaba.fcgi)",
    ),
    proxy: Optional[str] = typer.Option(
        None,
        help="HTTP(S)/SOCKS прокси в формате URL, если требуется обход блокировок",
    ),
    verbose: bool = typer.Option(False, help="Включить расширенные логи"),
) -> None:
    configure_logging(verbose)
    ensure_single_source(html_file, json_file)

    if html_file:
        console.print(f"Читаем HTML из {html_file}...")
        loader = load_thread_from_html
    elif json_file:
        console.print(f"Читаем JSON из {json_file}...")
        loader = load_thread_from_json
    else:
        target_url = base_url or DVACH_API
        console.print(f"Скачиваем /{board}/{thread} через {target_url}...")
        client = DvachClient(
            base_url=target_url,
            cookie=cookie,
            user_agent=user_agent,
            proxy=proxy,
        )
        try:
            records = list(client.iter_posts(board=board, thread=thread))
        except DvachAPIError as exc:
            console.print(f"[red]{exc}")
            if verbose:
                console.print("Попробуйте сохранить HTML/JSON вручную и воспользоваться --html-file или --json-file")
            raise typer.Exit(code=1)
        return _write_records(records, out, append)

    try:
        records = list(loader(html_file or json_file, board=board, thread=thread))  # type: ignore[arg-type]
    except DvachAPIError as exc:
        console.print(f"[red]{exc}")
        raise typer.Exit(code=1)

    return _write_records(records, out, append)


def _write_records(records, out: Path, append: bool) -> None:
    if not records:
        console.print("[yellow]Нет постов для сохранения")
        raise typer.Exit(code=1)
    if append:
        count = append_jsonl(records, out)
    else:
        count = export_jsonl(records, out)
    console.print(f"[green]Сохранено {count} постов в {out}")


if __name__ == "__main__":
    app()
