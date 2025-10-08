"""Utilities for ingesting data from dvach/2ch.hk threads."""
from __future__ import annotations

import html
import json
import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

import requests
from bs4 import BeautifulSoup

logger = logging.getLogger(__name__)

DVACH_PRIMARY_API = "https://2ch.hk/makaba/makaba.fcgi"
DVACH_FALLBACK_API = "https://2ch.org/makaba/makaba.fcgi"


class DvachAPIError(RuntimeError):
    """Raised when the dvach API returns an unexpected response."""



@dataclass
class PostRecord:
    """Lightweight representation of a dvach post."""

    board: str
    thread: str
    post_id: str
    comment: str

    def to_json(self) -> str:
        return json.dumps(self.__dict__, ensure_ascii=False)


class DvachClient:
    """Client for fetching threads from 2ch.hk."""

    def __init__(
        self,
        session: Optional[requests.Session] = None,
        base_url: Optional[str] = None,
        cookie: Optional[str] = None,
        user_agent: Optional[str] = None,
        extra_base_urls: Optional[Iterable[str]] = None,
    ) -> None:
        self.session = session or requests.Session()
        configured_urls = []
        if base_url:
            configured_urls.append(base_url)
        configured_urls.append(DVACH_PRIMARY_API)
        configured_urls.append(DVACH_FALLBACK_API)
        if extra_base_urls:
            configured_urls.extend(extra_base_urls)
        # preserve order while removing duplicates
        seen = set()
        self.base_urls = []
        for url in configured_urls:
            if url and url not in seen:
                self.base_urls.append(url)
                seen.add(url)
        # Some dvach endpoints return HTML unless a "real" user agent is provided.
        default_agent = "anime-chatbot/0.1 (+https://github.com/testtonconnect)"
        self.session.headers.setdefault("User-Agent", user_agent or default_agent)
        self.session.headers.setdefault("Accept", "application/json")
        self.session.headers.setdefault("X-Requested-With", "XMLHttpRequest")
        if cookie:
            self.session.headers.setdefault("Cookie", cookie)

    def fetch_thread(self, board: str, thread: str) -> dict:
        params = {
            "task": "get_thread",
            "board": board,
            "thread": thread,
            "json": "1",
        }
        last_error: Optional[Exception] = None
        for base in self.base_urls:
            try:
                logger.debug("Запрашиваем Makaba %s для /%s/%s", base, board, thread)
                response = self.session.get(base, params=params, timeout=20)
                response.raise_for_status()
            except requests.RequestException as exc:
                logger.debug("Запрос к %s завершился ошибкой: %s", base, exc)
                last_error = exc
                continue

            try:
                payload = response.json()
            except ValueError as exc:
                text_sample = response.text[:2000]
                logger.debug("dvach API non-JSON response from %s: %s", base, text_sample)
                html_lower = text_sample.lower()
                if "captcha" in html_lower or "cloudflare" in html_lower:
                    last_error = DvachAPIError(
                        "Makaba вернул HTML с защитой (капча/Cloudflare). Попробуйте "
                        "сменить IP или использовать сохранённый вручную JSON/HTML."
                    )
                elif "2ch.org" in html_lower and base.rstrip("/").endswith("2ch.hk/makaba/makaba.fcgi"):
                    logger.debug("Ответ указал на редирект 2ch.org, пробуем запасной домен")
                    last_error = exc
                    continue
                else:
                    last_error = DvachAPIError(
                        "Makaba вернул HTML вместо JSON. Попробуйте другой домен или ручную загрузку."
                    )
                continue

            if not payload:
                last_error = DvachAPIError("Makaba вернул пустой ответ")
                continue

            return payload

        if isinstance(last_error, DvachAPIError):
            raise last_error
        raise DvachAPIError(f"Не удалось получить тред /{board}/{thread}: {last_error}")

    def iter_posts(self, board: str, thread: str) -> Iterable[PostRecord]:
        payload = self.fetch_thread(board, thread)
        threads = payload.get("threads") or []
        if not threads:
            raise DvachAPIError("Ответ не содержит постов. Проверьте номер треда.")
        for post in threads[0].get("posts", []):
            comment = clean_comment(post.get("comment", ""))
            if not comment.strip():
                continue
            yield PostRecord(
                board=board,
                thread=thread,
                post_id=str(post.get("num", "")),
                comment=comment,
            )


def clean_comment(raw_html: str) -> str:
    """Convert dvach HTML comment to plain text."""

    text = html.unescape(raw_html)
    text = re.sub(r"<br ?/?>", "\n", text, flags=re.IGNORECASE)
    text = re.sub(r"<[^>]+>", "", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def export_jsonl(records: Iterable[PostRecord], path: Path) -> int:
    """Write records to JSON Lines file and return count."""

    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(record.to_json() + "\n")
            count += 1
    return count


def load_thread_from_json(path: Path, board: str, thread: str) -> Iterable[PostRecord]:
    """Load dvach posts from a previously downloaded JSON payload."""

    payload = json.loads(path.read_text(encoding="utf-8"))
    threads = []
    if isinstance(payload, dict):
        if "threads" in payload:
            threads = payload.get("threads") or []
        elif "posts" in payload:
            threads = [{"posts": payload.get("posts", [])}]
    if not threads:
        raise DvachAPIError("Файл JSON не содержит ожидаемых данных 2ch.hk")
    for post in threads[0].get("posts", []):
        comment = clean_comment(post.get("comment", ""))
        if not comment:
            continue
        yield PostRecord(
            board=board,
            thread=thread,
            post_id=str(post.get("num", "")),
            comment=comment,
        )


def load_thread_from_html(path: Path, board: str, thread: str) -> Iterable[PostRecord]:
    """Parse dvach HTML thread saved from a browser."""

    soup = BeautifulSoup(path.read_text(encoding="utf-8"), "html.parser")
    post_nodes = soup.select('[id^="post-"]')
    if not post_nodes:
        raise DvachAPIError(
            "Не удалось найти посты в HTML. Убедитесь, что сохранена страница треда 2ch.hk."
        )
    for node in post_nodes:
        post_id = node.get("id", "").replace("post-", "")
        if not post_id:
            continue
        message = node.select_one(".post__message") or node.select_one(".post-message")
        if not message:
            continue
        comment_html = message.decode_contents()
        comment = clean_comment(comment_html)
        if not comment:
            continue
        yield PostRecord(board=board, thread=thread, post_id=post_id, comment=comment)


def append_jsonl(records: Iterable[PostRecord], path: Path) -> int:
    """Append records to JSON Lines file."""

    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("a", encoding="utf-8") as f:
        for record in records:
            f.write(record.to_json() + "\n")
            count += 1
    return count
