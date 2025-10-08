"""Utilities for ingesting data from dvach/2ch.hk threads."""
from __future__ import annotations

import html
import json
import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Sequence

import requests
from bs4 import BeautifulSoup

logger = logging.getLogger(__name__)

MAKABA_PATH = "/makaba/makaba.fcgi"
DEFAULT_MAKABA_HOSTS: Sequence[str] = (
    "https://2ch.hk",
    "https://2ch.pm",
    "https://2ch.org",
)


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
        base_urls: Optional[Iterable[str]] = None,
        cookie: Optional[str] = None,
        user_agent: Optional[str] = None,
    ) -> None:
        self.session = session or requests.Session()
        host_candidates: Sequence[str]
        if base_urls:
            host_candidates = list(base_urls)
        elif base_url:
            host_candidates = [base_url]
        else:
            host_candidates = DEFAULT_MAKABA_HOSTS
        self.base_urls: List[str] = [self._normalize_base_url(host) for host in host_candidates]
        # Some dvach endpoints return HTML unless a "real" user agent is provided.
        default_agent = "anime-chatbot/0.1 (+https://github.com/testtonconnect)"
        self.session.headers.setdefault("User-Agent", user_agent or default_agent)
        self.session.headers.setdefault("Accept", "application/json")
        self.session.headers.setdefault("X-Requested-With", "XMLHttpRequest")
        if cookie:
            self.session.headers.setdefault("Cookie", cookie)

    @staticmethod
    def _normalize_base_url(host: str) -> str:
        host = host.strip()
        if not host:
            raise ValueError("Пустой адрес Makaba")
        if host.endswith(MAKABA_PATH):
            return host
        if host.startswith("http://") or host.startswith("https://"):
            base = host.rstrip("/")
        else:
            base = f"https://{host.strip('/')}"
        return f"{base}{MAKABA_PATH}"

    def fetch_thread(self, board: str, thread: str) -> dict:
        params = {
            "task": "get_thread",
            "board": board,
            "thread": thread,
            "json": "1",
        }
        errors: List[str] = []
        for base_url in self.base_urls:
            try:
                response = self.session.get(base_url, params=params, timeout=20)
                response.raise_for_status()
            except requests.RequestException as exc:
                errors.append(f"{base_url}: {exc}")
                continue

            try:
                payload = response.json()
            except ValueError:
                hint = self._describe_html_issue(response.text)
                logger.debug("dvach API non-JSON response from %s: %s", base_url, response.text[:500])
                errors.append(f"{base_url}: {hint}")
                continue

            if not payload:
                errors.append(f"{base_url}: пустой ответ")
                continue

            return payload

        error_text = (
            "Не удалось получить JSON от 2ch. Попробуйте указать --api-host, "
            "сохранить страницу вручную (--html-file/--json-file) или повторить запрос позже."
        )
        if errors:
            error_text = f"{error_text}\nПодробности:\n- " + "\n- ".join(errors)
        raise DvachAPIError(error_text)

    def _describe_html_issue(self, text: str) -> str:
        lowered = text.lower()
        if "captcha" in lowered:
            return "требуется пройти капчу"
        if "2ch.org" in lowered or "location.replace('https://2ch.org" in lowered:
            return "редирект на 2ch.org — укажите --api-host 2ch.org"
        if "cloudflare" in lowered:
            return "ответ с защитой Cloudflare"
        return "получен HTML вместо JSON"

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
