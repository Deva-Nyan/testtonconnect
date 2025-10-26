"""Lightweight memory component backed by dvach threads."""
from __future__ import annotations

import json
import random
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, FrozenSet


_WORD_RE = re.compile(r"\w+", flags=re.UNICODE)


@dataclass
class MemoryPost:
    """Simplified representation of a dvach post used for recall."""

    post_id: str
    text: str
    tokens: FrozenSet[str]


class DvachMemory:
    """Load dvach posts and retrieve snippets relevant to a prompt."""

    def __init__(self, posts: Iterable[MemoryPost]) -> None:
        self._posts: List[MemoryPost] = list(posts)

    @classmethod
    def from_jsonl(cls, path: Path) -> "DvachMemory":
        posts: List[MemoryPost] = []
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                data = json.loads(line)
                text = str(data.get("comment", "")).strip()
                if not text:
                    continue
                post_id = str(data.get("post_id") or data.get("num") or data.get("id") or "")
                tokens = frozenset(_tokenize(text))
                if not tokens:
                    continue
                posts.append(MemoryPost(post_id=post_id, text=text, tokens=tokens))
        return cls(posts)

    def retrieve(self, query: str, limit: int = 3) -> List[str]:
        if not self._posts or limit <= 0:
            return []
        query_tokens = set(_tokenize(query))
        if not query_tokens:
            return [self._posts[0].text]
        scored = []
        for post in self._posts:
            overlap = len(query_tokens.intersection(post.tokens))
            if overlap:
                scored.append((overlap, post))
        if not scored:
            sample = random.sample(self._posts, k=min(limit, len(self._posts)))
            return [post.text for post in sample]
        scored.sort(key=lambda item: item[0], reverse=True)
        top_posts = [post.text for _, post in scored[:limit]]
        return top_posts


def _tokenize(text: str) -> List[str]:
    return [token.lower() for token in _WORD_RE.findall(text)]


__all__ = ["DvachMemory"]
