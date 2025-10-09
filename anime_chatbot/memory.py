"""Simple JSONL-backed factual memory for the chatbot."""
from __future__ import annotations

import json
import os
import re
from typing import Dict, List

FACT_PATTERNS = [
    (re.compile(r"\bменя зовут\s+([A-Za-zА-Яа-яЁё\-]+)", re.I), "bot_name"),
    (re.compile(r"\bя\s+(?:твоя|ваша)\s+вайфу\b", re.I), "relation_waifu"),
    (re.compile(r"\bты\s+мой\s+любимый\s+человек\b", re.I), "affection"),
]


class JsonlMemory:
    """Persist simple structured facts extracted from dialogue."""

    def __init__(self, path: str = "data/memory.jsonl") -> None:
        self.path = path
        directory = os.path.dirname(path)
        if directory:
            os.makedirs(directory, exist_ok=True)
        if not os.path.exists(path):
            open(path, "w", encoding="utf-8").close()

    def add_fact(self, fact: Dict) -> None:
        with open(self.path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(fact, ensure_ascii=False) + "\n")

    def load(self) -> List[Dict]:
        facts: List[Dict] = []
        if not os.path.exists(self.path):
            return facts
        with open(self.path, "r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    facts.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
        return facts

    def extract_and_store(self, text: str) -> None:
        for regex, kind in FACT_PATTERNS:
            match = regex.search(text)
            if match:
                value = match.group(1) if match.groups() else True
                self.add_fact({"kind": kind, "value": value})

    def retrieve(self, query: str, k: int = 5) -> List[str]:
        facts = self.load()[-k:]
        return [f"• {fact['kind']}: {fact.get('value', True)}" for fact in facts]


__all__ = ["JsonlMemory", "FACT_PATTERNS"]
