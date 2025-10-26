"""Anime-style chatbot built on top of Hugging Face transformers."""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

from .config import ChatbotConfig
from .persona import PersonaSettings
from .memory import DvachMemory

Conversation = List[Tuple[str, str]]


logger = logging.getLogger(__name__)


@dataclass
class AnimeChatbot:
    """Utility class wrapping a text-generation pipeline."""

    config: ChatbotConfig
    persona: Optional[PersonaSettings] = None
    memory: Optional[DvachMemory] = None
    history: Conversation = None  # type: ignore

    def __post_init__(self) -> None:
        persona_prompt = (
            self.persona.build_system_prompt(self.config.system_prompt)
            if self.persona
            else self.config.system_prompt
        )
        self.system_prompt = persona_prompt.strip()
        self.history = []
        device = 0 if (self.config.device == "cuda" or self.config.device == "gpu") else -1
        if self.config.device and self.config.device not in {"cpu", "cuda", "gpu"}:
            device = self.config.device  # allow manual index
        self.tokenizer = AutoTokenizer.from_pretrained(self.config.model_name)
        self.model = AutoModelForCausalLM.from_pretrained(self.config.model_name)
        self.generator = pipeline(
            "text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
            device=device,
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    # -------------------- Public API --------------------
    def reset_history(self) -> None:
        self.history.clear()

    def chat(self, message: str) -> str:
        """Generate a reply to a user message and update conversation history."""

        prompt = self._build_prompt(message)
        response = self._generate(prompt)
        reply = self._extract_reply(prompt, response)
        if reply == self.config.fallback_reply:
            logger.debug("Primary generation returned fallback reply, retrying with minimal prompt")
            simple_prompt = self._build_minimal_prompt(message)
            alt_response = self._generate(simple_prompt)
            alt_reply = self._extract_reply(simple_prompt, alt_response)
            if alt_reply != self.config.fallback_reply:
                reply = alt_reply
            else:
                logger.debug("Minimal prompt also returned fallback reply")
        self.history.append((message, reply))
        self._trim_history()
        return reply

    def preload_greeting(self) -> Tuple[str, str]:
        """Return the welcome message and store it in history."""

        welcome = self.config.welcome_message
        self.history.append(("system", welcome))
        return ("system", welcome)

    # -------------------- Internal helpers --------------------
    def _trim_history(self) -> None:
        if len(self.history) > self.config.history_turns:
            self.history = self.history[-self.config.history_turns :]

    def _build_prompt(self, message: str) -> str:
        conversation_lines: List[str] = [f"System: {self.system_prompt}"]
        if self.memory:
            snippets = self.memory.retrieve(message, limit=self.config.memory_snippets)
            if snippets:
                for idx, snippet in enumerate(snippets, start=1):
                    conversation_lines.append(f"Context {idx}: {snippet}")
        for user, bot in self.history:
            if user == "system":
                conversation_lines.append(f"Assistant: {bot}")
            else:
                conversation_lines.append(f"User: {user}")
                conversation_lines.append(f"Assistant: {bot}")
        conversation_lines.append(f"User: {message}")
        conversation_lines.append("Assistant:")
        return "\n".join(conversation_lines)

    def _build_minimal_prompt(self, message: str) -> str:
        return f"System: {self.system_prompt}\nUser: {message}\nAssistant:"

    def _generate(self, prompt: str) -> str:
        gen_cfg = self.config.generation
        outputs = self.generator(
            prompt,
            max_new_tokens=gen_cfg.max_new_tokens,
            temperature=gen_cfg.temperature,
            top_p=gen_cfg.top_p,
            repetition_penalty=gen_cfg.repetition_penalty,
            pad_token_id=self.tokenizer.eos_token_id,
            return_full_text=False,
        )
        return outputs[0]["generated_text"]

    def _extract_reply(self, prompt: str, full_text: str) -> str:
        generated = full_text
        if not generated:
            logger.debug("Model returned empty string for prompt")
        stop_tokens = self.config.generation.stop_tokens
        for token in stop_tokens:
            if token in generated:
                generated = generated.split(token)[0]
        cleaned = generated.strip()
        if not cleaned:
            cleaned = self.config.fallback_reply
        return cleaned

    def export_history(self) -> Sequence[Tuple[str, str]]:
        """Return a copy of the current conversation."""

        return list(self.history)
