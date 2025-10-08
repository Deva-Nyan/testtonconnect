"""Anime-style chatbot built on top of Hugging Face transformers."""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

from transformers import AutoModelForCausalLM, AutoTokenizer

try:
    import torch
except ImportError as exc:  # pragma: no cover - torch is optional at import time
    raise RuntimeError(
        "Torch must be installed to use AnimeChatbot. Запустите scripts/bootstrap_env.*"
    ) from exc

from .config import ChatbotConfig
from .persona import PersonaSettings
from .memory import DvachMemory

Conversation = List[Tuple[str, str]]


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
        if self.config.device and self.config.device not in {"cpu", "cuda", "gpu"}:
            device_target = self.config.device
        elif self.config.device in {"cuda", "gpu"}:
            device_target = "cuda"
        else:
            device_target = "cpu"
        self.tokenizer = AutoTokenizer.from_pretrained(self.config.model_name)
        self.model = AutoModelForCausalLM.from_pretrained(self.config.model_name)
        self._device = torch.device(device_target)
        try:
            self.model.to(self._device)
        except RuntimeError:
            # Fallback to CPU if CUDA requested but unavailable.
            self._device = torch.device("cpu")
            self.model.to(self._device)
        self.model.eval()
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self._fallback_reply = self.config.fallback_reply.strip()

    # -------------------- Public API --------------------
    def reset_history(self) -> None:
        self.history.clear()

    def chat(self, message: str) -> str:
        """Generate a reply to a user message and update conversation history."""

        prompt = self._build_prompt(message)
        response = self._generate(prompt)
        reply = self._extract_reply(response)
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

    def _generate(self, prompt: str) -> str:
        gen_cfg = self.config.generation
        prepared_prompt = f"{prompt}{self.tokenizer.eos_token}" if self.tokenizer.eos_token else prompt
        inputs = self.tokenizer(
            prepared_prompt,
            return_tensors="pt",
            add_special_tokens=False,
        ).to(self._device)
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                do_sample=True,
                max_new_tokens=gen_cfg.max_new_tokens,
                min_new_tokens=gen_cfg.min_new_tokens,
                temperature=gen_cfg.temperature,
                top_p=gen_cfg.top_p,
                repetition_penalty=gen_cfg.repetition_penalty,
                pad_token_id=self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )
        generated_ids = outputs[0][inputs["input_ids"].shape[-1] :]
        if generated_ids.numel() == 0:
            return ""
        text = self.tokenizer.decode(generated_ids, skip_special_tokens=True)
        return text

    def _extract_reply(self, generated: str) -> str:
        stripped = generated.strip()
        cleaned = stripped
        stop_tokens = self.config.generation.stop_tokens
        for token in stop_tokens:
            if token in cleaned:
                before = cleaned.split(token)[0].strip()
                if before:
                    cleaned = before
                    break
        if not cleaned:
            cleaned = stripped
        if not cleaned:
            cleaned = self._fallback_reply or self.config.fallback_reply
        return cleaned

    def export_history(self) -> Sequence[Tuple[str, str]]:
        """Return a copy of the current conversation."""

        return list(self.history)
