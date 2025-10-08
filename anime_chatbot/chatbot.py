"""Anime-style chatbot built on top of Hugging Face transformers."""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.generation.logits_process import (
    LogitsProcessorList,
    NoBadWordsLogitsProcessor,
)

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
        if self.tokenizer.pad_token is None and self.tokenizer.eos_token is not None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        if self.tokenizer.pad_token_id is not None:
            self.model.config.pad_token_id = self.tokenizer.pad_token_id
        if self.tokenizer.eos_token_id is not None:
            self.model.config.eos_token_id = self.tokenizer.eos_token_id
        tokenizer_vocab = len(self.tokenizer)
        if tokenizer_vocab != self.model.config.vocab_size:
            added = getattr(self.tokenizer, "added_tokens_encoder", {})
            if added:
                self.model.resize_token_embeddings(tokenizer_vocab)
            else:
                self.model.config.vocab_size = tokenizer_vocab
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
        eos = self.tokenizer.eos_token or "\n"
        conversation_lines: List[str] = []
        if self.system_prompt:
            conversation_lines.append(f"{self.system_prompt}{eos}")
        if self.memory:
            snippets = self.memory.retrieve(message, limit=self.config.memory_snippets)
            if snippets:
                for idx, snippet in enumerate(snippets, start=1):
                    conversation_lines.append(f"Контекст {idx}: {snippet}{eos}")
        for user, bot in self.history:
            if user == "system":
                conversation_lines.append(f"Bot: {bot}{eos}")
            else:
                conversation_lines.append(f"User: {user}{eos}")
                conversation_lines.append(f"Bot: {bot}{eos}")
        conversation_lines.append(f"User: {message}{eos}")
        conversation_lines.append("Bot:")
        return "".join(conversation_lines)

    def _generate(self, prompt: str) -> str:
        gen_cfg = self.config.generation
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            add_special_tokens=False,
        ).to(self._device)
        pad_id = (
            self.tokenizer.pad_token_id
            or self.model.config.pad_token_id
            or self.tokenizer.eos_token_id
            or self.model.config.eos_token_id
        )
        if pad_id is None:
            raise RuntimeError("pad_token_id could not be resolved for generation")
        self.model.config.pad_token_id = pad_id
        if self.model.config.eos_token_id is None:
            self.model.config.eos_token_id = pad_id
        bad_patterns = [
            "@@ПЕРВЫЙ@@",
            "@@ВТОРОЙ@@",
            "FIRST@@",
            "SECOND@@",
            "@@",
        ]
        bad_word_ids = []
        for pattern in bad_patterns:
            token_ids = self.tokenizer.encode(pattern, add_special_tokens=False)
            if token_ids:
                bad_word_ids.append(token_ids)
        logits_processor = LogitsProcessorList()
        if bad_word_ids:
            logits_processor.append(
                NoBadWordsLogitsProcessor(
                    bad_words_ids=bad_word_ids,
                    eos_token_id=self.model.config.eos_token_id,
                )
            )
        generate_kwargs = {
            "do_sample": True,
            "max_new_tokens": gen_cfg.max_new_tokens,
            "temperature": gen_cfg.temperature,
            "top_p": gen_cfg.top_p,
            "repetition_penalty": gen_cfg.repetition_penalty,
            "no_repeat_ngram_size": gen_cfg.no_repeat_ngram_size,
            "pad_token_id": pad_id,
            "eos_token_id": self.model.config.eos_token_id,
        }
        if len(logits_processor) > 0:
            generate_kwargs["logits_processor"] = logits_processor
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                **generate_kwargs,
            )
        generated_ids = outputs[0][inputs["input_ids"].shape[-1] :]
        if generated_ids.numel() == 0:
            return ""
        text = self.tokenizer.decode(generated_ids, skip_special_tokens=True)
        return text

    def _extract_reply(self, generated: str) -> str:
        text = generated.strip()
        stop_tokens = self.config.generation.stop_tokens
        for token in stop_tokens:
            if token and token in text:
                candidate = text.split(token)[0].strip()
                if candidate:
                    text = candidate
                    break
        if not text and generated:
            text = generated.strip()
        if not text:
            return self._fallback_reply or self.config.fallback_reply
        text = self._clean_reply_text(text)
        if not text:
            return self._fallback_reply or self.config.fallback_reply
        return text

    def export_history(self) -> Sequence[Tuple[str, str]]:
        """Return a copy of the current conversation."""

        return list(self.history)

    def _clean_reply_text(self, text: str) -> str:
        cleaned = text.strip()
        if not cleaned:
            return ""
        cleaned = re.sub(
            r"@@\s*(ПЕРВЫЙ|ВТОРОЙ|FIRST|SECOND)\s*@@",
            "",
            cleaned,
            flags=re.IGNORECASE,
        )
        cleaned = re.sub(r"nya_[a-zA-Z0-9_]+", "ня~", cleaned)
        cleaned = re.sub(r"@@", "", cleaned)
        cleaned = re.sub(r"[~]{3,}", "~~", cleaned)
        cleaned = re.sub(
            r"(ня+~?)(\s*\1){2,}",
            r"\1 \1",
            cleaned,
            flags=re.IGNORECASE,
        )
        cleaned = cleaned.split("\n\n")[0].strip()
        cleaned = cleaned[:400].rstrip()
        return cleaned.strip()
