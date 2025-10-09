"""Anime-style chatbot built on top of Hugging Face transformers."""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.generation.logits_process import LogitsProcessorList, NoBadWordsLogitsProcessor
from transformers.generation.stopping_criteria import StoppingCriteria, StoppingCriteriaList

try:
    import torch
except ImportError as exc:  # pragma: no cover - torch is optional at import time
    raise RuntimeError(
        "Torch must be installed to use AnimeChatbot. Запустите scripts/bootstrap_env.*"
    ) from exc

from .config import ChatbotConfig
from .consistency import enforce_consistency
from .memory import JsonlMemory
from .summary import make_running_summary


_ROLE_REGEX = re.compile(
    r"(?:^|\n)\s*(?:User|Bot|Пользователь|Бот|[A-ZА-ЯЁ][\w'`~\-]{1,20})\s*:\s*$",
    re.IGNORECASE,
)


class _StopOnRoleCue(StoppingCriteria):
    """Stop generation once the model starts emitting a new speaker tag."""

    def __init__(self, tokenizer, tail_tokens: int = 80):
        self._tokenizer = tokenizer
        self._tail_tokens = tail_tokens

    def __call__(self, input_ids, scores, **kwargs) -> bool:  # type: ignore[override]
        if input_ids.size(0) == 0:
            return False
        tail = input_ids[0][-self._tail_tokens :].tolist()
        if not tail:
            return False
        decoded = self._tokenizer.decode(tail)
        return bool(_ROLE_REGEX.search(decoded))


@dataclass
class AnimeChatbot:
    """Utility class wrapping a text-generation pipeline."""

    config: ChatbotConfig
    memory: Optional[JsonlMemory] = None
    history: List[str] = field(default_factory=list)
    running_summary: Optional[str] = None

    def __post_init__(self) -> None:
        self.system_prompt = self.config.system_prompt.strip()
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
            self._device = torch.device("cpu")
            self.model.to(self._device)
        self.model.eval()
        if self.tokenizer.pad_token is None and self.tokenizer.eos_token is not None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        pad_id = self.tokenizer.pad_token_id or self.tokenizer.eos_token_id
        if pad_id is not None:
            self.model.config.pad_token_id = pad_id
            if self.model.config.eos_token_id is None:
                self.model.config.eos_token_id = pad_id
        elif self.model.config.eos_token_id is not None:
            self.model.config.pad_token_id = self.model.config.eos_token_id
        tokenizer_vocab = len(self.tokenizer)
        model_vocab = getattr(self.model.config, "vocab_size", tokenizer_vocab)
        if tokenizer_vocab > model_vocab:
            self.model.resize_token_embeddings(
                tokenizer_vocab,
                mean_resizing=False,
            )
        elif tokenizer_vocab < model_vocab:
            self.model.config.vocab_size = tokenizer_vocab
        ban_patterns = [
            "@@",
            "@@ПЕРВЫЙ@@",
            "@@ВТОРОЙ@@",
            "FIRST@@",
            "SECOND@@",
            "Dream:",
            "Bot_didnt_y",
            "Botanicula:",
            "Uzi:",
            "Uzio:",
            "User:",
            "Bot:",
            "Пользователь:",
            "Бот:",
            "Я:",
            "Мы:",
            "Они:",
        ]
        ban_token_sequences: List[List[int]] = []
        for pattern in ban_patterns:
            token_ids = self.tokenizer.encode(pattern, add_special_tokens=False)
            if token_ids:
                ban_token_sequences.append(token_ids)
        self._logits_processors = LogitsProcessorList()
        if ban_token_sequences:
            self._logits_processors.append(
                NoBadWordsLogitsProcessor(
                    bad_words_ids=ban_token_sequences,
                    eos_token_id=self.model.config.eos_token_id,
                )
            )
        self._stopping_criteria = StoppingCriteriaList([_StopOnRoleCue(self.tokenizer)])

    # -------------------- Public API --------------------
    def reset_history(self) -> None:
        self.history.clear()
        self.running_summary = None

    def chat(self, message: str) -> str:
        """Generate a reply to a user message and update conversation history."""

        user_text = message.strip()
        if not user_text:
            return self.config.fallback_reply
        if self.history and len(self.history) % self.config.summary_interval == 0:
            self.running_summary = make_running_summary(self.history, self.running_summary)
        messages = self._build_messages(user_text)
        prompt = self._format_messages(messages)
        generated = self._generate(prompt)
        reply = self._extract_reply(generated)
        reply = enforce_consistency(reply)
        if not reply:
            reply = self.config.fallback_reply
        self.history.extend([user_text, reply])
        if self.memory:
            self.memory.extract_and_store(reply)
        return reply

    def preload_greeting(self) -> Tuple[str, str]:
        """Return the welcome message without affecting dialogue history."""

        welcome = self.config.welcome_message.strip()
        return ("system", welcome)

    def export_history(self) -> Sequence[Tuple[str, str]]:
        """Return a copy of the current conversation as (user, bot) pairs."""

        pairs: List[Tuple[str, str]] = []
        items = self.history
        for idx in range(0, len(items), 2):
            try:
                user_msg = items[idx]
                bot_msg = items[idx + 1]
            except IndexError:
                break
            pairs.append((user_msg, bot_msg))
        return pairs

    # -------------------- Internal helpers --------------------
    def _build_messages(self, user_text: str) -> List[Dict[str, str]]:
        context_parts = [self.system_prompt]
        if self.memory:
            facts = self.memory.retrieve(user_text, k=self.config.memory_snippets)
        else:
            facts = []
        if facts:
            context_parts.append("[Контекст фактов]\n" + "\n".join(facts))
        if self.running_summary:
            context_parts.append("[Короткая сводка]\n" + self.running_summary)
        system_message = "\n\n".join(part for part in context_parts if part)
        messages: List[Dict[str, str]] = [{"role": "system", "content": system_message}]
        tail = self.history[-self.config.history_tail :]
        for idx, utterance in enumerate(tail):
            role = "user" if idx % 2 == 0 else "assistant"
            messages.append({"role": role, "content": utterance})
        messages.append({"role": "user", "content": user_text})
        return messages

    def _format_messages(self, messages: List[Dict[str, str]]) -> str:
        eos = self.tokenizer.eos_token or "\n"
        parts: List[str] = []
        for item in messages:
            content = item["content"].strip()
            if not content:
                continue
            if item["role"] == "system":
                parts.append(f"{content}{eos}")
            elif item["role"] == "user":
                parts.append(f"User: {content}{eos}")
            else:
                parts.append(f"Bot: {content}{eos}")
        parts.append("Bot:")
        return "".join(parts)

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
        generate_kwargs = {
            "do_sample": True,
            "max_new_tokens": gen_cfg.max_new_tokens,
            "temperature": gen_cfg.temperature,
            "top_p": gen_cfg.top_p,
            "top_k": gen_cfg.top_k,
            "repetition_penalty": gen_cfg.repetition_penalty,
            "no_repeat_ngram_size": gen_cfg.no_repeat_ngram_size,
            "pad_token_id": pad_id,
            "eos_token_id": self.model.config.eos_token_id,
        }
        if len(self._logits_processors) > 0:
            generate_kwargs["logits_processor"] = self._logits_processors
        if len(self._stopping_criteria) > 0:
            generate_kwargs["stopping_criteria"] = self._stopping_criteria
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
            return self.config.fallback_reply
        text = self._clean_reply_text(text)
        if not text:
            return self.config.fallback_reply
        return text

    def _clean_reply_text(self, text: str) -> str:
        cleaned = text.strip()
        if not cleaned:
            return ""
        cleaned = cleaned.replace("\r\n", "\n")
        cleaned = self._ensure_single_voice(cleaned)
        cleaned = cleaned.lstrip("-•—–* ")
        cleaned = re.sub(
            r"@@\s*(ПЕРВЫЙ|ВТОРОЙ|FIRST|SECOND)\s*@@",
            "",
            cleaned,
            flags=re.IGNORECASE,
        )
        cleaned = re.sub(r"nya_[a-zA-Z0-9_]+", "ня~", cleaned)
        cleaned = re.sub(r"@@", "", cleaned)
        cleaned = re.sub(r"Bot_didnt_y", "", cleaned)
        cleaned = re.sub(
            r"(?i)(?:^|[\s\-–—])+"
            r"(?:User|Bot|Пользователь|Бот|Dream|Uzi|Uzio|Botanicula|[A-ZА-ЯЁ][\w'`~\-]{1,20})"
            r"\s*:\s*",
            " ",
            cleaned,
        )
        cleaned = re.sub(r"\*[^\n]{0,60}\*", "", cleaned)
        cleaned = re.sub(r"_[^\n]{0,40}_", "", cleaned)
        cleaned = re.sub(r"\([^\n]{0,60}\)", "", cleaned)
        cleaned = re.sub(r"\s*(?:—|–|-)\s*$", "", cleaned)
        cleaned = re.sub(r"[~]{3,}", "~~", cleaned)
        cleaned = re.sub(
            r"(ня+~?)(\s*\1){2,}",
            r"\1 \1",
            cleaned,
            flags=re.IGNORECASE,
        )
        cleaned = re.sub(r"\s*Bot:\s*$", "", cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r"\s*User:\s*$", "", cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r"[A-Za-z_]{10,}\b", "", cleaned)
        cleaned = re.sub(r"\s{2,}", " ", cleaned)
        cleaned = cleaned.split("\n\n")[0].strip()
        cleaned = cleaned[:300].rstrip()
        return cleaned.strip()

    def _ensure_single_voice(self, text: str) -> str:
        """Strip stray speaker prefixes and keep the main utterance only."""

        segments = []
        for raw_line in text.splitlines():
            line = raw_line.strip()
            if not line:
                continue
            line = re.sub(
                r"^(?:[-•—–*\s]*)(?:User|Bot|Пользователь|Бот|Я|Ты|Он|Она|Мы|Они|[A-ZА-ЯЁ][\w'`~\-]{0,20})\s*:\s*",
                "",
                line,
            )
            if line:
                segments.append(line)
        if not segments:
            return ""
        primary = segments[0]
        for segment in segments[1:]:
            if re.match(
                r"^(?:User|Bot|Пользователь|Бот|Я|Ты|Он|Она|Мы|Они|[A-ZА-ЯЁ][\w'`~\-]{0,20})\s*:",
                segment,
                flags=re.IGNORECASE,
            ):
                break
            if not re.match(r"^[*!?_]", segment):
                primary += " " + segment
        primary = re.sub(
            r"(?:^|\s)(?:User|Bot|Пользователь|Бот|Я|Ты|Он|Она|Мы|Они|[A-ZА-ЯЁ][\w'`~\-]{0,20})\s*:\s*",
            " ",
            primary,
        )
        return primary.strip()


__all__ = ["AnimeChatbot"]
