"""Fine-tuning utilities for the anime chatbot."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List

from datasets import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)

from .config import ChatbotConfig


@dataclass
class TrainingSample:
    prompt: str
    completion: str

    @classmethod
    def from_dialogue(cls, system_prompt: str, user: str, assistant: str) -> "TrainingSample":
        prompt = f"System: {system_prompt}\nUser: {user}\nAssistant:"
        return cls(prompt=prompt, completion=assistant)


def load_jsonl(path: Path) -> List[dict]:
    """Load JSONL file into list of dictionaries."""

    rows: List[dict] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))  # type: ignore[name-defined]
    return rows


def build_dataset(samples: Iterable[TrainingSample]) -> Dataset:
    """Create Hugging Face dataset from training samples."""

    data = {"prompt": [], "completion": []}
    for sample in samples:
        data["prompt"].append(sample.prompt)
        data["completion"].append(sample.completion)
    return Dataset.from_dict(data)


def fine_tune(
    dataset: Dataset,
    config: ChatbotConfig,
    output_dir: Path,
    epochs: int = 2,
    lr: float = 5e-5,
) -> None:
    """Run a simple causal LM fine-tuning."""

    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    model = AutoModelForCausalLM.from_pretrained(config.model_name)

    def tokenize(batch):
        inputs = tokenizer(
            [p + " " + c for p, c in zip(batch["prompt"], batch["completion"])],
            truncation=True,
            padding="max_length",
            max_length=256,
            return_tensors="pt",
        )
        inputs["labels"] = inputs["input_ids"].clone()
        return inputs

    tokenized = dataset.map(tokenize, batched=True, remove_columns=dataset.column_names)

    args = TrainingArguments(
        output_dir=str(output_dir),
        per_device_train_batch_size=2,
        num_train_epochs=epochs,
        learning_rate=lr,
        fp16=True,
        logging_steps=10,
        save_strategy="epoch",
    )

    trainer = Trainer(model=model, args=args, train_dataset=tokenized)
    trainer.train()


__all__ = [
    "TrainingSample",
    "load_jsonl",
    "build_dataset",
    "fine_tune",
]
