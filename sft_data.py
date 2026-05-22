"""Streaming dataset pipeline for Gemma-3 SFT."""

from dataclasses import dataclass
from typing import Callable, Optional
from datasets import load_dataset, interleave_datasets, IterableDataset
from transformers import AutoTokenizer


@dataclass
class DatasetConfig:
    hf_path: str
    hf_name: Optional[str] = None
    split: str = "train"
    sampling_weight: float = 1.0
    max_samples: Optional[int] = None
    formatter: Optional[Callable] = None


def format_smoltalk2(example: dict, tokenizer: AutoTokenizer, max_length: int) -> dict:
    messages = list(example["messages"])
    custom_instructions = example.get("chat_template_kwargs", {}).get("custom_instructions", "")
    has_system = any(m.get("role") == "system" for m in messages)
    if custom_instructions and not has_system:
        messages.insert(0, {"role": "system", "content": custom_instructions})
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
    return {"text": text, "_n_tokens": len(tokenizer.encode(text))}


def format_input_output(example: dict, tokenizer: AutoTokenizer, max_length: int) -> dict:
    messages = [
        {"role": "user", "content": example["input"]},
        {"role": "assistant", "content": example["output"]},
    ]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
    return {"text": text, "_n_tokens": len(tokenizer.encode(text))}


def format_glm(example: dict, tokenizer: AutoTokenizer, max_length: int) -> dict:
    convos = example["conversations"]
    messages = [
        {"role": "user", "content": convos[0]["value"]},
        {"role": "assistant", "content": convos[1]["value"]},
    ]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
    return {"text": text, "_n_tokens": len(tokenizer.encode(text))}


def build_streaming_dataset(
    configs: list[DatasetConfig],
    tokenizer: AutoTokenizer,
    max_length: int = 4096,
    shuffle_buffer: int = 10000,
) -> IterableDataset:
    streams = []
    for cfg in configs:
        ds = load_dataset(cfg.hf_path, cfg.hf_name, split=cfg.split, streaming=True)
        if cfg.formatter is not None:
            ds = ds.map(lambda x, f=cfg.formatter, t=tokenizer, m=max_length: f(x, t, m))
        ds = ds.filter(lambda x, m=max_length: x["_n_tokens"] < m)
        if cfg.max_samples is not None:
            ds = ds.take(cfg.max_samples)
        streams.append(ds)

    weights = [c.sampling_weight for c in configs]
    combined = interleave_datasets(streams, probabilities=weights, stopping_strategy="all_exhausted")
    combined = combined.shuffle(buffer_size=shuffle_buffer, seed=42)
    return combined
