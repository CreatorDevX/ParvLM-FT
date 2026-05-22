"""Streaming dataset pipeline for Qwen3.5 SFT.

Each dataset config has a formatter that converts examples to formatted text
using the Qwen3.5 chat template. Then filters to <4096 tokens.
"""

from dataclasses import dataclass, field
from typing import Callable, Optional
from datasets import load_dataset, interleave_datasets, IterableDataset
from transformers import AutoProcessor


@dataclass
class DatasetConfig:
    hf_path: str
    hf_name: Optional[str] = None
    split: str = "train"
    sampling_weight: float = 1.0
    max_samples: Optional[int] = None
    formatter: Optional[Callable] = None  # (example) -> {"text": str}


def format_smoltalk2(example: dict, processor: AutoProcessor) -> dict:
    raw_kwargs = example.get("chat_template_kwargs", {})

    # Qwen3.5 template only understands enable_thinking and tools.
    # custom_instructions needs to go into messages as a system message.
    messages = list(example["messages"])
    custom_instructions = raw_kwargs.get("custom_instructions", "")
    has_system = any(m.get("role") == "system" for m in messages)
    if custom_instructions and not has_system:
        messages.insert(0, {"role": "system", "content": custom_instructions})

    valid_kwargs = {}
    if "enable_thinking" in raw_kwargs:
        valid_kwargs["enable_thinking"] = raw_kwargs["enable_thinking"]

    text = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False,
        **valid_kwargs,
    )
    return {"text": text}


def format_input_output(example: dict, processor: AutoProcessor,
                        enable_thinking: bool = True) -> dict:
    messages = [
        {"role": "user", "content": example["input"]},
        {"role": "assistant", "content": example["output"]},
    ]
    text = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False,
        enable_thinking=enable_thinking,
    )
    return {"text": text}


def format_glm(example: dict, processor: AutoProcessor) -> dict:
    convos = example["conversations"]
    messages = [
        {"role": "user", "content": convos[0]["value"]},
        {"role": "assistant", "content": convos[1]["value"]},
    ]
    text = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False,
        enable_thinking=True,
    )
    return {"text": text}


def build_streaming_dataset(
    configs: list[DatasetConfig],
    processor: AutoProcessor,
    max_length: int = 4096,
    shuffle_buffer: int = 10000,
) -> IterableDataset:
    streams = []
    for cfg in configs:
        ds = load_dataset(
            cfg.hf_path,
            cfg.hf_name,
            split=cfg.split,
            streaming=True,
        )

        if cfg.formatter is not None:
            ds = ds.map(lambda x, f=cfg.formatter, p=processor: f(x, p))

        text_filter = lambda x, p=processor, m=max_length: len(
            p(x["text"], return_tensors=None)["input_ids"]
        ) < m
        ds = ds.filter(text_filter)

        ds = ds.shuffle(buffer_size=shuffle_buffer, seed=42)

        if cfg.max_samples is not None:
            ds = ds.take(cfg.max_samples)

        streams.append(ds)

    if len(streams) == 1:
        return streams[0]

    weights = [c.sampling_weight for c in configs]
    return interleave_datasets(
        streams,
        probabilities=weights,
        stopping_strategy="all_exhausted",
    )
