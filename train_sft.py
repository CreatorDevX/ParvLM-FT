"""Qwen3.5-2B SFT with LoRA rank 64 on TPU v5e-8.

Usage (Kaggle notebook):
  model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3.5-2B-Base", ...)
  notebook_launcher(train_sft, (model,), num_processes=8)
"""

from dataclasses import dataclass, field
from typing import Optional
import torch
from transformers import AutoProcessor, AutoModelForCausalLM
from peft import LoraConfig
from trl import SFTTrainer, SFTConfig

from sft_data import (
    DatasetConfig,
    format_smoltalk2,
    format_input_output,
    format_glm,
    build_streaming_dataset,
)


def format_codex(example: dict, processor: AutoProcessor) -> dict:
    return format_input_output(example, processor, enable_thinking=True)


def _make_datasets():
    return [
        DatasetConfig(
            hf_path="HuggingFaceTB/smoltalk2",
            hf_name="SFT",
            split="smoltalk_smollm3_systemchats_30k_no_think",
            sampling_weight=0.09,
            max_samples=10000,
            formatter=format_smoltalk2,
        ),
        DatasetConfig(
            hf_path="HuggingFaceTB/smoltalk2",
            hf_name="SFT",
            split="smoltalk_smollm3_everyday_conversations_no_think",
            sampling_weight=0.02,
            max_samples=None,
            formatter=format_smoltalk2,
        ),
        DatasetConfig(
            hf_path="HuggingFaceTB/smoltalk2",
            hf_name="SFT",
            split="hermes_function_calling_v1_no_think",
            sampling_weight=0.08,
            max_samples=None,
            formatter=format_smoltalk2,
        ),
        DatasetConfig(
            hf_path="HuggingFaceTB/smoltalk2",
            hf_name="SFT",
            split="OpenHermes_2.5_no_think",
            sampling_weight=0.18,
            max_samples=20000,
            formatter=format_smoltalk2,
        ),
        DatasetConfig(
            hf_path="Modotte/CodeX-2M-Thinking",
            split="train",
            sampling_weight=0.45,
            max_samples=50000,
            formatter=format_codex,
        ),
        DatasetConfig(
            hf_path="Jackrong/GLM-5.1-Reasoning-1M-Cleaned",
            hf_name="main",
            split="train",
            sampling_weight=0.18,
            max_samples=20000,
            formatter=format_glm,
        ),
    ]


@dataclass
class SFTArguments:
    model_name: str = "Qwen/Qwen3.5-2B-Base"
    output_dir: str = "./sft-checkpoints"
    max_seq_length: int = 4096
    per_device_batch: int = 2
    gradient_accumulation: int = 4
    learning_rate: float = 2e-4
    num_train_epochs: int = 1
    max_steps: int = -1
    logging_steps: int = 10
    save_steps: int = 200
    lora_r: int = 64
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    enable_thinking: bool = False

    datasets: list = field(default_factory=_make_datasets)


def get_lora_config(args: SFTArguments) -> LoraConfig:
    return LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
        bias="none",
        task_type="CAUSAL_LM",
    )


def train_sft(model: torch.nn.Module, args: Optional[SFTArguments] = None):
    if args is None:
        args = SFTArguments()

    processor = AutoProcessor.from_pretrained(
        args.model_name,
        trust_remote_code=True,
    )

    # Patch Qwen3.5 template default to disable thinking for no_think subsets
    processor.tokenizer.chat_template = processor.tokenizer.chat_template.replace(
        "{%- set enable_thinking = enable_thinking if enable_thinking is defined else true %}",
        "{%- set enable_thinking = enable_thinking if enable_thinking is defined else false %}",
    )

    train_dataset = build_streaming_dataset(
        args.datasets,
        processor,
        max_length=args.max_seq_length,
    )

    sft_config = SFTConfig(
        output_dir=args.output_dir,
        max_seq_length=args.max_seq_length,
        per_device_train_batch_size=args.per_device_batch,
        gradient_accumulation_steps=args.gradient_accumulation,
        learning_rate=args.learning_rate,
        num_train_epochs=args.num_train_epochs,
        max_steps=args.max_steps,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        bf16=True,
        gradient_checkpointing=False,
        packing=False,
        dataloader_drop_last=True,
        remove_unused_columns=False,
        report_to=["tensorboard"],
        ddp_find_unused_parameters=False,
    )

    peft_config = get_lora_config(args)

    trainer = SFTTrainer(
        model=model,
        args=sft_config,
        train_dataset=train_dataset,
        processing_class=processor,
        peft_config=peft_config,
    )

    trainer.train()
    trainer.save_model(args.output_dir)
