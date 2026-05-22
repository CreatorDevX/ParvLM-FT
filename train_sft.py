"""Gemma-3-270m SFT with LoRA rank 64.

Supports single GPU and distributed (accelerate launch).
"""

from dataclasses import dataclass, field
from typing import Optional
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig
from trl import SFTTrainer, SFTConfig

from sft_data import (
    DatasetConfig,
    format_smoltalk2,
    format_input_output,
    format_glm,
    build_streaming_dataset,
)

# Roles: SYSTEM / user / agent
GEMMA_CHAT_TEMPLATE = (
    "{{ bos_token }}"
    "{% for message in messages %}"
    "{% if message['role'] == 'system' %}"
    "<start_of_turn>SYSTEM\n{{ message['content'] }}<end_of_turn>\n"
    "{% elif message['role'] == 'user' %}"
    "<start_of_turn>user\n{{ message['content'] }}<end_of_turn>\n"
    "{% elif message['role'] in ['assistant', 'model', 'agent'] %}"
    "<start_of_turn>agent\n{{ message['content'] }}<end_of_turn>{{ eos_token }}"
    "{% endif %}"
    "{% endfor %}"
    "{% if add_generation_prompt %}<start_of_turn>agent\n{% endif %}"
)


def format_codex(example: dict, tokenizer: AutoTokenizer, max_length: int) -> dict:
    return format_input_output(example, tokenizer, max_length)


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
            max_samples=25000,
            formatter=format_codex,
        ),
        DatasetConfig(
            hf_path="Jackrong/GLM-5.1-Reasoning-1M-Cleaned",
            hf_name="main",
            split="train",
            sampling_weight=0.18,
            max_samples=10000,
            formatter=format_glm,
        ),
    ]


@dataclass
class SFTArguments:
    model_name: str = "google/gemma-3-270m"
    output_dir: str = "./sft-checkpoints"
    per_device_batch: int = 4
    gradient_accumulation: int = 4
    learning_rate: float = 2e-4
    num_train_epochs: int = 1
    max_steps: int = -1
    logging_steps: int = 1
    save_steps: int = 200
    lora_r: int = 64
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    max_length: int = 4096
    gradient_checkpointing: bool = True
    num_workers: int = 0
    optim: str = "adamw_8bit"

    datasets: list = field(default_factory=_make_datasets)


def get_lora_config(args: SFTArguments) -> LoraConfig:
    return LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
        modules_to_save=["embed_tokens"],
        bias="none",
        task_type="CAUSAL_LM",
    )


def train_sft(args: Optional[SFTArguments] = None):
    if args is None:
        args = SFTArguments()

    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
    tokenizer.chat_template = GEMMA_CHAT_TEMPLATE

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype="auto",
        trust_remote_code=True,
    )
    model.config.use_cache = not args.gradient_checkpointing

    train_dataset = build_streaming_dataset(
        args.datasets,
        tokenizer,
        max_length=args.max_length,
    )

    sft_config = SFTConfig(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.per_device_batch,
        gradient_accumulation_steps=args.gradient_accumulation,
        learning_rate=args.learning_rate,
        num_train_epochs=args.num_train_epochs,
        max_steps=args.max_steps,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        bf16=True,
        gradient_checkpointing=args.gradient_checkpointing,
        optim=args.optim,
        packing=False,
        dataloader_drop_last=True,
        remove_unused_columns=True,
        report_to=["tensorboard"],
        ddp_find_unused_parameters=False,
        dataloader_num_workers=args.num_workers,
        logging_first_step=True,
        log_level="info",
    )

    peft_config = get_lora_config(args)

    trainer = SFTTrainer(
        model=model,
        args=sft_config,
        train_dataset=train_dataset,
        processing_class=tokenizer,
        peft_config=peft_config,
    )

    trainer.train()
    trainer.save_model(args.output_dir)
