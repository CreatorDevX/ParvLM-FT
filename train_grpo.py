"""GRPO on MMLU + GSM8K for Qwen3.5-2B.

Continues from SFT checkpoint. Uses rule-based rewards
for verifiable reasoning problems.

Usage (Kaggle notebook):
  model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3.5-2B-Base", ...)
  notebook_launcher(train_grpo, (model,), num_processes=8)
"""

from dataclasses import dataclass
from typing import Optional
import torch
from datasets import load_dataset, interleave_datasets
from transformers import AutoProcessor
from peft import LoraConfig, PeftModel
from trl import GRPOTrainer, GRPOConfig
from rewards import gsm8k_reward, mmlu_reward, format_reward


@dataclass
class GRPOArguments:
    model_name: str = "Qwen/Qwen3.5-2B-Base"
    sft_checkpoint: Optional[str] = "./sft-checkpoints"
    output_dir: str = "./grpo-checkpoints"
    max_prompt_length: int = 2048
    max_completion_length: int = 1024
    num_generations: int = 4
    per_device_batch: int = 1
    learning_rate: float = 1e-6
    max_steps: int = 300
    logging_steps: int = 10
    save_steps: int = 50
    lora_r: int = 64
    lora_alpha: int = 32
    enable_thinking: bool = True


def prepare_mmlu_dataset(processor, split: str = "auxiliary_train"):
    ds = load_dataset("cais/mmlu", "all", split=split, streaming=True)

    def format_mmlu(example, p=processor):
        choices = "\n".join(
            f"{chr(65+i)}. {c}" for i, c in enumerate(example["choices"])
        )
        messages = [
            {"role": "user", "content": (
                f"Question: {example['question']}\n"
                f"{choices}\n"
                f"Answer the letter (A, B, C, or D) of the correct choice."
            )},
        ]
        prompt = p.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
            enable_thinking=False,
        )
        answer = chr(65 + example["answer"])
        return {"prompt": prompt, "answer": answer}

    return ds.map(format_mmlu)


def prepare_gsm8k_dataset(processor, split: str = "train"):
    ds = load_dataset("gsm8k", "main", split=split, streaming=True)

    def format_gsm8k(example, p=processor):
        messages = [
            {"role": "user", "content": (
                f"Question: {example['question']}\n"
                "Solve the problem step by step. "
                "Put your final answer after ####."
            )},
        ]
        prompt = p.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
            enable_thinking=True,
        )
        answer = example["answer"].split("####")[-1].strip()
        return {"prompt": prompt, "answer": answer}

    return ds.map(format_gsm8k)


def train_grpo(model: torch.nn.Module, args: Optional[GRPOArguments] = None):
    if args is None:
        args = GRPOArguments()

    processor = AutoProcessor.from_pretrained(
        args.model_name,
        trust_remote_code=True,
    )
    processor.chat_template = (
        "{% for message in messages %}"
        "{% set role = 'agent' if message['role'] == 'assistant' else message['role'] %}"
        "<|im_start|>{{ role }}\n"
        "{% if role == 'agent' and enable_thinking is defined and not enable_thinking %}"
        "<think>\n\n</think>\n\n"
        "{% endif %}"
        "{{ message['content'] }}<|im_end|>\n"
        "{% endfor %}"
        "{% if add_generation_prompt %}"
        "<|im_start|>agent\n"
        "{% if enable_thinking is defined and not enable_thinking %}"
        "<think>\n\n</think>\n\n"
        "{% endif %}"
        "{% endif %}"
    )
    processor.tokenizer.chat_template = processor.chat_template

    # Load SFT checkpoint and merge into base for GRPO
    if args.sft_checkpoint is not None:
        model = PeftModel.from_pretrained(model, args.sft_checkpoint)
        model = model.merge_and_unload()

    # Prepare datasets (processor formats prompts with chat template)
    mmlu_ds = prepare_mmlu_dataset(processor, "auxiliary_train")
    gsm8k_ds = prepare_gsm8k_dataset(processor, "train")

    mmlu_ds = mmlu_ds.take(5000)
    gsm8k_ds = gsm8k_ds.take(5000)

    # Add task label for routing
    mmlu_ds = mmlu_ds.map(lambda x, t="mmlu": {**x, "task": t})
    gsm8k_ds = gsm8k_ds.map(lambda x, t="gsm8k": {**x, "task": t})

    train_dataset = interleave_datasets(
        [mmlu_ds, gsm8k_ds],
        probabilities=[0.5, 0.5],
        stopping_strategy="all_exhausted",
    )

    def routed_reward(prompts, completions, **kwargs):
        g_r = gsm8k_reward(prompts, completions, **kwargs)
        m_r = mmlu_reward(prompts, completions, **kwargs)
        f_r = format_reward(prompts, completions, **kwargs)
        return [
            (g if g is not None else 0) + (m if m is not None else 0) + f * 0.1
            for g, m, f in zip(g_r, m_r, f_r)
        ]

    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
        bias="none",
        task_type="CAUSAL_LM",
    )

    grpo_config = GRPOConfig(
        output_dir=args.output_dir,
        max_prompt_length=args.max_prompt_length,
        max_completion_length=args.max_completion_length,
        num_generations=args.num_generations,
        per_device_train_batch_size=args.per_device_batch,
        learning_rate=args.learning_rate,
        max_steps=args.max_steps,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        bf16=True,
        remove_unused_columns=False,
        report_to=["tensorboard"],
    )

    trainer = GRPOTrainer(
        model=model,
        processing_class=processor,
        reward_funcs=[routed_reward],
        args=grpo_config,
        train_dataset=train_dataset,
        peft_config=lora_config,
    )

    trainer.train()
    trainer.save_model(args.output_dir)
