"""GRPO on MMLU + GSM8K for Gemma-3-270m.

Continues from SFT checkpoint. Supports distributed execution.
"""

from dataclasses import dataclass
from typing import Optional
from datasets import load_dataset, interleave_datasets
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig, PeftModel
from trl import GRPOTrainer, GRPOConfig
from rewards import gsm8k_reward, mmlu_reward, format_reward

# Roles: SYSTEM / user / agent — must match train_sft.py
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


@dataclass
class GRPOArguments:
    model_name: str = "google/gemma-3-270m"
    sft_checkpoint: Optional[str] = "./sft-checkpoints"
    output_dir: str = "./grpo-checkpoints"
    max_prompt_length: int = 2048
    max_completion_length: int = 1024
    num_generations: int = 4
    per_device_batch: int = 1
    learning_rate: float = 1e-6
    max_steps: int = 300
    logging_steps: int = 1
    save_steps: int = 50
    lora_r: int = 64
    lora_alpha: int = 32
    gradient_checkpointing: bool = True
    num_workers: int = 0
    optim: str = "adamw_8bit"


def prepare_mmlu_dataset(tokenizer, split: str = "auxiliary_train"):
    ds = load_dataset("cais/mmlu", "all", split=split, streaming=True)

    def format_mmlu(example, t=tokenizer):
        choices = "\n".join(
            f"{chr(65+i)}. {c}" for i, c in enumerate(example["choices"])
        )
        messages = [
            {"role": "user", "content": (
                f"Question: {example['question']}\n"
                f"{choices}\n"
                "Answer the letter (A, B, C, or D) of the correct choice."
            )},
        ]
        prompt = t.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        answer = chr(65 + example["answer"])
        return {"prompt": prompt, "answer": answer}

    return ds.map(format_mmlu)


def prepare_gsm8k_dataset(tokenizer, split: str = "train"):
    ds = load_dataset("gsm8k", "main", split=split, streaming=True)

    def format_gsm8k(example, t=tokenizer):
        messages = [
            {"role": "user", "content": (
                f"Question: {example['question']}\n"
                "Solve the problem step by step. "
                "Put your final answer after ####."
            )},
        ]
        prompt = t.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        answer = example["answer"].split("####")[-1].strip()
        return {"prompt": prompt, "answer": answer}

    return ds.map(format_gsm8k)


def train_grpo(args: Optional[GRPOArguments] = None):
    if args is None:
        args = GRPOArguments()

    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
    tokenizer.chat_template = GEMMA_CHAT_TEMPLATE

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype="auto",
        trust_remote_code=True,
    )
    model.config.use_cache = not args.gradient_checkpointing

    if args.sft_checkpoint is not None:
        model = PeftModel.from_pretrained(model, args.sft_checkpoint)
        model = model.merge_and_unload()

    mmlu_ds = prepare_mmlu_dataset(tokenizer, "auxiliary_train")
    gsm8k_ds = prepare_gsm8k_dataset(tokenizer, "train")

    mmlu_ds = mmlu_ds.take(5000)
    gsm8k_ds = gsm8k_ds.take(5000)

    mmlu_ds = mmlu_ds.map(lambda x: {**x, "task": "mmlu"})
    gsm8k_ds = gsm8k_ds.map(lambda x: {**x, "task": "gsm8k"})

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
        modules_to_save=["embed_tokens"],
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
        gradient_checkpointing=args.gradient_checkpointing,
        optim=args.optim,
        remove_unused_columns=True,
        report_to=["tensorboard"],
        dataloader_num_workers=args.num_workers,
        logging_first_step=True,
        log_level="info",
    )

    trainer = GRPOTrainer(
        model=model,
        processing_class=tokenizer,
        reward_funcs=[routed_reward],
        args=grpo_config,
        train_dataset=train_dataset,
        peft_config=lora_config,
    )

    trainer.train()
    trainer.save_model(args.output_dir)
