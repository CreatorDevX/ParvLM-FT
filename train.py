"""Single entry point for Gemma-3-270m SFT + GRPO pipeline.

Usage:
    python train.py --phase all                          # SFT then GRPO
    python train.py --phase sft                          # SFT only
    accelerate launch --num_processes 2 train.py          # multi-GPU
    python train.py --lr 3e-4 --per-device-batch 8       # custom
"""

import argparse
import os

from train_sft import SFTArguments, train_sft
from train_grpo import GRPOArguments, train_grpo


def parse_args():
    parser = argparse.ArgumentParser(description="Gemma-3-270m SFT + GRPO")

    # Pipeline control
    parser.add_argument("--phase", choices=["sft", "grpo", "all"], default="all")

    # Model / paths
    parser.add_argument("--model", default="google/gemma-3-270m")
    parser.add_argument("--output-dir", default="./output")
    parser.add_argument("--sft-checkpoint", default=None)

    # SFT hyperparams
    parser.add_argument("--per-device-batch", type=int, default=4)
    parser.add_argument("--grad-accum", type=int, default=4)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--lora-r", type=int, default=32)
    parser.add_argument("--max-len", type=int, default=4096)
    parser.add_argument("--epochs", type=float, default=1.0)
    parser.add_argument("--max-steps", type=int, default=-1)

    # Training speed / memory
    parser.add_argument("--gradient-checkpointing", action="store_true", default=True)
    parser.add_argument("--gradient-checkpointing-off", action="store_false", dest="gradient_checkpointing")
    parser.add_argument("--num-workers", type=int, default=0)

    # Logging / saving
    parser.add_argument("--save-steps", type=int, default=200)
    parser.add_argument("--logging-steps", type=int, default=10)

    # GRPO-specific
    parser.add_argument("--grpo-batch", type=int, default=1)
    parser.add_argument("--grpo-steps", type=int, default=300)
    parser.add_argument("--num-generations", type=int, default=4)
    parser.add_argument("--max-prompt-len", type=int, default=2048)
    parser.add_argument("--max-completion-len", type=int, default=1024)
    parser.add_argument("--grpo-lr", type=float, default=1e-6)

    return parser.parse_args()


def main():
    args = parse_args()

    sft_dir = os.path.join(args.output_dir, "sft")
    grpo_dir = os.path.join(args.output_dir, "grpo")

    if args.phase in ("sft", "all"):
        print("=" * 60)
        print("Phase 1: SFT")
        print("=" * 60)

        sft_args = SFTArguments(
            model_name=args.model,
            output_dir=sft_dir,
            per_device_batch=args.per_device_batch,
            gradient_accumulation=args.grad_accum,
            learning_rate=args.lr,
            num_train_epochs=args.epochs,
            max_steps=args.max_steps,
            logging_steps=args.logging_steps,
            save_steps=args.save_steps,
            lora_r=args.lora_r,
            max_length=args.max_len,
            gradient_checkpointing=args.gradient_checkpointing,
            num_workers=args.num_workers,
        )
        train_sft(sft_args)

    if args.phase in ("grpo", "all"):
        print("=" * 60)
        print("Phase 2: GRPO")
        print("=" * 60)

        checkpoint = args.sft_checkpoint or sft_dir

        grpo_args = GRPOArguments(
            model_name=args.model,
            sft_checkpoint=checkpoint,
            output_dir=grpo_dir,
            max_prompt_length=args.max_prompt_len,
            max_completion_length=args.max_completion_len,
            num_generations=args.num_generations,
            per_device_batch=args.grpo_batch,
            learning_rate=args.grpo_lr,
            max_steps=args.grpo_steps,
            logging_steps=args.logging_steps,
            save_steps=args.save_steps,
            lora_r=args.lora_r,
            gradient_checkpointing=args.gradient_checkpointing,
            num_workers=args.num_workers,
        )
        train_grpo(grpo_args)


if __name__ == "__main__":
    main()
