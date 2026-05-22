"""Kaggle TPU v5e-8 notebook for Qwen3.5-2B.

Two phases:
  Phase 1 — SFT (6 datasets, LoRA rank 64, ~1-2 hours)
  Phase 2 — GRPO on MMLU+GSM8K, continues from SFT checkpoint (~2-3 hours)

Running Phase 2 requires a separate Kaggle session (9h limit).
"""

# %% [markdown]
# # Phase 1: SFT

# %% [markdown]
# ## Cell 1: Install

# %%
import subprocess, sys
def install(pkg):
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", pkg])

install("transformers>=4.49.0 accelerate>=1.5.0 trl>=0.18.0")
install("peft>=0.14.0 datasets>=3.3.0 sentencepiece tensorboard")

# %% [markdown]
# ## Cell 2: Environment

# %%
import os
os.environ["PJRT_DEVICE"] = "TPU"
os.environ["XLA_USE_BF16"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# %% [markdown]
# ## Cell 3: Imports

# %%
import torch
import torch_xla
from accelerate import notebook_launcher
from transformers import AutoModelForCausalLM, AutoProcessor

# %% [markdown]
# ## Cell 4: Load model (OUTSIDE training function — required for TPU fork)

# %%
MODEL_NAME = "Qwen/Qwen3.5-2B-Base"
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.bfloat16,
    trust_remote_code=True,
)
print(f"Model loaded: {MODEL_NAME}")

# %% [markdown]
# ## Cell 5: Run SFT

# %%
from train_sft import SFTArguments, train_sft

sft_args = SFTArguments(
    model_name=MODEL_NAME,
    output_dir="./sft-checkpoints",
    max_seq_length=4096,
    per_device_batch=1,
    gradient_accumulation=4,
    learning_rate=2e-4,
    num_train_epochs=1,
    logging_steps=10,
    save_steps=200,
    lora_r=64,
)

notebook_launcher(train_sft, (model, sft_args), num_processes=8)


# %% [markdown]
# ---
# # Phase 2: GRPO
#
# **Requires a separate session.** Stop here, save checkpoints, open a new Kaggle notebook
# with TPU v5e-8, upload this same code, and run from Cell 6 below.
#
# Phase 2 loads the SFT checkpoints saved above, merges LoRA into base,
# and runs GRPO on MMLU + GSM8K with rule-based rewards.

# %% [markdown]
# ## Cell 6: Imports (Phase 2)

# %%
import torch
import torch_xla
from accelerate import notebook_launcher
from transformers import AutoModelForCausalLM, AutoProcessor

# %% [markdown]
# ## Cell 7: Load base model (same as Phase 1)

# %%
MODEL_NAME = "Qwen/Qwen3.5-2B-Base"
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.bfloat16,
    trust_remote_code=True,
)
print("Base model loaded for GRPO")

# %% [markdown]
# ## Cell 8: Run GRPO
#
# Loads SFT checkpoint from `./sft-checkpoints`, merges, trains with GRPO.

# %%
from train_grpo import GRPOArguments, train_grpo

grpo_args = GRPOArguments(
    model_name=MODEL_NAME,
    sft_checkpoint="./sft-checkpoints",
    output_dir="./grpo-checkpoints",
    max_prompt_length=2048,
    max_completion_length=1024,
    num_generations=4,
    per_device_batch=1,
    learning_rate=1e-6,
    max_steps=300,
    logging_steps=10,
    save_steps=50,
    lora_r=64,
    enable_thinking=True,
)

notebook_launcher(train_grpo, (model, grpo_args), num_processes=8)
