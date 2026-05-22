"""Reward functions for GRPO on MMLU and GSM8K.

Each reward function takes (prompts, completions, **kwargs) and returns
a list of float rewards. Used by GRPOTrainer.
"""

import re
from typing import Union


def extract_answer_gsm8k(text: str) -> Union[str, None]:
    """Extract final numeric answer from GSM8K-style completion.

    Looks for '#### <number>' pattern. Falls back to last number.
    """
    match = re.search(r"####\s*(-?[\d,]+\.?\d*)", text)
    if match:
        return match.group(1).replace(",", "")
    # fallback: last number in text
    numbers = re.findall(r"-?[\d,]+\.?\d*", text)
    if numbers:
        return numbers[-1].replace(",", "")
    return None


def extract_answer_mmlu(text: str) -> Union[str, None]:
    """Extract multiple-choice answer from MMLU-style completion.

    Looks for 'Answer: X' or lone 'X' at end, where X in A-D.
    """
    match = re.search(r"(?:answer|Answer|ANSWER)\s*:\s*([A-D])", text)
    if match:
        return match.group(1)
    # fallback: last standalone letter A-D
    letters = re.findall(r"\b([A-D])\b", text)
    if letters:
        return letters[-1]
    return None


def gsm8k_reward(prompts: list, completions: list, **kwargs) -> list[float]:
    """Binary reward: 1.0 if extracted answer matches ground truth."""
    rewards = []
    for prompt, completion in zip(prompts, completions):
        extracted = extract_answer_gsm8k(completion)
        # Ground truth from dataset is passed through kwargs
        ground_truth = kwargs.get("answer", [None] * len(prompts))
        if extracted and ground_truth and extracted == ground_truth[0]:
            rewards.append(1.0)
        else:
            rewards.append(0.0)
    return rewards


def mmlu_reward(prompts: list, completions: list, **kwargs) -> list[float]:
    """Binary reward: 1.0 if extracted answer matches ground truth."""
    rewards = []
    for prompt, completion in zip(prompts, completions):
        extracted = extract_answer_mmlu(completion)
        ground_truth = kwargs.get("answer", [None] * len(prompts))
        if extracted and ground_truth and extracted == ground_truth[0]:
            rewards.append(1.0)
        else:
            rewards.append(0.0)
    return rewards


def format_reward(prompts: list, completions: list, **kwargs) -> list[float]:
    """Soft format reward: encourages structured thinking output.

    Gives partial credit for presence of  tags.
    """
    rewards = []
    for completion in completions:
        score = 0.0
        if "<think>" in completion and "</think>" in completion:
            score += 0.5
            # Check there's content between think tags
            inner = completion.split("<think>")[1].split("</think>")[0].strip()
            if inner:
                score += 0.3
        if "####" in completion or "answer" in completion.lower():
            score += 0.2
        rewards.append(min(score, 1.0))
    return rewards
