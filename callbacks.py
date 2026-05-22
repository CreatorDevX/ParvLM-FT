"""Training callbacks: periodic inference logging to wandb."""

import torch
from transformers import TrainerCallback

DEFAULT_PROMPTS = [
    "Explain the concept of gradient descent in simple terms.",
    "Write a Python function to check if a string is a palindrome.",
    "Tell me a short story about a robot that learns to paint.",
]

DEFAULT_GEN_KWARGS = {
    "max_new_tokens": 128,
    "temperature": 0.7,
    "do_sample": True,
}


class InferenceCallback(TrainerCallback):
    """Generate from fixed prompts every N steps and log to wandb."""

    def __init__(self, tokenizer, prompts=None, gen_kwargs=None, every_n_steps=250):
        self.tokenizer = tokenizer
        self.prompts = prompts or DEFAULT_PROMPTS
        self.gen_kwargs = gen_kwargs or DEFAULT_GEN_KWARGS
        self.every_n_steps = every_n_steps

    def on_step_end(self, args, state, control, model=None, **kwargs):
        if state.global_step == 0 or state.global_step % self.every_n_steps != 0:
            return control
        if not state.is_world_process_zero:
            return control

        model.eval()
        device = next(model.parameters()).device

        for i, prompt in enumerate(self.prompts):
            messages = [{"role": "user", "content": prompt}]
            formatted = self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            inputs = self.tokenizer(formatted, return_tensors="pt").to(device)
            with torch.no_grad():
                outputs = model.generate(**inputs, **self.gen_kwargs)
            response = self.tokenizer.decode(
                outputs[0][inputs.input_ids.shape[1]:],
                skip_special_tokens=True,
            )

            try:
                import wandb
                wandb.log({
                    f"inference/prompt_{i}": wandb.Html(
                        f"<b>Prompt:</b> {prompt}<br><b>Response:</b> {response}"
                    )
                }, step=state.global_step)
            except ImportError:
                print(f"\n[Step {state.global_step}] Prompt {i}: {prompt}")
                print(f"Response: {response}\n")

        model.train()
        return control
