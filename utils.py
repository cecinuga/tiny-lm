import torch
import time
from model import GPTConfig, GPT
today = time.strftime("%Y%m%d")

def model_arch(config: GPTConfig):
    return f"L{config.n_layer}H{config.n_head}E{config.n_embd}"

def checkpoint_name(step, config:GPTConfig, date=today, prefix="final"):
    return f"{prefix}_{today}_{model_arch(config)}_{step}"

def save_checkpoint(model: GPT, config: GPTConfig, step, stoi, itos, prefix="checkpoint"):
    torch.save(
        {
            "step": step,
            "model_state_dict": model.state_dict(),
            "config": config,
            "stoi": stoi,
            "itos": itos,
        },
        f"checkpoints/promessi_sposi/{checkpoint_name(step, config, prefix=prefix)}.pt",
    )

def static_vars(**kwargs):
    def decorate(func):
        for k in kwargs:
            setattr(func, k, kwargs[k])
        return func
    return decorate
