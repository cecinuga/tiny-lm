from dataclasses import dataclass
import math
import torch
import time
from model import GPTConfig, GPT
today = time.strftime("%Y%m%d")

@dataclass
class TrainConfig:
    n_layer: int = 6
    n_head: int = 6
    n_embd: int = 384
    block_size: int = 256
    batch_size: int = 64
    max_steps: int = 2500
    data: str = "data/promessi_sposi.txt"
    out_checkpoint: str = "checkpoints/"

def model_arch(config: GPTConfig):
    return f"L{config.n_layer}H{config.n_head}E{config.n_embd}"

def checkpoint_name(step, config:GPTConfig, date=today, prefix="final"):
    return f"{prefix}_{today}_{model_arch(config)}_{step}"

def save_checkpoint(model: GPT, config: GPTConfig, step, stoi, itos, output_dir:str="checkpoints", prefix="checkpoint"):
    torch.save(
        {
            "step": step,
            "model_state_dict": model.state_dict(),
            "config": config,
            "stoi": stoi,
            "itos": itos,
        },
        f"{output_dir}/{checkpoint_name(step, config, prefix=prefix)}.pt",
    )

def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")

def get_batch(block_size, batch_size, split_tokens, device=None):
    ix = torch.randint(len(split_tokens) - block_size - 1, (batch_size,))
    x = torch.stack([split_tokens[i : i + block_size] for i in ix]).to(device)
    y = torch.stack([split_tokens[i + 1 : i + block_size + 1] for i in ix]).to(device)
    return x, y

def get_lr(step, warmup_steps, max_steps, max_lr, min_lr):
    if step < warmup_steps:
        return max_lr * (step + 1) / warmup_steps
    if step >= max_steps:
        return min_lr

    progress = (step - warmup_steps) / (max_steps - warmup_steps)
    return min_lr + 0.5 * (max_lr - min_lr) * (1 + math.cos(math.pi * progress))

def load_data(config: TrainConfig, device):
    with open(config.data, "r", encoding="latin-1") as f:
        text = f.read()

    chars = sorted(set(text))
    vocab_size = len(chars)
    stoi = {c: i for i, c in enumerate(chars)}
    itos = {i: c for c, i in stoi.items()}

    tokens = torch.tensor([stoi[c] for c in text], dtype=torch.long)
    print(f"dataset: {len(tokens):,} chars, vocab_size: {vocab_size}")

    n = int(0.9 * len(tokens))
    get_train = lambda: get_batch(config.block_size, config.batch_size, tokens[:n], device)
    get_val = lambda: get_batch(config.block_size, config.batch_size, tokens[n:], device)
    return get_train, get_val, vocab_size, stoi, itos
