from train_types import TrainConfig, ArtifactConfig, default_artifact, default_checkpoint, default_loss_log, default_sampling
import math
import torch
import time
from model import GPTConfig, GPT

today = time.strftime("%Y%m%d")

def static_vars(**kwargs):
    """Decorator that attaches keyword arguments as persistent attributes on the decorated function."""
    def decorate(func):
        for k in kwargs:
            setattr(func, k, kwargs[k])
        return func
    return decorate

def validate_train_config(config: TrainConfig):
    """Raise a ValueError if the config is invalid."""
    if config.n_layer <= 0:
        raise ValueError("n_layer must be a positive integer")
    if config.n_head <= 0:
        raise ValueError("n_head must be a positive integer")
    if config.n_embd <= 0:
        raise ValueError("n_embd must be a positive integer")
    if config.block_size <= 0:
        raise ValueError("block_size must be a positive integer")
    if config.batch_size <= 0:
        raise ValueError("batch_size must be a positive integer")
    if config.max_steps <= 0:
        raise ValueError("max_steps must be a positive integer")
    if config.n_embd % config.n_head != 0:
        raise ValueError("n_embd must be divisible by n_head")
    if config.n_embd % config.batch_size != 0:
        raise ValueError("n_embd must be divisible by batch_size")

def validate_artifact_config(artifact_config: ArtifactConfig):
    if artifact_config.no_artifact and artifact_config.out_artifact != default_artifact:
        raise ValueError("out_artifact must be specified when artifact is enabled")
    if (artifact_config.no_checkpoint or artifact_config.no_artifact) and artifact_config.out_checkpoint != default_checkpoint:
        raise ValueError("out_checkpoint must be specified when checkpoint is enabled (and artifact is enabled)")
    if (artifact_config.no_loss_log or artifact_config.no_artifact) and artifact_config.out_loss_log != default_loss_log:
        raise ValueError("out_loss_log must be specified when loss log is enabled (and artifact is enabled)")
    if (artifact_config.no_sampling or artifact_config.no_artifact) and artifact_config.out_sampling != default_sampling:
        raise ValueError("out_sampling must be specified when sampling is enabled (and artifact is enabled)")

def model_arch(config: GPTConfig):
    """Return a compact architecture string, e.g. 'L6H6E384'."""
    return f"L{config.n_layer}H{config.n_head}E{config.n_embd}"

def checkpoint_name(step, config:GPTConfig, date=today, prefix="final"):
    """Build a checkpoint filename from prefix, date, architecture string, and step number."""
    return f"{prefix}_{date}_{model_arch(config)}_{step}"

def save_checkpoint(model: GPT, config: GPTConfig, step, stoi, itos, output_dir:str="checkpoints", prefix="checkpoint"):
    """Serialize model weights, config, and char vocabulary to a .pt file under artifacts/checkpoints/."""
    torch.save(
        {
            "step": step,
            "model_state_dict": model.state_dict(),
            "config": config,
            "stoi": stoi,
            "itos": itos,
        },
        f"artifacts/checkpoints/{output_dir}/{checkpoint_name(step, config, prefix=prefix)}.pt",
    )

def get_device():
    """Return the best available device: MPS (Apple Silicon), CUDA, or CPU."""
    if torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")

def get_batch(block_size, batch_size, split_tokens, device=None):
    """Sample a random batch of (input, target) token sequences from a flat token tensor."""
    ix = torch.randint(len(split_tokens) - block_size - 1, (batch_size,))
    x = torch.stack([split_tokens[i : i + block_size] for i in ix]).to(device)
    y = torch.stack([split_tokens[i + 1 : i + block_size + 1] for i in ix]).to(device)
    return x, y

def get_lr(step, warmup_steps, max_steps, max_lr, min_lr):
    """Compute the learning rate for the current step using linear warmup followed by cosine decay."""
    if step < warmup_steps:
        return max_lr * (step + 1) / warmup_steps
    if step >= max_steps:
        return min_lr

    progress = (step - warmup_steps) / (max_steps - warmup_steps)
    return min_lr + 0.5 * (max_lr - min_lr) * (1 + math.cos(math.pi * progress))

def load_data(config: TrainConfig, device):
    """
    Load a text file, build a character-level vocabulary, and tokenize the corpus.

    Returns train/val batch samplers, vocab size, and char↔index mappings (stoi, itos).
    The split is 90% train / 10% val.
    """
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
