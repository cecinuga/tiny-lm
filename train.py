import json
import time
import torch
import math
from tqdm import tqdm
from inference import generate
from model import GPT, GPTConfig

today = time.strftime("%Y%m%d")

def checkpoint_name(step, config:GPTConfig, date=today, prefix="final"):
    return f"{prefix}_{today}_L{config.n_layer}H{config.n_head}E{config.n_embd}_{step}"

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


def load_data(filepath, block_size, batch_size, device):
    with open(filepath, "r") as f:
        text = f.read()

    chars = sorted(set(text))
    vocab_size = len(chars)
    stoi = {c: i for i, c in enumerate(chars)}
    itos = {i: c for c, i in stoi.items()}

    tokens = torch.tensor([stoi[c] for c in text], dtype=torch.long)
    print(f"dataset: {len(tokens):,} chars, vocab_size: {vocab_size}")

    n = int(0.9 * len(tokens))
    get_train = lambda: get_batch(block_size, batch_size, tokens[:n], device)
    get_val = lambda: get_batch(block_size, batch_size, tokens[n:], device)
    return get_train, get_val, vocab_size, stoi, itos


def train(
    data_path,
    model:GPT,
    config:GPTConfig,
    max_steps=5000,
    batch_size=64,
    n_layer=6,
    n_head=6,
    n_embd=384,
    block_size=256,
    device=None
):
    get_train_batch, get_val_batch, vocab_size, stoi, itos = load_data(
        data_path, block_size, batch_size, device
    )

    config.vocab_size = vocab_size

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)

    max_lr = 1e-3
    min_lr = max_lr * 0.1
    warmup_steps = 100

    loss_log = {"steps": [], "train": [], "val": [], "perplexity": []}

    pbar = tqdm(range(max_steps), desc="Training")
    for step in pbar:
        if step % 100 == 0:
            model.eval()
            with torch.no_grad():
                val_losses = []
                for _ in range(20):
                    x, y = get_val_batch()
                    _, loss = model(x, y)
                    val_losses.append(loss.item())
                val_loss = sum(val_losses) / len(val_losses)
                perplexity = math.exp(val_loss)
                tqdm.write(f"Steps {step:5d} | val loss: {val_loss:.4f} | perplexity: {perplexity:.1f}")
            model.train()

        lr = get_lr(step, warmup_steps, max_steps, max_lr, min_lr)
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr

        x, y = get_train_batch()
        _, loss = model(x, y)
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        pbar.set_postfix(loss=f"{loss.item():.4f}", lr=f"{lr:.2e}")

        loss_log["steps"].append(step)
        loss_log["train"].append(loss.item())
        if step % 100 == 0:
            loss_log["val"].append(val_loss)
            loss_log["perplexity"].append(perplexity)

        if step > 0 and step % 100 == 0:
            model.eval()
            sample = generate.generate(
                model, "To be or not", stoi, itos, max_new_tokens=100, temperature=0.8
            )
            tqdm.write(f"\n--- Step {step} sample ---\n{sample}\n---\n")
            model.train()

        if step > 0 and step % 1000 == 0:
            torch.save(
                {
                    "step": step,
                    "model_state_dict": model.state_dict(),
                    "config": config,
                    "stoi": stoi,
                    "itos": itos,
                },
                f"checkpoints/{checkpoint_name(step, config, prefix="_checkpoint")}.pt",
            )

    torch.save(
        {
            "step": max_steps,
            "model_state_dict": model.state_dict(),
            "config": config,
            "stoi": stoi,
            "itos": itos,
        },
        f"checkpoints/{checkpoint_name(max_steps, config, prefix="_checkpoint_final")}.pt",
    )

    with open("loss_log.json", "w") as f:
        json.dump(loss_log, f)

    return model, stoi, itos


if __name__ == "__main__":
    device = get_device()
    print(f"Using device: {device}")

    n_layer = 6
    n_head = 6
    n_embd = 384
    block_size = 256
    config = GPTConfig(
        block_size=block_size,
        n_layer=n_layer,
        n_embd=n_embd,
        n_head=n_head,
    )
    model = GPT(config).to(device)
    print(
        f"Model: {n_layer}L/{n_head}H/{n_embd}D, "
        f"{sum(p.numel() for p in model.parameters()) / 1e6:.1f}M params"
    )
    train("./data/shakespeare.txt", model=model, config=config, n_layer=n_layer, n_head=n_head, n_embd=n_embd, device=device)
