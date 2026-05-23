from math import ceil
from utils import save_checkpoint, model_arch, today
from attr import s
import argparse
import json
import time
import torch
import math
import utils
from tqdm import tqdm
import inference
from model import GPT, GPTConfig

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
    with open(filepath, "r", encoding="latin-1") as f:
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

def sampling(model: GPT, stoi, itos, step: int):
    model.eval()
    sample = inference.generate(
        model, "Come stai ?", stoi, itos, max_new_tokens=100, temperature=0.8
    )
    tqdm.write(f"\n--- Step {step} sample ---\n{sample}\n---\n")
    model.train()

@utils.static_vars(best_val_loss_log=[float('inf')], best_val_loss=float('inf'), patience=0)
def health_check(model: GPT, config: GPTConfig, step:int, stoi, itos, val_loss: float) -> bool:
    #print(val_loss, health_check.patience)
    print(health_check.best_val_loss_log)
    if val_loss < health_check.best_val_loss_log[-1]:
        health_check.patience = 0
        health_check.best_val_loss = val_loss
        health_check.best_val_loss_log.append(val_loss)
        save_checkpoint(model, config, step, stoi, itos, "best")
    else:
        health_check.patience += 1

    if health_check.patience > ceil(len(health_check.best_val_loss_log)/2):
        print(f"Overfitting detected, stopping early!!!")
        print(f"actual_val_loss={val_loss}, best_val_loss={health_check.best_val_loss}, patience={health_check.patience}")
        return False

    return True


def train(
    args: argparse.Namespace,
    device=None
):
    get_train_batch, get_val_batch, vocab_size, stoi, itos = load_data(
        args.data, args.block, args.batch, device
    )

    config = GPTConfig(
        block_size=args.block,
        n_layer=args.layer,
        vocab_size=vocab_size,
        n_embd=args.embd,
        n_head=args.head,
    )
    model = GPT(config).to(device)

    print(
        f"Model: {args.layer}L/{args.head}H/{args.embd}D, "
        f"{sum(p.numel() for p in model.parameters()) / 1e6:.1f}M params"
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)

    max_lr = 1e-3
    min_lr = max_lr * 0.1
    warmup_steps = 100
    health = True
    valuation_step = int(args.max_steps/10)
    health_step = int(args.max_steps/25)

    loss_log = {"steps": [], "train": [], "val": [], "perplexity": []}

    pbar = tqdm(range(args.max_steps), desc="Training")
    for step in pbar:
        # Evaluation
        if step % valuation_step == 0:
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

        lr = get_lr(step, warmup_steps, args.max_steps, max_lr, min_lr)
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr

        x, y = get_train_batch()
        _, loss = model(x, y)
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        pbar.set_postfix(perplexity=f"{perplexity}", loss=f"{loss.item():.4f}", lr=f"{lr:.2e}")

        loss_log["steps"].append(step)
        loss_log["train"].append(loss.item())
        if step % valuation_step == 0:
            loss_log["val"].append(val_loss)
            loss_log["perplexity"].append(perplexity)

        if step > 0 and step % valuation_step == 0:
            sampling(model, stoi, itos, step)

        if step > 0 and step % health_step == 0:
            health = health_check(model, config, step, stoi, itos, val_loss)

        if health is False:
            break

    if health is True:
        save_checkpoint(model, config, args.max_steps, stoi, itos, "final_checkpoint")

    with open(f"loss_logs/loss_log_{today}_{model_arch(config)}.json", "w") as f:
        json.dump(loss_log, f)

    return model, stoi, itos

if __name__ == "__main__":
    device = get_device()
    print(f"Using device: {device}")

    parser = argparse.ArgumentParser(description="Train a GPT model")
    parser.add_argument("--data", type=str, default="data/promessi_sposi.txt", help="Path to dataset file (e.g. shakespeare.txt)")
    parser.add_argument("--layer", type=int, default=6, help="Number of layers")
    parser.add_argument("--head", type=int, default=6, help="Number of heads")
    parser.add_argument("--embd",  type=int, default=384, help="Embedding dimension")
    parser.add_argument("--block", type=int, default=256, help="Block size")
    parser.add_argument("--batch", type=int, default=64, help="Batch size")
    parser.add_argument("--max-steps", type=int, default=2500, help="Maximum number of training steps")
    args = parser.parse_args()

    for _, (name, value) in enumerate(args.__dict__.items()):
        print(f"{name} = {value}")
    print("-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+")

    train(args, device=device)
