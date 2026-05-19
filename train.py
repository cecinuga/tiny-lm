from attr import s
import argparse
import json
import time
import torch
import math
from tqdm import tqdm
import inference
from model import GPT, GPTConfig

today = time.strftime("%Y%m%d")

def checkpoint_name(step, config:GPTConfig, date=today, prefix="final"):
    return f"{prefix}_{today}_L{config.n_layer}H{config.n_head}E{config.n_embd}_{step}"

def save_checkpoint(model: GPT, config: GPTConfig, step, stoi, itos, prefix="checkpoint"):
    torch.save(
        {
            "step": step,
            "model_state_dict": model.state_dict(),
            "config": config,
            "stoi": stoi,
            "itos": itos,
        },
        f"checkpoints/{checkpoint_name(step, config, prefix=prefix)}.pt",
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

    best_val_loss = float('inf')
    patience = 5          # quante valutazioni consecutive senza miglioramento
    patience_counter = 0
    early_stop = False

    loss_log = {"steps": [], "train": [], "val": [], "perplexity": []}

    pbar = tqdm(range(args.max_steps), desc="Training")
    for step in pbar:
        # Evaluation
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

        lr = get_lr(step, warmup_steps, args.max_steps, max_lr, min_lr)
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr

        x, y = get_train_batch()
        _, loss = model(x, y)
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        pbar.set_postfix(perplexity=f"{perplexity}",loss=f"{loss.item():.4f}", lr=f"{lr:.2e}")

        loss_log["steps"].append(step)
        loss_log["train"].append(loss.item())
        if step % 100 == 0:
            loss_log["val"].append(val_loss)
            loss_log["perplexity"].append(perplexity)
            if val_loss < best_val_loss:
                best_val_loss = val_loss

        if step > 0 and step % 100 == 0:
            model.eval()
            sample = inference.generate(
                model, "Come stai ?", stoi, itos, max_new_tokens=100, temperature=0.8
            )
            tqdm.write(f"\n--- Step {step} sample ---\n{sample}\n---\n")
            model.train()

        if step > 0 and step % 1000 == 0:
            save_checkpoint(model, config, step, stoi, itos, "checkpoint")

        if step > 1500 and step % 100 == 0:
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                save_checkpoint(model, config, step, stoi, itos, "best")
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print("Overfitting detected, stopping early!!!")
                    break

    if not early_stop:
        save_checkpoint(model, config, args.max_steps, stoi, itos, "final_checkpoint")

    with open("loss_log.json", "w") as f:
        json.dump(loss_log, f)

    return model, stoi, itos

if __name__ == "__main__":
    device = get_device()
    print(f"Using device: {device}")

    parser = argparse.ArgumentParser(description="Train a GPT model")
    parser.add_argument("--data", help="Path to dataset file (e.g. shakespeare.txt)")
    parser.add_argument("--layer", default=6, help="Number of layers")
    parser.add_argument("--head",  default=6, help="Number of heads")
    parser.add_argument("--embd",  default=384, help="Embedding dimension")
    parser.add_argument("--block", default=256, help="Block size")
    parser.add_argument("--batch", default=64, help="Batch size")
    parser.add_argument("--max-steps", default=5000, help="Maximum number of training steps")
    args = parser.parse_args()

    train(args, device=device)
