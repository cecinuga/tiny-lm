from dataclasses import dataclass
from train_utils import static_vars, load_data, save_checkpoint, model_arch, today, get_device, get_lr, TrainConfig
from math import floor
import argparse
import json
import time
import torch
import math
from tqdm import tqdm
import inference
from model import GPT, GPTConfig

def sampling(model: GPT, stoi, itos, step: int):
    """Generate a short text sample from the model and print it to stdout via tqdm."""
    model.eval()
    sample = inference.generate(
        model, "Come stai ?", stoi, itos, max_new_tokens=100, temperature=0.8
    )
    tqdm.write(f"\n--- Step {step} sample ---\n{sample}\n---\n")
    model.train()

@static_vars(best_val_loss=float('inf'), patience=1, patience_counter=0)
def early_stop(val_loss: float) -> bool:
    """
    Track validation loss history and return True if overfitting is detected.

    Overfitting is signaled when the number of consecutive non-improving evaluations
    exceeds half the count of improvements seen so far.
    """
    if val_loss < early_stop.best_val_loss:
        early_stop.patience = 0
        early_stop.patience_counter += 1
        early_stop.best_val_loss = val_loss
    else:
        early_stop.patience += 1

    if early_stop.patience > floor(len(early_stop.patience_counter)/2):
        print(f"Overfitting detected, stopping early!!!")
        print(f"actual_val_loss={val_loss}, best_val_loss={early_stop.best_val_loss}, patience={early_stop.patience}")
        return True

    return False


def train(
    train_config: TrainConfig,
    device=None
):
    """
    Run the full training loop: load data, build model, optimize, evaluate, and checkpoint.

    Evaluates every max_steps/10 steps; saves the best checkpoint after each evaluation
    and a final checkpoint at the end. Stops early if overfitting is detected.
    Returns the trained model along with stoi and itos vocabulary mappings.
    """
    get_train_batch, get_val_batch, vocab_size, stoi, itos = load_data(train_config, device)

    model_config = GPTConfig(
        vocab_size=vocab_size,
        n_layer=train_config.n_layer,
        n_head=train_config.n_head,
        n_embd=train_config.n_embd,
        block_size=train_config.block_size,
    )
    model = GPT(model_config).to(device)

    print(
        f"Model: {train_config.n_layer}L/{train_config.n_head}H/{train_config.n_embd}D, "
        f"{sum(p.numel() for p in model.parameters()) / 1e6:.1f}M params"
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)

    max_lr = 1e-3
    min_lr = max_lr * 0.1
    warmup_steps = 100
    early_stopped = False
    valuation_step = int(train_config.max_steps/10)

    loss_log = {"steps": [], "train": [], "val": [], "perplexity": []}

    pbar = tqdm(range(train_config.max_steps), desc="Training")
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

        lr = get_lr(step, warmup_steps, train_config.max_steps, max_lr, min_lr)
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr

        x, y = get_train_batch()
        _, loss = model(x, y)
        optimizer.zero_grad()
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        pbar.set_postfix(gnorm=f"{grad_norm:.2f}", perplexity=f"{perplexity:.1f}", loss=f"{loss.item():.4f}", lr=f"{lr:.2e}")

        loss_log["steps"].append(step)
        loss_log["train"].append(loss.item())
        if step % valuation_step == 0:
            loss_log["val"].append(val_loss)
            loss_log["perplexity"].append(perplexity)

        if step > 0 and step % valuation_step == 0:
            sampling(model, stoi, itos, step)

        if step > 0 and step % valuation_step == 0:
            early_stopped = early_stop(val_loss)
            if early_stopped is False:
                save_checkpoint(model, model_config, step, stoi, itos, train_config.out_checkpoint, "best")

        if early_stopped is True:
            break

    if early_stopped is False:
        save_checkpoint(model, model_config, train_config.max_steps, stoi, itos, train_config.out_checkpoint, "final_checkpoint")

    with open(f"artifacts/loss_logs/loss_log_{today}_{model_arch(model_config)}.json", "w") as f:
        json.dump(loss_log, f)

    return model, stoi, itos

if __name__ == "__main__":
    device = get_device()
    print(f"Using device: {device}")

    parser = argparse.ArgumentParser(description="Train a GPT model")
    parser.add_argument("-d", "--data", type=str, default="data/promessi_sposi.txt", help="Path to dataset file (e.g. shakespeare.txt)")
    parser.add_argument("-l", "--layer", type=int, default=6, help="Number of layers")
    parser.add_argument("-H", "--head", type=int, default=6, help="Number of heads")
    parser.add_argument("-e", "--embd", type=int, default=384, help="Embedding dimension")
    parser.add_argument("-b", "--block-size", type=int, default=256, help="Block size")
    parser.add_argument("-B", "--batch-size", type=int, default=64, help="Batch size")
    parser.add_argument("--max-steps", type=int, default=2500, help="Maximum number of training steps")
    parser.add_argument("-o-chk", "--out-checkpoint", type=str, default="checkpoints/", help="Path to checkpoint file")
    args = parser.parse_args()

    train_config = TrainConfig(
        data=args.data,
        n_layer=args.layer,
        n_head=args.head,
        n_embd=args.embd,
        block_size=args.block_size,
        batch_size=args.batch_size,
        max_steps=args.max_steps,
        out_checkpoint=args.out_checkpoint,
    )

    for _, (name, value) in enumerate(train_config.__dict__.items()):
        print(f"{name} = {value}")
    print("-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+")

    train(train_config, device=device)
