from six import b
from matplotlib.rcsetup import validate_any
from dataclasses import dataclass
from train_utils import static_vars, load_data, save_checkpoint, model_arch, today, get_device, get_lr, validate_train_config, validate_artifact_config
from train_types import TrainConfig, ArtifactConfig
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

    if early_stop.patience > floor(early_stop.patience_counter/2):
        print(f"Overfitting detected, stopping early!!!")
        print(f"actual_val_loss={val_loss}, best_val_loss={early_stop.best_val_loss}, patience={early_stop.patience}")
        return True

    return False


def train(
    train_config: TrainConfig,
    artifact_config: ArtifactConfig,
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
                save_checkpoint(model, model_config, step, stoi, itos, artifact_config.out_checkpoint, "best")

        if early_stopped is True:
            break

    if early_stopped is False:
        save_checkpoint(model, model_config, train_config.max_steps, stoi, itos, artifact_config.out_checkpoint, "final_checkpoint")

    with open(f"artifacts/loss_logs/loss_log_{today}_{model_arch(model_config)}.json", "w") as f:
        json.dump(loss_log, f)

    return model, stoi, itos

if __name__ == "__main__":
    device = get_device()
    print(f"Using device: {device}")

    train_parser = argparse.ArgumentParser(description="Train a GPT model")
    train_parser.add_argument("-d", "--data", type=str, default="data/promessi_sposi.txt", help="Path to dataset file (e.g. shakespeare.txt)")
    train_parser.add_argument("-l", "--layer", type=int, default=6, help="Number of layers")
    train_parser.add_argument("-H", "--head", type=int, default=6, help="Number of heads")
    train_parser.add_argument("-e", "--embd", type=int, default=384, help="Embedding dimension")
    train_parser.add_argument("-b", "--block-size", type=int, default=256, help="Block size")
    train_parser.add_argument("-B", "--batch-size", type=int, default=64, help="Batch size")
    train_parser.add_argument("--max-steps", type=int, default=2500, help="Maximum number of training steps")
    train_parser.add_argument("-n-save", "--n-save-interval", type=int, default=10, help="Number of time program saves an artifact (shared between all artifacts)")
    train_parser.add_argument("-o-art", "--out-artifact", type=str, default="artifacts/", help="Root artifact folder")
    train_parser.add_argument("-o-chk", "--out-checkpoint", type=str, default="checkpoints/", help="checkpoint folder, is prefixed to artifacts folder")
    train_parser.add_argument("-o-l", "--out-loss-log", type=str, default="loss_logs/", help="loss log folder, is prefixed to artifacts folder")
    train_parser.add_argument("-o-s", "--out-sampling", type=str, default="sampling/", help="sampling folder, is prefixed to artifacts folder")
    train_parser.add_argument("-no-a", "--no-artifact", action="store_true", help="Disable artifact saving")
    train_parser.add_argument("-no-c", "--no-checkpoint", action="store_true", help="Disable checkpoint saving")
    train_parser.add_argument("-no-ll", "--no-loss-log", action="store_true", help="Disable loss log saving")
    train_parser.add_argument("-no-s", "--no-sampling", action="store_true", help="Disable sampling saving")
    train_args = train_parser.parse_args()

    train_config = TrainConfig(
        data=train_args.data,
        n_layer=train_args.layer,
        n_head=train_args.head,
        n_embd=train_args.embd,
        block_size=train_args.block_size,
        batch_size=train_args.batch_size,
        max_steps=train_args.max_steps,
    )
    validate_train_config(train_config)

    artifact_config = ArtifactConfig(
        out_artifact=train_args.out_artifact,
        out_checkpoint=train_args.out_checkpoint,
        out_loss_log=train_args.out_loss_log,
        out_sampling=train_args.out_sampling,
        n_save_interval=train_args.n_save_interval,
        no_artifact=train_args.no_artifact,
        no_checkpoint=train_args.no_checkpoint,
        no_loss_log=train_args.no_loss_log,
        no_sampling=train_args.no_sampling,
    )
    validate_artifact_config(artifact_config)

    for _, (name, value) in enumerate(train_config.__dict__.items()):
        print(f"{name} = {value}")
    print("-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+")

    for _, (name, value) in enumerate(artifact_config.__dict__.items()):
        print(f"{name} = {value}")
    print("-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+")

    train(train_config, artifact_config, device=device)
