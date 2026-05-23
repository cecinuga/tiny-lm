from dataclasses import dataclass
from train_utils import load_data, save_checkpoint, model_arch, today, get_device, get_lr, TrainConfig
from math import ceil
from attr import s
import argparse
import json
import time
import torch
import math
import utils
from tqdm import tqdm
import inference
from automapper import mapper
from model import GPT, GPTConfig

def sampling(model: GPT, stoi, itos, step: int):
    model.eval()
    sample = inference.generate(
        model, "Come stai ?", stoi, itos, max_new_tokens=100, temperature=0.8
    )
    tqdm.write(f"\n--- Step {step} sample ---\n{sample}\n---\n")
    model.train()

@utils.static_vars(best_val_loss_log=[float('inf')], best_val_loss=float('inf'), patience=0)
def overfit_detector(model: GPT, config: GPTConfig, step:int, stoi, itos, val_loss: float) -> bool:
    """
    returns True if the model is overfitting, False otherwise.
    """
    #print(val_loss, overfit_detector.patience)
    if val_loss < overfit_detector.best_val_loss_log[-1]:
        overfit_detector.patience = 0
        overfit_detector.best_val_loss = val_loss
        overfit_detector.best_val_loss_log.append(val_loss)
    else:
        overfit_detector.patience += 1

    if overfit_detector.patience > ceil(len(overfit_detector.best_val_loss_log)/2):
        print(f"Overfitting detected, stopping early!!!")
        print(f"actual_val_loss={val_loss}, best_val_loss={overfit_detector.best_val_loss}, patience={overfit_detector.patience}")
        return True

    return False


def train(
    train_config: TrainConfig,
    device=None
):
    get_train_batch, get_val_batch, vocab_size, stoi, itos = load_data(train_config, device)

    model_config = mapper.to(GPTConfig).map(train_config, fields_mapping={"vocab_size": vocab_size})
    model = GPT(model_config).to(device)

    print(
        f"Model: {train_config.n_layer}L/{train_config.n_head}H/{train_config.n_embd}D, "
        f"{sum(p.numel() for p in model.parameters()) / 1e6:.1f}M params"
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)

    max_lr = 1e-3
    min_lr = max_lr * 0.1
    warmup_steps = 100
    overfitted = False
    valuation_step = int(train_config.max_steps/10)
    health_step = int(train_config.max_steps/25)

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
            overfitted = overfit_detector(model, model_config, step, stoi, itos, val_loss)
            if overfitted is False:
                save_checkpoint(model, model_config, step, stoi, itos, train_config.out_checkpoint, "best")

        if overfitted is True:
            break

    if overfitted is False:
        save_checkpoint(model, model_config, train_config.max_steps, stoi, itos, train_config.out_checkpoint, "final_checkpoint")

    with open(f"loss_logs/loss_log_{today}_{model_arch(model_config)}.json", "w") as f:
        json.dump(loss_log, f)

    return model, stoi, itos

if __name__ == "__main__":
    device = get_device()
    print(f"Using device: {device}")

    parser = argparse.ArgumentParser(description="Train a GPT model")
    parser.add_argument("-d", "--data", type=str, default="data/promessi_sposi.txt", help="Path to dataset file (e.g. shakespeare.txt)")
    parser.add_argument("-l", "--n-layer", type=int, default=6, help="Number of layers")
    parser.add_argument("-H", "--n-head", type=int, default=6, help="Number of heads")
    parser.add_argument("-e", "--n-embd", type=int, default=384, help="Embedding dimension")
    parser.add_argument("-b", "--block-size", type=int, default=256, help="Block size")
    parser.add_argument("-B", "--batch-size", type=int, default=64, help="Batch size")
    parser.add_argument("--max-steps", type=int, default=2500, help="Maximum number of training steps")
    parser.add_argument("-o-chk", "--out-checkpoint", type=str, default="checkpoints/", help="Path to checkpoint file")
    args = parser.parse_args()

    train_config = mapper.to(TrainConfig).map(args)

    for _, (name, value) in enumerate(train_config.__dict__.items()):
        print(f"{name} = {value}")
    print("-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+")

    train(train_config, device=device)
