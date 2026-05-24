import json
from statistics import mode
import os
from dataclasses import dataclass
import inference
import time
import torch
from tqdm import tqdm
from model import GPTConfig, GPT

today = time.strftime("%Y%m%d")

default_artifact = "artifacts/"
default_checkpoint = "checkpoints/"
default_loss_log = "loss_logs/"
default_sample = "samples/"

@dataclass
class ArtifactConfig:
    """Configuration for saving model artifacts during training."""
    n_save_interval: int = 10
    out_artifact: str = default_artifact
    out_checkpoint: str = default_checkpoint
    out_loss_log: str = default_loss_log
    out_sample: str = default_sample
    no_artifact: bool = False
    no_checkpoint: bool = False
    no_loss_log: bool = False
    no_sample: bool = False

def validate_artifact_config(artifact_config: ArtifactConfig):
    if artifact_config.no_artifact and artifact_config.out_artifact != default_artifact:
        raise ValueError("out_artifact must be specified when artifact is enabled")
    if (artifact_config.no_checkpoint or artifact_config.no_artifact) and artifact_config.out_checkpoint != default_checkpoint:
        raise ValueError("out_checkpoint must be specified when checkpoint is enabled (and artifact is enabled)")
    if (artifact_config.no_loss_log or artifact_config.no_artifact) and artifact_config.out_loss_log != default_loss_log:
        raise ValueError("out_loss_log must be specified when loss log is enabled (and artifact is enabled)")
    if (artifact_config.no_sample or artifact_config.no_artifact) and artifact_config.out_sample != default_sample:
        raise ValueError("out_sampling must be specified when sampling is enabled (and artifact is enabled)")

def model_arch(config: GPTConfig):
    """Return a compact architecture string, e.g. 'L6H6E384'."""
    return f"L{config.n_layer}H{config.n_head}E{config.n_embd}"

def checkpoint_name(step, config:GPTConfig, date=today, prefix="check"):
    """Build a checkpoint filename from prefix, date, architecture string, and step number."""
    assert isinstance(step, int), f"step must be int, got {type(step).__name__}"
    return f"{prefix}_{date}_{model_arch(config)}_{step}"

def save_checkpoint(model: GPT, model_config: GPTConfig, artifact_config: ArtifactConfig, step, stoi, itos, prefix="check"):
    """Serialize model weights, config, and char vocabulary to a .pt file under artifacts/checkpoints/."""
    if artifact_config.no_artifact or artifact_config.no_checkpoint:
        return

    path = f"{artifact_config.out_artifact}{today}/{artifact_config.out_checkpoint}"
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(
        {
            "step": step,
            "model_state_dict": model.state_dict(),
            "config": model_config,
            "stoi": stoi,
            "itos": itos,
        },
        f"{path}{checkpoint_name(step, model_config, prefix=prefix)}.pt",
    )

def sampling(model: GPT, stoi, itos):
    """Generate a short text sample from the model and print it to stdout via tqdm."""
    model.eval()
    sample = inference.generate(
        model, "Come stai ?", stoi, itos, max_new_tokens=100, temperature=0.8
    )
    model.train()
    return sample

def save_sample(model: GPT, model_config: GPTConfig, artifact_config: ArtifactConfig, stoi, itos, step: int):
    """Generate a short text sample from the model and save it to a file under artifacts/samples/."""
    if artifact_config.no_artifact or artifact_config.no_sample:
        return
    sample = sampling(model, stoi, itos)
    tqdm.write(f"\n--- Step {step} sample ---\n{sample}\n---\n")

    name = f"sample_{today}_{model_arch(model_config)}.txt"
    path = f"{artifact_config.out_artifact}{today}/{artifact_config.out_sample}"
    os.makedirs(os.path.dirname(path), exist_ok=True)

    with open(f"{path}{name}", "a") as f:
        f.write(f"--- Step {step} sample ---\n")
        f.write(sample)
        f.write("\n---\n")

def save_loss_log(model_config: GPTConfig, artifact_config: ArtifactConfig, loss_log):
    """Save the loss log to a file."""
    if artifact_config.no_artifact or artifact_config.no_loss_log:
        return

    path = f"{artifact_config.out_artifact}{today}/{artifact_config.out_loss_log}"
    os.makedirs(os.path.dirname(path), exist_ok=True)

    with open(f"{path}loss_log_{model_arch(model_config)}.json", "w") as f:
        json.dump(loss_log, f)
