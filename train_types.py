from dataclasses import dataclass

default_artifact = "artifacts/"
default_checkpoint = "checkpoints/"
default_loss_log = "loss_logs/"
default_sampling = "sampling/"

@dataclass
class TrainConfig:
    """Hyperparameters and I/O paths for a single training run."""
    n_layer: int = 6
    n_head: int = 6
    n_embd: int = 384
    block_size: int = 256
    batch_size: int = 64
    max_steps: int = 2500
    data: str = "data/promessi_sposi.txt"

@dataclass
class ArtifactConfig:
    """Configuration for saving model artifacts during training."""
    n_save_interval: int = 10
    out_artifact: str = default_artifact
    out_checkpoint: str = default_checkpoint
    out_loss_log: str = default_loss_log
    out_sampling: str = default_sampling
    no_artifact: bool = False
    no_checkpoint: bool = False
    no_loss_log: bool = False
    no_sampling: bool = False
