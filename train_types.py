from dataclasses import dataclass

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
