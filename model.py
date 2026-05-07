import torch
import torch.nn as nn
from dataclasses import dataclass

class GPTConfig:
    n_head:     int = 6   # number of attention heads
    n_emdb:     int = 384 # embedding dimension
    n_layer:    int = 6   # number of transformer blocks
    vocab_size: int = 65  # character-level: 65 unique chars in Shakespeare
    block_size: int = 256 # max sequence length (context window)

class GPT(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.config = config
        self.transformer = nn.ModuleDict(dict(
            wte  = nn.Embedding(config.vocab_size, config.n_emdb), # token embeddings
            wpe  = nn.Embedding(config.block_size, config.n_emdb), # position embeddings
            h    = nn.ModuleList([torch.Block(config) for _ in range(config.n_layer)]),
            ln_f = nn.LayerNorm(config.n_emdb),
        ))
        self.lm_head = nn.Linear(config.n_emdb, config.vocab_size, bias=False)
        self.transformer.wte.weight = self.lm_head.weight
