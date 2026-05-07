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

    def forward(self, idx, targets=None):
        B, T = idx.shape
        pos = torch.arange(0, T, device=idx.device)

        tok_emb = self.transformer.wte(idx) # (B, T, n_embd)
        pos_emb = self.transformer.wpe(pos) # (T, n_embd)
        x = tok_emb + pos_emb # (B, T, n_embd) — broadcasting adds position info

        for block in self.transformer.h:
            x = block(x)

        x = self.transformer.ln_f(x)
        logits = self.lm_head(x)

        loss = None
        if targets is not None:
            loss = nn.functional.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1)
            )

        return logits, loss