# tiny-lm

A tiny GPT-style language model trained from scratch on Shakespeare, runnable on a laptop.

## What this is

tiny-lm implements a character-level GPT with ~11.5M parameters. It learns to generate Shakespeare-like text by predicting the next character one step at a time. The architecture follows the original transformer paper: multi-head causal self-attention + feed-forward blocks with residual connections and layer normalization.

## Installation

```bash
# Requires Python 3.12+ and uv
uv sync
```

## Usage

**Train:**
```bash
python train.py
```
Checkpoints are saved in `checkpoints/` every 1000 steps and at the end of training.

**Generate text from a checkpoint:**
```bash
python generate.py --checkpoint checkpoints/<your_checkpoint>.pt --prompt "To be or not"
```

Options:
| Flag | Default | Description |
|------|---------|-------------|
| `--max_new_tokens` | 200 | How many characters to generate |
| `--temperature` | 0.8 | Higher = more random, lower = more deterministic |
| `--top_k` | 40 | Only sample from the top-k most likely next tokens |
| `--seed` | None | Fix for reproducible output |

## Architecture

```
Input characters
      ↓
Token embeddings + Positional embeddings
      ↓
6× Transformer Block:
   LayerNorm → CausalSelfAttention → Residual
   LayerNorm → MLP (4× expansion, GELU) → Residual
      ↓
Final LayerNorm → Linear head → logits → cross-entropy loss
```

Key choices:
- **Character-level tokenization** — vocabulary of 65 chars, no external tokenizer needed
- **Weight tying** — the token embedding matrix and the output projection share weights (reduces parameters, improves generalization)
- **Pre-norm** — LayerNorm before each sub-layer (more stable than post-norm)
- **Learning rate schedule** — linear warmup (100 steps) + cosine decay

Default config: 6 layers, 6 heads, 384 embedding dim, 256 context window, batch size 64.

## Known Bugs (educational)

These are real bugs in the current code — finding and fixing them is a great learning exercise:

### Bug 1 — `get_lr()` returns the wrong value (`train.py:35`)

```python
# current (wrong): returns a float between 0.0 and 1.0, not a learning rate
progress = (step - warmup_steps) / (max_steps - warmup_steps)
return progress

# correct: apply the cosine decay formula
import math
return min_lr + 0.5 * (max_lr - min_lr) * (1 + math.cos(math.pi * progress))
```

**Why it matters:** The optimizer receives a near-zero "learning rate" for most of training, so the model barely learns after the warmup phase. This is the single biggest performance bug in the project.

### Bug 2 — Validation uses training data (`train.py:52`)

```python
# current (wrong): both lambdas use tokens[:n] (the training split)
get_val = lambda: get_batch(block_size, batch_size, tokens[:n], device)

# correct: use the held-out 10%
get_val = lambda: get_batch(block_size, batch_size, tokens[n:], device)
```

**Why it matters:** The reported validation loss is actually training loss. You cannot detect overfitting this way because the model is evaluated on data it has already seen.

### Bug 3 — `model.py` runs code on every import (`model.py:102-105`)

```python
# current (wrong): executes at import time
config = GPTConfig()
model = GPT(config)
n_params = sum(p.numel() for p in model.parameters())
print(f"parameters: {n_params / 1e6:.1F}M")

# correct: guard with __main__
if __name__ == "__main__":
    config = GPTConfig()
    model = GPT(config)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"parameters: {n_params / 1e6:.1F}M")
```

**Why it matters:** Every `import model` (e.g., from `train.py`) allocates a full model on CPU and prints to stdout. It wastes memory and pollutes logs.

## Improvement Ideas

These are not bugs — the code works — but implementing them will teach you important ML concepts:

### 1. Add perplexity to logs
Perplexity is `exp(cross_entropy_loss)` and is more interpretable than raw loss. A perplexity of 5 means the model is as uncertain as if it had to choose between 5 equally likely options at each step.

```python
import math
perplexity = math.exp(val_loss)
tqdm.write(f"Step {step:5d} | val loss: {val_loss:.4f} | perplexity: {perplexity:.1f}")
```

### 2. Detect overfitting
After fixing Bug 2, compare training and validation loss at each eval step. If training loss keeps dropping but val loss stops improving (or rises), the model is overfitting — it is memorizing rather than generalizing.

A simple early stopping rule: stop training if val loss has not improved in the last N evaluations.

### 3. Add `argparse` to `train.py`
Right now hyperparameters are hardcoded. Adding a CLI lets you run experiments without editing source:

```bash
python train.py --n_layer 4 --n_head 4 --n_embd 128 --max_steps 2000
```

### 4. Try a different dataset
Shakespeare is ~1MB. The model memorizes it quickly. Try a larger, noisier dataset — Project Gutenberg novels, Wikipedia dumps, or the TinyStories dataset on HuggingFace — to see how the model generalizes to a harder distribution.

### 5. Subword tokenization
Character-level tokenization is simple but inefficient: each token carries little information, so the model needs a long context to understand meaning. Real language models (GPT, LLaMA) use BPE tokenizers that split text into subwords (e.g., "Shake" + "speare"). `tiktoken` is already in the dependencies — try replacing the character tokenizer with `tiktoken.get_encoding("gpt2")`.

### 6. Experiment tracking
Saving losses to `loss_log.json` is a start. Plot them with matplotlib to see the learning curves. For bigger experiments, tools like [Weights & Biases](https://wandb.ai) or TensorBoard let you compare runs, visualize attention patterns, and track GPU utilization.

### 7. Gradient norm monitoring
The code already clips gradients (`clip_grad_norm_`). Also log the gradient norm before clipping — a suddenly large gradient norm often signals numerical instability or a bad batch.

```python
grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
pbar.set_postfix(loss=f"{loss.item():.4f}", lr=f"{lr:.2e}", gnorm=f"{grad_norm:.2f}")
```

## TODO

- [x] Checkpoint saving with config metadata
- [ ] Fix Bug 1: cosine decay learning rate
- [ ] Fix Bug 2: correct validation split
- [ ] Fix Bug 3: `__main__` guard in `model.py`
- [ ] Detect overfitting (compare train vs val loss curves)
- [ ] Add `argparse` to `train.py`
- [ ] Early stopping
