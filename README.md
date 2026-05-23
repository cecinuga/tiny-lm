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
python inference.py --checkpoint checkpoints/<your_checkpoint>.pt --prompt "To be or not"
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

## Improvement Ideas

### 0. Add TrainConfig class in train.py and insert variable used by training

### 1. Saving health_check must be done after each `max_step/10` epochs

### 2. Improove overfitting detection system
The term `value` refers to a general term that is used to measure the performance of the model.
Save the last `n` `value` in an array, initialized with only `neutral value`, each successive element for being part to the array it must be greater than the last `n` values.
To detect overfitting the actual `value` must fail the aforementioned control `m` times in a row, `m` is computed as `n / 2`.
- 2.1 `value` = actual best_val_loss; `neutral value` = `float('inf')`
- 2.2 `value` = gap between actual best_val_loss and train_loss; `neutral value` = `0`: the control is inverted, if to much value is appended to array the overfitting will be detected
 
### 3. Saving inference sampling in a text file during training 
I wanna save sampling inference in a unique text file, separating each sample with a delimiter.
Training samplice not work

### 4. Train on Wikipedia-ITA
Shakespeare is ~1MB. The model memorizes it quickly. `data/promessi_sposi.txt` is already included as an alternative. For something larger, Wikipedia-ITA dumps

### 5. Add BPE tokenization

### 6. Subword tokenization
Character-level tokenization is simple but inefficient: each token carries little information, so the model needs a long context to understand meaning. Real language models (GPT, LLaMA) use BPE tokenizers that split text into subwords (e.g., "Shake" + "speare"). `tiktoken` is already in the dependencies — try replacing the character tokenizer with `tiktoken.get_encoding("gpt2")`.

### 7. Experiment tracking
Saving losses to `loss_log.json` is a start. Plot them with matplotlib to see the learning curves. For bigger experiments, tools like [Weights & Biases](https://wandb.ai) or TensorBoard let you compare runs, visualize attention patterns, and track GPU utilization.

### 8. Gradient norm monitoring
The code already clips gradients (`clip_grad_norm_`). Also log the gradient norm before clipping — a suddenly large gradient norm often signals numerical instability or a bad batch.

```python
grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
pbar.set_postfix(loss=f"{loss.item():.4f}", lr=f"{lr:.2e}", gnorm=f"{grad_norm:.2f}")
```

### 7. Add streaming in inference
Stream the output tokens one by one rather than waiting for the model to generate all at once.
