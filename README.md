# tiny-lm

A tiny GPT-style language model trained from scratch on any text file, runnable on a laptop.

## What this is

tiny-lm implements a character-level GPT with ~11.5M parameters. It learns to generate text in the style of the training corpus by predicting the next character one step at a time. The architecture follows the original transformer paper: multi-head causal self-attention + feed-forward blocks with residual connections and layer normalization.

## Installation

```bash
# Requires Python 3.12+ and uv
uv sync
```

## Usage

**Train:**
```bash
python train.py [options]
```
| Flag | Default | Description |
|------|---------|-------------|
| `-d`, `--data` | `data/promessi_sposi.txt` | Path to training dataset |
| `-l`, `--layer` | 6 | Number of transformer layers |
| `-H`, `--head` | 6 | Number of attention heads |
| `-e`, `--embd` | 384 | Embedding dimension |
| `-b`, `--block-size` | 256 | Context window size |
| `-B`, `--batch-size` | 64 | Batch size |
| `--max-steps` | 2500 | Training steps |

Artifact output is controlled separately:

| Flag | Default | Description |
|------|---------|-------------|
| `-n-save`, `--n-save-interval` | 10 | How many times artifacts are saved during training |
| `-o-art`, `--out-artifact` | `artifacts/` | Root artifact directory |
| `-o-chk`, `--out-checkpoint` | `checkpoints/` | Checkpoint subfolder (under artifact root) |
| `-o-l`, `--out-loss-log` | `loss_logs/` | Loss log subfolder |
| `-o-s`, `--out-sampling` | `sampling/` | Sampling subfolder |
| `-no-a`, `--no-artifact` | — | Disable all artifact saving |
| `-no-c`, `--no-checkpoint` | — | Disable checkpoint saving |
| `-no-ll`, `--no-loss-log` | — | Disable loss log saving |
| `-no-s`, `--no-sampling` | — | Disable sampling saving |

**Plot learning curves:**
```bash
python loss_plot.py
```
Displays train loss (raw + smoothed), val loss, and perplexity from a `loss_log.json` file. Edit `LOG_PATH` at the top of the file to point to a different run. Requires `matplotlib` (`uv add matplotlib`).

**Generate text from a checkpoint:**
```bash
python inference.py --checkpoint artifacts/checkpoints/<your_checkpoint>.pt --prompt "To be or not"
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
- **Character-level tokenization** — vocabulary size depends on dataset, no external tokenizer needed
- **Weight tying** — the token embedding matrix and the output projection share weights (reduces parameters, improves generalization)
- **Pre-norm** — LayerNorm before each sub-layer (more stable than post-norm)
- **Learning rate schedule** — linear warmup (100 steps) + cosine decay

Default config: 6 layers, 6 heads, 384 embedding dim, 256 context window, batch size 64.

## Improvement Ideas

### 0. Improved artifacts handling
0. artifacts folder structure must be refactored: artifacts/`today`
  1. every sub folder of artifacts must be in artifacts/`today`
  2. every artifact name must be unique
  3. every artifact name must not contain any date
2. Check for duplicate file names before saving checkpoints and loss logs 
3. Save inference sampling in a text file during training
5. Implement disable mid-checkpoint saving flag

### 8. Saving inference sampling in a text file during training 
I wanna save sampling inference in a unique text file, separating each sample with a delimiter.
Training samplice not work

### 9. Train on Wikipedia-ITA
Shakespeare is ~1MB. The model memorizes it quickly. `data/promessi_sposi.txt` is already included as an alternative. For something larger, Wikipedia-ITA dumps

### 10. Add BPE tokenization
Character-level tokenization is simple but inefficient: each token carries little information, so the model needs a long context to understand meaning. Real language models (GPT, LLaMA) use BPE tokenizers that split text into subwords (e.g., "Shake" + "speare"). `tiktoken` is already in the dependencies — try replacing the character tokenizer with `tiktoken.get_encoding("gpt2")`.

### 12. Experiment tracking
Loss curves are already plotted by `loss_plot.py`. For bigger experiments, tools like [Weights & Biases](https://wandb.ai) or TensorBoard let you compare runs, visualize attention patterns, and track GPU utilization.

### 14. Add streaming in inference
Stream the output tokens one by one rather than waiting for the model to generate all at once.
