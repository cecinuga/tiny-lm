# tiny-lm

A from-scratch, GPT-style language model you can train on **any** plain-text file — and at **any size** your hardware can afford. Two short files of model code, one training loop, one streaming sampler.

The architecture is fully parametric. There is no fixed model size: depth (`n_layer`), width (`n_embd`), attention heads (`n_head`) and context window (`block_size`) are all command-line flags. Run a 0.5M-parameter toy on a CPU in a minute, or scale to a multi-hundred-million-parameter model on a GPU — same code, different flags.

---

## Quick start (reproduce it on your machine)

**Prerequisites**

- Python **3.12+** (the repo pins 3.14 via `.python-version`, but `>=3.12` is enough)
- [`uv`](https://docs.astral.sh/uv/) for dependency management
- `git`
- A GPU is optional. The code auto-selects **CUDA → MPS (Apple Silicon) → CPU**; PyTorch is installed by `uv sync`.

**Setup**

```bash
git clone https://github.com/cecinuga/tiny-lm.git
cd tiny-lm
uv sync                 # creates .venv and installs torch, numpy, matplotlib, tqdm
```

**Smoke test** (~1 minute on CPU — trains a tiny model for 100 steps):

```bash
./test_train.sh
# equivalently:
uv run train.py --layer 2 --head 2 --embd 128 --block-size 64 --max-steps 100
```

**Train** a model on the bundled corpus (or point `--data` at your own `.txt`):

```bash
uv run train.py --data data/promessi_sposi.txt --max-steps 2500
```

Checkpoints, text samples and a loss log are written under `artifacts/<YYYYMMDD>/`.

**Generate** text from a checkpoint (tokens stream to stdout as they are produced):

```bash
uv run inference.py \
  --checkpoint artifacts/<YYYYMMDD>/checkpoints/<your_checkpoint>.pt \
  --prompt "Quel ramo del lago" --max_new_tokens 300
```

**Plot** the training curves (edit `LOG_PATH` at the top of the file to pick a run):

```bash
uv run loss_plot.py
```

> **Sizing constraints (read before going big).** Two invariants are enforced at startup: `n_embd` must be divisible by `n_head`, **and** `n_embd` must be divisible by `--batch-size`. Memory and time scale with `block_size²` (attention is quadratic) and with model width/depth — pick numbers your device can hold.

---

## Overview

tiny-lm trains a **character-level**, decoder-only transformer ("GPT") to predict the next character of a corpus, one step at a time. After training it can continue any prompt in the style of the text it saw. The architecture is the classic transformer block — multi-head causal self-attention + feed-forward MLP, with residual connections and pre-LayerNorm — and nothing more, which is the point: it's small enough to read end to end and modify.

It's deliberately honest about its scope. The tokenizer is character-level (no BPE, no external vocab — see roadmap), attention is dense/quadratic, and training is single-process (no distributed/multi-GPU). It is an educational, hackable codebase, not a production training stack — but the model itself has no built-in size ceiling.

## Main entities

| Entity | File | Role |
|--------|------|------|
| `GPTConfig` | `model.py` | Dataclass of the five numbers that fully define the architecture (`n_layer`, `n_head`, `n_embd`, `block_size`, `vocab_size`). |
| `GPT` | `model.py` | The model: token + positional embeddings → `n_layer` stacked `Block`s → final LayerNorm → linear head. Output head is **weight-tied** to the token embedding. |
| `Block` / `CausalSelfAttention` / `MLP` | `model.py` | One transformer layer: pre-norm causal attention (fused QKV, flash via `scaled_dot_product_attention`) + pre-norm 4×-expansion GELU MLP, each wrapped in a residual. |
| `TrainConfig` + `load_data` | `train_utils.py` | Run hyperparameters; loads a text file, builds the char vocab, tokenizes, and yields random 90/10 train/val batches. Also holds the LR schedule (`get_lr`) and device picker (`get_device`). |
| `train()` | `train.py` | The training loop: AdamW, warmup+cosine LR, periodic eval, perplexity, gradient clipping, early stopping, and checkpointing. CLI entry point. |
| `generate()` | `inference.py` | Autoregressive sampler with temperature + top-k. A **generator** that yields tokens one by one, so `inference.py` streams output live. CLI entry point. |
| `ArtifactConfig` + `save_*` | `artifact_utils.py` | Where and whether to persist checkpoints, samples, and loss logs. Filenames encode the architecture, e.g. `L6H6E384`. |

## How they interact

The two entry points share `model.py` and run two distinct flows.

**Training** (`uv run train.py …`):

```
text file ──load_data──▶ char vocab (stoi/itos) + tokens ──get_batch──▶ (x, y)
                                                                          │
        GPTConfig ───────────────▶ GPT ◀───────────────────────────────┘
                                    │  forward → logits, cross-entropy loss
                                    ▼
            AdamW + warmup/cosine LR + grad-clip ── step ──┐
                                    ▲                       │
        every max_steps/10:  eval on val split ────────────┘
            ├─ perplexity = exp(val_loss)
            ├─ save_sample / save_checkpoint("best")
            └─ early_stop(): stop if val loss stops improving
                                    │
                                    ▼
        save_checkpoint("final") + save_loss_log  →  artifacts/<date>/
```

A checkpoint is a self-contained `.pt` bundle: model weights **plus** its `GPTConfig` **plus** the `stoi`/`itos` vocab. That is the only contract between the two flows.

**Inference** (`uv run inference.py …`): load the checkpoint → rebuild the exact `GPT` from the stored config → encode the prompt with the stored vocab → `generate()` yields one token per step (each conditioned on the last `block_size` tokens), decoded and flushed to stdout immediately.

## Entry points

- **Users** run two CLIs: `train.py` (produce a model) and `inference.py` (use it). `loss_plot.py` visualizes a run.
- **Developers** start at `model.py` (the architecture, ~110 lines) and `train.py::train` (the loop). `train_utils.py` and `artifact_utils.py` hold data/IO plumbing.

## `train.py --help`

```
usage: train.py [-h] [-d DATA] [-l LAYER] [-H HEAD] [-e EMBD] [-b BLOCK_SIZE]
                [-B BATCH_SIZE] [--max-steps MAX_STEPS]
                [-n-save N_SAVE_INTERVAL] [-o-art OUT_ARTIFACT]
                [-o-chk OUT_CHECKPOINT] [-o-l OUT_LOSS_LOG] [-o-s OUT_SAMPLE]
                [-no-a] [-no-c] [-no-ll] [-no-s]

Train a GPT model

options:
  -h, --help            show this help message and exit
  -d, --data DATA       Path to dataset file (e.g. shakespeare.txt)
  -l, --layer LAYER     Number of layers
  -H, --head HEAD       Number of heads
  -e, --embd EMBD       Embedding dimension
  -b, --block-size BLOCK_SIZE
                        Block size
  -B, --batch-size BATCH_SIZE
                        Batch size
  --max-steps MAX_STEPS
                        Maximum number of training steps
  -n-save, --n-save-interval N_SAVE_INTERVAL
                        Number of time program saves an artifact (shared
                        between all artifacts)
  -o-art, --out-artifact OUT_ARTIFACT
                        Root artifact folder
  -o-chk, --out-checkpoint OUT_CHECKPOINT
                        checkpoint folder, is prefixed to artifacts folder
  -o-l, --out-loss-log OUT_LOSS_LOG
                        loss log folder, is prefixed to artifacts folder
  -o-s, --out-sample OUT_SAMPLE
                        sampling folder, is prefixed to artifacts folder
  -no-a, --no-artifact  Disable artifact saving
  -no-c, --no-checkpoint
                        Disable checkpoint saving
  -no-ll, --no-loss-log
                        Disable loss log saving
  -no-s, --no-sample    Disable sampling saving
```

## `inference.py --help`

```
usage: inference.py [-h] [--checkpoint CHECKPOINT] [--prompt PROMPT]
                    [--max_new_tokens MAX_NEW_TOKENS]
                    [--temperature TEMPERATURE] [--top_k TOP_K] [--seed SEED]

Generate text from a trained GPT checkpoint

options:
  -h, --help            show this help message and exit
  --checkpoint CHECKPOINT
                        Path to checkpoint file (e.g. checkpoint_final.pt)
  --prompt PROMPT       Starting text for generation
  --max_new_tokens MAX_NEW_TOKENS
                        Number of tokens to generate
  --temperature TEMPERATURE
                        Sampling temperature (lower = more deterministic)
  --top_k TOP_K         Only sample from top-k most likely tokens
  --seed SEED           Random seed for reproducibility
```

## Design choices

- **Character-level tokenization** — vocabulary is derived from the dataset itself, so no external tokenizer is needed and the model is corpus-agnostic.
- **Weight tying** — the token-embedding matrix and the output projection share weights (fewer parameters, better generalization).
- **Pre-norm transformer** — LayerNorm before each sub-layer, more stable to train than post-norm.
- **Flash attention** — uses `torch.nn.functional.scaled_dot_product_attention` with a causal mask instead of a hand-rolled softmax.
- **LR schedule** — linear warmup (100 steps) followed by cosine decay to 10% of the peak LR.
- **Early stopping** — training halts when the validation loss stops improving, guarding small models against memorizing small corpora.

## Repository layout

```
model.py          GPT architecture (GPTConfig, GPT, Block, attention, MLP)
train.py          training loop + CLI
inference.py      streaming text generation + CLI
train_utils.py    TrainConfig, data loading/tokenization, LR schedule, device
artifact_utils.py ArtifactConfig, checkpoint/sample/loss-log persistence
loss_plot.py      matplotlib plot of a loss log
data/             bundled corpora (promessi_sposi.txt, shakespeare.txt)
artifacts/        training outputs, grouped by date
test_train.sh     tiny end-to-end smoke run
```

## Roadmap

Honest list of what's missing, roughly in order of payoff:

- **Unique artifact names** — loss logs and samples currently overwrite each other across runs of the same architecture on the same day; filenames should include the step (or a run id).
- **BPE / subword tokenization** — character-level is simple but information-poor. Swapping in `tiktoken.get_encoding("gpt2")` would let the model use context far more efficiently.
- **Larger corpora** — the bundled texts are ~1 MB and easily memorized; a Wikipedia-scale dump would exercise the "arbitrary size" claim properly.
- **Experiment tracking** — `loss_plot.py` covers a single run; tools like Weights & Biases or TensorBoard would help compare many.
