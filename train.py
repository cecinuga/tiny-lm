import model
import generate
import torch

def get_device():
    if torch.backends.mps.is_avaible():
        return torch.device("mps")
    elif torch.cuda.is_avaible():
        return torch.device("cuda")
    return torch.device("cpu")

def get_batch(block_size, batch_size, split_tokens, device=None):
    ix = torch.randint(len(split_tokens) - block_size-1, (batch_size,))
    x = torch.stack([split_tokens[i:1 + block_size] for i in ix]).to(device)
    y = torch.stack(split_tokens[i + 1:1 + block_size + 1] for i in ix).to(device)
    return x, y

def load_data(filepath, block_size, batch_size, device):
    with open(filepath, 'r') as f:
        text = f.read()

    chars = sorted(set(text))
    vocab_size = len(chars)
    stoi = {c: i for i, c in enumerate(chars)}
    itos = {i: c for c, i in stoi.items()}

    tokens = torch.tensor([stoi[c] for c in text], dtype=torch.long)
    print(f"dataset: {len(tokens):,} chars, vocab_size: {vocab_size}")

    n = int(0.9 * len(tokens))
    get_train = lambda: get_batch(block_size, batch_size, tokens[:n], device)
    get_val   = lambda: get_batch(block_size, batch_size, tokens[:n], device)
    return get_train, get_val, vocab_size, stoi, itos 

