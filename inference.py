import torch
from sys import stdout
from model import GPT, GPTConfig

@torch.no_grad()
def generate(model:GPT, prompt, stoi, itos, max_new_tokens=200, temperature=0.8, top_k=40):
    """
    Autoregressively generate text from a prompt using top-k sampling with temperature.

    Characters in the prompt not present in the vocabulary are silently skipped.
    Returns the full string (prompt + generated tokens).
    """
    device = next(model.parameters()).device

    tokens = [stoi[c] for c in prompt if c in stoi]
    idx = torch.tensor([tokens], dtype=torch.long, device=device)

    model.eval()
    for _ in range(max_new_tokens):
        idx_cond = idx[:, -model.config.block_size :]

        logits, _ = model(idx_cond)
        logits = logits[:, -1, :] / temperature

        if top_k > 0:
            values, _ = torch.topk(logits, top_k)
            logits[logits < values[:, -1:]] = float("-inf")

        probs = torch.softmax(logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)
        idx = torch.cat([idx, next_token], dim=-1)
        yield next_token


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Generate text from a trained GPT checkpoint")
    parser.add_argument("--checkpoint", help="Path to checkpoint file (e.g. checkpoint_final.pt)")
    parser.add_argument("--prompt", default="To be or not", help="Starting text for generation")
    parser.add_argument("--max_new_tokens", type=int, default=200, help="Number of tokens to generate")
    parser.add_argument("--temperature", type=float, default=0.8, help="Sampling temperature (lower = more deterministic)")
    parser.add_argument("--top_k", type=int, default=40, help="Only sample from top-k most likely tokens")
    parser.add_argument("--seed", type=int, default=None, help="Random seed for reproducibility")
    args = parser.parse_args()

    if args.seed is not None:
        torch.manual_seed(args.seed)

    torch.serialization.add_safe_globals([GPTConfig])
    checkpoint = torch.load(args.checkpoint, weights_only=True)

    config = checkpoint["config"]
    stoi = checkpoint["stoi"]
    itos = checkpoint["itos"]

    model = GPT(config)
    model.load_state_dict(checkpoint["model_state_dict"])

    for token in generate(model, args.prompt, stoi, itos,
                      max_new_tokens=args.max_new_tokens,
                      temperature=args.temperature,
                      top_k=args.top_k):
        c = itos[token.item()]
        stdout.write(f"{c}")
        stdout.flush()
    print()
