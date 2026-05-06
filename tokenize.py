text = open("./data/shakespeare.txt").read()
chars = sorted(set(text))
vocab_size = len(chars) # 65 for Sheakspeare

stoi = { k: i for i, k in enumerate(chars) }
itos = { i: k for i, k in stoi.items() }

def encode(s: str):
    """transforms a string into its int-vector representation according to character level reference stystem.

    encode('Hello') -> [20, 43, 50, 50, 53]"""
    return [stoi[c] for c in s]


def decode(ids: list[int]):
    """transforms a int-vector into it's string representation according to character level reference stystem.
    
    decode("[20, 43, 50, 50, 53]") -> 'Hello'"""
    return "".join([itos[i] for i in ids])