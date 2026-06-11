import sys
from pathlib import Path

import torch

# Add parent directory to path so we can import src
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.model import Transformer


def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def print_shape(name, tensor):
    print(f"{name:<30} {list(tensor.shape)}")


def main():
    device = get_device()

    batch_size = 2
    src_seq_len = 5
    tgt_seq_len = 6

    src_vocab_size = 100
    tgt_vocab_size = 120

    d_model = 16
    num_heads = 4
    num_layers = 2
    d_ff = 64
    max_seq_length = 100
    dropout = 0.1

    src_pad_idx = 0
    tgt_pad_idx = 0

    model = Transformer(
        src_vocab_size=src_vocab_size,
        tgt_vocab_size=tgt_vocab_size,
        d_model=d_model,
        num_heads=num_heads,
        num_layers=num_layers,
        d_ff=d_ff,
        max_seq_length=max_seq_length,
        dropout=dropout,
        src_pad_idx=src_pad_idx,
        tgt_pad_idx=tgt_pad_idx,
    ).to(device)

    src = torch.randint(
        low=1,
        high=src_vocab_size,
        size=(batch_size, src_seq_len),
        device=device,
    )

    tgt = torch.randint(
        low=1,
        high=tgt_vocab_size,
        size=(batch_size, tgt_seq_len),
        device=device,
    )

    # Simulate padding tokens
    src[0, -1] = src_pad_idx
    tgt[0, -2:] = tgt_pad_idx

    src_mask, tgt_mask = model.create_masks(src, tgt)

    logits = model(src, tgt)

    print("Full Transformer forward pass")
    print_shape("src", src)
    print_shape("tgt", tgt)
    print_shape("src_mask", src_mask)
    print_shape("tgt_mask", tgt_mask)
    print_shape("logits", logits)
    print()

    print("Expected:")
    print(f"src     = [{batch_size}, {src_seq_len}]")
    print(f"tgt     = [{batch_size}, {tgt_seq_len}]")
    print(f"logits  = [{batch_size}, {tgt_seq_len}, {tgt_vocab_size}]")
    print()

    print("Example src:")
    print(src)
    print()

    print("Example tgt:")
    print(tgt)
    print()

    print("Target mask for first sample:")
    print(tgt_mask[0, 0])


if __name__ == "__main__":
    main()