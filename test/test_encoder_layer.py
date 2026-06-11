import sys
from pathlib import Path

import torch

# Add parent directory to path so we can import src
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.model import EncoderLayer


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
    d_model = 16
    num_heads = 4
    d_ff = 64
    dropout = 0.1

    encoder_layer = EncoderLayer(
        d_model=d_model,
        num_heads=num_heads,
        d_ff=d_ff,
        dropout=dropout,
    ).to(device)

    x = torch.randn(batch_size, src_seq_len, d_model, device=device)

    output = encoder_layer(x)

    print("EncoderLayer without mask")
    print_shape("input", x)
    print_shape("output", output)
    print()

    print("Expected:")
    print(f"input  = [{batch_size}, {src_seq_len}, {d_model}]")
    print(f"output = [{batch_size}, {src_seq_len}, {d_model}]")
    print()

    # Optional: test with source padding mask
    # Shape expected by attention scores:
    # scores = [B, num_heads, query_len, key_len]
    #
    # src_mask should be broadcastable to:
    # [B, num_heads, src_seq_len, src_seq_len]
    #
    # Here we create:
    # [B, 1, 1, src_seq_len]
    #
    # Meaning:
    # For each query token, do not attend to masked key positions.
    src_mask = torch.ones(batch_size, 1, 1, src_seq_len, device=device)

    # Simulate padding at the last position of the first sample
    src_mask[0, :, :, -1] = 0

    output_with_mask = encoder_layer(x, src_mask=src_mask)

    print("EncoderLayer with src_mask")
    print_shape("src_mask", src_mask)
    print_shape("output_with_mask", output_with_mask)


if __name__ == "__main__":
    main()