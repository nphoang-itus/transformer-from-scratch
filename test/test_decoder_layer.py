import sys
from pathlib import Path

import torch

# Add parent directory to path so we can import src
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.model import DecoderLayer


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
    d_model = 16
    num_heads = 4
    d_ff = 64
    dropout = 0.1

    decoder_layer = DecoderLayer(
        d_model=d_model,
        num_heads=num_heads,
        d_ff=d_ff,
        dropout=dropout,
    ).to(device)

    # Decoder input
    x = torch.randn(batch_size, tgt_seq_len, d_model, device=device)

    # Encoder output
    encoder_output = torch.randn(batch_size, src_seq_len, d_model, device=device)

    # Case 1: without masks
    output = decoder_layer(
        x=x,
        encoder_output=encoder_output,
    )

    print("DecoderLayer without masks")
    print_shape("decoder input x", x)
    print_shape("encoder_output", encoder_output)
    print_shape("output", output)
    print()

    print("Expected:")
    print(f"x              = [{batch_size}, {tgt_seq_len}, {d_model}]")
    print(f"encoder_output = [{batch_size}, {src_seq_len}, {d_model}]")
    print(f"output         = [{batch_size}, {tgt_seq_len}, {d_model}]")
    print()

    # Case 2: with masks

    # Source mask:
    # Shape should be broadcastable to cross-attention scores:
    # scores = [B, num_heads, tgt_seq_len, src_seq_len]
    #
    # src_mask = [B, 1, 1, src_seq_len]
    src_mask = torch.ones(batch_size, 1, 1, src_seq_len, device=device)

    # Simulate padding in source:
    # First sample masks the last source token.
    src_mask[0, :, :, -1] = 0

    # Target look-ahead mask:
    # Shape should be broadcastable to decoder self-attention scores:
    # scores = [B, num_heads, tgt_seq_len, tgt_seq_len]
    #
    # tgt_mask = [1, 1, tgt_seq_len, tgt_seq_len]
    tgt_mask = torch.tril(
        torch.ones(tgt_seq_len, tgt_seq_len, device=device)
    ).unsqueeze(0).unsqueeze(0)

    output_with_masks = decoder_layer(
        x=x,
        encoder_output=encoder_output,
        src_mask=src_mask,
        tgt_mask=tgt_mask,
    )

    print("DecoderLayer with src_mask and tgt_mask")
    print_shape("src_mask", src_mask)
    print_shape("tgt_mask", tgt_mask)
    print_shape("output_with_masks", output_with_masks)
    print()

    print("Target look-ahead mask:")
    print(tgt_mask[0, 0])


if __name__ == "__main__":
    main()