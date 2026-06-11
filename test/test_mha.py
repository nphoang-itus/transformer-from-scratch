import sys
from pathlib import Path

import torch

# Add parent directory to path so we can import src
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.model import MultiHeadAttention


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

    attention = MultiHeadAttention(d_model, num_heads).to(device)

    # Case 1: Encoder self-attention
    # query, key, value all come from src
    src = torch.randn(batch_size, src_seq_len, d_model, device=device)

    encoder_output = attention(src, src, src)

    print("Case 1: Encoder self-attention")
    print_shape("src", src)
    print_shape("encoder_output", encoder_output)
    print()

    # Case 2: Decoder self-attention
    # query, key, value all come from tgt
    tgt = torch.randn(batch_size, tgt_seq_len, d_model, device=device)

    decoder_self_output = attention(tgt, tgt, tgt)

    print("Case 2: Decoder self-attention")
    print_shape("tgt", tgt)
    print_shape("decoder_self_output", decoder_self_output)
    print()

    # Case 3: Encoder-decoder attention
    # query comes from decoder
    # key and value come from encoder
    cross_output = attention(tgt, src, src)

    print("Case 3: Encoder-decoder attention")
    print_shape("query/tgt", tgt)
    print_shape("key/value/src", src)
    print_shape("cross_output", cross_output)


if __name__ == "__main__":
    main()