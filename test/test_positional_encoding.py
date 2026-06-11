import sys
from pathlib import Path

import torch

# Add parent directory to path so we can import src
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.model import PositionalEncoding


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
    seq_len = 5
    d_model = 16
    max_seq_length = 100

    positional_encoding = PositionalEncoding(
        d_model=d_model,
        max_seq_length=max_seq_length,
    ).to(device)

    x = torch.zeros(batch_size, seq_len, d_model, device=device)
    output = positional_encoding(x)

    print("Positional Encoding")
    print_shape("input", x)
    print_shape("pe buffer", positional_encoding.pe)
    print_shape("output", output)
    print()

    print("First token positional vector:")
    print(output[0, 0])
    print()

    print("Second token positional vector:")
    print(output[0, 1])
    print()

    print("Check same position across different batch items:")
    print("output[0, 1] == output[1, 1]:", torch.allclose(output[0, 1], output[1, 1]))


if __name__ == "__main__":
    main()