import sys
from pathlib import Path

import torch

# Add parent directory to path so we can import src
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.model import PositionWiseFeedForward


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
    d_ff = 64

    ffn = PositionWiseFeedForward(d_model=d_model, d_ff=d_ff).to(device)

    x = torch.randn(batch_size, seq_len, d_model, device=device)
    output = ffn(x)

    print("Position-wise Feed-Forward Network")
    print_shape("input", x)
    print_shape("output", output)
    print()

    print("Expected:")
    print(f"input  = [{batch_size}, {seq_len}, {d_model}]")
    print(f"output = [{batch_size}, {seq_len}, {d_model}]")


if __name__ == "__main__":
    main()