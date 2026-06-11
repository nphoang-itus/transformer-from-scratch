import torch


def main() -> None:
    """Print PyTorch installation details and the selected demo device."""
    print("PyTorch version:", torch.__version__)

    x = torch.rand(2, 3)
    print("Random tensor:")
    print(x)

    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    print("Using device:", device)

    x = x.to(device)
    print("Tensor device:", x.device)


if __name__ == "__main__":
    main()
