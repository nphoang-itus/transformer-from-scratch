from __future__ import annotations

import torch

try:
    from .model import Transformer
except ImportError:  # Allows `python src/some_script.py`.
    from model import Transformer


PAD_IDX = 0
SOS_IDX = 1
EOS_IDX = 2


def get_device() -> torch.device:
    """Return the best available PyTorch device for these small demos."""
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def print_shape(name: str, tensor: torch.Tensor) -> None:
    """Print a compact tensor shape line for educational scripts."""
    print(f"{name:<35} {list(tensor.shape)}")


def generate_copy_batch(
    batch_size: int,
    min_len: int,
    max_len: int,
    vocab_size: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Generate padded source/target tensors for the copy task.

    Token ids 0, 1, and 2 are reserved for PAD, SOS, and EOS.

    Returns:
        src: [B, max_len]
        tgt: [B, max_len + 2], where each row is [SOS] + src_tokens + [EOS] + PAD...
    """
    src_batch: list[torch.Tensor] = []
    tgt_batch: list[torch.Tensor] = []

    for _ in range(batch_size):
        seq_len = int(torch.randint(min_len, max_len + 1, (1,)).item())
        tokens = torch.randint(low=3, high=vocab_size, size=(seq_len,))

        src = torch.full((max_len,), PAD_IDX, dtype=torch.long)
        src[:seq_len] = tokens

        tgt = torch.full((max_len + 2,), PAD_IDX, dtype=torch.long)
        tgt[0] = SOS_IDX
        tgt[1 : seq_len + 1] = tokens
        tgt[seq_len + 1] = EOS_IDX

        src_batch.append(src)
        tgt_batch.append(tgt)

    return torch.stack(src_batch).to(device), torch.stack(tgt_batch).to(device)


def build_model(config: dict[str, int | float], device: torch.device) -> Transformer:
    """Create a Transformer from a checkpoint/training config and move it to device."""
    model = Transformer(
        src_vocab_size=int(config["src_vocab_size"]),
        tgt_vocab_size=int(config["tgt_vocab_size"]),
        d_model=int(config["d_model"]),
        num_heads=int(config["num_heads"]),
        num_layers=int(config["num_layers"]),
        d_ff=int(config["d_ff"]),
        max_seq_length=int(config["max_seq_length"]),
        dropout=float(config["dropout"]),
        src_pad_idx=int(config["src_pad_idx"]),
        tgt_pad_idx=int(config["tgt_pad_idx"]),
    )
    return model.to(device)
