import argparse
import sys
from pathlib import Path

import torch

ROOT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT_DIR / "src"
sys.path.append(str(SRC_DIR))

from checkpoint import load_checkpoint
from inference import greedy_decode, trim_after_eos
from model import Transformer


PAD_IDX = 0
SOS_IDX = 1
EOS_IDX = 2


def get_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def build_model(config: dict, device: torch.device) -> Transformer:
    model = Transformer(
        src_vocab_size=config["src_vocab_size"],
        tgt_vocab_size=config["tgt_vocab_size"],
        d_model=config["d_model"],
        num_heads=config["num_heads"],
        num_layers=config["num_layers"],
        d_ff=config["d_ff"],
        max_seq_length=config["max_seq_length"],
        dropout=config["dropout"],
        src_pad_idx=config["src_pad_idx"],
        tgt_pad_idx=config["tgt_pad_idx"],
    )

    return model.to(device)


def prepare_src_tensor(
    src_tokens: list[int],
    max_src_len: int,
    device: torch.device,
) -> torch.Tensor:
    """
    Convert user input token ids into padded tensor.

    Example:
        src_tokens = [13, 13, 7, 8]
        max_src_len = 6

        result = [[13, 13, 7, 8, 0, 0]]
    """
    if len(src_tokens) == 0:
        raise ValueError("src must contain at least one token")

    if len(src_tokens) > max_src_len:
        raise ValueError(
            f"src is too long. Got {len(src_tokens)} tokens, "
            f"but max allowed is {max_src_len}."
        )

    for token in src_tokens:
        if token in [PAD_IDX, SOS_IDX, EOS_IDX]:
            raise ValueError(
                f"Invalid normal token {token}. "
                f"0, 1, 2 are reserved for PAD, SOS, EOS."
            )

    src = torch.full(
        size=(1, max_src_len),
        fill_value=PAD_IDX,
        dtype=torch.long,
        device=device,
    )

    src[0, : len(src_tokens)] = torch.tensor(
        src_tokens,
        dtype=torch.long,
        device=device,
    )

    return src


def decode_output(token_ids: list[int]) -> list[int]:
    """
    For copy task, remove special tokens from generated sequence.

    Example:
        [1, 13, 13, 7, 8, 2] -> [13, 13, 7, 8]
    """
    result = []

    for token in token_ids:
        if token == SOS_IDX:
            continue
        if token == EOS_IDX:
            break
        if token == PAD_IDX:
            continue

        result.append(token)

    return result


def parse_args():
    parser = argparse.ArgumentParser(
        description="Load trained Transformer checkpoint and run copy-task inference."
    )

    parser.add_argument(
        "--checkpoint",
        type=str,
        default=str(ROOT_DIR / "checkpoints" / "copy_transformer.pt"),
        help="Path to model checkpoint.",
    )

    parser.add_argument(
        "--src",
        type=int,
        nargs="+",
        required=True,
        help="Source token ids, e.g. --src 13 13 7 8",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    device = get_device()
    checkpoint_path = Path(args.checkpoint)

    print("Using device:", device)
    print("Checkpoint:", checkpoint_path)
    print()

    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {checkpoint_path}\n"
            f"Run this first:\n"
            f"python src/train_copy_with_checkpoint.py"
        )

    # First load checkpoint metadata to get config.
    raw_checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
    )

    config = raw_checkpoint.get("config")

    if config is None:
        raise ValueError(
            "Checkpoint does not contain config. "
            "Please save checkpoint with config first."
        )

    model = build_model(config, device)

    metadata = load_checkpoint(
        path=checkpoint_path,
        model=model,
        optimizer=None,
        device=device,
    )

    model.eval()

    max_seq_length = config["max_seq_length"]

    # In this copy task:
    # target = [SOS] + src + [EOS]
    # so max source length = max_seq_length - 2
    max_src_len = max_seq_length - 2

    src = prepare_src_tensor(
        src_tokens=args.src,
        max_src_len=max_src_len,
        device=device,
    )

    with torch.no_grad():
        generated = greedy_decode(
            model=model,
            src=src,
            max_decode_len=max_src_len + 1,
            sos_idx=SOS_IDX,
            eos_idx=EOS_IDX,
            pad_idx=PAD_IDX,
        )

    generated_raw = generated[0].tolist()
    generated_trimmed = trim_after_eos(
        generated_raw,
        eos_idx=EOS_IDX,
        pad_idx=PAD_IDX,
    )

    decoded_tokens = decode_output(generated_trimmed)

    print("Loaded checkpoint metadata:")
    print("step:", metadata["step"])
    print("loss:", metadata["loss"])
    print()

    print("Input:")
    print("src tokens:         ", args.src)
    print("src padded tensor:  ", src[0].tolist())
    print()

    print("Generated:")
    print("raw:                ", generated_raw)
    print("trimmed:            ", generated_trimmed)
    print("decoded copy output:", decoded_tokens)


if __name__ == "__main__":
    main()