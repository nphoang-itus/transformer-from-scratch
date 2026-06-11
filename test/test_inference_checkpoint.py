import sys
from pathlib import Path

import torch
import torch.optim as optim

# Add parent directory to path so we can import src
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.checkpoint import load_checkpoint, save_checkpoint
from src.inference import greedy_decode, trim_after_eos
from src.model import Transformer


PAD_IDX = 0
SOS_IDX = 1
EOS_IDX = 2


def get_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def print_shape(name: str, tensor: torch.Tensor):
    print(f"{name:<30} {list(tensor.shape)}")


def build_model(config: dict, device: torch.device) -> Transformer:
    return Transformer(
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
    ).to(device)


def main():
    device = get_device()
    print("Using device:", device)
    print()

    config = {
        "src_vocab_size": 30,
        "tgt_vocab_size": 30,
        "d_model": 16,
        "num_heads": 4,
        "num_layers": 2,
        "d_ff": 64,
        "max_seq_length": 10,
        "dropout": 0.0,
        "src_pad_idx": PAD_IDX,
        "tgt_pad_idx": PAD_IDX,
    }

    model = build_model(config, device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    model.eval()

    src = torch.tensor(
        [
            [7, 11, 15, 0, 0, 0],
            [9, 12, 18, 21, 23, 0],
        ],
        dtype=torch.long,
        device=device,
    )

    tgt_input = torch.tensor(
        [
            [1, 7, 11, 15, 2, 0, 0],
            [1, 9, 12, 18, 21, 23, 2],
        ],
        dtype=torch.long,
        device=device,
    )

    with torch.no_grad():
        logits_before = model(src, tgt_input)

    checkpoint_path = Path(__file__).parent.parent / "checkpoints" / "test_transformer.pt"
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    save_checkpoint(
        path=checkpoint_path,
        model=model,
        optimizer=optimizer,
        step=123,
        loss=0.456,
        config=config,
    )

    loaded_model = build_model(config, device)
    loaded_optimizer = optim.Adam(loaded_model.parameters(), lr=1e-3)

    metadata = load_checkpoint(
        path=checkpoint_path,
        model=loaded_model,
        optimizer=loaded_optimizer,
        device=device,
    )

    loaded_model.eval()

    with torch.no_grad():
        logits_after = loaded_model(src, tgt_input)

    same_logits = torch.allclose(
        logits_before,
        logits_after,
        atol=1e-5,
    )

    print("Checkpoint test")
    print("checkpoint path:", checkpoint_path)
    print("loaded step:", metadata["step"])
    print("loaded loss:", metadata["loss"])
    print("same logits before/after load:", same_logits)
    print()

    print("Inference test")
    generated = greedy_decode(
        model=loaded_model,
        src=src,
        max_decode_len=8,
    )

    print_shape("src", src)
    print_shape("generated", generated)
    print()

    for i in range(src.size(0)):
        raw_tokens = generated[i].tolist()
        trimmed_tokens = trim_after_eos(raw_tokens)

        print(f"Sample {i + 1}")
        print("src:       ", src[i].tolist())
        print("generated: ", raw_tokens)
        print("trimmed:   ", trimmed_tokens)
        print()


if __name__ == "__main__":
    main()