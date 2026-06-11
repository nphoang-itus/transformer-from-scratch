import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim

ROOT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT_DIR / "src"
sys.path.append(str(SRC_DIR))

from checkpoint import load_checkpoint, save_checkpoint
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


def generate_copy_batch(
    batch_size: int,
    min_len: int,
    max_len: int,
    vocab_size: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Generate copy-task data.

    Example:
        src = [7, 12, 5, PAD, PAD]
        tgt = [SOS, 7, 12, 5, EOS, PAD, PAD]
    """
    src_batch = []
    tgt_batch = []

    for _ in range(batch_size):
        seq_len = int(
            torch.randint(
                low=min_len,
                high=max_len + 1,
                size=(1,),
            ).item()
        )

        tokens = torch.randint(
            low=3,
            high=vocab_size,
            size=(seq_len,),
        )

        src = torch.full((max_len,), PAD_IDX, dtype=torch.long)
        src[:seq_len] = tokens

        tgt = torch.full((max_len + 2,), PAD_IDX, dtype=torch.long)
        tgt[0] = SOS_IDX
        tgt[1 : seq_len + 1] = tokens
        tgt[seq_len + 1] = EOS_IDX

        src_batch.append(src)
        tgt_batch.append(tgt)

    src_batch = torch.stack(src_batch).to(device)
    tgt_batch = torch.stack(tgt_batch).to(device)

    return src_batch, tgt_batch


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


def train_on_fixed_batch(
    model: Transformer,
    optimizer: optim.Optimizer,
    criterion: nn.Module,
    fixed_src: torch.Tensor,
    fixed_tgt: torch.Tensor,
    vocab_size: int,
    num_steps: int,
) -> float:
    """
    Overfit one fixed batch to verify the Transformer can learn.
    """
    model.train()

    last_loss = 0.0

    for step in range(1, num_steps + 1):
        tgt_input = fixed_tgt[:, :-1]
        tgt_label = fixed_tgt[:, 1:]

        logits = model(fixed_src, tgt_input)

        loss = criterion(
            logits.reshape(-1, vocab_size),
            tgt_label.reshape(-1),
        )

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        last_loss = loss.item()

        if step == 1 or step % 50 == 0:
            print(f"step {step:04d} | loss = {last_loss:.4f}")

    return last_loss


def print_decode_results(
    model: Transformer,
    src: torch.Tensor,
    target: torch.Tensor,
    max_decode_len: int,
):
    model.eval()

    with torch.no_grad():
        generated = greedy_decode(
            model=model,
            src=src,
            max_decode_len=max_decode_len,
            sos_idx=SOS_IDX,
            eos_idx=EOS_IDX,
            pad_idx=PAD_IDX,
        )

    for i in range(src.size(0)):
        raw_generated = generated[i].tolist()
        trimmed_generated = trim_after_eos(
            raw_generated,
            eos_idx=EOS_IDX,
            pad_idx=PAD_IDX,
        )

        print()
        print(f"Sample {i + 1}")
        print("src:               ", src[i].tolist())
        print("target:            ", target[i].tolist())
        print("generated raw:     ", raw_generated)
        print("generated trimmed: ", trimmed_generated)


def main():
    device = get_device()
    print("Using device:", device)
    print()

    config = {
        "src_vocab_size": 20,
        "tgt_vocab_size": 20,
        "d_model": 32,
        "num_heads": 4,
        "num_layers": 2,
        "d_ff": 128,
        "max_seq_length": 8,
        "dropout": 0.0,
        "src_pad_idx": PAD_IDX,
        "tgt_pad_idx": PAD_IDX,
    }

    vocab_size = config["src_vocab_size"]
    max_src_len = 6
    batch_size = 16
    num_steps = 800
    learning_rate = 1e-3

    checkpoint_path = ROOT_DIR / "checkpoints" / "copy_transformer.pt"

    model = build_model(config, device)

    optimizer = optim.Adam(
        model.parameters(),
        lr=learning_rate,
    )

    criterion = nn.CrossEntropyLoss(
        ignore_index=PAD_IDX,
    )

    fixed_src, fixed_tgt = generate_copy_batch(
        batch_size=batch_size,
        min_len=3,
        max_len=max_src_len,
        vocab_size=vocab_size,
        device=device,
    )

    print("Fixed training sample:")
    print("src:", fixed_src[0].tolist())
    print("tgt:", fixed_tgt[0].tolist())
    print()

    print("Training...")
    final_loss = train_on_fixed_batch(
        model=model,
        optimizer=optimizer,
        criterion=criterion,
        fixed_src=fixed_src,
        fixed_tgt=fixed_tgt,
        vocab_size=vocab_size,
        num_steps=num_steps,
    )

    print()
    print("Saving checkpoint...")

    save_checkpoint(
        path=checkpoint_path,
        model=model,
        optimizer=optimizer,
        step=num_steps,
        loss=final_loss,
        config=config,
    )

    print("Saved to:", checkpoint_path)
    print()

    print("Decoding with original trained model:")
    print_decode_results(
        model=model,
        src=fixed_src[:5],
        target=fixed_tgt[:5],
        max_decode_len=max_src_len + 1,
    )

    print()
    print("=" * 80)
    print()

    print("Loading checkpoint into a fresh model...")

    loaded_model = build_model(config, device)
    loaded_optimizer = optim.Adam(
        loaded_model.parameters(),
        lr=learning_rate,
    )

    metadata = load_checkpoint(
        path=checkpoint_path,
        model=loaded_model,
        optimizer=loaded_optimizer,
        device=device,
    )

    print("Loaded checkpoint metadata:")
    print("step:", metadata["step"])
    print("loss:", metadata["loss"])
    print()

    print("Decoding with loaded model:")
    print_decode_results(
        model=loaded_model,
        src=fixed_src[:5],
        target=fixed_tgt[:5],
        max_decode_len=max_src_len + 1,
    )


if __name__ == "__main__":
    main()