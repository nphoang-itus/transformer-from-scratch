from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim

ROOT_DIR = Path(__file__).resolve().parents[1]

try:
    from .checkpoint import load_checkpoint, save_checkpoint
    from .inference import greedy_decode, trim_after_eos
    from .model import Transformer
    from .utils import (
        EOS_IDX,
        PAD_IDX,
        SOS_IDX,
        build_model,
        generate_copy_batch,
        get_device,
    )
except ImportError:  # Allows `python src/train_copy_with_checkpoint.py`.
    from checkpoint import load_checkpoint, save_checkpoint
    from inference import greedy_decode, trim_after_eos
    from model import Transformer
    from utils import (
        EOS_IDX,
        PAD_IDX,
        SOS_IDX,
        build_model,
        generate_copy_batch,
        get_device,
    )


def train_on_fixed_batch(
    model: Transformer,
    optimizer: optim.Optimizer,
    criterion: nn.Module,
    fixed_src: torch.Tensor,
    fixed_tgt: torch.Tensor,
    vocab_size: int,
    num_steps: int,
) -> float:
    """Overfit one fixed copy-task batch to verify the Transformer can learn."""
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


def put_copy_example_in_batch(
    src_batch: torch.Tensor,
    tgt_batch: torch.Tensor,
    tokens: list[int],
) -> None:
    """Overwrite the first batch item with a known copy-task example.

    src_batch: [B, S]
    tgt_batch: [B, S + 2]
    """
    if len(tokens) > src_batch.size(1):
        raise ValueError("example tokens are longer than the source sequence length")

    src_batch[0].fill_(PAD_IDX)
    tgt_batch[0].fill_(PAD_IDX)

    src_batch[0, : len(tokens)] = torch.tensor(
        tokens,
        dtype=torch.long,
        device=src_batch.device,
    )
    tgt_batch[0, 0] = SOS_IDX
    tgt_batch[0, 1 : len(tokens) + 1] = torch.tensor(
        tokens,
        dtype=torch.long,
        device=tgt_batch.device,
    )
    tgt_batch[0, len(tokens) + 1] = EOS_IDX


def print_decode_results(
    model: Transformer,
    src: torch.Tensor,
    target: torch.Tensor,
    max_decode_len: int,
) -> None:
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
    put_copy_example_in_batch(
        src_batch=fixed_src,
        tgt_batch=fixed_tgt,
        tokens=[13, 13, 7, 8],
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
