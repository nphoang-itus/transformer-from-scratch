import torch
import torch.nn as nn
import torch.optim as optim

try:
    from .inference import greedy_decode
    from .model import Transformer
    from .utils import PAD_IDX, generate_copy_batch, get_device
except ImportError:  # Allows `python src/overfit_copy_task.py`.
    from inference import greedy_decode
    from model import Transformer
    from utils import PAD_IDX, generate_copy_batch, get_device


def main():
    device = get_device()
    print("Using device:", device)

    vocab_size = 20
    max_src_len = 6
    batch_size = 16

    d_model = 32
    num_heads = 4
    num_layers = 2
    d_ff = 128
    max_seq_length = max_src_len + 2
    dropout = 0.0

    learning_rate = 1e-3
    num_steps = 800

    model = Transformer(
        src_vocab_size=vocab_size,
        tgt_vocab_size=vocab_size,
        d_model=d_model,
        num_heads=num_heads,
        num_layers=num_layers,
        d_ff=d_ff,
        max_seq_length=max_seq_length,
        dropout=dropout,
        src_pad_idx=PAD_IDX,
        tgt_pad_idx=PAD_IDX,
    ).to(device)

    criterion = nn.CrossEntropyLoss(ignore_index=PAD_IDX)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Fixed batch for overfit sanity check
    fixed_src, fixed_tgt = generate_copy_batch(
        batch_size=batch_size,
        min_len=3,
        max_len=max_src_len,
        vocab_size=vocab_size,
        device=device,
    )

    print()
    print("Fixed training sample:")
    print("src:", fixed_src[0].tolist())
    print("tgt:", fixed_tgt[0].tolist())
    print()

    model.train()

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

        if step == 1 or step % 50 == 0:
            print(f"step {step:04d} | loss = {loss.item():.4f}")

    print()
    print("Greedy decoding on fixed training samples:")

    model.eval()

    with torch.no_grad():
        generated = greedy_decode(
            model=model,
            src=fixed_src[:5],
            max_decode_len=max_src_len + 1,
        )

    for i in range(5):
        print()
        print(f"Sample {i + 1}")
        print("src:       ", fixed_src[i].tolist())
        print("target:    ", fixed_tgt[i].tolist())
        print("generated: ", generated[i].tolist())


if __name__ == "__main__":
    main()
