import torch
import torch.nn as nn
import torch.optim as optim

try:
    from .model import Transformer
    from .utils import PAD_IDX, generate_copy_batch, get_device
except ImportError:  # Allows `python src/train_dummy.py`.
    from model import Transformer
    from utils import PAD_IDX, generate_copy_batch, get_device


def main():
    device = get_device()
    print("Using device:", device)

    vocab_size = 50
    max_src_len = 8

    d_model = 32
    num_heads = 4
    num_layers = 2
    d_ff = 128
    max_seq_length = max_src_len + 2
    dropout = 0.1

    batch_size = 32
    num_steps = 300
    learning_rate = 1e-3

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

    model.train()

    for step in range(1, num_steps + 1):
        src, tgt = generate_copy_batch(
            batch_size=batch_size,
            min_len=3,
            max_len=max_src_len,
            vocab_size=vocab_size,
            device=device,
        )

        # Teacher forcing:
        # tgt      = [SOS, token1, token2, ..., EOS, PAD]
        # tgt_in   = [SOS, token1, token2, ...]
        # tgt_label= [token1, token2, ..., EOS]
        tgt_input = tgt[:, :-1]
        tgt_label = tgt[:, 1:]

        logits = model(src, tgt_input)

        # logits:    [B, T, vocab_size]
        # tgt_label: [B, T]
        loss = criterion(
            logits.reshape(-1, vocab_size),
            tgt_label.reshape(-1),
        )

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if step == 1 or step % 25 == 0:
            print(f"step {step:03d} | loss = {loss.item():.4f}")

    print()
    print("Training finished.")

    # Quick inspection
    model.eval()

    with torch.no_grad():
        src, tgt = generate_copy_batch(
            batch_size=1,
            min_len=5,
            max_len=max_src_len,
            vocab_size=vocab_size,
            device=device,
        )

        tgt_input = tgt[:, :-1]
        tgt_label = tgt[:, 1:]

        logits = model(src, tgt_input)
        predictions = logits.argmax(dim=-1)

        print()
        print("Example after training:")
        print("src:        ", src[0].tolist())
        print("tgt_input:  ", tgt_input[0].tolist())
        print("tgt_label:  ", tgt_label[0].tolist())
        print("prediction: ", predictions[0].tolist())


if __name__ == "__main__":
    main()
