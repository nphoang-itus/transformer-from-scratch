import torch
import torch.nn as nn
import torch.optim as optim

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
    Copy task.

    Example:
        src = [7, 12, 5, PAD, PAD]
        tgt = [SOS, 7, 12, 5, EOS, PAD, PAD]
    """
    src_batch = []
    tgt_batch = []

    for _ in range(batch_size):
        seq_len = int(torch.randint(
            low=min_len,
            high=max_len + 1,
            size=(1,),
        ).item())

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


def greedy_decode(
    model: Transformer,
    src: torch.Tensor,
    max_decode_len: int,
) -> torch.Tensor:
    """
    Generate target sequence autoregressively.

    src: [B, src_seq_len]

    return:
        generated token ids: [B, generated_len]
    """
    model.eval()

    batch_size = src.size(0)
    device = src.device

    src_mask = model.create_src_mask(src)
    encoder_output = model.encode(src, src_mask)

    generated = torch.full(
        (batch_size, 1),
        SOS_IDX,
        dtype=torch.long,
        device=device,
    )

    for _ in range(max_decode_len):
        tgt_mask = model.create_tgt_mask(generated)

        decoder_output = model.decode(
            tgt=generated,
            encoder_output=encoder_output,
            src_mask=src_mask,
            tgt_mask=tgt_mask,
        )

        # Use only the last decoder position to predict next token
        last_hidden = decoder_output[:, -1, :]
        logits = model.fc_out(last_hidden)

        next_token = logits.argmax(dim=-1, keepdim=True)

        generated = torch.cat([generated, next_token], dim=1)

        # Stop early if all samples already generated EOS
        if torch.all(next_token.squeeze(1) == EOS_IDX):
            break

    return generated


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