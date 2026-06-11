import sys
from pathlib import Path

import torch
import torch.nn as nn

# Add parent directory to path so we can import src
sys.path.insert(0, str(Path(__file__).parent.parent))

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
    print(f"{name:<35} {list(tensor.shape)}")


def main():
    device = get_device()
    print("Using device:", device)
    print()

    vocab_size = 30
    d_model = 16
    num_heads = 4
    num_layers = 2
    d_ff = 64
    max_seq_length = 10
    dropout = 0.0

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

    model.eval()

    # Batch size = 2
    # Source length = 6
    src = torch.tensor(
        [
            [7, 11, 15, 0, 0, 0],
            [9, 12, 18, 21, 23, 0],
        ],
        dtype=torch.long,
        device=device,
    )

    # Full target includes SOS and EOS.
    # Target length = 8
    tgt_full = torch.tensor(
        [
            [1, 7, 11, 15, 2, 0, 0, 0],
            [1, 9, 12, 18, 21, 23, 2, 0],
        ],
        dtype=torch.long,
        device=device,
    )

    # Teacher forcing shift
    tgt_input = tgt_full[:, :-1]
    tgt_label = tgt_full[:, 1:]

    print("=== Raw token ids ===")
    print_shape("src", src)
    print_shape("tgt_full", tgt_full)
    print_shape("tgt_input", tgt_input)
    print_shape("tgt_label", tgt_label)
    print()

    print("src:")
    print(src)
    print()

    print("tgt_input:")
    print(tgt_input)
    print()

    print("tgt_label:")
    print(tgt_label)
    print()

    # 1. Create masks
    src_mask, tgt_mask = model.create_masks(src, tgt_input)

    print("=== Masks ===")
    print_shape("src_mask", src_mask)
    print_shape("tgt_mask", tgt_mask)
    print()

    print("src_mask[0, 0, 0]:")
    print(src_mask[0, 0, 0])
    print()

    print("tgt_mask[0, 0]:")
    print(tgt_mask[0, 0])
    print()

    # 2. Encoder trace
    print("=== Encoder trace ===")

    src_emb = model.src_embedding(src)
    print_shape("src embedding", src_emb)

    src_emb_scaled = src_emb * (d_model ** 0.5)
    print_shape("src embedding scaled", src_emb_scaled)

    src_pos = model.positional_encoding(src_emb_scaled)
    print_shape("src + positional encoding", src_pos)

    encoder_x = model.dropout(src_pos)
    print_shape("encoder input", encoder_x)

    for i, encoder_layer in enumerate(model.encoder_layers, start=1):
        encoder_x = encoder_layer(encoder_x, src_mask=src_mask)
        print_shape(f"encoder layer {i} output", encoder_x)

    encoder_output = encoder_x
    print_shape("encoder_output final", encoder_output)
    print()

    # 3. Decoder trace
    print("=== Decoder trace ===")

    tgt_emb = model.tgt_embedding(tgt_input)
    print_shape("tgt embedding", tgt_emb)

    tgt_emb_scaled = tgt_emb * (d_model ** 0.5)
    print_shape("tgt embedding scaled", tgt_emb_scaled)

    tgt_pos = model.positional_encoding(tgt_emb_scaled)
    print_shape("tgt + positional encoding", tgt_pos)

    decoder_x = model.dropout(tgt_pos)
    print_shape("decoder input", decoder_x)

    for i, decoder_layer in enumerate(model.decoder_layers, start=1):
        decoder_x = decoder_layer(
            x=decoder_x,
            encoder_output=encoder_output,
            src_mask=src_mask,
            tgt_mask=tgt_mask,
        )
        print_shape(f"decoder layer {i} output", decoder_x)

    decoder_output = decoder_x
    print_shape("decoder_output final", decoder_output)
    print()

    # 4. Final projection
    print("=== Final projection ===")

    logits = model.fc_out(decoder_output)

    print_shape("logits", logits)
    print()

    # 5. Loss shape
    print("=== Loss reshape ===")

    criterion = nn.CrossEntropyLoss(ignore_index=PAD_IDX)

    logits_for_loss = logits.reshape(-1, vocab_size)
    labels_for_loss = tgt_label.reshape(-1)

    print_shape("logits_for_loss", logits_for_loss)
    print_shape("labels_for_loss", labels_for_loss)

    loss = criterion(logits_for_loss, labels_for_loss)

    print()
    print("loss:", loss.item())


if __name__ == "__main__":
    main()