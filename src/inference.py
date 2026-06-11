import torch

from model import Transformer


PAD_IDX = 0
SOS_IDX = 1
EOS_IDX = 2


def greedy_decode(
    model: Transformer,
    src: torch.Tensor,
    max_decode_len: int,
    sos_idx: int = SOS_IDX,
    eos_idx: int = EOS_IDX,
    pad_idx: int = PAD_IDX,
) -> torch.Tensor:
    """
    Autoregressive greedy decoding.

    src:
        [B, src_seq_len]

    return:
        generated token ids [B, generated_len]

    Behavior:
        - Starts with SOS.
        - Appends one predicted token per step.
        - Once a sample generates EOS, future tokens for that sample become PAD.
        - Stops early when all samples have generated EOS.
    """
    model.eval()

    batch_size = src.size(0)
    device = src.device

    src_mask = model.create_src_mask(src)
    encoder_output = model.encode(src, src_mask)

    generated = torch.full(
        size=(batch_size, 1),
        fill_value=sos_idx,
        dtype=torch.long,
        device=device,
    )

    finished = torch.zeros(
        batch_size,
        dtype=torch.bool,
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

        last_hidden = decoder_output[:, -1, :]
        logits = model.fc_out(last_hidden)

        next_token = logits.argmax(dim=-1)

        # If a sample already finished, append PAD instead of repeated EOS.
        next_token = torch.where(
            finished,
            torch.full_like(next_token, pad_idx),
            next_token,
        )

        generated = torch.cat(
            [generated, next_token.unsqueeze(1)],
            dim=1,
        )

        finished = finished | (next_token == eos_idx)

        if finished.all():
            break

    return generated


def trim_after_eos(
    token_ids: list[int],
    eos_idx: int = EOS_IDX,
    pad_idx: int = PAD_IDX,
) -> list[int]:
    """
    Trim a generated sequence after the first EOS.
    Also removes trailing PAD after EOS.

    Example:
        [1, 13, 7, 2, 0, 0] -> [1, 13, 7, 2]
    """
    if eos_idx in token_ids:
        eos_pos = token_ids.index(eos_idx)
        return token_ids[: eos_pos + 1]

    while token_ids and token_ids[-1] == pad_idx:
        token_ids.pop()

    return token_ids