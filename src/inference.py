import torch

try:
    from .model import Transformer
    from .utils import EOS_IDX, PAD_IDX, SOS_IDX
except ImportError:  # Allows `python src/inference.py` style local imports.
    from model import Transformer
    from utils import EOS_IDX, PAD_IDX, SOS_IDX


def greedy_decode(
    model: Transformer,
    src: torch.Tensor,
    max_decode_len: int,
    sos_idx: int = SOS_IDX,
    eos_idx: int = EOS_IDX,
    pad_idx: int = PAD_IDX,
) -> torch.Tensor:
    """Generate target token ids with autoregressive greedy decoding.

    Args:
        model: Encoder-decoder Transformer.
        src: Source token ids with shape [B, S].
        max_decode_len: Maximum number of tokens to append after SOS.

    Returns:
        Generated token ids with shape [B, generated_len]. The first token is SOS.
        Once a sample generates EOS, later generated positions for that sample are PAD.
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
    """Trim a generated sequence after the first EOS or trailing PAD tokens.

    Example:
        [1, 13, 7, 2, 0, 0] -> [1, 13, 7, 2]
    """
    if eos_idx in token_ids:
        eos_pos = token_ids.index(eos_idx)
        return token_ids[: eos_pos + 1]

    while token_ids and token_ids[-1] == pad_idx:
        token_ids.pop()

    return token_ids
