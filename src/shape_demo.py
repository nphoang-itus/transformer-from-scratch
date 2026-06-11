import math

import torch
import torch.nn as nn

try:
    from .utils import get_device, print_shape
except ImportError:  # Allows `python src/shape_demo.py`.
    from utils import get_device, print_shape


def main() -> None:
    """Walk through the core tensor shapes used by multi-head attention."""
    device = get_device()

    batch_size = 2
    src_seq_len = 5
    tgt_seq_len = 6
    src_vocab_size = 100
    tgt_vocab_size = 120
    d_model = 16
    num_heads = 4
    d_k = d_model // num_heads

    print("Device:", device)
    print("d_k:", d_k)
    print()

    src = torch.randint(0, src_vocab_size, (batch_size, src_seq_len), device=device)
    tgt = torch.randint(0, tgt_vocab_size, (batch_size, tgt_seq_len), device=device)

    print_shape("src token ids", src)
    print_shape("tgt token ids", tgt)
    print()

    src_embedding = nn.Embedding(src_vocab_size, d_model).to(device)
    tgt_embedding = nn.Embedding(tgt_vocab_size, d_model).to(device)

    src_emb = src_embedding(src)
    tgt_emb = tgt_embedding(tgt)

    print_shape("src_emb", src_emb)
    print_shape("tgt_emb", tgt_emb)
    print()

    w_q = nn.Linear(d_model, d_model).to(device)
    w_k = nn.Linear(d_model, d_model).to(device)
    w_v = nn.Linear(d_model, d_model).to(device)

    q = w_q(src_emb)
    k = w_k(src_emb)
    v = w_v(src_emb)

    print_shape("Q before split heads", q)
    print_shape("K before split heads", k)
    print_shape("V before split heads", v)
    print()

    # [B, S, d_model] -> [B, S, num_heads, d_k] -> [B, num_heads, S, d_k]
    q = q.view(batch_size, src_seq_len, num_heads, d_k).transpose(1, 2)
    k = k.view(batch_size, src_seq_len, num_heads, d_k).transpose(1, 2)
    v = v.view(batch_size, src_seq_len, num_heads, d_k).transpose(1, 2)

    print_shape("Q after split heads", q)
    print_shape("K after split heads", k)
    print_shape("V after split heads", v)
    print()

    # Q: [B, num_heads, S, d_k]
    # K.transpose: [B, num_heads, d_k, S]
    # scores: [B, num_heads, S, S]
    scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d_k)
    print_shape("attention scores", scores)
    print()

    attention_weights = torch.softmax(scores, dim=-1)
    print_shape("attention weights", attention_weights)
    print()

    # attention_weights: [B, num_heads, S, S]
    # V: [B, num_heads, S, d_k]
    # context: [B, num_heads, S, d_k]
    context = torch.matmul(attention_weights, v)
    print_shape("context per head", context)
    print()

    # [B, num_heads, S, d_k] -> [B, S, num_heads, d_k] -> [B, S, d_model]
    context = context.transpose(1, 2).contiguous()
    context = context.view(batch_size, src_seq_len, d_model)
    print_shape("context concat heads", context)
    print()

    w_o = nn.Linear(d_model, d_model).to(device)
    output = w_o(context)
    print_shape("attention output", output)


if __name__ == "__main__":
    main()
