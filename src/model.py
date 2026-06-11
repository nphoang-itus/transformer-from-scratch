import math

import torch
import torch.nn as nn

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int):
        super().__init__()

        if d_model % num_heads != 0:
            raise ValueError("d_model must be divisible by num_heads")

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        # Linear layers to project input embeddings into Q, K, V spaces
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)

        # Final projection after concatenating all heads
        self.W_o = nn.Linear(d_model, d_model)

    def split_heads(self, x: torch.Tensor) -> torch.Tensor:
        """
        Convert:
            [B, seq_len, d_model]
        into:
            [B, num_heads, seq_len, d_k]
        """
        batch_size, seq_len, _ = x.size()

        x = x.view(batch_size, seq_len, self.num_heads, self.d_k)
        x = x.transpose(1, 2)

        return x

    def combine_heads(self, x: torch.Tensor) -> torch.Tensor:
        """
        Convert:
            [B, num_heads, seq_len, d_k]
        back into:
            [B, seq_len, d_model]
        """
        batch_size, _, seq_len, _ = x.size()

        x = x.transpose(1, 2).contiguous()
        x = x.view(batch_size, seq_len, self.d_model)

        return x

    def scaled_dot_product_attention(
        self,
        Q: torch.Tensor,
        K: torch.Tensor,
        V: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Q: [B, num_heads, query_len, d_k]
        K: [B, num_heads, key_len, d_k]
        V: [B, num_heads, key_len, d_k]

        scores:            [B, num_heads, query_len, key_len]
        attention_weights: [B, num_heads, query_len, key_len]
        output:            [B, num_heads, query_len, d_k]
        """
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)

        if mask is not None:
            # Masked positions get a very negative value before softmax
            scores = scores.masked_fill(mask == 0, -1e9)

        attention_weights = torch.softmax(scores, dim=-1)

        output = torch.matmul(attention_weights, V)

        return output

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        query: [B, query_len, d_model]
        key:   [B, key_len, d_model]
        value: [B, key_len, d_model]

        output: [B, query_len, d_model]
        """
        Q = self.W_q(query)
        K = self.W_k(key)
        V = self.W_v(value)

        Q = self.split_heads(Q)
        K = self.split_heads(K)
        V = self.split_heads(V)

        attention_output = self.scaled_dot_product_attention(Q, K, V, mask)

        output = self.combine_heads(attention_output)
        output = self.W_o(output)

        return output