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

class PositionWiseFeedForward(nn.Module):
    def __init__(self, d_model: int, d_ff: int):
        super().__init__()

        self.fc1 = nn.Linear(d_model, d_ff)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(d_ff, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x:      [B, seq_len, d_model]
        output: [B, seq_len, d_model]
        """
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)

        return x
    
class PositionalEncoding(nn.Module):
    pe: torch.Tensor  # Type annotation to help Pylance understand pe is a tensor

    def __init__(self, d_model: int, max_seq_length: int):
        super().__init__()

        if d_model % 2 != 0:
            raise ValueError("d_model must be even for this positional encoding implementation")

        # pe: [max_seq_length, d_model]
        pe = torch.zeros(max_seq_length, d_model)

        # position: [max_seq_length, 1]
        position = torch.arange(0, max_seq_length, dtype=torch.float).unsqueeze(1)

        # div_term: [d_model / 2]
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float)
            * (-math.log(10000.0) / d_model)
        )

        # Even dimensions: 0, 2, 4, ...
        pe[:, 0::2] = torch.sin(position * div_term)

        # Odd dimensions: 1, 3, 5, ...
        pe[:, 1::2] = torch.cos(position * div_term)

        # Add batch dimension:
        # [max_seq_length, d_model] -> [1, max_seq_length, d_model]
        pe = pe.unsqueeze(0)

        # Register as buffer:
        # - saved with model state_dict
        # - moved to device when calling model.to(device)
        # - not trained as parameter
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x:      [B, seq_len, d_model]
        output: [B, seq_len, d_model]
        """
        seq_len = x.size(1)

        x = x + self.pe[:, :seq_len, :]

        return x
    
class EncoderLayer(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int,
        dropout: float,
    ):
        super().__init__()

        self.self_attention = MultiHeadAttention(
            d_model=d_model,
            num_heads=num_heads,
        )

        self.feed_forward = PositionWiseFeedForward(
            d_model=d_model,
            d_ff=d_ff,
        )

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        src_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        x:        [B, src_seq_len, d_model]
        src_mask: optional mask for source sequence

        output:   [B, src_seq_len, d_model]
        """

        # 1. Multi-head self-attention
        # query = key = value = x
        attention_output = self.self_attention(
            query=x,
            key=x,
            value=x,
            mask=src_mask,
        )

        # 2. Residual connection + LayerNorm
        x = self.norm1(x + self.dropout(attention_output))

        # 3. Position-wise feed-forward
        feed_forward_output = self.feed_forward(x)

        # 4. Residual connection + LayerNorm
        x = self.norm2(x + self.dropout(feed_forward_output))

        return x