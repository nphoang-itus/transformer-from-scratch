import math

import torch
import torch.nn as nn


class MultiHeadAttention(nn.Module):
    """Multi-head attention implemented from linear projections and matmul.

    Input tensors use batch-first shape [B, seq_len, d_model].
    Output has shape [B, query_len, d_model].
    """

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
        """Split the last dimension into independent attention heads.

        Args:
            x: [B, seq_len, d_model]

        Returns:
            [B, num_heads, seq_len, d_k]
        """
        batch_size, seq_len, _ = x.size()

        # [B, seq_len, d_model] -> [B, seq_len, num_heads, d_k]
        x = x.view(batch_size, seq_len, self.num_heads, self.d_k)
        # [B, seq_len, num_heads, d_k] -> [B, num_heads, seq_len, d_k]
        x = x.transpose(1, 2)

        return x

    def combine_heads(self, x: torch.Tensor) -> torch.Tensor:
        """Concatenate attention heads back into d_model.

        Args:
            x: [B, num_heads, seq_len, d_k]

        Returns:
            [B, seq_len, d_model]
        """
        batch_size, _, seq_len, _ = x.size()

        # [B, num_heads, seq_len, d_k] -> [B, seq_len, num_heads, d_k]
        x = x.transpose(1, 2).contiguous()
        # [B, seq_len, num_heads, d_k] -> [B, seq_len, d_model]
        x = x.view(batch_size, seq_len, self.d_model)

        return x

    def scaled_dot_product_attention(
        self,
        Q: torch.Tensor,
        K: torch.Tensor,
        V: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Compute scaled dot-product attention for all heads.

        Q: [B, num_heads, query_len, d_k]
        K: [B, num_heads, key_len, d_k]
        V: [B, num_heads, key_len, d_k]
        mask: broadcastable to [B, num_heads, query_len, key_len]

        Returns:
            output: [B, num_heads, query_len, d_k]
        """
        # Q: [B, num_heads, query_len, d_k]
        # K.transpose: [B, num_heads, d_k, key_len]
        # scores: [B, num_heads, query_len, key_len]
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)

        if mask is not None:
            # Masked positions get a very negative value before softmax.
            scores = scores.masked_fill(mask == 0, -1e9)

        # attention_weights: [B, num_heads, query_len, key_len]
        attention_weights = torch.softmax(scores, dim=-1)
        # output: [B, num_heads, query_len, d_k]
        output = torch.matmul(attention_weights, V)

        return output

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Apply multi-head attention.

        query: [B, query_len, d_model]
        key: [B, key_len, d_model]
        value: [B, key_len, d_model]
        mask: broadcastable to [B, num_heads, query_len, key_len]

        Returns:
            [B, query_len, d_model]
        """
        # Project each token embedding into query, key, and value spaces.
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
    """Two-layer feed-forward network applied independently at each position.

    Input: [B, seq_len, d_model]
    Output: [B, seq_len, d_model]
    """

    def __init__(self, d_model: int, d_ff: int):
        super().__init__()

        self.fc1 = nn.Linear(d_model, d_ff)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(d_ff, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply d_model -> d_ff -> d_model to every token position.

        Args:
            x: [B, seq_len, d_model]

        Returns:
            [B, seq_len, d_model]
        """
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)

        return x


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding added to token embeddings.

    Input: [B, seq_len, d_model]
    Output: [B, seq_len, d_model]
    """

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
        """Add the first seq_len positional vectors to x.

        Args:
            x: [B, seq_len, d_model]

        Returns:
            [B, seq_len, d_model]
        """
        seq_len = x.size(1)

        x = x + self.pe[:, :seq_len, :]

        return x


class EncoderLayer(nn.Module):
    """One Transformer encoder block.

    The block performs source self-attention, residual + LayerNorm, position-wise
    feed-forward, and another residual + LayerNorm.
    """

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
        """Encode source hidden states.

        Args:
            x: [B, S, d_model]
            src_mask: Optional source mask [B, 1, 1, S]

        Returns:
            [B, S, d_model]
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


class DecoderLayer(nn.Module):
    """One Transformer decoder block.

    The block performs masked target self-attention, cross-attention over encoder
    outputs, and a position-wise feed-forward network, each with residual + LayerNorm.
    """

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

        self.cross_attention = MultiHeadAttention(
            d_model=d_model,
            num_heads=num_heads,
        )

        self.feed_forward = PositionWiseFeedForward(
            d_model=d_model,
            d_ff=d_ff,
        )

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)

        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        encoder_output: torch.Tensor,
        src_mask: torch.Tensor | None = None,
        tgt_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Decode target hidden states using target and source context.

        Args:
            x: [B, T, d_model]
            encoder_output: [B, S, d_model]
            src_mask: Optional source padding mask [B, 1, 1, S]
            tgt_mask: Optional target mask [B, 1, T, T]

        Returns:
            [B, T, d_model]
        """

        # 1. Masked self-attention over target sequence
        self_attention_output = self.self_attention(
            query=x,
            key=x,
            value=x,
            mask=tgt_mask,
        )

        # 2. Residual connection + LayerNorm
        x = self.norm1(x + self.dropout(self_attention_output))

        # 3. Cross-attention:
        # query comes from decoder
        # key and value come from encoder
        cross_attention_output = self.cross_attention(
            query=x,
            key=encoder_output,
            value=encoder_output,
            mask=src_mask,
        )

        # 4. Residual connection + LayerNorm
        x = self.norm2(x + self.dropout(cross_attention_output))

        # 5. Position-wise feed-forward
        feed_forward_output = self.feed_forward(x)

        # 6. Residual connection + LayerNorm
        x = self.norm3(x + self.dropout(feed_forward_output))

        return x


class Transformer(nn.Module):
    """Educational encoder-decoder Transformer from scratch.

    Source and target inputs are integer token ids. The model returns logits for
    each target input position.

    src: [B, S]
    tgt: [B, T]
    logits: [B, T, tgt_vocab_size]
    """

    def __init__(
        self,
        src_vocab_size: int,
        tgt_vocab_size: int,
        d_model: int,
        num_heads: int,
        num_layers: int,
        d_ff: int,
        max_seq_length: int,
        dropout: float,
        src_pad_idx: int = 0,
        tgt_pad_idx: int = 0,
    ):
        super().__init__()

        self.d_model = d_model
        self.src_pad_idx = src_pad_idx
        self.tgt_pad_idx = tgt_pad_idx

        self.src_embedding = nn.Embedding(src_vocab_size, d_model)
        self.tgt_embedding = nn.Embedding(tgt_vocab_size, d_model)

        self.positional_encoding = PositionalEncoding(
            d_model=d_model,
            max_seq_length=max_seq_length,
        )

        self.encoder_layers = nn.ModuleList(
            [
                EncoderLayer(
                    d_model=d_model,
                    num_heads=num_heads,
                    d_ff=d_ff,
                    dropout=dropout,
                )
                for _ in range(num_layers)
            ]
        )

        self.decoder_layers = nn.ModuleList(
            [
                DecoderLayer(
                    d_model=d_model,
                    num_heads=num_heads,
                    d_ff=d_ff,
                    dropout=dropout,
                )
                for _ in range(num_layers)
            ]
        )

        self.fc_out = nn.Linear(d_model, tgt_vocab_size)
        self.dropout = nn.Dropout(dropout)

    def create_src_mask(self, src: torch.Tensor) -> torch.Tensor:
        """
        src:      [B, src_seq_len]
        src_mask: [B, 1, 1, src_seq_len]

        1 = real token, will be attended to
        0 = padding token, will be masked
        """
        src_mask = (src != self.src_pad_idx).unsqueeze(1).unsqueeze(2)

        return src_mask

    def create_tgt_mask(self, tgt: torch.Tensor) -> torch.Tensor:
        """
        tgt:      [B, tgt_seq_len]
        tgt_mask: [B, 1, tgt_seq_len, tgt_seq_len]

        Combines:
        - padding mask
        - look-ahead mask
        """
        batch_size, tgt_seq_len = tgt.size()

        tgt_padding_mask = (tgt != self.tgt_pad_idx).unsqueeze(1).unsqueeze(2)

        look_ahead_mask = torch.tril(
            torch.ones(
                tgt_seq_len,
                tgt_seq_len,
                device=tgt.device,
                dtype=torch.bool,
            )
        )

        look_ahead_mask = look_ahead_mask.unsqueeze(0).unsqueeze(0)

        tgt_mask = tgt_padding_mask & look_ahead_mask

        return tgt_mask

    def create_masks(
        self,
        src: torch.Tensor,
        tgt: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Create source and target masks.

        Args:
            src: [B, S]
            tgt: [B, T]

        Returns:
            src_mask: [B, 1, 1, S]
            tgt_mask: [B, 1, T, T]
        """
        src_mask = self.create_src_mask(src)
        tgt_mask = self.create_tgt_mask(tgt)

        return src_mask, tgt_mask

    def encode(
        self,
        src: torch.Tensor,
        src_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        src:      [B, src_seq_len]
        src_mask: [B, 1, 1, src_seq_len]

        output:   [B, src_seq_len, d_model]
        """
        src = self.src_embedding(src) * math.sqrt(self.d_model)
        src = self.positional_encoding(src)
        src = self.dropout(src)

        for encoder_layer in self.encoder_layers:
            src = encoder_layer(src, src_mask=src_mask)

        return src

    def decode(
        self,
        tgt: torch.Tensor,
        encoder_output: torch.Tensor,
        src_mask: torch.Tensor,
        tgt_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        tgt:            [B, tgt_seq_len]
        encoder_output: [B, src_seq_len, d_model]
        src_mask:       [B, 1, 1, src_seq_len]
        tgt_mask:       [B, 1, tgt_seq_len, tgt_seq_len]

        output:         [B, tgt_seq_len, d_model]
        """
        tgt = self.tgt_embedding(tgt) * math.sqrt(self.d_model)
        tgt = self.positional_encoding(tgt)
        tgt = self.dropout(tgt)

        for decoder_layer in self.decoder_layers:
            tgt = decoder_layer(
                x=tgt,
                encoder_output=encoder_output,
                src_mask=src_mask,
                tgt_mask=tgt_mask,
            )

        return tgt

    def forward(
        self,
        src: torch.Tensor,
        tgt: torch.Tensor,
    ) -> torch.Tensor:
        """
        src:    [B, src_seq_len]
        tgt:    [B, tgt_seq_len]

        logits: [B, tgt_seq_len, tgt_vocab_size]
        """
        src_mask, tgt_mask = self.create_masks(src, tgt)

        encoder_output = self.encode(
            src=src,
            src_mask=src_mask,
        )

        decoder_output = self.decode(
            tgt=tgt,
            encoder_output=encoder_output,
            src_mask=src_mask,
            tgt_mask=tgt_mask,
        )

        logits = self.fc_out(decoder_output)

        return logits
