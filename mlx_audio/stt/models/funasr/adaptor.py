# Copyright © 2025 FunASR (original model implementation)
# Copyright © Anthony DePasquale (MLX port)
# Ported to MLX from https://github.com/modelscope/FunASR
# License: licenses/funasr.txt

"""
Audio Adaptor for Fun-ASR model.

Projects the audio encoder output to the LLM embedding dimension.
Uses downsampling, linear projection, and transformer blocks.
"""

from dataclasses import dataclass
from typing import Optional, Tuple

import mlx.core as mx
import mlx.nn as nn

from .encoder import PositionwiseFeedForward


@dataclass
class AudioAdaptorConfig:
    """Configuration for the audio adaptor.

    Defaults match the published Fun-ASR-Nano-2512 (``downsample_rate: 1``,
    ``use_low_frame_rate: true``).
    """

    downsample_rate: int = (
        1  # Downsample by grouping this many frames (1 = no downsample)
    )
    encoder_dim: int = 512  # Input dimension from encoder
    llm_dim: int = 1024  # Output dimension for LLM
    ffn_dim: int = 2048  # Intermediate projection dimension
    n_layer: int = 2  # Number of transformer blocks
    attention_heads: int = 8  # Attention heads
    dropout: float = 0.0
    # When true, the LLM only consumes the first ``fake_token_len`` frames of
    # the adaptor output. The count is computed as three successive
    # ``1 + (T - 1) // 2`` reductions on the LFR frame count.
    use_low_frame_rate: bool = True


class MultiHeadedAttention(nn.Module):
    """
    Standard multi-head attention.
    """

    def __init__(
        self,
        n_head: int,
        n_feat: int,
        dropout: float = 0.0,
    ):
        super().__init__()
        assert n_feat % n_head == 0
        self.d_k = n_feat // n_head
        self.h = n_head
        self.n_feat = n_feat

        self.linear_q = nn.Linear(n_feat, n_feat, bias=True)
        self.linear_k = nn.Linear(n_feat, n_feat, bias=True)
        self.linear_v = nn.Linear(n_feat, n_feat, bias=True)
        self.linear_out = nn.Linear(n_feat, n_feat, bias=True)

        self.dropout = nn.Dropout(dropout)

    def __call__(
        self,
        query: mx.array,
        key: mx.array,
        value: mx.array,
        mask: Optional[mx.array] = None,
    ) -> mx.array:
        batch_size = query.shape[0]

        q = self.linear_q(query)
        k = self.linear_k(key)
        v = self.linear_v(value)

        # Reshape for multi-head attention
        q = q.reshape(batch_size, -1, self.h, self.d_k).transpose(0, 2, 1, 3)
        k = k.reshape(batch_size, -1, self.h, self.d_k).transpose(0, 2, 1, 3)
        v = v.reshape(batch_size, -1, self.h, self.d_k).transpose(0, 2, 1, 3)

        # Convert mask to additive format for fast attention if provided
        attn_mask = None
        if mask is not None:
            attn_mask = mx.where(mask == 0, mx.array(float("-inf")), mx.array(0.0))

        # Use fast scaled dot-product attention
        context = mx.fast.scaled_dot_product_attention(
            q, k, v, scale=self.d_k**-0.5, mask=attn_mask
        )

        # Apply dropout after attention
        context = self.dropout(context)

        # Reshape back
        context = context.transpose(0, 2, 1, 3).reshape(batch_size, -1, self.n_feat)

        return self.linear_out(context)


class EncoderLayer(nn.Module):
    """
    Transformer encoder layer with pre-norm (matches original FunASR).

    Structure (pre-norm):
    - LayerNorm -> Self-Attention -> Dropout -> Residual
    - LayerNorm -> Feed-Forward -> Dropout -> Residual
    """

    def __init__(
        self,
        size: int,
        self_attn: MultiHeadedAttention,
        feed_forward: PositionwiseFeedForward,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.norm1 = nn.LayerNorm(size)
        self.norm2 = nn.LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def __call__(
        self,
        x: mx.array,
        mask: Optional[mx.array] = None,
    ) -> mx.array:
        # Self-attention with pre-norm (matches original FunASR)
        residual = x
        x = self.norm1(x)
        x = self.self_attn(x, x, x, mask)
        x = residual + self.dropout(x)

        # Feed-forward with pre-norm (matches original FunASR)
        residual = x
        x = self.norm2(x)
        x = self.feed_forward(x)
        x = residual + self.dropout(x)

        return x


class AudioAdaptor(nn.Module):
    """
    Audio Adaptor that projects encoder output to LLM embedding space.

    Architecture (matches original FunASR Transformer adaptor):
    - Downsample by grouping k consecutive frames
    - linear1: encoder_dim * k -> ffn_dim (with ReLU)
    - linear2: ffn_dim -> llm_dim
    - blocks: n_layer transformer blocks for refinement

    The downsampling reduces sequence length by factor k while
    increasing feature dimension.
    """

    def __init__(self, config: AudioAdaptorConfig):
        super().__init__()
        self.config = config
        self.k = config.downsample_rate

        # Linear projections (after downsampling, input dim is encoder_dim * k)
        self.linear1 = nn.Linear(config.encoder_dim * self.k, config.ffn_dim, bias=True)
        self.linear2 = nn.Linear(config.ffn_dim, config.llm_dim, bias=True)

        # Transformer blocks (optional, based on n_layer)
        self.blocks = None
        if config.n_layer > 0:
            # FFN dimension in transformer blocks is llm_dim // 4 (matches original)
            block_ffn_dim = config.llm_dim // 4
            self.blocks = [
                EncoderLayer(
                    size=config.llm_dim,
                    self_attn=MultiHeadedAttention(
                        n_head=config.attention_heads,
                        n_feat=config.llm_dim,
                        dropout=config.dropout,
                    ),
                    feed_forward=PositionwiseFeedForward(
                        d_model=config.llm_dim,
                        d_ff=block_ffn_dim,
                        dropout=config.dropout,
                    ),
                    dropout=config.dropout,
                )
                for _ in range(config.n_layer)
            ]

    def __call__(
        self,
        x: mx.array,
        lengths: Optional[mx.array] = None,
    ) -> Tuple[mx.array, mx.array]:
        """
        Forward pass through the adaptor.

        Parameters
        ----------
        x : mx.array, shape = (batch, seq, encoder_dim)
            Encoder output
        lengths : mx.array, optional
            Sequence lengths

        Returns
        -------
        Tuple[mx.array, mx.array]
            - Projected features for LLM input, shape = (batch, seq//k, llm_dim)
            - Output lengths
        """
        batch_size, seq_len, dim = x.shape

        # Pad sequence to be divisible by k
        chunk_num = (seq_len - 1) // self.k + 1
        pad_num = chunk_num * self.k - seq_len
        if pad_num > 0:
            x = mx.pad(x, [(0, 0), (0, pad_num), (0, 0)])

        # Reshape to group k consecutive frames
        # (batch, seq, dim) -> (batch, seq//k, dim*k)
        x = x.reshape(batch_size, chunk_num, dim * self.k)

        # Linear projections with ReLU (matches original)
        x = self.linear1(x)
        x = nn.relu(x)
        x = self.linear2(x)

        # Compute output lengths
        if lengths is not None:
            out_lengths = (lengths - 1) // self.k + 1
        else:
            out_lengths = mx.full((batch_size,), chunk_num, dtype=mx.int32)

        # Create padding mask for transformer blocks
        mask = None
        if lengths is not None and self.blocks is not None:
            # Create mask from lengths
            max_len = x.shape[1]
            indices = mx.arange(max_len)[None, :]
            mask = indices < out_lengths[:, None]
            # Expand for attention: (batch, 1, 1, seq)
            mask = mask[:, None, None, :]

        # Apply transformer blocks
        if self.blocks is not None:
            for block in self.blocks:
                x = block(x, mask)

        return x, out_lengths


def create_adaptor_from_config(config_dict: dict) -> AudioAdaptor:
    """
    Create an audio adaptor from a configuration dictionary.

    Parameters
    ----------
    config_dict : dict
        Configuration dictionary

    Returns
    -------
    AudioAdaptor
        Initialized adaptor
    """
    config = AudioAdaptorConfig(
        downsample_rate=config_dict.get("downsample_rate", 1),
        encoder_dim=config_dict.get("encoder_dim", 512),
        llm_dim=config_dict.get("llm_dim", 1024),
        ffn_dim=config_dict.get("ffn_dim", 2048),
        n_layer=config_dict.get("n_layer", 2),
        attention_heads=config_dict.get("attention_heads", 8),
        dropout=config_dict.get("dropout", 0.0),
        use_low_frame_rate=config_dict.get("use_low_frame_rate", True),
    )
    return AudioAdaptor(config)
