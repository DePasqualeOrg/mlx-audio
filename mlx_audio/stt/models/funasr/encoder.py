# Copyright © 2025 FunASR (original model implementation)
# Copyright © Anthony DePasquale (MLX port)
# Ported to MLX from https://github.com/modelscope/FunASR
# License: licenses/funasr.txt

"""
SenseVoice Encoder for Fun-ASR model.

Implements SANM (Self-Attention with Memory) encoder blocks with FSMN
(Feedforward Sequential Memory Network) for local context modeling.
"""

import math
from dataclasses import dataclass
from typing import Optional, Tuple

import mlx.core as mx
import mlx.nn as nn


@dataclass
class SenseVoiceEncoderConfig:
    """Configuration for SenseVoice encoder."""

    input_dim: int = 560  # 80 * 7 (n_mels * lfr_m)
    encoder_dim: int = 512
    num_heads: int = 4
    ffn_dim: int = 2048
    kernel_size: int = 11  # FSMN kernel size
    sanm_shift: int = 0  # SANM shift for asymmetric context
    num_encoders0: int = 1  # Initial encoder layers
    num_encoders: int = 49  # Main encoder layers
    num_tp_encoders: int = 20  # Time-pooling encoder layers
    dropout: float = 0.0


def sinusoidal_position_encoding(
    timesteps: int, depth: int, dtype: mx.Dtype = mx.float32
) -> mx.array:
    """Match upstream ``SinusoidalPositionEncoder``: 1-based positions,
    ``log(10000)/(depth/2 - 1)`` log timescale increment, then concatenate
    ``[sin, cos]`` along the feature axis. Returns shape ``(1, timesteps,
    depth)`` so it broadcasts over the batch dimension."""
    half_depth = depth // 2
    log_timescale_increment = math.log(10000.0) / (half_depth - 1)
    inv_timescales = mx.exp(
        mx.arange(half_depth, dtype=mx.float32) * (-log_timescale_increment)
    )
    positions = mx.arange(1, timesteps + 1, dtype=mx.float32)
    scaled_time = positions[:, None] * inv_timescales[None, :]
    encoding = mx.concatenate([mx.sin(scaled_time), mx.cos(scaled_time)], axis=-1)
    return encoding[None, :, :].astype(dtype)


class MultiHeadedAttentionSANM(nn.Module):
    """
    Self-Attention with Memory (SANM).

    Combines standard multi-head attention with FSMN for local context.
    The FSMN output is added AFTER computing attention (as a residual).

    This matches the original FunASR implementation exactly.
    """

    def __init__(
        self,
        n_head: int,
        in_feat: int,
        n_feat: int,
        kernel_size: int = 11,
        sanm_shift: int = 0,
        dropout: float = 0.0,
    ):
        super().__init__()
        assert n_feat % n_head == 0
        self.d_k = n_feat // n_head
        self.h = n_head
        self.n_feat = n_feat

        # Combined Q/K/V projection
        self.linear_q_k_v = nn.Linear(in_feat, n_feat * 3, bias=True)

        # Output projection
        self.linear_out = nn.Linear(n_feat, n_feat, bias=True)

        # FSMN block - depthwise conv with no padding in conv itself
        # Padding is applied explicitly before conv
        self.fsmn_block = nn.Conv1d(
            in_channels=n_feat,
            out_channels=n_feat,
            kernel_size=kernel_size,
            stride=1,
            padding=0,  # No padding in conv - we pad explicitly
            groups=n_feat,  # Depthwise
            bias=False,
        )

        # Compute padding amounts
        left_padding = (kernel_size - 1) // 2
        if sanm_shift > 0:
            left_padding = left_padding + sanm_shift
        right_padding = kernel_size - 1 - left_padding
        self.left_padding = left_padding
        self.right_padding = right_padding
        self.kernel_size = kernel_size

        self.dropout = nn.Dropout(dropout)

    def _forward_fsmn(
        self,
        inputs: mx.array,
        mask: Optional[mx.array] = None,
    ) -> mx.array:
        """
        Apply FSMN to inputs.

        Parameters
        ----------
        inputs : mx.array, shape = (batch, seq, dim)
            The unprojected value tensor
        mask : mx.array, optional
            Mask tensor of shape (batch, seq) or (batch, 1, seq)

        Returns
        -------
        mx.array, shape = (batch, seq, dim)
            FSMN output with local context
        """
        b, t, d = inputs.shape

        # Apply mask if provided
        if mask is not None:
            if mask.ndim == 3:
                mask = mask.reshape(b, -1, 1)
            elif mask.ndim == 2:
                mask = mask[:, :, None]
            inputs = inputs * mask

        # Transpose for conv1d: (batch, seq, dim) -> (batch, dim, seq)
        x = inputs.swapaxes(1, 2)

        # Apply explicit padding
        if self.left_padding > 0 or self.right_padding > 0:
            x = mx.pad(x, [(0, 0), (0, 0), (self.left_padding, self.right_padding)])

        # Apply depthwise conv
        # MLX conv1d expects (batch, seq, channels), so transpose
        x = x.swapaxes(1, 2)  # (batch, padded_seq, dim)
        x = self.fsmn_block(x)  # (batch, seq, dim)

        # Add residual connection
        x = x + inputs

        # Apply dropout
        x = self.dropout(x)

        # Apply mask again
        if mask is not None:
            x = x * mask

        return x

    def __call__(
        self,
        x: mx.array,
        mask: Optional[mx.array] = None,
    ) -> mx.array:
        """
        Forward pass for SANM attention.

        Parameters
        ----------
        x : mx.array, shape = (batch, seq, in_feat)
            Input tensor
        mask : mx.array, optional
            Attention mask

        Returns
        -------
        mx.array, shape = (batch, seq, n_feat)
            Output tensor
        """
        batch_size, seq_len, _ = x.shape

        # Combined Q/K/V projection
        q_k_v = self.linear_q_k_v(x)

        # Split into q, k, v
        q, k, v = mx.split(q_k_v, 3, axis=-1)

        # Apply FSMN to unprojected value (before multi-head reshape)
        fsmn_memory = self._forward_fsmn(v, mask)

        # Reshape for multi-head attention
        # (batch, seq, n_feat) -> (batch, n_head, seq, d_k)
        q_h = q.reshape(batch_size, seq_len, self.h, self.d_k).transpose(0, 2, 1, 3)
        k_h = k.reshape(batch_size, seq_len, self.h, self.d_k).transpose(0, 2, 1, 3)
        v_h = v.reshape(batch_size, seq_len, self.h, self.d_k).transpose(0, 2, 1, 3)

        # Convert mask to additive format for fast attention if provided
        attn_mask = None
        if mask is not None:
            if mask.ndim == 2:
                attn_mask = mask[:, None, None, :]  # (batch, 1, 1, seq)
            elif mask.ndim == 3:
                attn_mask = mask[:, None, :, :]  # (batch, 1, seq, seq)
            else:
                attn_mask = mask
            # Convert boolean/binary mask to additive mask (0 -> -inf for masked positions)
            attn_mask = mx.where(attn_mask == 0, mx.array(float("-inf")), mx.array(0.0))

        # Use fast scaled dot-product attention
        context = mx.fast.scaled_dot_product_attention(
            q_h, k_h, v_h, scale=self.d_k**-0.5, mask=attn_mask
        )

        # Apply dropout after attention
        context = self.dropout(context)

        # Reshape back: (batch, n_head, seq, d_k) -> (batch, seq, n_feat)
        context = context.transpose(0, 2, 1, 3).reshape(
            batch_size, seq_len, self.n_feat
        )

        # Output projection
        att_outs = self.linear_out(context)

        # Add FSMN memory AFTER attention (key difference from naive implementation)
        return att_outs + fsmn_memory


class PositionwiseFeedForward(nn.Module):
    """
    Positionwise feed-forward network.

    Uses ReLU activation (matching the original FunASR implementation).
    """

    def __init__(
        self,
        d_model: int,
        d_ff: int,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.w_1 = nn.Linear(d_model, d_ff, bias=True)
        self.w_2 = nn.Linear(d_ff, d_model, bias=True)
        self.dropout = nn.Dropout(dropout)

    def __call__(self, x: mx.array) -> mx.array:
        # Original: w_2(dropout(relu(w_1(x)))) - single dropout after activation
        return self.w_2(self.dropout(nn.relu(self.w_1(x))))


class EncoderLayerSANM(nn.Module):
    """
    Single SANM encoder layer.

    Structure (pre-norm):
    - LayerNorm -> Self-Attention (SANM) -> Dropout -> Residual
    - LayerNorm -> Feed-Forward -> Dropout -> Residual
    """

    def __init__(
        self,
        in_size: int,
        size: int,
        n_head: int,
        d_ff: int,
        kernel_size: int = 11,
        sanm_shift: int = 0,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.in_size = in_size
        self.size = size

        self.norm1 = nn.LayerNorm(in_size)
        self.self_attn = MultiHeadedAttentionSANM(
            n_head=n_head,
            in_feat=in_size,
            n_feat=size,
            kernel_size=kernel_size,
            sanm_shift=sanm_shift,
            dropout=dropout,
        )

        self.norm2 = nn.LayerNorm(size)
        self.feed_forward = PositionwiseFeedForward(size, d_ff, dropout)

        self.dropout = nn.Dropout(dropout)

    def __call__(
        self,
        x: mx.array,
        mask: Optional[mx.array] = None,
    ) -> mx.array:
        # Self-attention with pre-norm
        residual = x
        x = self.norm1(x)
        x = self.self_attn(x, mask)
        x = self.dropout(x)

        # Add residual (only if dimensions match)
        if self.in_size == self.size:
            x = x + residual

        # Feed-forward with pre-norm
        residual = x
        x = self.norm2(x)
        x = self.feed_forward(x)
        x = x + residual

        return x


class SenseVoiceEncoder(nn.Module):
    """
    Full SenseVoice encoder with three encoder stacks.

    Architecture:
    - Scale input by sqrt(output_size)
    - encoders0: 1 layer, processes input from 560 -> 512 dims
    - encoders: 49 layers, main encoder at 512 dims
    - after_norm: LayerNorm before time-pooling
    - tp_encoders: 20 layers, time-pooling encoder at 512 dims
    - tp_norm: Final LayerNorm

    The encoder uses SANM (Self-Attention with Memory) blocks which
    combine standard attention with FSMN for local context modeling.
    """

    def __init__(self, config: SenseVoiceEncoderConfig):
        super().__init__()
        self.config = config
        self._output_size = config.encoder_dim

        # Initial encoder layer(s) - handles input dimension change
        self.encoders0 = [
            EncoderLayerSANM(
                in_size=config.input_dim if i == 0 else config.encoder_dim,
                size=config.encoder_dim,
                n_head=config.num_heads,
                d_ff=config.ffn_dim,
                kernel_size=config.kernel_size,
                sanm_shift=config.sanm_shift,
                dropout=config.dropout,
            )
            for i in range(config.num_encoders0)
        ]

        # Main encoder layers
        self.encoders = [
            EncoderLayerSANM(
                in_size=config.encoder_dim,
                size=config.encoder_dim,
                n_head=config.num_heads,
                d_ff=config.ffn_dim,
                kernel_size=config.kernel_size,
                sanm_shift=config.sanm_shift,
                dropout=config.dropout,
            )
            for _ in range(config.num_encoders)
        ]

        # Time-pooling encoder layers
        self.tp_encoders = [
            EncoderLayerSANM(
                in_size=config.encoder_dim,
                size=config.encoder_dim,
                n_head=config.num_heads,
                d_ff=config.ffn_dim,
                kernel_size=config.kernel_size,
                sanm_shift=config.sanm_shift,
                dropout=config.dropout,
            )
            for _ in range(config.num_tp_encoders)
        ]

        # Normalization layers
        self.after_norm = nn.LayerNorm(config.encoder_dim)
        self.tp_norm = nn.LayerNorm(config.encoder_dim)

    def output_size(self) -> int:
        return self._output_size

    def __call__(
        self,
        x: mx.array,
        lengths: Optional[mx.array] = None,
    ) -> Tuple[mx.array, mx.array]:
        """
        Forward pass through the encoder.

        Parameters
        ----------
        x : mx.array, shape = (batch, seq, input_dim)
            LFR-processed audio features
        lengths : mx.array, optional
            Sequence lengths for each batch item

        Returns
        -------
        Tuple[mx.array, mx.array]
            - Encoder output of shape (batch, seq, encoder_dim)
            - Output lengths
        """
        batch_size, seq_len, _ = x.shape

        if lengths is None:
            lengths = mx.full((batch_size,), seq_len, dtype=mx.int32)

        # Scale input by sqrt(output_size) and add sinusoidal position
        # encoding (upstream ``SenseVoiceEncoderSmall.forward`` applies
        # ``self.embed = SinusoidalPositionEncoder()`` after the scale).
        x = x * math.sqrt(self._output_size)
        x = x + sinusoidal_position_encoding(seq_len, x.shape[-1], dtype=x.dtype)

        # Create attention mask from lengths if needed
        mask = None  # For full attention, no mask needed

        # Initial encoder(s)
        for layer in self.encoders0:
            x = layer(x, mask)

        # Main encoder
        for layer in self.encoders:
            x = layer(x, mask)

        # Apply after_norm
        x = self.after_norm(x)

        # Time-pooling encoder
        for layer in self.tp_encoders:
            x = layer(x, mask)

        # Final normalization
        x = self.tp_norm(x)

        return x, lengths


def create_encoder_from_config(config_dict: dict) -> SenseVoiceEncoder:
    """
    Create a SenseVoice encoder from a configuration dictionary.

    Parameters
    ----------
    config_dict : dict
        Configuration dictionary with encoder parameters

    Returns
    -------
    SenseVoiceEncoder
        Initialized encoder
    """
    config = SenseVoiceEncoderConfig(
        input_dim=config_dict.get("input_dim", 560),
        encoder_dim=config_dict.get("encoder_dim", 512),
        num_heads=config_dict.get("num_heads", 4),
        ffn_dim=config_dict.get("ffn_dim", 2048),
        kernel_size=config_dict.get("kernel_size", 11),
        sanm_shift=config_dict.get("sanm_shift", 0),
        num_encoders0=config_dict.get("num_encoders0", 1),
        num_encoders=config_dict.get("num_encoders", 49),
        num_tp_encoders=config_dict.get("num_tp_encoders", 20),
        dropout=config_dict.get("dropout", 0.0),
    )
    return SenseVoiceEncoder(config)
