# Copyright © 2025 FunASR (original model implementation)
# Copyright © Anthony DePasquale (MLX port)
# Ported to MLX from https://github.com/modelscope/FunASR
# License: licenses/funasr.txt

"""
Audio preprocessing for Fun-ASR model.

Implements a `torchaudio.compliance.kaldi.fbank`-equivalent frontend (matching
the upstream `WavFrontend` defaults: `dither=1.0`, `snip_edges=True`,
`upsacle_samples=True`, hamming window, HTK mel scale, no Slaney area
normalization, plus the kaldi defaults `remove_dc_offset=True`,
`preemphasis_coefficient=0.97`, `round_to_power_of_two=True`, `low_freq=20`)
followed by Low Frame Rate (LFR) frame stacking.
"""

import math
from typing import Optional, Union

import mlx.core as mx
import numpy as np

from mlx_audio.stt.utils import load_audio

# Audio hyperparameters for Fun-ASR
SAMPLE_RATE = 16000
N_FFT = 400  # 25ms window at 16kHz
HOP_LENGTH = 160  # 10ms hop
N_MELS = 80

# LFR (Low Frame Rate) parameters
LFR_M = 7  # Stack every 7 frames
LFR_N = 6  # Subsample by factor of 6


def _next_power_of_two(n: int) -> int:
    """Smallest power of two that is >= ``n``."""
    if n <= 1:
        return max(1, n)
    return 1 << (n - 1).bit_length()


def _hamming_window(length: int) -> mx.array:
    """Hamming window: ``w[k] = 0.54 - 0.46 * cos(2π k / (N - 1))``."""
    if length == 1:
        return mx.array([1.0])
    n = mx.arange(length, dtype=mx.float32)
    return 0.54 - 0.46 * mx.cos(2.0 * math.pi * n / float(length - 1))


def _kaldi_mel_filterbank(
    sample_rate: int,
    fft_length: int,
    n_mels: int,
    f_min: float = 20.0,
    f_max: Optional[float] = None,
) -> mx.array:
    """
    Kaldi-compatible HTK mel filterbank: triangles built in mel space, no
    Slaney area normalization, FFT bin centers as the analysis frequencies.

    Matches ``torchaudio.compliance.kaldi.get_mel_banks``. Returns a matrix of
    shape ``(n_mels, fft_length // 2)`` — the Nyquist bin is dropped.
    """
    if f_max is None:
        f_max = sample_rate / 2.0

    n_freqs = fft_length // 2

    fft_bin_width = float(sample_rate) / float(fft_length)
    all_freqs_hz = mx.arange(n_freqs, dtype=mx.float32) * fft_bin_width

    # ``2595·log10(1 + f/700)`` and ``1127·ln(1 + f/700)`` differ by a constant
    # factor that cancels in the rising/falling-edge slope ratios below.
    all_freqs_mel = 2595.0 * mx.log10(1.0 + all_freqs_hz / 700.0)

    mel_low = 2595.0 * math.log10(1.0 + f_min / 700.0)
    mel_high = 2595.0 * math.log10(1.0 + f_max / 700.0)
    mel_delta = (mel_high - mel_low) / float(n_mels + 1)

    bin_idx = mx.arange(n_mels, dtype=mx.float32)
    left_mel = mel_low + bin_idx * mel_delta  # (n_mels,)
    center_mel = mel_low + (bin_idx + 1.0) * mel_delta
    right_mel = mel_low + (bin_idx + 2.0) * mel_delta

    # Broadcast to (n_freqs, n_mels)
    mel = all_freqs_mel[:, None]
    up_slope = (mel - left_mel[None, :]) / (center_mel - left_mel)[None, :]
    down_slope = (right_mel[None, :] - mel) / (right_mel - center_mel)[None, :]
    filterbank = mx.maximum(mx.zeros_like(up_slope), mx.minimum(up_slope, down_slope))

    # (n_mels, n_freqs)
    return filterbank.T


def log_mel_spectrogram(
    audio: Union[str, np.ndarray, mx.array],
    n_mels: int = N_MELS,
    n_fft: int = N_FFT,
    hop_length: int = HOP_LENGTH,
    sample_rate: int = SAMPLE_RATE,
    dither: float = 0.0,
) -> mx.array:
    """
    Compute kaldi-compatible log-mel filterbank features.

    Parameters
    ----------
    audio : str, np.ndarray, or mx.array
        Path to audio or float waveform in [-1, 1] at ``sample_rate``.
    n_mels : int
        Number of mel filterbank bins.
    n_fft : int
        Window length in samples (e.g. 400 = 25ms at 16kHz). The actual FFT
        length is rounded up to the next power of two.
    hop_length : int
        Frame stride in samples.
    sample_rate : int
        Audio sample rate in Hz.
    dither : float
        Gaussian dither standard deviation at int16 magnitude. The upstream
        kaldi default is ``1.0``; pass ``0.0`` for deterministic output.

    Returns
    -------
    mx.array, shape ``(num_frames, n_mels)``
        Log-mel features.
    """
    if isinstance(audio, str):
        audio = load_audio(audio, sr=sample_rate)
    if isinstance(audio, np.ndarray):
        audio = mx.array(audio)
    if audio.ndim > 1:
        audio = audio.reshape(-1)

    # Upscale float waveform to int16 magnitude (``upsacle_samples=True``).
    audio = audio * float(1 << 15)

    if audio.shape[0] < n_fft:
        raise ValueError(
            f"Audio is too short for one frame ({audio.shape[0]} < {n_fft})"
        )

    # ``snip_edges=True`` framing — no padding, drop trailing partial frame.
    num_frames = (audio.shape[0] - n_fft) // hop_length + 1
    frames = mx.as_strided(audio, shape=(num_frames, n_fft), strides=(hop_length, 1))

    if dither != 0.0:
        frames = frames + mx.random.normal(shape=(num_frames, n_fft)) * dither

    # Per-frame DC offset removal.
    frames = frames - mx.mean(frames, axis=-1, keepdims=True)

    # Per-frame pre-emphasis with replicate padding:
    # ``y[i] = x[i] - 0.97 * x[i-1]``, ``y[0] = (1 - 0.97) * x[0] = 0.03 * x[0]``.
    shifted = mx.concatenate([frames[:, 0:1], frames[:, : n_fft - 1]], axis=-1)
    frames = frames - 0.97 * shifted

    # Hamming window.
    frames = frames * _hamming_window(n_fft)

    # Zero-pad to next power of two for the FFT (``round_to_power_of_two``).
    fft_length = _next_power_of_two(n_fft)
    pad = fft_length - n_fft
    if pad > 0:
        frames = mx.concatenate(
            [frames, mx.zeros((num_frames, pad), dtype=frames.dtype)], axis=-1
        )

    spec = mx.fft.rfft(frames, n=fft_length, axis=-1)

    # Drop the Nyquist bin to match kaldi (``num_fft_bins = fft_length / 2``).
    n_freqs = fft_length // 2
    spec = spec[:, :n_freqs]
    power = mx.abs(spec) ** 2

    filters = _kaldi_mel_filterbank(
        sample_rate=sample_rate, fft_length=fft_length, n_mels=n_mels, f_min=20.0
    )
    mel_spec = mx.matmul(power, filters.T)

    eps = float(np.finfo(np.float32).eps)
    return mx.log(mx.maximum(mel_spec, eps))


def apply_lfr(
    features: mx.array,
    lfr_m: int = LFR_M,
    lfr_n: int = LFR_N,
) -> mx.array:
    """
    Apply Low Frame Rate (LFR) processing.

    Stacks ``lfr_m`` consecutive frames and subsamples by ``lfr_n``. Uses
    vectorized gather operations.

    Parameters
    ----------
    features : mx.array, shape ``(n_frames, n_mels)``
    lfr_m : int
        Number of frames to stack (default 7).
    lfr_n : int
        Subsampling factor (default 6).

    Returns
    -------
    mx.array, shape ``(ceil(n_frames / lfr_n), n_mels * lfr_m)``
    """
    T, n_mels = features.shape

    T_lfr = int(math.ceil(T / lfr_n))

    left_pad = (lfr_m - 1) // 2
    if left_pad > 0:
        left_padding = mx.broadcast_to(features[0:1], (left_pad, n_mels))
        features = mx.concatenate([left_padding, features], axis=0)

    T_padded = features.shape[0]
    total_needed = (T_lfr - 1) * lfr_n + lfr_m
    if total_needed > T_padded:
        right_pad = total_needed - T_padded
        right_padding = mx.broadcast_to(features[-1:], (right_pad, n_mels))
        features = mx.concatenate([features, right_padding], axis=0)

    start_indices = mx.arange(T_lfr) * lfr_n
    offsets = mx.arange(lfr_m)
    indices = start_indices[:, None] + offsets[None, :]

    gathered = features[indices]
    return gathered.reshape(T_lfr, -1)


def apply_cmvn(
    features: mx.array,
    cmvn_mean: Optional[mx.array] = None,
    cmvn_istd: Optional[mx.array] = None,
) -> mx.array:
    """
    Apply Cepstral Mean and Variance Normalization.

    With ``cmvn_mean`` and ``cmvn_istd`` provided, applies the precomputed
    transform ``(x + mean) * istd`` (note: ``cmvn_mean`` is the negative of
    the actual mean, following kaldi's convention). Otherwise applies
    per-utterance normalization.

    Fun-ASR-Nano-2512 ships with ``cmvn_file: null``, so per-utterance
    normalization is **not** what the model was trained for and should not be
    used in inference unless a CMVN file is loaded.
    """
    if cmvn_mean is not None and cmvn_istd is not None:
        return (features + cmvn_mean) * cmvn_istd

    mean = mx.mean(features, axis=0, keepdims=True)
    std = mx.sqrt(mx.var(features, axis=0, keepdims=True)) + 1e-6
    return (features - mean) / std


def preprocess_audio(
    audio: Union[str, np.ndarray, mx.array],
    n_mels: int = N_MELS,
    lfr_m: int = LFR_M,
    lfr_n: int = LFR_N,
    cmvn_mean: Optional[mx.array] = None,
    cmvn_istd: Optional[mx.array] = None,
    apply_normalization: bool = False,
    dither: float = 0.0,
) -> mx.array:
    """
    Full audio preprocessing pipeline for Fun-ASR.

    Steps:
      1. Compute kaldi-compatible log-mel filterbank features.
      2. Apply LFR (frame stacking and subsampling).
      3. Optionally apply CMVN — disabled by default to match the
         Fun-ASR-Nano-2512 config (``cmvn_file: null``).

    Parameters
    ----------
    audio : str, np.ndarray, or mx.array
        Audio path or waveform.
    n_mels : int
        Mel bins (default 80).
    lfr_m : int
        LFR frame stacking count (default 7).
    lfr_n : int
        LFR subsampling factor (default 6).
    cmvn_mean, cmvn_istd : mx.array, optional
        Precomputed CMVN statistics.
    apply_normalization : bool
        Whether to apply CMVN. Default ``False`` to match upstream config.
    dither : float
        Gaussian dither at int16 magnitude. The upstream kaldi default is
        ``1.0``; pass ``0.0`` for deterministic output. Defaults to ``0.0``
        here so callers opt in.

    Returns
    -------
    mx.array, shape ``(ceil(T / lfr_n), n_mels * lfr_m)``
    """
    mel = log_mel_spectrogram(audio, n_mels=n_mels, dither=dither)
    feats = apply_lfr(mel, lfr_m=lfr_m, lfr_n=lfr_n)
    if apply_normalization:
        feats = apply_cmvn(feats, cmvn_mean, cmvn_istd)
    return feats


def compute_feature_lengths(
    audio_lengths: mx.array,
    hop_length: int = HOP_LENGTH,
    lfr_n: int = LFR_N,
) -> mx.array:
    """
    Compute output feature lengths after preprocessing.

    Parameters
    ----------
    audio_lengths : mx.array
        Lengths of input audio in samples.
    hop_length : int
        Hop length for STFT (default 160).
    lfr_n : int
        LFR subsampling factor (default 6).

    Returns
    -------
    mx.array
        Output feature lengths.
    """
    n_frames = audio_lengths // hop_length
    out_len = (n_frames + lfr_n - 1) // lfr_n
    return out_len.astype(mx.int32)
