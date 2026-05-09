"""
Tests for the Fun-ASR Python MLX port.

Two layers:

- ``TestFunASRFrontend`` — unit tests that compare the kaldi-style frontend
  in :mod:`mlx_audio.stt.models.funasr.audio` against
  :func:`torchaudio.compliance.kaldi.fbank` on synthetic audio. These are
  fast and don't load the model.
- ``TestFunASRTranscription`` — end-to-end test that downloads the LJ Speech
  sample and the 4-bit Fun-ASR-Nano model, then asserts the transcription
  matches the reference text. Network and model-load required; skipped if
  the model can't be loaded.
"""

import os
import unittest
from pathlib import Path
from urllib.request import urlretrieve

import mlx.core as mx
import numpy as np

LJ_URL = "https://keithito.com/LJ-Speech-Dataset/LJ037-0171.wav"
LJ_REFERENCE = (
    "The examination and testimony of the experts enabled the commission to "
    "conclude that five shots may have been fired"
)
MODEL_REPO = "mlx-community/Fun-ASR-Nano-2512-4bit"


def _fixture_path(name: str) -> Path:
    """Cached path under /tmp for the test audio sample."""
    return Path("/tmp") / name


def _ensure_lj_speech() -> Path:
    path = _fixture_path("lj037-0171.wav")
    if not path.exists():
        urlretrieve(LJ_URL, str(path))
    return path


class TestFunASRFrontend(unittest.TestCase):
    """Verify the FunASR frontend against ``torchaudio.compliance.kaldi.fbank``."""

    def test_log_mel_matches_kaldi_fbank(self):
        try:
            import torch
            import torchaudio.compliance.kaldi as kaldi
        except ImportError:
            self.skipTest("torchaudio not installed")

        from mlx_audio.stt.models.funasr.audio import log_mel_spectrogram

        np.random.seed(0)
        # Two seconds of low-amplitude noise — enough frames to exercise the
        # framing math without being slow.
        audio_np = np.random.randn(32000).astype(np.float32) * 0.1

        torch_wav = torch.from_numpy(audio_np).unsqueeze(0) * (1 << 15)
        torch_fbank = kaldi.fbank(
            torch_wav,
            num_mel_bins=80,
            frame_length=25,
            frame_shift=10,
            dither=0.0,
            energy_floor=0.0,
            window_type="hamming",
            sample_frequency=16000,
            snip_edges=True,
        ).numpy()

        mlx_fbank = np.asarray(log_mel_spectrogram(mx.array(audio_np), dither=0.0))

        self.assertEqual(mlx_fbank.shape, tuple(torch_fbank.shape))
        max_abs_diff = float(np.max(np.abs(torch_fbank - mlx_fbank)))
        # The two paths use slightly different float reductions (HTK vs the
        # kaldi 1127·ln formulation, mx.fft.rfft vs torch FFT). 1e-3 is far
        # tighter than anything that would matter to the model.
        self.assertLess(
            max_abs_diff,
            1e-3,
            f"FunASR frontend diverges from kaldi.fbank: max abs diff "
            f"{max_abs_diff}",
        )

    def test_apply_lfr_shape(self):
        from mlx_audio.stt.models.funasr.audio import apply_lfr

        # 100 mel frames -> ceil(100/6) = 17 LFR frames, each 7*80 = 560 dims.
        feats = mx.zeros((100, 80))
        out = apply_lfr(feats, lfr_m=7, lfr_n=6)
        self.assertEqual(out.shape, (17, 560))


class TestFunASRTranscription(unittest.TestCase):
    """End-to-end transcription test on the LJ Speech sample."""

    @classmethod
    def setUpClass(cls):
        if os.environ.get("MLX_AUDIO_SKIP_MODEL_TESTS"):
            raise unittest.SkipTest(
                "MLX_AUDIO_SKIP_MODEL_TESTS set; skipping model load"
            )

        try:
            from mlx_audio.stt.models.funasr import Model
        except ImportError as e:
            raise unittest.SkipTest(f"FunASR import failed: {e}")

        try:
            cls.model = Model.from_pretrained(MODEL_REPO)
        except Exception as e:
            raise unittest.SkipTest(f"Could not load {MODEL_REPO}: {e}")

        cls.audio_path = _ensure_lj_speech()

    def test_lj_speech_transcription(self):
        """Reference transcription should match exactly (greedy decoding)."""
        result = self.model.generate(str(self.audio_path), max_tokens=128)

        # Greedy + frontend determinism (dither=0 by default in
        # ``preprocess_audio``) means the transcription should be character-
        # stable. Allow a trailing period since punctuation can vary.
        normalized = result.text.rstrip(".").strip()
        self.assertEqual(
            normalized,
            LJ_REFERENCE,
            f"Transcription drifted from reference. Got: {result.text!r}",
        )

    def test_fake_token_len_used(self):
        """``encode_audio`` returns ``fake_token_len`` frames, not the full
        adaptor output (regression test for the issue-#4 long-audio bug)."""
        from mlx_audio.stt.models.funasr.audio import preprocess_audio

        features = preprocess_audio(str(self.audio_path), apply_normalization=False)
        lfr_frames = features.shape[0]
        expected = self.model._fake_token_len(lfr_frames)

        embeddings = self.model.encode_audio(str(self.audio_path))
        # Shape: (1, audio_token_len, llm_dim)
        self.assertEqual(embeddings.shape[0], 1)
        self.assertEqual(embeddings.shape[1], expected)
        self.assertLess(
            embeddings.shape[1],
            lfr_frames,
            "encode_audio should slice down from the LFR frame count",
        )


if __name__ == "__main__":
    unittest.main()
