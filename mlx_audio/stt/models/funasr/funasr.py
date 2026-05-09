# Copyright © 2025 FunASR (original model implementation)
# Copyright © Anthony DePasquale (MLX port)
# Ported to MLX from https://github.com/modelscope/FunASR
# License: licenses/funasr.txt

"""
Fun-ASR-Nano model implementation for MLX.

The model combines:
- Audio frontend (kaldi-style mel filterbank + LFR frame stacking)
- SenseVoice encoder (SANM-based)
- Audio adaptor (encoder_dim -> llm_dim projection + transformer blocks)
- Qwen3 LLM decoder

Inference follows upstream PyTorch (`fun_asr_nano/model.py`):
- The system message is the literal Qwen3 default ``"You are a helpful assistant."``.
- The user message is a Chinese instruction (``语音转写：`` or
  ``语音转写成{language}：``) followed by the audio embeddings.
- The audio embeddings are spliced directly into the LLM input embedding
  stream — the literal speech-marker strings are never tokenized and never
  reach the LLM, because they decompose to multi-piece BPE.
- When ``use_low_frame_rate`` is set on the adaptor, only the first
  ``fake_token_len`` frames of the adaptor output are consumed.
"""

import json
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Generator, List, Optional, Tuple, Union

import mlx.core as mx
import mlx.nn as nn
import numpy as np

from mlx_audio.stt.utils import get_model_path

from .adaptor import AudioAdaptor, AudioAdaptorConfig
from .audio import LFR_M, LFR_N, N_MELS, SAMPLE_RATE, preprocess_audio
from .encoder import SenseVoiceEncoder, SenseVoiceEncoderConfig
from .qwen3 import Qwen3Config, Qwen3ForCausalLM


@dataclass
class STTOutput:
    """Output from speech-to-text generation."""

    text: str
    segments: Optional[List[dict]] = None
    language: Optional[str] = None
    task: Optional[str] = None
    duration: Optional[float] = None
    tokens: Optional[List[int]] = None


# Task types
TASK_TRANSCRIBE = "transcribe"
TASK_TRANSLATE = "translate"

# Language codes -> Chinese display names used in the upstream prompt template.
# Upstream `fun_asr_nano/model.py::get_prompt` interpolates the language name
# directly into ``语音转写成{language}：``. Unknown codes fall through to
# ``语音转写：`` (no language hint).
LANGUAGE_PROMPT_NAMES = {
    "zh": "中文",
    "en": "英语",
    "ja": "日语",
    "ko": "韩语",
    "es": "西班牙语",
    "fr": "法语",
    "de": "德语",
    "it": "意大利语",
    "pt": "葡萄牙语",
    "ru": "俄语",
    "ar": "阿拉伯语",
    "th": "泰语",
    "vi": "越南语",
    "yue": "粤语",
}

# English-display-name lookup. Documentation only — does not affect prompting.
SUPPORTED_LANGUAGES = {
    "zh": "Chinese",
    "en": "English",
    "ja": "Japanese",
    "ko": "Korean",
    "es": "Spanish",
    "fr": "French",
    "de": "German",
    "it": "Italian",
    "pt": "Portuguese",
    "ru": "Russian",
    "ar": "Arabic",
    "th": "Thai",
    "vi": "Vietnamese",
    "yue": "Cantonese",
    "auto": "Auto-detect",
}


@dataclass
class FunASRConfig:
    """Configuration for Fun-ASR model."""

    # Audio processing
    sample_rate: int = 16000
    n_mels: int = 80
    lfr_m: int = 7
    lfr_n: int = 6

    # Encoder config
    encoder: SenseVoiceEncoderConfig = field(
        default_factory=lambda: SenseVoiceEncoderConfig()
    )

    # Adaptor config
    adaptor: AudioAdaptorConfig = field(default_factory=lambda: AudioAdaptorConfig())

    # LLM config
    llm: Qwen3Config = field(default_factory=lambda: Qwen3Config())

    # Special token strings
    sos_token: str = "<|startofspeech|>"
    eos_token: str = "<|endofspeech|>"
    im_start_token: str = "<|im_start|>"
    im_end_token: str = "<|im_end|>"

    # Generation defaults
    max_tokens: int = 512
    temperature: float = 0.0

    @classmethod
    def from_dict(cls, config_dict: dict) -> "FunASRConfig":
        """Create config from dictionary."""
        encoder_config = SenseVoiceEncoderConfig(
            input_dim=config_dict.get("encoder", {}).get("input_dim", 560),
            encoder_dim=config_dict.get("encoder", {}).get("encoder_dim", 512),
            num_heads=config_dict.get("encoder", {}).get("num_heads", 4),
            ffn_dim=config_dict.get("encoder", {}).get("ffn_dim", 2048),
            kernel_size=config_dict.get("encoder", {}).get("kernel_size", 11),
            num_encoders0=config_dict.get("encoder", {}).get("num_encoders0", 1),
            num_encoders=config_dict.get("encoder", {}).get("num_encoders", 49),
            num_tp_encoders=config_dict.get("encoder", {}).get("num_tp_encoders", 20),
            dropout=config_dict.get("encoder", {}).get("dropout", 0.0),
        )

        adaptor_config = AudioAdaptorConfig(
            downsample_rate=config_dict.get("adaptor", {}).get("downsample_rate", 1),
            encoder_dim=config_dict.get("adaptor", {}).get("encoder_dim", 512),
            llm_dim=config_dict.get("adaptor", {}).get("llm_dim", 1024),
            ffn_dim=config_dict.get("adaptor", {}).get("ffn_dim", 2048),
            n_layer=config_dict.get("adaptor", {}).get("n_layer", 2),
            attention_heads=config_dict.get("adaptor", {}).get("attention_heads", 8),
            dropout=config_dict.get("adaptor", {}).get("dropout", 0.0),
            use_low_frame_rate=config_dict.get("adaptor", {}).get(
                "use_low_frame_rate", True
            ),
        )

        llm_config = Qwen3Config(
            vocab_size=config_dict.get("llm", {}).get("vocab_size", 151936),
            hidden_size=config_dict.get("llm", {}).get("hidden_size", 1024),
            num_hidden_layers=config_dict.get("llm", {}).get("num_hidden_layers", 28),
            num_attention_heads=config_dict.get("llm", {}).get(
                "num_attention_heads", 16
            ),
            num_key_value_heads=config_dict.get("llm", {}).get(
                "num_key_value_heads", 8
            ),
            intermediate_size=config_dict.get("llm", {}).get("intermediate_size", 3072),
            max_position_embeddings=config_dict.get("llm", {}).get(
                "max_position_embeddings", 40960
            ),
            rope_theta=config_dict.get("llm", {}).get("rope_theta", 1000000.0),
            rms_norm_eps=config_dict.get("llm", {}).get("rms_norm_eps", 1e-6),
            tie_word_embeddings=config_dict.get("llm", {}).get(
                "tie_word_embeddings", False
            ),
            head_dim=config_dict.get("llm", {}).get("head_dim", 128),
        )

        return cls(
            sample_rate=config_dict.get("sample_rate", 16000),
            n_mels=config_dict.get("n_mels", 80),
            lfr_m=config_dict.get("lfr_m", 7),
            lfr_n=config_dict.get("lfr_n", 6),
            encoder=encoder_config,
            adaptor=adaptor_config,
            llm=llm_config,
            sos_token=config_dict.get("sos_token", "<|startofspeech|>"),
            eos_token=config_dict.get("eos_token", "<|endofspeech|>"),
            im_start_token=config_dict.get("im_start_token", "<|im_start|>"),
            im_end_token=config_dict.get("im_end_token", "<|im_end|>"),
            max_tokens=config_dict.get("max_tokens", 512),
            temperature=config_dict.get("temperature", 0.0),
        )


class Model(nn.Module):
    """
    Fun-ASR-Nano main model.

    Combines audio encoder, adaptor, and LLM decoder for end-to-end speech
    recognition. Inference path mirrors upstream PyTorch
    (`fun_asr_nano/model.py`): system + user prompt with embedded audio,
    optionally sliced to ``fake_token_len`` when ``use_low_frame_rate=True``.
    """

    def __init__(self, config: FunASRConfig):
        super().__init__()
        self.config = config

        self.audio_encoder = SenseVoiceEncoder(config.encoder)
        self.audio_adaptor = AudioAdaptor(config.adaptor)
        self.llm = Qwen3ForCausalLM(config.llm)

        # Tokenizer (set during loading)
        self._tokenizer = None

        # Token IDs that signal end-of-generation. Only single-ID encodings
        # are added — the speech markers (``<|startofspeech|>`` etc.) are NOT
        # added tokens; they decompose to multi-piece BPE and must never be
        # treated as stop tokens.
        self._eos_token_ids: set = set()

    # ------------------------------------------------------------------
    # Audio encoding
    # ------------------------------------------------------------------

    @staticmethod
    def _fake_token_len(lfr_frame_count: int) -> int:
        """
        Number of audio tokens the LLM expects when ``use_low_frame_rate=True``.

        Mirrors upstream ``fun_asr_nano/model.py``: three successive
        ``1 + (x - 1) // 2`` reductions on the LFR frame count, with the last
        one expressed as ``(x - 1) // 2 + 1``.
        """
        olens1 = 1 + (lfr_frame_count - 1) // 2
        olens2 = 1 + (olens1 - 1) // 2
        return (olens2 - 1) // 2 + 1

    def encode_audio(
        self,
        audio: Union[str, np.ndarray, mx.array, Path],
    ) -> mx.array:
        """
        Encode audio to LLM-space embeddings, sliced to ``fake_token_len``
        when ``use_low_frame_rate`` is enabled on the adaptor.

        Returns shape ``(1, audio_token_len, llm_dim)``.
        """
        if isinstance(audio, Path):
            audio = str(audio)

        # CMVN is intentionally disabled — the upstream Fun-ASR-Nano-2512
        # config has ``cmvn_file: null``, so per-utterance normalization
        # would shift the feature distribution away from training.
        features = preprocess_audio(
            audio,
            n_mels=self.config.n_mels,
            lfr_m=self.config.lfr_m,
            lfr_n=self.config.lfr_n,
            apply_normalization=False,
        )

        if features.ndim == 2:
            batched = features[None, ...]
        else:
            batched = features

        encoder_out, lengths = self.audio_encoder(batched)
        adapted, _ = self.audio_adaptor(encoder_out, lengths)

        # Slice to the audio-token count the LLM was trained to consume.
        lfr_frame_count = features.shape[0]
        if self.config.adaptor.use_low_frame_rate:
            audio_token_len = self._fake_token_len(lfr_frame_count)
        else:
            audio_token_len = adapted.shape[1]
        truncated_len = min(audio_token_len, adapted.shape[1])
        return adapted[:, :truncated_len, :]

    # ------------------------------------------------------------------
    # Prompt construction
    # ------------------------------------------------------------------

    def _build_prompt_parts(
        self,
        task: str = TASK_TRANSCRIBE,
        language: str = "auto",
        target_language: str = "en",
        initial_prompt: Optional[str] = None,
    ) -> Tuple[List[int], List[int]]:
        """
        Build the prompt as the pair of token sequences that go before and
        after the audio embeddings.

        The upstream PyTorch reference (``fun_asr_nano/model.py::get_prompt``
        and ``generate_chatml``) renders:

        ```
        <|im_start|>system
        You are a helpful assistant.<|im_end|>
        <|im_start|>user
        {prompt}<|startofspeech|>!{audio}<|endofspeech|><|im_end|>
        <|im_start|>assistant
        ```

        where ``{prompt}`` is ``"语音转写："`` (auto) or
        ``"语音转写成{lang}："`` (with a Chinese language name).

        We never tokenize the speech markers — they decompose to BPE pieces
        like ``<``/``|``/``start``/... in the Qwen3 tokenizer and were never
        meant to be in the LLM input. Instead we tokenize the text up to and
        after the marker span and splice the audio embeddings between them.

        Returns ``(pre_ids, post_ids)``. The full LLM input is then
        ``embed(pre_ids) + audio_embeddings + embed(post_ids)``.
        """
        # Single language slot in the upstream prompt — we repurpose it as
        # the target language for the (best-effort) translate task.
        prompt_lang = target_language if task == TASK_TRANSLATE else language
        lang_name = LANGUAGE_PROMPT_NAMES.get(prompt_lang)

        if lang_name is None:
            user_instruction = "语音转写："
        else:
            user_instruction = f"语音转写成{lang_name}："

        if initial_prompt:
            user_instruction = f"{initial_prompt}\n\n{user_instruction}"

        pre_text = (
            f"{self.config.im_start_token}system\n"
            "You are a helpful assistant."
            f"{self.config.im_end_token}\n"
            f"{self.config.im_start_token}user\n"
            f"{user_instruction}"
        )
        post_text = (
            f"{self.config.im_end_token}\n" f"{self.config.im_start_token}assistant\n"
        )

        pre_ids = self._tokenizer.encode(pre_text, add_special_tokens=False)
        post_ids = self._tokenizer.encode(post_text, add_special_tokens=False)
        return pre_ids, post_ids

    def _splice_embeddings(
        self,
        pre_ids: List[int],
        audio_embeddings: mx.array,
        post_ids: List[int],
    ) -> mx.array:
        """
        Embed the prompt token sequences and concatenate them around the
        audio embeddings. Returns shape ``(1, total, llm_dim)``.
        """
        embed = self.llm.get_input_embeddings()
        pre = embed(mx.array(pre_ids, dtype=mx.int32))
        post = embed(mx.array(post_ids, dtype=mx.int32))
        audio = (
            audio_embeddings.squeeze(0)
            if audio_embeddings.ndim == 3
            else audio_embeddings
        )
        combined = mx.concatenate([pre, audio, post], axis=0)
        return combined[None, ...]

    # ------------------------------------------------------------------
    # Sampling
    # ------------------------------------------------------------------

    def _sample_next_token(
        self,
        logits: mx.array,
        temperature: float = 0.0,
        top_p: float = 0.95,
        top_k: int = 0,
    ) -> mx.array:
        """Sample a token from logits ``(batch, seq, vocab)``."""
        logits = logits[:, -1, :]

        if temperature == 0:
            return mx.argmax(logits, axis=-1)

        logits = logits / temperature

        if top_k > 0:
            top_k_logits, top_k_indices = mx.topk(logits, k=top_k)
            mask = mx.full_like(logits, float("-inf"))
            mask = mask.at[..., top_k_indices].set(top_k_logits)
            logits = mask

        if top_p < 1.0:
            sorted_logits = mx.sort(logits, axis=-1)[:, ::-1]
            sorted_indices = mx.argsort(logits, axis=-1)[:, ::-1]
            cumulative_probs = mx.cumsum(mx.softmax(sorted_logits, axis=-1), axis=-1)

            sorted_to_remove = cumulative_probs > top_p
            # Keep at least one token by shifting right.
            sorted_to_remove = mx.concatenate(
                [
                    mx.zeros((logits.shape[0], 1), dtype=mx.bool_),
                    sorted_to_remove[:, :-1],
                ],
                axis=-1,
            )
            for b in range(logits.shape[0]):
                indices_to_remove = sorted_indices[b][sorted_to_remove[b]]
                logits = logits.at[b, indices_to_remove].set(float("-inf"))

        probs = mx.softmax(logits, axis=-1)
        return mx.random.categorical(mx.log(probs + 1e-10))

    # ------------------------------------------------------------------
    # Generation
    # ------------------------------------------------------------------

    def stream_generate(
        self,
        audio: Union[str, np.ndarray, mx.array, Path],
        *,
        max_tokens: int = 512,
        temperature: float = 0.0,
        top_p: float = 0.95,
        top_k: int = 0,
        language: str = "auto",
        task: str = TASK_TRANSCRIBE,
        target_language: str = "en",
        initial_prompt: Optional[str] = None,
    ) -> Generator[Tuple[int, mx.array], None, None]:
        """Yield ``(token_id, logits)`` tuples until EOS or ``max_tokens``."""
        audio_embeddings = self.encode_audio(audio)

        pre_ids, post_ids = self._build_prompt_parts(
            task=task,
            language=language,
            target_language=target_language,
            initial_prompt=initial_prompt,
        )
        input_embeddings = self._splice_embeddings(pre_ids, audio_embeddings, post_ids)

        cache: Optional[List] = None

        # Prefill.
        logits, cache = self.llm(
            input_embeddings=input_embeddings,
            cache=cache,
        )
        mx.async_eval(logits, cache)

        for _ in range(max_tokens):
            token = self._sample_next_token(logits, temperature, top_p, top_k)

            # Run the next step before extracting the token ID, so the GPU
            # work overlaps with the host-side ``.item()`` sync.
            next_input = self.llm.get_input_embeddings()(token.reshape(1, 1))
            logits, cache = self.llm(
                input_embeddings=next_input,
                cache=cache,
            )
            mx.async_eval(logits, cache)

            token_id = int(token.item())

            if token_id in self._eos_token_ids:
                break

            yield token_id, logits

    def generate(
        self,
        audio: Union[str, np.ndarray, mx.array, Path],
        *,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        top_p: float = 0.95,
        top_k: int = 0,
        language: str = "auto",
        task: str = TASK_TRANSCRIBE,
        target_language: str = "en",
        initial_prompt: Optional[str] = None,
        verbose: bool = False,
        stream: bool = False,
        **kwargs,
    ) -> Union[STTOutput, Generator[str, None, STTOutput]]:
        """
        Generate transcription or translation from audio.

        Parameters
        ----------
        audio : str, Path, np.ndarray, or mx.array
            Audio input (file path or waveform).
        max_tokens : int, optional
            Maximum tokens to generate (default: from config).
        temperature : float, optional
            Sampling temperature; 0 for greedy (default: from config).
        top_p, top_k : float, int
            Nucleus / top-k sampling parameters.
        language : str
            Source language hint, or ``"auto"`` for no hint.
        task : str
            ``"transcribe"`` (default) or ``"translate"``. Translation reuses
            the upstream prompt's single language slot to nudge the LLM
            toward the target language; quality is best-effort since the
            model wasn't trained for cross-lingual output.
        target_language : str
            Target language for translation (default ``"en"``).
        initial_prompt : str, optional
            Prepended to the user instruction.
        verbose : bool
            Print tokens as they're generated.
        stream : bool
            If True, return a generator yielding text chunks.
        """
        if max_tokens is None:
            max_tokens = self.config.max_tokens
        if temperature is None:
            temperature = self.config.temperature

        if isinstance(audio, Path):
            audio = str(audio)

        start_time = time.time()

        if stream:
            return self._generate_stream(
                audio=audio,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                language=language,
                task=task,
                target_language=target_language,
                initial_prompt=initial_prompt,
                verbose=verbose,
            )

        tokens: List[int] = []
        for token_id, _ in self.stream_generate(
            audio,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            language=language,
            task=task,
            target_language=target_language,
            initial_prompt=initial_prompt,
        ):
            tokens.append(token_id)
            if verbose:
                print(self._tokenizer.decode([token_id]), end="", flush=True)

        if verbose:
            print()

        duration = time.time() - start_time
        text = self._clean_output(self._tokenizer.decode(tokens))

        detected_language = (
            language if language != "auto" else self._detect_language_from_text(text)
        )

        mx.clear_cache()

        return STTOutput(
            text=text,
            language=detected_language,
            task=task,
            duration=duration,
            tokens=tokens,
            segments=None,
        )

    def _generate_stream(
        self,
        audio: Union[str, np.ndarray, mx.array],
        max_tokens: int,
        temperature: float,
        top_p: float,
        top_k: int,
        language: str,
        task: str,
        target_language: str,
        initial_prompt: Optional[str],
        verbose: bool,
    ) -> Generator[str, None, STTOutput]:
        """Internal streaming generator."""
        start_time = time.time()
        tokens: List[int] = []

        for token_id, _ in self.stream_generate(
            audio,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            language=language,
            task=task,
            target_language=target_language,
            initial_prompt=initial_prompt,
        ):
            tokens.append(token_id)
            chunk = self._tokenizer.decode([token_id])

            if verbose:
                print(chunk, end="", flush=True)

            yield chunk

        if verbose:
            print()

        duration = time.time() - start_time
        text = self._clean_output(self._tokenizer.decode(tokens))
        detected_language = (
            language if language != "auto" else self._detect_language_from_text(text)
        )

        mx.clear_cache()

        return STTOutput(
            text=text,
            language=detected_language,
            task=task,
            duration=duration,
            tokens=tokens,
            segments=None,
        )

    # ------------------------------------------------------------------
    # Output cleanup / language detection
    # ------------------------------------------------------------------

    def _detect_language_from_text(self, text: str) -> str:
        """Best-effort script-based language tag for the output text."""
        if not text:
            return "unknown"

        cjk = sum(1 for c in text if "一" <= c <= "鿿")
        japanese = sum(1 for c in text if "぀" <= c <= "ヿ")
        korean = sum(1 for c in text if "가" <= c <= "힯")
        arabic = sum(1 for c in text if "؀" <= c <= "ۿ")
        thai = sum(1 for c in text if "฀" <= c <= "๿")
        cyrillic = sum(1 for c in text if "Ѐ" <= c <= "ӿ")

        total = len(text)
        if total == 0:
            return "unknown"

        if japanese / total > 0.1:
            return "ja"
        if korean / total > 0.1:
            return "ko"
        if cjk / total > 0.2:
            return "zh"
        if arabic / total > 0.2:
            return "ar"
        if thai / total > 0.2:
            return "th"
        if cyrillic / total > 0.2:
            return "ru"
        return "en"

    def _clean_output(self, text: str) -> str:
        """Strip ``<think>`` blocks and any leaked special-token strings."""
        text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
        for tok in [
            self.config.im_start_token,
            self.config.im_end_token,
            self.config.sos_token,
            self.config.eos_token,
            "<|endoftext|>",
        ]:
            text = text.replace(tok, "")
        return text.strip()

    # ------------------------------------------------------------------
    # Weight loading
    # ------------------------------------------------------------------

    def sanitize(self, weights: Dict) -> Dict:
        """
        Sanitize weights for loading. Handles Conv1d weight transposition.
        """
        sanitized = {}
        for k, v in weights.items():
            if "fsmn_block" in k and "conv.weight" in k:
                if v.ndim == 3 and v.shape[1] == 1:
                    v = v.squeeze(1)[..., None]
            elif "conv" in k and "weight" in k:
                if v.ndim == 3 and v.shape[-1] < v.shape[-2]:
                    v = v.swapaxes(-1, -2)
            sanitized[k] = v
        return sanitized

    @classmethod
    def from_pretrained(
        cls,
        path_or_hf_repo: str,
        *,
        dtype: mx.Dtype = mx.bfloat16,
        **kwargs,
    ) -> "Model":
        """Load model from a local path or Hugging Face repo."""
        from transformers import AutoTokenizer

        revision = kwargs.get("revision", None)
        force_download = kwargs.get("force_download", False)
        model_path = get_model_path(
            path_or_hf_repo, revision=revision, force_download=force_download
        )

        config_path = model_path / "config.json"
        config_dict = {}
        if config_path.exists():
            with open(config_path, "r") as f:
                config_dict = json.load(f)
            config = FunASRConfig.from_dict(config_dict)
        else:
            config = FunASRConfig()

        model = cls(config)

        # Apply quantization before loading weights if specified in config.
        if "quantization" in config_dict:
            q_config = config_dict["quantization"]
            q_bits = q_config.get("bits", 4)
            q_group_size = q_config.get("group_size", 64)
            q_components = set(q_config.get("quantized_components", []))

            def class_predicate(path: str, module) -> bool:
                if isinstance(module, nn.Linear):
                    for component in q_components:
                        if component in path:
                            return True
                return False

            nn.quantize(
                model,
                bits=q_bits,
                group_size=q_group_size,
                class_predicate=class_predicate,
            )

        # Tokenizer.
        try:
            model._tokenizer = AutoTokenizer.from_pretrained(
                path_or_hf_repo, trust_remote_code=True
            )
        except Exception:
            model._tokenizer = AutoTokenizer.from_pretrained(
                str(model_path), trust_remote_code=True
            )

        model._setup_eos_tokens()

        # Weights.
        weight_files = list(model_path.glob("*.safetensors"))
        if not weight_files:
            weight_files = list(model_path.glob("*.npz"))

        weights: Dict[str, mx.array] = {}
        for wf in weight_files:
            weights.update(mx.load(str(wf)))

        def should_cast(key: str, value: mx.array) -> bool:
            if key.endswith((".scales", ".biases")):
                return False
            if value.dtype == mx.uint32:  # Quantized weights
                return False
            return True

        weights = {
            k: v.astype(dtype) if should_cast(k, v) else v for k, v in weights.items()
        }

        model.load_weights(list(weights.items()))
        model.eval()

        return model

    def _setup_eos_tokens(self):
        """
        Build the set of token IDs that end generation.

        Only IDs from strings that tokenize to a single token are added —
        the speech markers (``<|startofspeech|>`` etc.) decompose to
        multi-piece BPE in the Qwen3 tokenizer and must never be treated as
        stop tokens.
        """
        if self._tokenizer is None:
            return

        eos_ids: set = set()

        if (
            getattr(self._tokenizer, "eos_token_id", None) is not None
            and self._tokenizer.eos_token_id >= 0
        ):
            eos_ids.add(self._tokenizer.eos_token_id)

        for tok in ["<|endoftext|>", "<|im_end|>", "</s>"]:
            try:
                encoded = self._tokenizer.encode(tok, add_special_tokens=False)
            except Exception:
                continue
            if len(encoded) == 1:
                eos_ids.add(encoded[0])

        self._eos_token_ids = eos_ids
