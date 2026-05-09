# Copyright © 2025 FunASR (original model implementation)
# Copyright © Anthony DePasquale (MLX port)
# Ported to MLX from https://github.com/modelscope/FunASR
# License: licenses/funasr.txt

"""
Weight conversion script for Fun-ASR-Nano.

Converts PyTorch weights from the original Fun-ASR model to MLX format.
Supports optional quantization for LLM and adaptor layers.
"""

import argparse
import json
import shutil
from pathlib import Path
from typing import Any, Dict, Set

import mlx.core as mx
import mlx.nn as nn
import numpy as np
from mlx.utils import tree_flatten, tree_unflatten

try:
    import torch
except ImportError:
    torch = None

try:
    from safetensors import safe_open
    from safetensors.numpy import save_file as safetensors_save
except ImportError:
    safe_open = None
    safetensors_save = None


def load_pytorch_weights(checkpoint_path: Path) -> Dict[str, np.ndarray]:
    """
    Load PyTorch weights from checkpoint.

    Parameters
    ----------
    checkpoint_path : Path
        Path to PyTorch checkpoint (.pt, .pth, or .bin file)

    Returns
    -------
    Dict[str, np.ndarray]
        Dictionary of weight arrays
    """
    if torch is None:
        raise ImportError(
            "PyTorch is required for weight conversion. Install with: pip install torch"
        )

    state_dict = torch.load(checkpoint_path, map_location="cpu")

    # Handle different checkpoint formats
    if "state_dict" in state_dict:
        state_dict = state_dict["state_dict"]
    elif "model" in state_dict:
        state_dict = state_dict["model"]

    # Convert to numpy
    weights = {}
    for key, value in state_dict.items():
        if isinstance(value, torch.Tensor):
            # Convert bfloat16 to float32 first (numpy doesn't support bfloat16)
            if value.dtype == torch.bfloat16:
                value = value.float()
            weights[key] = value.detach().cpu().numpy()
        else:
            weights[key] = value

    return weights


def load_safetensors_weights(weights_path: Path) -> Dict[str, np.ndarray]:
    """
    Load weights from safetensors format.

    Parameters
    ----------
    weights_path : Path
        Path to safetensors file

    Returns
    -------
    Dict[str, np.ndarray]
        Dictionary of weight arrays
    """
    if safe_open is None:
        raise ImportError(
            "safetensors is required. Install with: pip install safetensors"
        )

    weights = {}
    with safe_open(weights_path, framework="numpy") as f:
        for key in f.keys():
            weights[key] = f.get_tensor(key)

    return weights


def remap_weight_keys(weights: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    """
    Remap weight keys from PyTorch to MLX naming convention.

    Parameters
    ----------
    weights : Dict[str, np.ndarray]
        Original weights dictionary

    Returns
    -------
    Dict[str, np.ndarray]
        Remapped weights dictionary
    """
    remapped = {}

    for key, value in weights.items():
        new_key = key

        # Remove common prefixes
        if new_key.startswith("model."):
            new_key = new_key[6:]

        # Remap encoder keys
        # audio_encoder.encoders0.0.* -> audio_encoder.encoders0.0.*
        # audio_encoder.encoders.0.* -> audio_encoder.encoders.0.*
        # audio_encoder.tp_encoders.0.* -> audio_encoder.tp_encoders.0.*

        # Remap SANM attention keys
        # self_attn.linear_q_k_v -> self_attn.linear_q_k_v
        # self_attn.fsmn_block -> self_attn.fsmn_block
        # self_attn.linear_out -> self_attn.linear_out

        # Remap adaptor keys
        # audio_adaptor.linear1 -> audio_adaptor.linear1
        # audio_adaptor.linear2 -> audio_adaptor.linear2
        # audio_adaptor.blocks.0.* -> audio_adaptor.blocks.0.*

        # Remap LLM keys (Qwen3)
        # llm.model.embed_tokens -> llm.model.embed_tokens
        # llm.model.layers.0.* -> llm.model.layers.0.*
        # llm.model.norm -> llm.model.norm
        # llm.lm_head -> llm.lm_head (if not tied)

        remapped[new_key] = value

    return remapped


def transform_weights(weights: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    """
    Transform weights to MLX format.

    Handles Conv1d weight transposition and other format differences.

    Parameters
    ----------
    weights : Dict[str, np.ndarray]
        Input weights dictionary

    Returns
    -------
    Dict[str, np.ndarray]
        Transformed weights dictionary
    """
    transformed = {}

    for key, value in weights.items():
        # Handle FSMN conv weights
        # PyTorch: (out_channels, 1, kernel_size) for depthwise
        # MLX: (out_channels, kernel_size, 1) for depthwise
        if "fsmn_block" in key and "weight" in key:
            if value.ndim == 3 and value.shape[1] == 1:
                # (out, 1, kernel) -> (out, kernel, 1)
                value = np.transpose(value, (0, 2, 1))

        # Handle other Conv1d weights
        elif "conv" in key.lower() and "weight" in key:
            if value.ndim == 3:
                # PyTorch Conv1d: (out, in, kernel)
                # MLX Conv1d: (out, kernel, in)
                if value.shape[2] < value.shape[1]:
                    value = np.swapaxes(value, 1, 2)

        transformed[key] = value

    return transformed


def create_config(weights: Dict[str, np.ndarray]) -> Dict[str, Any]:
    """
    Infer model configuration from weights.

    Parameters
    ----------
    weights : Dict[str, np.ndarray]
        Model weights

    Returns
    -------
    Dict[str, Any]
        Model configuration dictionary matching FunASRConfig.from_dict() format
    """
    config = {
        "model_type": "funasr",
        "sample_rate": 16000,
        "n_mels": 80,
        "lfr_m": 7,
        "lfr_n": 6,
    }

    # Infer encoder config (matches SenseVoiceEncoderConfig)
    encoder_config = {
        "input_dim": 560,  # 80 * 7
        "encoder_dim": 512,
        "num_heads": 4,
        "ffn_dim": 2048,
        "kernel_size": 11,
        "num_encoders0": 1,
        "num_encoders": 49,
        "num_tp_encoders": 20,
        "dropout": 0.0,
    }

    # Try to infer from weights
    for key, value in weights.items():
        if "audio_encoder.encoders0.0.self_attn.linear_q_k_v.weight" in key:
            encoder_config["input_dim"] = value.shape[1]
        elif "audio_encoder.encoders.0.self_attn.linear_q_k_v.weight" in key:
            encoder_config["encoder_dim"] = value.shape[1]
        elif "audio_encoder.encoders.0.feed_forward.w_1.weight" in key:
            encoder_config["ffn_dim"] = value.shape[0]
        elif "audio_encoder.encoders.0.self_attn.fsmn_block" in key and "weight" in key:
            if value.ndim == 3:
                encoder_config["kernel_size"] = max(value.shape[1], value.shape[2])

    # Count encoder layers
    encoders0_indices = set()
    encoders_indices = set()
    tp_encoders_indices = set()
    for key in weights.keys():
        if "audio_encoder.encoders0." in key:
            parts = key.split(".")
            idx = parts.index("encoders0") + 1
            if idx < len(parts) and parts[idx].isdigit():
                encoders0_indices.add(int(parts[idx]))
        elif "audio_encoder.tp_encoders." in key:
            parts = key.split(".")
            idx = parts.index("tp_encoders") + 1
            if idx < len(parts) and parts[idx].isdigit():
                tp_encoders_indices.add(int(parts[idx]))
        elif "audio_encoder.encoders." in key:
            parts = key.split(".")
            idx = parts.index("encoders") + 1
            if idx < len(parts) and parts[idx].isdigit():
                encoders_indices.add(int(parts[idx]))

    if encoders0_indices:
        encoder_config["num_encoders0"] = max(encoders0_indices) + 1
    if encoders_indices:
        encoder_config["num_encoders"] = max(encoders_indices) + 1
    if tp_encoders_indices:
        encoder_config["num_tp_encoders"] = max(tp_encoders_indices) + 1

    config["encoder"] = encoder_config

    # Infer adaptor config (matches AudioAdaptorConfig)
    adaptor_config = {
        "downsample_rate": 1,
        "encoder_dim": 512,
        "llm_dim": 1024,
        "ffn_dim": 2048,
        "n_layer": 2,
        "attention_heads": 8,
        "dropout": 0.0,
        # Architecture flag — not deducible from weights, hard-coded per the
        # published Fun-ASR-Nano-2512 / MLT-Nano-2512 config.yaml.
        "use_low_frame_rate": True,
    }

    for key, value in weights.items():
        if "audio_adaptor.linear1.weight" in key:
            # linear1 input is encoder_dim * downsample_rate
            adaptor_config["ffn_dim"] = value.shape[0]
            input_dim = value.shape[1]
            adaptor_config["encoder_dim"] = encoder_config["encoder_dim"]
            adaptor_config["downsample_rate"] = (
                input_dim // encoder_config["encoder_dim"]
            )
        elif "audio_adaptor.linear2.weight" in key:
            adaptor_config["llm_dim"] = value.shape[0]

    # Count adaptor blocks
    block_indices = set()
    for key in weights.keys():
        if "audio_adaptor.blocks." in key:
            parts = key.split(".")
            idx = parts.index("blocks") + 1
            if idx < len(parts) and parts[idx].isdigit():
                block_indices.add(int(parts[idx]))
    if block_indices:
        adaptor_config["n_layer"] = max(block_indices) + 1

    config["adaptor"] = adaptor_config

    # Infer LLM config (matches Qwen3Config)
    llm_config = {
        "vocab_size": 151936,
        "hidden_size": 1024,
        "num_hidden_layers": 28,
        "num_attention_heads": 16,
        "num_key_value_heads": 8,
        "intermediate_size": 3072,
        "max_position_embeddings": 40960,
        "rope_theta": 1000000.0,
        "rms_norm_eps": 1e-6,
        "tie_word_embeddings": True,
        "head_dim": 64,
    }

    # First pass: get head_dim from q_norm if available (most reliable)
    for key, value in weights.items():
        if "llm.model.layers.0.self_attn.q_norm.weight" in key:
            llm_config["head_dim"] = value.shape[0]
            break

    # Second pass: infer other config values
    for key, value in weights.items():
        if "llm.model.embed_tokens.weight" in key:
            llm_config["vocab_size"] = value.shape[0]
            llm_config["hidden_size"] = value.shape[1]
        elif "llm.model.layers.0.mlp.gate_proj.weight" in key:
            llm_config["intermediate_size"] = value.shape[0]
        elif "llm.model.layers.0.self_attn.q_proj.weight" in key:
            q_dim = value.shape[0]
            head_dim = llm_config.get("head_dim", 128)
            llm_config["num_attention_heads"] = q_dim // head_dim
        elif "llm.model.layers.0.self_attn.k_proj.weight" in key:
            k_dim = value.shape[0]
            head_dim = llm_config.get("head_dim", 128)
            llm_config["num_key_value_heads"] = k_dim // head_dim

    # Count LLM layers
    layer_indices = set()
    for key in weights.keys():
        if "llm.model.layers." in key:
            parts = key.split(".")
            idx = parts.index("layers") + 1
            if idx < len(parts) and parts[idx].isdigit():
                layer_indices.add(int(parts[idx]))
    if layer_indices:
        llm_config["num_hidden_layers"] = max(layer_indices) + 1

    # Check for tied embeddings
    llm_config["tie_word_embeddings"] = "llm.lm_head.weight" not in weights

    config["llm"] = llm_config

    return config


def get_quantizable_layers() -> Set[str]:
    """
    Return patterns for layers that should be quantized.

    Quantizable layers (Linear layers with significant parameter count):
    - LLM attention projections: q_proj, k_proj, v_proj, o_proj
    - LLM MLP layers: gate_proj, up_proj, down_proj
    - Audio adaptor linear layers: linear1, linear2
    - Audio adaptor transformer attention projections

    Returns
    -------
    Set[str]
        Set of key patterns that should be quantized
    """
    return {
        # LLM layer patterns
        "llm.model.layers",
        # Audio adaptor patterns
        "audio_adaptor.linear1",
        "audio_adaptor.linear2",
        "audio_adaptor.blocks",
    }


def get_non_quantizable_patterns() -> Set[str]:
    """
    Return patterns for layers that should NOT be quantized.

    Non-quantizable layers:
    - Normalization layers (RMSNorm, LayerNorm) - precision-critical
    - Embeddings - lookup tables, less benefit from quantization
    - FSMN convolutions - audio-specific, small kernels
    - Bias terms - already small
    - Audio encoder - preserves audio feature quality

    Returns
    -------
    Set[str]
        Set of key patterns that should not be quantized
    """
    return {
        # Normalization layers
        "norm",
        "ln_",
        "layer_norm",
        # Embeddings
        "embed_tokens",
        "embedding",
        # FSMN convolutions
        "fsmn_block",
        # Audio encoder (keep full precision for audio features)
        "audio_encoder",
        # Bias terms
        ".bias",
    }


def should_quantize_weight(key: str) -> bool:
    """
    Determine if a weight should be quantized based on its key.

    Parameters
    ----------
    key : str
        Weight key name

    Returns
    -------
    bool
        True if the weight should be quantized
    """
    quantizable = get_quantizable_layers()
    non_quantizable = get_non_quantizable_patterns()

    # Check if in non-quantizable patterns first
    for pattern in non_quantizable:
        if pattern in key.lower():
            return False

    # Check if in quantizable patterns
    for pattern in quantizable:
        if pattern in key:
            # Only quantize weight matrices, not biases
            if "weight" in key and ".bias" not in key:
                return True

    return False


def quantize_weights(
    weights: Dict[str, np.ndarray],
    config: Dict[str, Any],
    q_bits: int = 4,
    q_group_size: int = 64,
) -> tuple:
    """
    Quantize model weights using MLX quantization.

    Only quantizes Linear layers in LLM and adaptor components,
    keeping embeddings, norms, and audio encoder at full precision.

    Parameters
    ----------
    weights : Dict[str, np.ndarray]
        Model weights as numpy arrays
    config : Dict[str, Any]
        Model configuration dictionary
    q_bits : int
        Quantization bits (4 or 8)
    q_group_size : int
        Quantization group size (default: 64)

    Returns
    -------
    tuple
        (quantized_weights, updated_config)
    """
    from .funasr import FunASRConfig, Model

    print(f"Quantizing to {q_bits}-bit (group_size={q_group_size})...")

    # Create model from config
    model_config = FunASRConfig.from_dict(config)
    model = Model(model_config)

    # Load weights into model
    mlx_weights = {k: mx.array(v) for k, v in weights.items()}
    model.load_weights(list(mlx_weights.items()))
    mx.eval(model.parameters())

    # Calculate original size
    orig_weights = dict(tree_flatten(model.parameters()))
    orig_size = sum(v.nbytes for v in orig_weights.values())
    print(f"Original size: {orig_size / 1e9:.2f} GB")

    # Define quantization predicate
    quantized_count = [0]

    def class_predicate(path: str, module) -> bool:
        """Only quantize Linear layers in LLM and adaptor."""
        if isinstance(module, nn.Linear):
            if should_quantize_weight(path + ".weight"):
                quantized_count[0] += 1
                return True
        return False

    # Apply quantization
    nn.quantize(
        model,
        bits=q_bits,
        group_size=q_group_size,
        class_predicate=class_predicate,
    )
    mx.eval(model.parameters())
    print(f"Quantized {quantized_count[0]} Linear layers")

    # Get quantized weights
    quantized_weights = dict(tree_flatten(model.parameters()))
    new_size = sum(v.nbytes for v in quantized_weights.values())
    print(f"Quantized size: {new_size / 1e9:.2f} GB")
    print(f"Reduction: {(1 - new_size / orig_size) * 100:.1f}%")

    # Update config with quantization info
    config["quantization"] = {
        "bits": q_bits,
        "group_size": q_group_size,
        "quantized_components": ["llm.model.layers", "audio_adaptor"],
    }

    return quantized_weights, config


def convert_funasr_weights(
    input_path: str,
    output_path: str,
    dtype: str = "bfloat16",
    quantize: bool = False,
    q_bits: int = 4,
    q_group_size: int = 64,
) -> None:
    """
    Convert Fun-ASR weights to MLX format.

    Parameters
    ----------
    input_path : str
        Path to input weights (PyTorch checkpoint or safetensors)
    output_path : str
        Path for output MLX model directory
    dtype : str
        Output dtype (float16, bfloat16, or float32)
    quantize : bool
        Whether to quantize weights (default: False)
    q_bits : int
        Quantization bits, 4 or 8 (default: 4)
    q_group_size : int
        Quantization group size (default: 64)
    """
    input_path = Path(input_path)
    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)

    # Load weights
    print(f"Loading weights from {input_path}...")
    if input_path.suffix == ".safetensors":
        weights = load_safetensors_weights(input_path)
    elif input_path.is_dir():
        # Look for safetensors files
        safetensors_files = list(input_path.glob("*.safetensors"))
        if safetensors_files:
            weights = {}
            for sf in safetensors_files:
                weights.update(load_safetensors_weights(sf))
        else:
            # Look for PyTorch files
            pt_files = (
                list(input_path.glob("*.pt"))
                + list(input_path.glob("*.pth"))
                + list(input_path.glob("*.bin"))
            )
            if pt_files:
                weights = load_pytorch_weights(pt_files[0])
            else:
                raise ValueError(f"No weight files found in {input_path}")
    else:
        weights = load_pytorch_weights(input_path)

    print(f"Loaded {len(weights)} weight tensors")

    # Remap keys
    print("Remapping weight keys...")
    weights = remap_weight_keys(weights)

    # Transform weights
    print("Transforming weights...")
    weights = transform_weights(weights)

    # Convert dtype
    dtype_map = {
        "float16": np.float16,
        "bfloat16": np.float16,  # numpy doesn't support bfloat16, use float16
        "float32": np.float32,
    }
    np_dtype = dtype_map.get(dtype, np.float16)

    print(f"Converting to {dtype}...")
    for key in weights:
        if weights[key].dtype in [np.float32, np.float64]:
            weights[key] = weights[key].astype(np_dtype)

    # Create config
    print("Creating config...")
    config = create_config(weights)

    # Quantize if requested
    if quantize:
        print("Quantizing weights...")
        weights, config = quantize_weights(weights, config, q_bits, q_group_size)
        is_quantized = True
    else:
        is_quantized = False

    # Save config
    config_path = output_path / "config.json"
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)
    print(f"Saved config to {config_path}")

    # Save weights
    weights_path = output_path / "model.safetensors"
    if is_quantized:
        # Quantized weights are MLX arrays, use mx.save_safetensors
        mx.save_safetensors(str(weights_path), weights, metadata={"format": "mlx"})
        print(f"Saved quantized weights to {weights_path}")
    elif safetensors_save is not None:
        safetensors_save(weights, str(weights_path))
        print(f"Saved weights to {weights_path}")
    else:
        # Fall back to npz
        weights_path = output_path / "model.npz"
        np.savez(str(weights_path), **weights)
        print(f"Saved weights to {weights_path}")

    # Copy tokenizer files if present
    if input_path.is_dir():
        tokenizer_files = [
            "tokenizer.json",
            "tokenizer_config.json",
            "special_tokens_map.json",
            "vocab.json",
            "merges.txt",
        ]
        # Search in root and subdirectories (e.g., Qwen3-0.6B/)
        search_dirs = [input_path] + list(input_path.iterdir())
        for tf in tokenizer_files:
            for search_dir in search_dirs:
                if not search_dir.is_dir():
                    continue
                src = search_dir / tf
                if src.exists():
                    shutil.copy(src, output_path / tf)
                    print(f"Copied {tf}")
                    break

    print(f"\nConversion complete! Model saved to {output_path}")


def generate_readme(output_dir: Path, model_id: str, upload_repo: str) -> None:
    """Generate README.md model card for Hugging Face."""
    from mlx_audio.version import __version__

    # Detect if this is an MLT (Multi-Language Transcription) model
    is_mlt = "MLT" in model_id or "MLT" in upload_repo

    if is_mlt:
        # MLT model: 31 languages with code-switching
        language_tags = """- multilingual"""
        features_table = """| Feature | Description |
|---------|-------------|
| **Multilingual** | Supports 31 languages with focus on East and Southeast Asian languages |
| **Chinese dialects** | Supports 7 major Chinese dialects |
| **Code-switching** | Handles mixed-language speech within sentences |
| **Translation** | Translate speech directly to English text |
| **Custom prompting** | Guide recognition with domain-specific context |
| **Streaming** | Real-time token-by-token output |"""
        language_table = f"""See [original model](https://huggingface.co/{model_id}) for the full list of supported languages."""
    else:
        # Standard model: 13 languages
        language_tags = """- multilingual"""
        features_table = """| Feature | Description |
|---------|-------------|
| **Multilingual** | Supports 13+ languages |
| **Translation** | Translate speech directly to English text |
| **Custom prompting** | Guide recognition with domain-specific context |
| **Streaming** | Real-time token-by-token output |"""
        language_table = f"""See [original model](https://huggingface.co/{model_id}) for the full list of supported languages."""

    card_text = f"""---
library_name: mlx-audio-plus
base_model:
- {model_id}
tags:
- mlx
- funasr
- speech-recognition
- speech-to-text
- stt
pipeline_tag: automatic-speech-recognition
language:
{language_tags}
---

# {upload_repo}

This model was converted to MLX format from [{model_id}](https://huggingface.co/{model_id}) using [mlx-audio-plus](https://github.com/DePasqualeOrg/mlx-audio-plus) version **{__version__}**.

## Features

{features_table}

## Installation

```bash
pip install -U mlx-audio-plus
```

## Usage

### Basic Transcription

```python
from mlx_audio.stt.models.funasr import Model

# Load the model
model = Model.from_pretrained("{upload_repo}")

# Transcribe audio
result = model.generate("audio.wav")
print(result.text)
# Output: "The quick brown fox jumps over the lazy dog."

print(f"Duration: {{result.duration:.2f}}s")
print(f"Language: {{result.language}}")
```

### Translation (Speech to English Text)

```python
# Translate Chinese/Japanese/etc. audio to English
result = model.generate(
    "chinese_speech.wav",
    task="translate",
    target_language="en"
)
print(result.text)  # English translation
```

### Custom Prompting

Provide context to improve recognition accuracy for specialized domains:

```python
# Medical transcription
result = model.generate(
    "doctor_notes.wav",
    initial_prompt="Medical consultation discussing cardiac symptoms and treatment options."
)

# Technical content
result = model.generate(
    "tech_podcast.wav",
    initial_prompt="Discussion about machine learning, APIs, and software development."
)
```

### Streaming Output

Get real-time output as the model generates:

```python
# Print tokens as they're generated
result = model.generate("audio.wav", verbose=True)
# Tokens stream to stdout in real-time

# Or use the streaming generator
for chunk in model.generate("audio.wav", stream=True):
    print(chunk, end="", flush=True)
```

## Supported Languages

{language_table}
"""
    card_path = output_dir / "README.md"
    with open(card_path, "w") as f:
        f.write(card_text)
    print(f"Created {card_path}")


def upload_to_hub(output_dir: Path, upload_repo: str) -> None:
    """Upload converted model to Hugging Face Hub."""
    from huggingface_hub import HfApi

    print(f"\nUploading to {upload_repo}...")
    api = HfApi()
    api.create_repo(repo_id=upload_repo, exist_ok=True)
    api.upload_folder(
        folder_path=str(output_dir),
        repo_id=upload_repo,
        repo_type="model",
    )
    print(f"Upload successful! Visit https://huggingface.co/{upload_repo}")


def convert_from_source(
    model_id: str,
    output_dir: Path = None,
    quantize: bool = False,
    q_bits: int = 4,
    q_group_size: int = 64,
    dtype: str = "float16",
    upload_repo: str = None,
    dry_run: bool = False,
) -> None:
    """
    Convert Fun-ASR PyTorch weights to MLX format.

    This function is called by the central conversion utility when it detects
    a Fun-ASR model, or can be called directly.

    Args:
        model_id: Hugging Face model ID or local path to weights
        output_dir: Output directory for MLX weights
        quantize: Whether to quantize weights
        q_bits: Quantization bits (default: 4)
        q_group_size: Quantization group size (default: 64)
        dtype: Data type for weights (float16, bfloat16, float32)
        upload_repo: Hugging Face repo to upload to
        dry_run: Generate files but skip upload
    """
    from huggingface_hub import snapshot_download

    # Determine output directory
    if output_dir is None:
        model_name = model_id.split("/")[-1] if "/" in model_id else "FunASR"
        suffix = f"{q_bits}bit" if quantize else "fp16"
        output_dir = Path(f"./{model_name}-{suffix}")

    output_dir = Path(output_dir)

    # Download model if it's a HuggingFace repo
    input_path = Path(model_id)
    if not input_path.exists():
        print(f"Downloading {model_id} from Hugging Face...")
        input_path = Path(
            snapshot_download(
                model_id,
                allow_patterns=[
                    "*.safetensors",
                    "*.pt",
                    "*.pth",
                    "*.bin",
                    "*.json",
                    "*.txt",
                ],
            )
        )
        print(f"Downloaded to: {input_path}")

    # Convert weights
    convert_funasr_weights(
        input_path=str(input_path),
        output_path=str(output_dir),
        dtype=dtype,
        quantize=quantize,
        q_bits=q_bits,
        q_group_size=q_group_size,
    )

    # Generate README (always, using derived repo name if not specified)
    readme_repo = upload_repo
    if readme_repo is None:
        # Derive repo name from output directory (e.g., mlx-community/FunASR-Nano-fp16)
        readme_repo = f"mlx-community/{output_dir.name}"
    print("\nGenerating README.md...")
    generate_readme(output_dir, model_id, readme_repo)

    # Print summary
    print(f"\n{'=' * 60}")
    print("Conversion complete!")
    print(f"{'=' * 60}")
    print(f"\nOutput directory: {output_dir}")
    print("\nFiles created:")
    for f in sorted(output_dir.iterdir()):
        if f.is_file():
            size_mb = f.stat().st_size / (1024 * 1024)
            print(f"  {f.name}: {size_mb:.1f} MB")

    # Upload to Hugging Face if requested (and not dry run)
    if upload_repo and not dry_run:
        upload_to_hub(output_dir, upload_repo)
    elif upload_repo:
        print(f"\nDry run - to upload to {upload_repo}, run without --dry-run")


def main():
    parser = argparse.ArgumentParser(
        description="Convert Fun-ASR weights to MLX format"
    )
    parser.add_argument(
        "--model-id",
        type=str,
        required=True,
        help="Hugging Face model ID or local path to weights",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory for MLX weights (default: ./{model-name}-{fp16|Nbit})",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="float16",
        choices=["float16", "bfloat16", "float32"],
        help="Output dtype (default: float16)",
    )
    parser.add_argument(
        "-q",
        "--quantize",
        action="store_true",
        help="Quantize LLM and adaptor weights (reduces model size)",
    )
    parser.add_argument(
        "--q-bits",
        type=int,
        default=4,
        choices=[4, 8],
        help="Quantization bits (default: 4)",
    )
    parser.add_argument(
        "--q-group-size",
        type=int,
        default=64,
        help="Quantization group size (default: 64)",
    )
    parser.add_argument(
        "--upload-repo",
        type=str,
        default=None,
        help="Hugging Face repo to upload to (e.g., mlx-community/FunASR-Nano)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Generate all files including README but skip upload",
    )

    args = parser.parse_args()
    convert_from_source(
        model_id=args.model_id,
        output_dir=args.output_dir,
        quantize=args.quantize,
        q_bits=args.q_bits,
        q_group_size=args.q_group_size,
        dtype=args.dtype,
        upload_repo=args.upload_repo,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()
