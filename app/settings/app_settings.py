"""
Global application settings and configuration.
"""
import os
from pathlib import Path

# Base directories
BASE_DIR = Path(__file__).parent.parent
APP_DIR = BASE_DIR / "app"
OUTPUTS_DIR = BASE_DIR / "outputs"
CONFIGS_DIR = BASE_DIR / "configs"

# Output directories
DISTILLED_DIR = OUTPUTS_DIR / "distilled"
QUANTIZED_DIR = OUTPUTS_DIR / "quantized"

# Create output directories if they don't exist
DISTILLED_DIR.mkdir(parents=True, exist_ok=True)
QUANTIZED_DIR.mkdir(parents=True, exist_ok=True)

# Model settings
DEFAULT_MODEL_TYPE = "HuggingFace folder"
DEFAULT_QUANT_TYPE = "INT8 (dynamic)"

# Supported model types
# Supported model types
MODEL_TYPES = [
    "HuggingFace folder",
    "HuggingFace Transformers (NLP/Vision/Audio)",
    "Diffusers (Image/Video/Audio Generation)",
    "Sentence-Transformers (Embeddings)",
    "Tokenizers (Rust/Python)",
    "Accelerate (Distributed)",
    "PEFT (LoRA/QLoRA)",
    "TRL (RLHF/DPO)",
    "ONNX Runtime",
    "TensorRT / TensorRT-LLM",
    "GGML / llama.cpp",
    "vLLM",
    "MLX (Apple Silicon)",
    "OpenVINO",
    "OpenCV",
    "Pillow",
    "PyTorch Vision",
    "SAM / Segment Anything",
    "ControlNet",
    "Torchaudio",
    "Whisper / WhisperX",
    "RVC / So-VITS",
    "Coqui TTS",
    "CLIP",
    "BLIP / BLIP-2",
    "LLaVA",
    "OpenCLIP",
    "SAM2",
    "HuggingFace Datasets",
    "Lightning / Fabric",
    "Weights & Biases",
    "DeepSpeed",
    "PyTorch .pt/.bin file",
]

# Supported quantization types
QUANT_TYPES = [
    "INT4 (weight-only)",
    "INT8 (dynamic)",
    "FP8",
]

# TorchAO quantization configs
TORCHAO_CONFIGS = {
    "INT4 (weight-only)": {
        "config_class": "Int4WeightOnlyConfig",
        "group_size": 128,
    },
    "INT8 (dynamic)": {
        "config_class": "Int8DynamicConfig",
    },
}

# UI settings
GRADIO_TITLE = "Distill & Quantize (TorchAO)"
GRADIO_DESCRIPTION = "Distill and quantize models using TorchAO"
GRADIO_THEME = "soft"


# Device management settings
MIN_VRAM_GB = 2.0  # Minimum VRAM required to use GPU (GB)
VRAM_THRESHOLD = 0.9  # Use CPU if model size > VRAM * threshold
AUTO_DEVICE_SWITCHING = True  # Automatically switch to CPU when VRAM is low
