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
    # --- 8-bit Formats ---
    "INT8 (dynamic)",
    "INT8 (static)",
    "INT8 (weight-only)",
    "UINT8",
    "FP8 (E4M3)",
    "FP8 (E5M2)",
    "NF8 (Normal-Float-8)",
    "MXFP8",
    "LLM.int8()",
    
    # --- 6-bit Formats ---
    "INT6",
    "UINT6",
    "GPTQ-INT6",
    
    # --- 5-bit Formats ---
    "INT5",
    "UINT5",
    "LLM.int5()",
    
    # --- 4-bit Formats ---
    "INT4 (weight-only)",
    "UINT4",
    "NF4 (NormalFloat-4)",
    "FP4",
    "FP4-E2M1",
    "FP4-E3M0",
    "QLoRA NF4",
    "QLoRA FP4",
    "GPTQ (4-bit)",
    "AWQ (4-bit)",
    "ZeroQuant (INT4)",
    "SmoothQuant (INT4)",
    "NormalFloat-4 (NF4)",
    "NormalFloat-8 (NF8)",
    "GPTQ (3-bit)",
    "INT3",
    "UINT3",
    "INT2",
    "UINT2",
    "Binary (-1, +1)",
    "Ternary (-1, 0, +1)",
    "FP16",
    "BF16",
    "FP32",
    "ONNX (INT8)",
    "ONNX (FP16)",
    "GGUF (Q4_K_M)",
    "GGUF (Q5_K_M)",
    "GGUF (Q8_0)",
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
    "SmoothQuant (INT4)": {
        "config_class": "SmoothQuantConfigWrapper",
        "alpha": 0.5,
    },
    "NormalFloat-4 (NF4)": {
        "config_class": "NF4ConfigWrapper",
    },
    "GPTQ (4-bit)": {
        "config_class": "GPTQConfig",
        "bits": 4,
    },
}

# UI settings
GRADIO_TITLE = "Distill & Quantize (TorchAO)"
GRADIO_DESCRIPTION = "Distill and quantize models using TorchAO"



# Device management settings
MIN_VRAM_GB = 2.0  # Minimum VRAM required to use GPU (GB)
VRAM_THRESHOLD = 0.9  # Use CPU if model size > VRAM * threshold
AUTO_DEVICE_SWITCHING = True  # Automatically switch to CPU when VRAM is low
