# QTinker - Distill & Quantize with TorchAO

A modern application for distilling and quantizing language models using TorchAO.

## Features

- 🎯 **Model Loading**: Support for HuggingFace models and PyTorch checkpoints
- 🧪 **Distillation**: Model distillation pipeline (placeholder implementation)
- ⚡ **Quantization**: TorchAO-based quantization (INT4 weight-only, INT8 dynamic)
- 🎨 **Gradio UI**: Beautiful web interface with live log output
- 📦 **Modular Structure**: Clean separation of concerns
- 🖥️ **Smart GPU/CPU Management**: Automatically uses GPU when available and falls back to CPU when VRAM is limited
- 💾 **Memory Efficient**: Intelligent device switching for large models to prevent OOM errors

## Project Structure

```
QTinker/
├── app/
│   ├── app.py              # Main entry point (Pinokio compatible)
│   ├── gradio_ui.py        # Full Gradio UI
│   ├── distilled/          # Output: Distilled models
│   └── quantized/          # Output: Quantized models
├── configs/
│   └── torchao_configs.py  # TorchAO quantization configurations
├── core/
│   └── logic.py            # Core distillation and quantization logic
├── settings/
│   └── app_settings.py     # Global application settings
├── outputs/                # Output directories
│   ├── distilled/
│   └── quantized/
├── requirements.txt        # Python dependencies
├── pyproject.toml          # Project configuration
└── README.md               # This file
```

## Installation

### Using pip

```bash
pip install -r requirements.txt
```

### Using uv (recommended for Pinokio)

```bash
uv pip install -r requirements.txt
```

### CUDA Support

If you need CUDA-specific PyTorch wheels, install them manually:

```bash
# For CUDA 11.8
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# For CUDA 12.1
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

## Usage

### Running the Application

```bash
python app/app.py
```

Or directly:

```bash
cd app
python gradio_ui.py
```

The Gradio interface will be available at `http://localhost:7860`

### Using the UI

1. **Model Path**: Enter the path to your model (HuggingFace folder or PyTorch file)
2. **Model Type**: Select the type of model you're loading
3. **Quantization Type**: Choose INT4 (weight-only) or INT8 (dynamic)
4. **Run**: Click "Run Distill + Quantize" to start the pipeline
5. **Monitor**: Watch the live log output for progress

### Programmatic Usage

```python
from core.logic import run_pipeline

# Run the pipeline
distilled_path, quantized_path = run_pipeline(
    model_path="microsoft/phi-2",
    model_type="HuggingFace folder",
    quant_type="INT8 (dynamic)",
    log_fn=print
)
```

## Configuration

### TorchAO Configs

Edit `configs/torchao_configs.py` to customize quantization settings:

```python
# Example: Change group size for INT4
Int4WeightOnlyConfig(group_size=64)  # Default is 128
```

### App Settings

Edit `settings/app_settings.py` to modify:
- Output directories
- Default model/quantization types
- UI theme and appearance

## Output Directories

- **Distilled Models**: `outputs/distilled/`
- **Quantized Models**: `outputs/quantized/`

Models are saved with their tokenizers (if available) in HuggingFace format.

## Dependencies

- `torch>=2.0.0` - PyTorch
- `torchao>=0.1.0` - TorchAO quantization library
- `transformers>=4.30.0` - HuggingFace transformers
- `gradio>=4.0.0` - Web UI framework
- `accelerate>=0.20.0` - Model acceleration utilities

## GPU/CPU Management

The application automatically manages device selection:

- **GPU Detection**: Automatically detects and uses CUDA (NVIDIA) or MPS (Apple Silicon) when available
- **VRAM Monitoring**: Monitors GPU memory usage and switches to CPU when VRAM is limited
- **Automatic Fallback**: Falls back to CPU if:
  - Less than 2GB VRAM is available
  - Model size exceeds 90% of available VRAM
  - GPU runs out of memory during processing
- **Memory Efficient**: Loads models on CPU first, then moves to GPU if appropriate
- **Cache Management**: Automatically clears GPU cache between operations

### Device Settings

You can adjust device management behavior in `settings/app_settings.py`:

```python
MIN_VRAM_GB = 2.0  # Minimum VRAM required to use GPU
VRAM_THRESHOLD = 0.9  # Use CPU if model size > VRAM * threshold
AUTO_DEVICE_SWITCHING = True  # Enable automatic device switching
```

## Notes

- The distillation step is currently a placeholder. Implement your own distillation logic in `core/logic.py`
- The app automatically handles GPU/CPU switching for large models to prevent OOM errors
- Quantized models are saved in the same format as the input model
- Device information is displayed in the UI and logs show which device is being used

## License

[Add your license here]
