# Project Structure Reference

## 📁 Complete Structure

```
QTinker/
├── 📦 app/
│   ├── app.py              # Main entry point (Pinokio compatible)
│   ├── gradio_ui.py        # Full Gradio UI with model picker, dropdowns, run button, live logs
│   ├── distill_quant_app.py # Old tkinter version (kept for reference)
│   ├── distilled/          # Legacy output (use outputs/ instead)
│   └── quantized/          # Legacy output (use outputs/ instead)
│
├── ⚙️ configs/
│   └── torchao_configs.py  # TorchAO quantization configurations
│
├── ⚙️ settings/
│   └── app_settings.py     # Global app settings (paths, defaults, UI config)
│
├── 🧩 core/
│   └── logic.py            # Core distillation and quantization logic
│
├── 📦 outputs/              # Output directories
│   ├── distilled/          # Distilled models saved here
│   └── quantized/          # Quantized models saved here
│
├── 📜 requirements.txt      # Python dependencies
├── 📜 pyproject.toml        # Project configuration
├── 📜 README.md            # Full documentation
└── 📜 PROJECT_STRUCTURE.md # This file
```

## 🎯 Key Components

### 1. **app/gradio_ui.py** - Full Gradio UI
- Model picker (text input)
- Model type dropdown
- Quantization dropdown
- Run button
- Live log output area
- Examples section

### 2. **core/logic.py** - Core Logic
- `load_model()` - Load models from various sources
- `distill_model()` - Distillation pipeline
- `apply_quantization()` - Apply TorchAO quantization
- `run_pipeline()` - Complete pipeline orchestration

### 3. **configs/torchao_configs.py** - TorchAO Configs
- `get_quantization_config()` - Get config by type
- Supports INT4 (weight-only) and INT8 (dynamic)

### 4. **settings/app_settings.py** - Global Settings
- Output directory paths
- Default model/quantization types
- UI theme and appearance settings

### 5. **outputs/** - Output Directories
- `outputs/distilled/` - Distilled models
- `outputs/quantized/` - Quantized models

## 🚀 Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run the app
python app/app.py
```

## 📝 Copy/Paste Quick Reference

### Import Pattern
```python
from core.logic import run_pipeline
from settings.app_settings import MODEL_TYPES, QUANT_TYPES
from configs.torchao_configs import get_quantization_config
```

### Run Pipeline
```python
distilled_path, quantized_path = run_pipeline(
    model_path="microsoft/phi-2",
    model_type="HuggingFace folder",
    quant_type="INT8 (dynamic)",
    log_fn=print
)
```

### Custom Quantization Config
```python
from configs.torchao_configs import get_quantization_config
config = get_quantization_config("INT4 (weight-only)")
```
