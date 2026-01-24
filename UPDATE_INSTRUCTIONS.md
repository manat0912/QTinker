# Update Instructions

To get the new features (Knowledge Distillation, Local LLM, FP8), you need to install the new dependencies.

## Quick Update (Recommended)

1. In Pinokio, click the **"Install"** tab
2. This will automatically install all new dependencies from `requirements.txt`

## Manual Update (Alternative)

If you prefer to update manually, run:

```bash
cd app
uv pip install pyyaml>=6.0 requests>=2.31.0
```

Or install all requirements:

```bash
cd app
uv pip install -r requirements.txt
```

## New Dependencies Added

- `pyyaml>=6.0` - For YAML configuration files
- `requests>=2.31.0` - For local LLM API integration (LM Studio, Ollama)

## New Features Available After Installation

✅ **Knowledge Distillation**
   - Teacher-student training mode
   - Configurable hyperparameters (epochs, batch size, learning rate, temperature, alpha)

✅ **Local LLM Integration**
   - Auto-detect LM Studio (port 1234)
   - Auto-detect Ollama (port 11434)
   - Support for custom API endpoints
   - Model listing and text generation

✅ **FP8 Quantization**
   - Added as quantization option in UI
   - Falls back gracefully if not supported by TorchAO

✅ **YAML Configuration**
   - All settings stored in `app/config/settings.yaml`
   - Persistent configuration across sessions

## Verification

After installation, you can verify by:
1. Starting the app (click "Start" in Pinokio)
2. Check the UI - you should see:
   - Distillation Mode selector
   - Teacher Model inputs
   - Local LLM section
   - FP8 in quantization dropdown
