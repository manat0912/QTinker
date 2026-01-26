# Stable Diffusion & Diffusers Models Guide

This guide explains how to use QTinker with Stable Diffusion models and other Diffusers-based models.

## Quick Start

### Option 1: Full Pipeline (Easiest)
If you have a complete Stable Diffusion model folder with `model_index.json`:

1. Open QTinker in Pinokio or run the web UI
2. In the model path field, enter the full path to the model folder
3. The app will **automatically detect** it's a Stable Diffusion model
4. Click "Run Distill + Quantize"

Example: `C:/path/to/stable-diffusion-v1-5` or `/home/user/models/SDXL`

### Option 2: Component Models
If you only have individual components (UNet, VAE, or Text Encoder):

1. Point to the folder containing the component subfolder
2. For example: `C:/path/to/model/unet` or `C:/path/to/model/vae`
3. The app will detect the component and load it appropriately

### Option 3: Raw Model Files
If you have a `.bin` or `.pt` file (raw state_dict):

1. Provide the full path to the file
2. The app will **analyze the weights** to determine the component type
3. It will **automatically wrap the state_dict** in the appropriate model class
4. Loading will proceed seamlessly

Example: `C:/models/unet_weights.bin` or `/home/user/models/pytorch_model.bin`

## Supported Stable Diffusion Versions

### Stable Diffusion 1.5
All community-maintained checkpoints and fine-tuned versions:
- RunwayML v1.5
- OpenAI v1.5
- Community checkpoints (Animix, SweetArt, etc.)
- VAE-swapped versions

**Expected folder structure:**
```
model-folder/
├── model_index.json
├── unet/
├── vae/
├── text_encoder/
├── tokenizer/
├── feature_extractor/
└── safety_checker/
```

### Stable Diffusion 2.x
SD 2.0 and 2.1 variants:
- Full pipeline (model_index.json)
- 768-pixel variants (768 unfrozen)
- Inpainting variants

### Stable Diffusion XL (SDXL)
SDXL 1.0 and community checkpoints:
- Base model + refiner (can load separately)
- VAE improvements
- Enhanced text encoding with multiple text encoders

**Note**: SDXL models are larger. Recommended VRAM:
- Full pipeline: 12GB+ VRAM
- With quantization: 8GB+ VRAM

### Other Diffusers Models
- ControlNet (for control-guided generation)
- Consistency Models
- Latent Diffusion variants
- Custom pipeline models

## Model Detection

QTinker automatically detects the model type by examining:

1. **`model_index.json`**: Indicates a full diffusers pipeline
2. **Folder structure**: Presence of `unet/`, `vae/`, `text_encoder/` subfolders
3. **Config files**: `config.json` files indicate component type
4. **State dict keys**: For raw `.bin` files, analyzes weight keys to determine component:
   - `"up_blocks"`, `"down_blocks"` → UNet
   - `"encoder"`, `"decoder"` → VAE
   - Others → Text Encoder or custom model

## Common Model Sources

### HuggingFace Hub
```
# Use any model from HuggingFace
# These work with direct HF IDs:
runwayml/stable-diffusion-v1-5
stabilityai/stable-diffusion-2-1
stabilityai/stable-diffusion-xl-base-1.0
```

### Local Downloads
Models you've already downloaded and stored locally:
```
C:/models/sd-v1-5-local/
C:/models/SDXL/
/home/user/diffusion-models/custom-checkpoint/
```

### Hugging Face Diffusers Format
Any model in diffusers format (not safetensors-only):
```
model-name/
├── model_index.json
├── unet/diffusion_pytorch_model.bin
├── vae/diffusion_pytorch_model.bin
└── ...
```

## Distillation with Stable Diffusion

### Teacher-Student Setup
For knowledge distillation with Stable Diffusion:

1. **Teacher**: Full SD pipeline or larger UNet variant
2. **Student**: Smaller/pruned UNet variant
3. The app will:
   - Load both models with auto-detection
   - Adapt the distillation strategy to work with diffusion components
   - Apply quantization afterward

### Example
```
Teacher Model: stabilityai/stable-diffusion-xl-base-1.0
Student Model: /local/path/to/pruned-unet/
Quantization: INT8 (dynamic)
```

## Quantization with Stable Diffusion

Quantization applies to any component:
- **UNet**: Typically the largest component (~2GB for SDXL)
- **VAE**: Smaller component (~0.5GB)
- **Text Encoder**: Smallest (~1GB for CLIP)
- **Full pipeline**: All components quantized

### Recommended Settings
```
Model Type: Stable Diffusion
Quantization: INT8 (dynamic)  ← Better for vision tasks
# or
Quantization: INT4 (weight-only)  ← Maximum compression
```

### Expected Compression
- INT4: 75% compression (~2GB → ~0.5GB for UNet)
- INT8: 50% compression (~2GB → ~1GB for UNet)

## Troubleshooting

### Error: "Could not detect model architecture"
**Solution**: 
- Ensure the folder structure is intact
- For component models, check that subfolders exist (unet/, vae/, etc.)
- For raw files, ensure it's a valid PyTorch checkpoint

### Error: "pytorch_model.bin not found"
**Solution**:
- The file might be named differently (model.bin, weights.bin, etc.)
- Or it might be a safetensors file instead (.safetensors)
- Use the full file path instead of folder path

### CUDA Out of Memory
**Solution**:
- SDXL requires significant VRAM. The app will auto-switch to CPU if needed
- Use INT4 quantization for aggressive compression
- Load only the UNet component instead of full pipeline
- Use a smaller base model (e.g., SD 1.5 instead of SDXL)

### Model loads slowly
**Solution**:
- First load is slower (compiling/tracing)
- Check that GPU is detected (see logs at startup)
- For HTTP models, ensure good internet connection
- Smaller models load faster

## Advanced: Loading Individual Components

### UNet Only
```
Path: /path/to/model/unet
or
Path: /path/to/model/unet/diffusion_pytorch_model.bin
```

### VAE Only
```
Path: /path/to/model/vae
or
Path: /path/to/model/vae/diffusion_pytorch_model.bin
```

### Text Encoder Only
```
Path: /path/to/model/text_encoder
or
Path: /path/to/model/text_encoder/pytorch_model.bin
```

## Memory Requirements

### Minimum VRAM by Model
- **SD 1.5**: 6GB GPU / 4GB + CPU swap
- **SD 2.1**: 8GB GPU / 6GB + CPU swap  
- **SDXL Base**: 12GB GPU / 8GB + CPU swap
- **SDXL with Refiner**: 16GB+ GPU

### With Quantization
- **SD 1.5 (INT8)**: 3GB GPU
- **SDXL (INT4)**: 6GB GPU

The app will automatically use CPU offloading if needed.

## Example Workflows

### Quick Test with Small Model
```
Model: stabilityai/stable-diffusion-2-base
Quantization: INT4
Expected time: ~5-10 minutes
Output size: ~500MB
```

### Production Setup
```
Teacher: stabilityai/stable-diffusion-xl-base-1.0
Student: /custom/pruned-unet-sdxl/
Quantization: INT8 (dynamic)
Expected time: ~30 minutes
Output: Distilled + Quantized models
```

### Component Quantization
```
Model: /path/to/sdxl-unet/
(Single component - faster)
Quantization: INT4 (weight-only)
Expected time: ~3-5 minutes
```

## See Also

- [Main README](README.md) - Full documentation
- [Troubleshooting Guide](README.md#troubleshooting) - Common issues
- [Model Types](README.md#supported-model-types) - All supported model types
