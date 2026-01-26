# QTinker Quick Reference: Model Loading

## TL;DR - Model Paths

Just provide the path. The app **auto-detects everything**.

### Stable Diffusion Models
```
C:\models\stable-diffusion-v1-5
/home/user/models/SDXL
C:\models\sd-unet
/models/pytorch_model.bin  ← Auto-wrapped!
```

### HuggingFace Models
```
microsoft/phi-2
bert-base-uncased
runwayml/stable-diffusion-v1-5  ← Auto-detected as SD!
/local/path/to/model
```

### PyTorch Files
```
C:\models\my_model.pt
/home/user/weights.bin
C:\checkpoint.pth
```

## Automatic Detection Matrix

| What You Have | Path to Provide | What Happens |
|---|---|---|
| Full SD pipeline | `model/` (has model_index.json) | ✓ Auto-loads as StableDiffusionPipeline |
| SD UNet only | `model/unet/` or `model/unet/config.json` | ✓ Auto-loads as UNet2DConditionModel |
| SD VAE only | `model/vae/` or `model/vae/config.json` | ✓ Auto-loads as AutoencoderKL |
| SD Text Encoder | `model/text_encoder/` or `.../config.json` | ✓ Auto-loads as CLIPTextModel |
| Raw UNet weights | `model/pytorch_model.bin` | ✓ Analyzes keys → Wraps as UNet |
| Raw VAE weights | `model/pytorch_model.bin` | ✓ Analyzes keys → Wraps as VAE |
| HuggingFace model | `model/` or `org/model-name` | ✓ Auto-loads via AutoModel |
| Custom PyTorch | `model.pt` or `model.bin` | ✓ Loads as module |

## Example Paths

### ✅ Correct
```
C:/models/stable-diffusion-v1-5              # SD full pipeline
C:/models/stable-diffusion-v1-5/unet         # SD component
/home/models/sdxl/                           # SDXL pipeline
microsoft/phi-2                               # HF Hub model
C:\Downloads\pytorch_model.bin                # Raw weights
/local/models/my-checkpoint/                  # Local HF model
```

### ❌ Problematic (but now works!)
```
C:/models/unet/pytorch_model.bin    # Raw UNet weights → Now auto-wrapped! ✓
C:/models/diffusion_pytorch_model.bin  # Component weights → Auto-detected! ✓
```

## Step-by-Step: Different Scenarios

### Scenario 1: I downloaded a Stable Diffusion model from CivitAI

**What you have**: Folder like `sd-v1-5-model-safetensors-lora` with `model_index.json`

**What to do**:
1. Copy full folder path
2. Paste into "Model Path" field
3. **Auto-detected as Stable Diffusion** ✓
4. Click "Run"

```
Path: C:\Downloads\sd-v1-5-model-safetensors-lora
↓
Detected: stable_diffusion
↓
Loads: StableDiffusionPipeline
✓ Success!
```

### Scenario 2: I have just a UNet component file

**What you have**: `pytorch_model.bin` that's just a UNet

**What to do**:
1. Provide path to the `.bin` file OR its folder
2. App analyzes the weights
3. **Detects it's a UNet** (looks for `up_blocks`, `down_blocks` keys)
4. **Automatically wraps it** in the correct class
5. Click "Run"

```
Path: C:\models\unet\pytorch_model.bin
↓
Detected: pytorch_weights
↓
Analyzes keys: Found "up_blocks" → UNet
↓
Auto-wraps: UNet2DConditionModel(state_dict)
✓ Success!
```

### Scenario 3: I have a HuggingFace model

**What you have**: Model folder with `config.json` or HF model ID

**Option A - Local folder**:
```
Path: C:\models\bert-base-uncased\
↓
Detected: huggingface_nlp
↓
Loads: AutoModel.from_pretrained()
✓ Success!
```

**Option B - HuggingFace Hub**:
```
Path: microsoft/phi-2
↓
Detected: huggingface_nlp (or dynamically loads from HF)
↓
Loads: AutoModel.from_pretrained("microsoft/phi-2")
✓ Success!
```

### Scenario 4: I have SDXL with separate components

**What you have**: 
```
sdxl-model/
├── unet/
│   ├── config.json
│   └── diffusion_pytorch_model.bin
├── vae/
│   ├── config.json
│   └── diffusion_pytorch_model.bin
└── text_encoder/
    ├── config.json
    └── pytorch_model.bin
```

**What to do**:
```
Path: C:\models\sdxl-model\
↓
Detected: stable_diffusion_component (sees unet/ folder)
↓
App chooses which to load:
  - UNet? → UNet2DConditionModel.from_pretrained("path/unet")
  - VAE? → AutoencoderKL.from_pretrained("path/vae")
  - Text Encoder? → CLIPTextModel.from_pretrained("path/text_encoder")
✓ Success!
```

## If Something Goes Wrong

| Error | Cause | Solution |
|---|---|---|
| File not found | Wrong path | Check path exists on disk |
| Model not found on HF | Network/auth | Check internet, use local path |
| Loaded object not Module | Old code issue | **✓ Fixed! Should work now** |
| Component not detected | Folder structure | Provide path to component folder, not file |
| CUDA OOM | Model too large | App auto-switches to CPU |
| Slow loading | First load | Normal, happens once |

## Advanced: Force Model Type

If auto-detection fails, you can force a model type in the UI dropdown:
- "HuggingFace folder"
- "Diffusers (Image/Video/Audio Generation)"
- "PyTorch .pt/.bin file"

But in 95% of cases, auto-detection works! Just let the app figure it out.

## Memory Requirements Cheat Sheet

| Model | Min VRAM | Recommended | With INT4 |
|---|---|---|---|
| SD 1.5 | 4GB | 6GB+ | 2-3GB |
| SD 2.1 | 6GB | 8GB+ | 3-4GB |
| SDXL | 8GB | 12GB+ | 6-8GB |
| Phi-2 | 2GB | 4GB+ | 1-2GB |
| BERT-base | 1GB | 2GB+ | 0.5-1GB |

App auto-switches to CPU if needed. No manual setup required.

## See Also

- [Full Stable Diffusion Guide](STABLE_DIFFUSION_GUIDE.md)
- [Main README](README.md)
- [Implementation Details](STABLE_DIFFUSION_FIX_SUMMARY.md)
