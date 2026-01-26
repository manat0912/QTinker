# Testing the Stable Diffusion Fix

## Test Cases to Verify the Fix Works

### Test 1: Load from the Error's Original Path
**Goal**: Verify the exact error case is now fixed

**Test Data**: 
```
Path: C:/pinokio/api/PersonaLiveStream/app/pretrained_weights/Temp/unet/pytorch_model.bin
Model Type: PyTorch .pt/.bin file (or auto-detect)
```

**Expected Result**:
```
✓ Loading PyTorch model from file...
✓ Detected raw state_dict - attempting to identify component type...
✓ Detected UNet component - creating wrapper...
✓ State_dict loaded and wrapped successfully
✓ Model successfully loaded on: CUDA (NVIDIA GeForce RTX 4070)
```

**Verification**: 
- No error about "not a torch.nn.Module"
- Model loads successfully
- Can proceed to distillation

---

### Test 2: Load Full Stable Diffusion Pipeline
**Goal**: Test loading a complete SD model

**Test Data**:
```
Path: C:\Models\stable-diffusion-v1-5
(folder with model_index.json)
```

**Expected Result**:
```
✓ Loading model from: C:\Models\stable-diffusion-v1-5
✓ Detected model architecture: stable_diffusion
✓ Loading Stable Diffusion model...
✓ Detecting Stable Diffusion model structure...
✓ Loading as Stable Diffusion pipeline...
✓ Loaded as Stable Diffusion (or SDXL) pipeline
✓ Model successfully loaded on: CUDA
```

**Verification**:
- Pipeline loads without errors
- Correct version detected (SD vs SDXL)
- Model ready for distillation

---

### Test 3: Load SD Component (UNet)
**Goal**: Test loading individual components

**Test Data**:
```
Path: C:\Models\sdxl\unet
(folder with config.json and diffusion_pytorch_model.bin)
```

**Expected Result**:
```
✓ Detected model architecture: stable_diffusion_component
✓ Loading Stable Diffusion model...
✓ Loading UNet component...
✓ Loaded UNet component
✓ Model successfully loaded on: CUDA
```

**Verification**:
- Component loads as UNet2DConditionModel
- Correct component type detected
- Ready for distillation

---

### Test 4: Load Raw State Dict (Analysis-based)
**Goal**: Test auto-wrapping of raw weights

**Test Data**:
```
Path: C:\Models\components\pytorch_model.bin
(raw weights file without model architecture)
```

**Expected Result**:
```
✓ Loading PyTorch model from file...
✓ Detected raw state_dict - attempting to identify component type...
✓ Detected UNet component - creating wrapper...
✓ State_dict loaded and wrapped successfully
✓ Model successfully loaded on: CUDA
```

**Verification**:
- App analyzes state dict keys
- Correctly identifies component type (UNet, VAE, etc.)
- Auto-wraps in appropriate class
- Model loads successfully

---

### Test 5: Load HuggingFace Model (Backward Compatibility)
**Goal**: Ensure existing functionality still works

**Test Data**:
```
Path: microsoft/phi-2
(or local HuggingFace model folder)
```

**Expected Result**:
```
✓ Loading model from: microsoft/phi-2
✓ Detected model architecture: huggingface_nlp
✓ Loading HuggingFace/Transformers model...
✓ Model loaded with device_map='auto'
✓ Model successfully loaded on: CUDA
```

**Verification**:
- HuggingFace models still work
- Auto-detection works correctly
- Backward compatibility maintained

---

### Test 6: Run Full Pipeline with SD Model
**Goal**: Test end-to-end distillation + quantization

**Test Data**:
```
Model Path: C:\Models\stable-diffusion-v1-5
Model Type: (auto-detected)
Quantization: INT8 (dynamic)
Distillation: placeholder (or teacher_student)
```

**Expected Result**:
```
=== Starting distill + quantize pipeline ===
==================================================
Loading model from: C:\Models\stable-diffusion-v1-5
Detected model architecture: stable_diffusion
✓ Model loaded on: CUDA
...
✓ Distilled model saved to: outputs/distilled/distilled_model
✓ Quantized model saved to: outputs/quantized/quantized_model
=== SUCCESS ===
```

**Verification**:
- Model loads without errors
- Distillation completes
- Quantization applies correctly
- Both output models saved

---

## Specific Test for Your Original Error

**Your Original Error**:
```
Loading model from: C:/pinokio/api/PersonaLiveStream/app/pretrained_weights/Temp/unet/pytorch_model.bin
ERROR: Loaded object is not a torch.nn.Module. Customize loader for your model.
```

**How to Test the Fix**:

1. **Gather the file**:
   ```
   cp C:/pinokio/api/PersonaLiveStream/app/pretrained_weights/Temp/unet/pytorch_model.bin \
      C:\pinokio\api\QTinker\test_models\pytorch_model.bin
   ```

2. **Load in QTinker**:
   - Open QTinker Gradio UI
   - Paste path: `C:\pinokio\api\QTinker\test_models\pytorch_model.bin`
   - Leave model type as "PyTorch .pt/.bin file"
   - Click "Load" or try to run pipeline

3. **Expected (New) Result**:
   ```
   ✓ Loading PyTorch model from file...
   ✓ Detected raw state_dict...
   ✓ Detected UNet component...
   ✓ State_dict loaded and wrapped successfully
   ✓ Model successfully loaded on: CUDA
   ```

4. **Verify**:
   - ✅ NO error about "not a torch.nn.Module"
   - ✅ Model loaded successfully
   - ✅ Can proceed with distillation/quantization

---

## Quick Test Script

If you want to test programmatically:

```python
from core.logic import load_model
from core.device_manager import get_device_manager

# Create a logging function
def log_fn(msg):
    print(msg)

# Get device manager
device_manager = get_device_manager(log_fn)

# Test 1: Raw state dict (your original error case)
print("\n=== Test 1: Raw UNet State Dict ===")
try:
    model, tokenizer = load_model(
        "C:/path/to/unet/pytorch_model.bin",
        "PyTorch .pt/.bin file",
        log_fn,
        device_manager
    )
    print("✓ Test 1 PASSED")
except Exception as e:
    print(f"✗ Test 1 FAILED: {e}")

# Test 2: Full SD pipeline
print("\n=== Test 2: Full SD Pipeline ===")
try:
    model, tokenizer = load_model(
        "C:/path/to/stable-diffusion-v1-5",
        "Diffusers (Image/Video/Audio Generation)",
        log_fn,
        device_manager
    )
    print("✓ Test 2 PASSED")
except Exception as e:
    print(f"✗ Test 2 FAILED: {e}")

# Test 3: HuggingFace model
print("\n=== Test 3: HuggingFace Model ===")
try:
    model, tokenizer = load_model(
        "microsoft/phi-2",
        "HuggingFace folder",
        log_fn,
        device_manager
    )
    print("✓ Test 3 PASSED")
except Exception as e:
    print(f"✗ Test 3 FAILED: {e}")

print("\n=== All Tests Complete ===")
```

---

## Debugging If Tests Fail

**If you get "detected model architecture: unknown"**:
- Check path exists and is readable
- Verify folder structure is intact
- Try with absolute path instead of relative

**If you get import errors**:
- Ensure `diffusers` is installed: `pip install diffusers`
- Ensure `transformers` is installed: `pip install transformers`

**If you get CUDA OOM**:
- App should auto-switch to CPU
- Check logs for which device is being used
- Try with a smaller model first

**If model doesn't wrap correctly**:
- Check state_dict keys: `torch.load(path, map_location='cpu').keys()`
- Verify component type (UNet has `up_blocks`, VAE has `encoder`, etc.)
- Report detailed keys in issue

---

## Test Environment

For best testing:
- **GPU**: NVIDIA RTX 4070 or better (or any CUDA GPU)
- **VRAM**: 12GB+ for SDXL, 6GB+ for SD 1.5
- **Models**: Download one of:
  - `stabilityai/stable-diffusion-v1-5` (2.2GB, 512x512)
  - `stabilityai/stable-diffusion-2-1` (5GB, 768x768)
  - `stabilityai/stable-diffusion-xl-base-1.0` (6.9GB)

---

## Success Criteria

✅ **The fix is working if**:
1. Your original UNet file loads without "not a torch.nn.Module" error
2. Full SD pipelines load with auto-detection
3. Components load from subfolders
4. HuggingFace models still work (backward compatibility)
5. Distillation + Quantization pipeline completes

✅ **All tests pass**: You're good to use the app with all Stable Diffusion models!

---

## Reporting Results

If tests fail, please report:
1. **Which test failed** (Test 1-6)
2. **Exact error message** (copy-paste from logs)
3. **Model path** (or model name if public)
4. **Your hardware** (GPU, VRAM, CUDA version)
5. **Your setup** (Pinokio version, Python version)

---

## Next Steps After Testing

Once tests pass:

1. **Use with your models**:
   ```
   Path: C:/pinokio/api/PersonaLiveStream/app/pretrained_weights/Temp/unet/
   (auto-detects as SD component)
   ↓
   Run distill + quantize
   ↓
   ✓ Output saved to outputs/
   ```

2. **Try different models**:
   - Different SD versions
   - Different quantization settings
   - Different distillation strategies

3. **Check documentation**:
   - [QUICK_REFERENCE_MODELS.md](QUICK_REFERENCE_MODELS.md) for quick start
   - [STABLE_DIFFUSION_GUIDE.md](STABLE_DIFFUSION_GUIDE.md) for detailed guide

---

## Support

For issues, see:
- [Troubleshooting in README](README.md#troubleshooting)
- [Stable Diffusion-specific section](README.md#stable-diffusion-model-loading-errors)
- [STABLE_DIFFUSION_FIX_SUMMARY.md](STABLE_DIFFUSION_FIX_SUMMARY.md) for technical details
