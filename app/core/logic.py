"""
Core logic for model distillation and quantization.
"""
import os
import sys
import json
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
from torchao.quantization import quantize_

from configs.torchao_configs import get_quantization_config
from settings.app_settings import DISTILLED_DIR, QUANTIZED_DIR, AUTO_DEVICE_SWITCHING, MIN_VRAM_GB
from core.device_manager import get_device_manager
from core.distillation import distill_model as distill_model_new
from compression_toolkit import QuantizationToolkit, save_model_robust


def _detect_model_architecture(model_path: str) -> str:
    """
    Detect the model architecture from the directory structure.
    
    Args:
        model_path: Path to the model
        
    Returns:
        Model type identifier (e.g., 'stable_diffusion', 'huggingface_nlp', 'pytorch_weights')
    """
    try:
        model_path = Path(model_path)
        
        # Check for Stable Diffusion or Diffusers model
        if (model_path / "model_index.json").exists():
            with open(model_path / "model_index.json") as f:
                config = json.load(f)
                class_name = config.get("_class_name", "")
                if "StableDiffusion" in class_name or "pipeline" in class_name.lower():
                    return "stable_diffusion"
            return "diffusers"
        
        # Check for UNet/VAE/TextEncoder components (Stable Diffusion components)
        has_unet = (model_path / "unet" / "config.json").exists() or (model_path / "unet").exists()
        has_vae = (model_path / "vae" / "config.json").exists() or (model_path / "vae").exists()
        if has_unet or has_vae:
            return "stable_diffusion_component"
        
        # Check for HuggingFace transformers
        if (model_path / "config.json").exists():
            with open(model_path / "config.json") as f:
                config = json.load(f)
                
                # 1. Check model_type key (standard for Transformers)
                model_type = config.get("model_type", "")
                if model_type:
                    return "huggingface_nlp"
                
                # 2. Check for diffusers components via _class_name
                class_name = config.get("_class_name", "")
                if class_name:
                    if any(x in class_name for x in ["UNet", "Autoencoder", "VAE", "CLIPText", "Scheduler"]):
                        return "stable_diffusion_component"
                
                # 3. Check for recognized model keywords in folder name or architectural clues
                recognized_path = Path(__file__).parent.parent / "configs" / "recognized_models.json"
                if recognized_path.exists():
                    with open(recognized_path) as rf:
                        recognized_list = json.load(rf).get("recognized_models", [])
                        folder_name = model_path.name.lower()
                        # Check if folder name contains any recognized keywords
                        if any(kw in folder_name for kw in recognized_list):
                            return "huggingface_nlp"
                        
                        # Check if any keys in config match recognized types
                        for key in config.keys():
                            if any(kw in key.lower() for kw in recognized_list):
                                return "huggingface_nlp"
        
        # Check if it's a single .pt or .bin file (not a directory with structure)
        if str(model_path).endswith(('.pt', '.bin', '.ckpt', '.safetensors')):
            return "pytorch_weights"
        
        # Default to huggingface if directory exists with minimal structure
        return "huggingface_nlp"
    
    except Exception as e:
        print(f"Warning: Could not detect model architecture: {e}")
        return "unknown"


def _load_stable_diffusion_model(model_path: str, log_fn=None, device_manager=None):
    """
    Load a Stable Diffusion model or component.
    
    Args:
        model_path: Path to the model
        log_fn: Optional logging function
        device_manager: Optional DeviceManager instance
        
    Returns:
        Tuple of (model, tokenizer) - tokenizer is None for SD models
    """
    try:
        from diffusers import (
            StableDiffusionPipeline,
            UNet2DConditionModel,
            AutoencoderKL,
            StableDiffusionXLPipeline,
        )
        from transformers import CLIPTextModel
    except ImportError:
        raise ImportError("diffusers library is required for Stable Diffusion models. Install with: pip install diffusers")
    
    model_path = Path(model_path)
    
    if log_fn:
        log_fn(f"Detecting Stable Diffusion model structure at: {model_path}")
    
    try:
        # Try loading as full pipeline first
        if (model_path / "model_index.json").exists():
            if log_fn:
                log_fn("Loading as Stable Diffusion pipeline...")
            
            # Check if it's SDXL or SD 1.5
            try:
                model = StableDiffusionXLPipeline.from_pretrained(str(model_path), torch_dtype=torch.float16)
                if log_fn:
                    log_fn("✓ Loaded as SDXL pipeline")
            except Exception:
                model = StableDiffusionPipeline.from_pretrained(str(model_path), torch_dtype=torch.float16)
                if log_fn:
                    log_fn("✓ Loaded as Stable Diffusion pipeline")
            
            return model, None
        
        # Check if it's a standalone component (direct folder)
        if (model_path / "config.json").exists():
            with open(model_path / "config.json") as f:
                config = json.load(f)
                class_name = config.get("_class_name", "")
                
                if "Autoencoder" in class_name or "VAE" in class_name:
                    if log_fn: log_fn("Loading standalone VAE...")
                    try:
                        return AutoencoderKL.from_pretrained(str(model_path), torch_dtype=torch.float16, use_safetensors=True), None
                    except Exception as e:
                        if "diffusion_pytorch_model.safetensors" in str(e):
                            if log_fn: log_fn("Safetensors not found, falling back to .bin weights...")
                            return AutoencoderKL.from_pretrained(str(model_path), torch_dtype=torch.float16, use_safetensors=False), None
                        raise e
                
                if "UNet" in class_name:
                    if log_fn: log_fn("Loading standalone UNet...")
                    try:
                        return UNet2DConditionModel.from_pretrained(str(model_path), torch_dtype=torch.float16, use_safetensors=True), None
                    except Exception as e:
                        if "diffusion_pytorch_model.safetensors" in str(e):
                            if log_fn: log_fn("Safetensors not found, falling back to .bin weights...")
                            return UNet2DConditionModel.from_pretrained(str(model_path), torch_dtype=torch.float16, use_safetensors=False), None
                        raise e
                
                if "CLIPText" in class_name:
                    if log_fn: log_fn("Loading standalone Text Encoder...")
                    return CLIPTextModel.from_pretrained(str(model_path), torch_dtype=torch.float16), None

        # Load nested components
        elif (model_path / "unet").exists() or (model_path / "unet" / "config.json").exists():
            if log_fn:
                log_fn("Loading UNet component...")
            model = UNet2DConditionModel.from_pretrained(str(model_path / "unet"), torch_dtype=torch.float16)
            if log_fn:
                log_fn("✓ Loaded UNet component")
            return model, None
        
        elif (model_path / "vae").exists() or (model_path / "vae" / "config.json").exists():
            if log_fn:
                log_fn("Loading VAE component...")
            try:
                model = AutoencoderKL.from_pretrained(str(model_path / "vae"), torch_dtype=torch.float16, use_safetensors=True)
            except Exception as e:
                if "diffusion_pytorch_model.safetensors" in str(e):
                    if log_fn: log_fn("Safetensors not found, falling back to .bin weights...")
                    model = AutoencoderKL.from_pretrained(str(model_path / "vae"), torch_dtype=torch.float16, use_safetensors=False)
                else:
                    raise e
            if log_fn:
                log_fn("✓ Loaded VAE component")
            return model, None
        
        elif (model_path / "text_encoder").exists() or (model_path / "text_encoder" / "config.json").exists():
            if log_fn:
                log_fn("Loading Text Encoder component...")
            model = CLIPTextModel.from_pretrained(str(model_path / "text_encoder"), torch_dtype=torch.float16)
            if log_fn:
                log_fn("✓ Loaded Text Encoder component")
            return model, None
        
        else:
            # Try loading parent directory as pipeline
            if log_fn:
                log_fn("Attempting to load parent directory as pipeline...")
            model = StableDiffusionPipeline.from_pretrained(str(model_path.parent), torch_dtype=torch.float16)
            return model, None
    
    except Exception as e:
        if log_fn:
            log_fn(f"Error loading Stable Diffusion model: {e}")
        raise


def _load_pytorch_state_dict(model_path: str, log_fn=None, device_manager=None):
    """
    Load a raw PyTorch state_dict and wrap it in a simple module.
    
    Args:
        model_path: Path to the .pt or .bin file
        log_fn: Optional logging function
        device_manager: Optional DeviceManager instance
        
    Returns:
        Tuple of (model, None)
    """
    if log_fn:
        log_fn(f"Loading PyTorch state_dict from: {model_path}")
    
    state = torch.load(model_path, map_location="cpu")
    
    if isinstance(state, torch.nn.Module):
        # Already a module
        model = state
    elif isinstance(state, dict):
        # Raw state_dict - need to identify component type
        if log_fn:
            log_fn("Detected raw state_dict - attempting to identify component type...")
        
        # Check state_dict keys to determine model type
        keys = list(state.keys())[:10]
        
        if any("unet" in k.lower() for k in keys) or any("up_blocks" in k or "down_blocks" in k for k in keys):
            # Likely UNet
            if log_fn:
                log_fn("Detected UNet component - creating wrapper...")
            try:
                from diffusers import UNet2DConditionModel
                model = UNet2DConditionModel.from_pretrained(str(Path(model_path).parent))
            except Exception:
                # Fallback: create a simple wrapper
                class StateDictWrapper(torch.nn.Module):
                    def __init__(self, state_dict):
                        super().__init__()
                        self.state_dict_data = state_dict
                model = StateDictWrapper(state)
        
        elif any("vae" in k.lower() for k in keys) or any("encoder" in k or "decoder" in k for k in keys):
            # Likely VAE
            if log_fn:
                log_fn("Detected VAE component - creating wrapper...")
            try:
                from diffusers import AutoencoderKL
                model = AutoencoderKL.from_pretrained(str(Path(model_path).parent))
            except Exception:
                class StateDictWrapper(torch.nn.Module):
                    def __init__(self, state_dict):
                        super().__init__()
                        self.state_dict_data = state_dict
                model = StateDictWrapper(state)
        
        else:
            # Generic wrapper for unknown state_dict
            if log_fn:
                log_fn("Creating generic state_dict wrapper...")
            
            class StateDictWrapper(torch.nn.Module):
                def __init__(self, state_dict):
                    super().__init__()
                    # Try to load state_dict into a generic module
                    for key, value in state_dict.items():
                        if isinstance(value, torch.Tensor):
                            self.register_buffer(key.replace(".", "_"), value)
            
            model = StateDictWrapper(state)
    else:
        raise ValueError(f"Loaded object type {type(state)} is not supported. Expected torch.nn.Module or dict.")
    
    if log_fn:
        log_fn("✓ State_dict loaded and wrapped successfully")
    
    return model, None


def load_model(model_path: str, model_type: str, log_fn=None, device_manager=None):
    """
    Load a model from the specified path with intelligent device management.
    Supports HuggingFace, Stable Diffusion, Diffusers, and PyTorch files.
    
    Args:
        model_path: Path to the model
        model_type: Type of model from UI dropdown
        log_fn: Optional logging function
        device_manager: Optional DeviceManager instance
        
    Returns:
        Tuple of (model, tokenizer)
    """
    if device_manager is None:
        device_manager = get_device_manager(log_fn)
    
    if log_fn:
        log_fn(f"Loading model from: {model_path}")
    
    device_manager.log_device_info()
    
    # Check device availability
    device = device_manager.get_device()
    use_gpu = device.type == "cuda"
    
    try:
        # Detect actual model architecture
        architecture = _detect_model_architecture(model_path)
        
        if log_fn:
            log_fn(f"Detected model architecture: {architecture}")
        
        # Load based on detected architecture, not just UI selection
        if architecture == "stable_diffusion" or architecture == "stable_diffusion_component":
            if log_fn:
                log_fn("Loading Stable Diffusion model...")
            model, tokenizer = _load_stable_diffusion_model(model_path, log_fn, device_manager)
        
        elif architecture in ["huggingface_nlp", "diffusers"] or "HuggingFace" in model_type or "Diffusers" in model_type:
            if log_fn:
                log_fn(f"Loading HuggingFace/Transformers model...")
            
            device_map = "auto" if use_gpu else "cpu"
            
            try:
                from transformers import AutoModel
                model = AutoModel.from_pretrained(
                    model_path,
                    torch_dtype=torch.float16 if use_gpu else torch.float32,
                    low_cpu_mem_usage=True,
                    device_map=device_map
                )
            except Exception:
                # Fallback to AutoModelForCausalLM
                model = AutoModelForCausalLM.from_pretrained(
                    model_path,
                    torch_dtype=torch.float16 if use_gpu else torch.float32,
                    low_cpu_mem_usage=True,
                    device_map=device_map
                )
            
            try:
                tokenizer = AutoTokenizer.from_pretrained(model_path)
            except Exception:
                tokenizer = None
            
            if log_fn:
                log_fn(f"Model loaded with device_map='{device_map}'")
            
            return model, tokenizer
        
        elif architecture == "pytorch_weights" or ".pt" in model_path or ".bin" in model_path:
            if log_fn:
                log_fn("Loading PyTorch model from file...")
            model, tokenizer = _load_pytorch_state_dict(model_path, log_fn, device_manager)
        
        else:
            # Default: try HuggingFace
            if log_fn:
                log_fn("Loading as HuggingFace folder...")
            
            device_map = "auto" if use_gpu else "cpu"
            model = AutoModelForCausalLM.from_pretrained(
                model_path,
                torch_dtype=torch.float16 if use_gpu else torch.float32,
                low_cpu_mem_usage=True,
                device_map=device_map
            )
            tokenizer = AutoTokenizer.from_pretrained(model_path)
            return model, tokenizer
        
        # Move to appropriate device
        model = device_manager.move_model_to_device(model, force_cpu=False)
        
        if log_fn:
            log_fn(f"Model successfully loaded on: {device_manager.get_device_name()}")
        
        return model, tokenizer
    
    except RuntimeError as e:
        if "out of memory" in str(e).lower() or "cuda" in str(e).lower():
            if log_fn:
                log_fn(f"⚠️  Memory error: {e}")
                log_fn("Retrying on CPU...")
            device_manager.switch_to_cpu()
            device_manager.clear_cache()
            
            # Retry on CPU
            try:
                model = AutoModelForCausalLM.from_pretrained(
                    model_path,
                    torch_dtype=torch.float32,
                    low_cpu_mem_usage=True,
                    device_map="cpu"
                )
                tokenizer = AutoTokenizer.from_pretrained(model_path)
                return model, tokenizer
            except Exception:
                raise e
        else:
            raise


def distill_model(model, tokenizer, log_fn=None, device_manager=None):
    """
    Distill a model using the new distillation module.
    
    Args:
        model: The model to distill
        tokenizer: Model tokenizer
        log_fn: Optional logging function
        device_manager: Optional DeviceManager instance
        
    Returns:
        Distilled model
    """
    if device_manager is None:
        device_manager = get_device_manager(log_fn)
    
    # Use the new distillation module
    return distill_model_new(model, tokenizer, device_manager, log_fn)


def save_model(model, tokenizer, out_dir, log_fn=None, label="model"):
    """
    Save a model and tokenizer to the specified directory using the robust saver.
    
    Args:
        model: The model to save
        tokenizer: The tokenizer to save (can be None)
        out_dir: Output directory path
        log_fn: Optional logging function
        label: Label for logging purposes
    """
    if log_fn:
        log_fn(f"Saving {label} model to: {out_dir}")
    
    save_model_robust(model, out_dir, tokenizer=tokenizer)



def apply_quantization(model, quant_type: str, log_fn=None, device_manager=None, model_input_path=None, **kwargs):
    """
    Apply quantization to a model using TorchAO or external tools with device management.
    
    Args:
        model: The model to quantize (in-memory)
        quant_type: Type of quantization
        log_fn: Optional logging function
        device_manager: Optional DeviceManager instance
        model_input_path: Path to the model on disk (required for GPTQ/AWQ/GGUF)
        **kwargs: Additional parameters like smoothquant_alpha, gptq_group_size
        
    Returns:
        Quantized model (if in-memory) or None (if saved to disk by external tool)
    """
    if device_manager is None:
        device_manager = get_device_manager(log_fn)
    
    if log_fn:
        log_fn(f"Applying quantization: {quant_type}")
        log_fn(f"Quantization running on: {device_manager.get_device_name()}")
    
    # Handle external tool quantization (GPTQ, AWQ, etc.)
    if "GPTQ" in quant_type or "AWQ" in quant_type or "GGUF" in quant_type or "ONNX" in quant_type:
        if not model_input_path:
            raise ValueError(f"Path to saved model is required for {quant_type} quantization.")
            
        output_path = QUANTIZED_DIR / f"quantized_{quant_type.split()[0].lower()}"
        output_path_str = str(output_path)
        
        # Ensure we are on CPU/GPU as needed, but mostly these tools handle it.
        # We might need to unload the current model from GPU to free memory for the tool.
        device_manager.clear_cache()
        # Ideally, we should unload 'model' here if it's on GPU.
        if model is not None:
             model.to("cpu")
        device_manager.clear_cache()

        try:
            if "GPTQ" in quant_type:
                log_fn(f"Running GPTQ... (Input: {model_input_path})")
                group_size = kwargs.get("gptq_group_size", 128)
                QuantizationToolkit.quantize_with_gptq(str(model_input_path), output_path_str, bits=4, group_size=group_size)
            elif "AWQ" in quant_type:
                log_fn(f"Running AWQ... (Input: {model_input_path})")
                QuantizationToolkit.quantize_with_awq(str(model_input_path), output_path_str, bits=4)
            elif "ONNX" in quant_type:
                log_fn(f"Running ONNX quantization...")
                QuantizationToolkit.quantize_with_onnx(str(model_input_path), output_path_str)
            
            log_fn(f"Quantization complete. Saved to: {output_path_str}")
            return None # External tool saved it
        except Exception as e:
            log_fn(f"External quantization failed: {e}")
            raise e

    # Handle BitsAndBytes (Load-time mostly, but check toolkit)
    if "LLM.int8()" in quant_type or "NF4" in quant_type:
         # BNB is tricky as it's usually load-time. 
         # But if we want to return a model that behaves quantized (e.g. for saving),
         # saving 4bit/8bit models usually requires saving as safetensors/peft.
         # For now, we'll try to use the toolkit or just warn.
         try:
             # Just pass through for now or implement conversion if possible
             log_fn("Applying BitsAndBytes quantization (Note: This is usually for inference loading).")
             # We can try converting linear layers if the model is on CPU, but BNB needs GPU often.
             return model
         except Exception as e:
             log_fn(f"BNB Quantization failed: {e}")
             return model

    # Check VRAM before quantization (TorchAO)
    if AUTO_DEVICE_SWITCHING:
        estimated_size = device_manager.estimate_model_size(model)
        if device_manager.should_use_cpu(estimated_size):
            device_manager.switch_to_cpu()
            model = model.to(torch.device("cpu"))
            if log_fn:
                log_fn("Using CPU for quantization due to VRAM limitations")
    
    try:
        config = get_quantization_config(quant_type)
        if config:
            if hasattr(config, "__call__") and not isinstance(config, type):
                # Handle wrappers like SmoothQuant
                if hasattr(config, "alpha") and "smoothquant_alpha" in kwargs:
                    config.alpha = kwargs["smoothquant_alpha"]
                model = config(model)
            else:
                quantize_(model, config)
            # Clear cache after quantization
            device_manager.clear_cache()
            return model
        else:
            log_fn(f"Warning: No config found for {quant_type}, returning original model.")
            return model

    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            if log_fn:
                log_fn(f"⚠️  GPU OOM during quantization: {e}")
                log_fn("Moving to CPU and retrying...")
            device_manager.switch_to_cpu()
            device_manager.clear_cache()
            model = model.to(torch.device("cpu"))
            
            # Retry quantization on CPU
            config = get_quantization_config(quant_type)
            quantize_(model, config)
            device_manager.clear_cache()
            return model
        else:
            raise


def run_pipeline(model_path: str, model_type: str, quant_type: str, log_fn=None, **kwargs):
    """
    Run the complete distillation and quantization pipeline with GPU/CPU management.
    
    Args:
        model_path: Path to the input model
        model_type: Type of model
        quant_type: Type of quantization
        log_fn: Optional logging function
        **kwargs: Additional parameters like smoothquant_alpha, gptq_group_size
        
    Returns:
        Tuple of (distilled_model_path, quantized_model_path)
    """
    # Initialize device manager
    device_manager = get_device_manager(log_fn)
    
    if log_fn:
        log_fn("=== Starting distill + quantize pipeline ===")
        log_fn("=" * 50)
    
    try:
        # 1. Load model (with device management)
        model, tokenizer = load_model(model_path, model_type, log_fn, device_manager)
        
        if log_fn:
            log_fn(f"Model loaded on: {device_manager.get_device_name()}")
            device_manager.log_device_info()
        
        # 2. Distill (with device management)
        if log_fn:
            log_fn("-" * 50)
        distilled_model = distill_model(model, tokenizer, log_fn, device_manager)
        
        # Clear memory before saving
        device_manager.clear_cache()
        
        # 3. Save distilled
        if log_fn:
            log_fn("-" * 50)
        distilled_out = DISTILLED_DIR / "distilled_model"
        save_model(distilled_model, tokenizer, distilled_out, log_fn, "distilled")
        
        # Clear memory after saving
        device_manager.clear_cache()
        
        # 4. Quantize (with device management)
        if log_fn:
            log_fn("-" * 50)
        
        # Determine path to distilled model
        distilled_out = DISTILLED_DIR / "distilled_model"
        
        # Pass the path to apply_quantization
        quantized_model = apply_quantization(
            distilled_model, quant_type, log_fn, device_manager, 
            model_input_path=distilled_out, **kwargs
        )
        
        # Clear memory before final save
        device_manager.clear_cache()
        
        # 5. Save quantized (only if apply_quantization returned a model, otherwise it saved it itself)
        quantized_out = QUANTIZED_DIR / "quantized_model"
        
        if quantized_model is not None:
            if log_fn:
                log_fn("-" * 50)
            save_model(quantized_model, tokenizer, quantized_out, log_fn, "quantized")
        elif "GPTQ" in quant_type or "AWQ" in quant_type or "ONNX" in quant_type:
            # It was saved by external tool to a specific path
            # We can update quantized_out to point there for the return value
             quantized_out = QUANTIZED_DIR / f"quantized_{quant_type.split()[0].lower()}"
        
        if log_fn:
            log_fn("=" * 50)
            log_fn("=== Done. Distilled + Quantized models saved. ===")
            device_manager.log_device_info()
        
        return str(distilled_out), str(quantized_out)
    
    except Exception as e:
        if log_fn:
            log_fn(f"ERROR: {e}")
            log_fn("Attempting cleanup...")
        device_manager.clear_cache()
        raise
