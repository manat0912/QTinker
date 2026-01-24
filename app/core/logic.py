"""
Core logic for model distillation and quantization.
"""
import os
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from torchao.quantization import quantize_

from configs.torchao_configs import get_quantization_config
from settings.app_settings import DISTILLED_DIR, QUANTIZED_DIR, AUTO_DEVICE_SWITCHING, MIN_VRAM_GB
from core.device_manager import get_device_manager
from core.distillation import distill_model as distill_model_new


def load_model(model_path: str, model_type: str, log_fn=None, device_manager=None):
    """
    Load a model from the specified path with intelligent device management.
    
    Args:
        model_path: Path to the model
        model_type: Type of model ("HuggingFace folder" or "PyTorch .pt/.bin file")
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
    
    # Determine device for loading
    load_on_cpu = False
    if AUTO_DEVICE_SWITCHING:
        # For large models, we'll load on CPU first then move if needed
        load_on_cpu = True
    
    try:
        if model_type == "HuggingFace folder":
            # Load on CPU first to check size, then move to GPU if appropriate
            if log_fn:
                log_fn("Loading model on CPU first (will move to GPU if VRAM allows)...")
            
            model = AutoModelForCausalLM.from_pretrained(
                model_path,
                torch_dtype=torch.float16,
                low_cpu_mem_usage=True,
                device_map="cpu"  # Load on CPU first
            )
            tokenizer = AutoTokenizer.from_pretrained(model_path)
            
            # Estimate model size
            estimated_size = device_manager.estimate_model_size(model)
            if log_fn:
                log_fn(f"Estimated model size: {estimated_size:.2f}GB")
            
            # Check if we should use GPU or CPU
            if AUTO_DEVICE_SWITCHING and device_manager.should_use_cpu(estimated_size):
                device_manager.switch_to_cpu()
                load_on_cpu = True
            else:
                # Try to move to GPU
                try:
                    model = device_manager.move_model_to_device(model, force_cpu=load_on_cpu)
                except RuntimeError as e:
                    if "out of memory" in str(e).lower():
                        if log_fn:
                            log_fn(f"⚠️  GPU OOM during model loading: {e}")
                        device_manager.switch_to_cpu()
                        model = model.to(torch.device("cpu"))
                        if log_fn:
                            log_fn("Model loaded on CPU due to VRAM limitations")
            
            return model, tokenizer
        
        elif model_type == "PyTorch .pt/.bin file":
            # Load on CPU first
            if log_fn:
                log_fn("Loading PyTorch model on CPU...")
            
            state = torch.load(model_path, map_location="cpu")
            if isinstance(state, torch.nn.Module):
                model = state
            else:
                raise ValueError(
                    "Loaded object is not a torch.nn.Module. "
                    "Customize loader for your model."
                )
            
            # Estimate size and decide device
            estimated_size = device_manager.estimate_model_size(model)
            if log_fn:
                log_fn(f"Estimated model size: {estimated_size:.2f}GB")
            
            if AUTO_DEVICE_SWITCHING and device_manager.should_use_cpu(estimated_size):
                device_manager.switch_to_cpu()
                load_on_cpu = True
            
            # Move to appropriate device
            model = device_manager.move_model_to_device(model, force_cpu=load_on_cpu)
            tokenizer = None
            return model, tokenizer
        
        else:
            raise ValueError(f"Unsupported model type: {model_type}")
    
    except RuntimeError as e:
        if "out of memory" in str(e).lower() or "cuda" in str(e).lower():
            if log_fn:
                log_fn(f"⚠️  Memory error: {e}")
                log_fn("Retrying on CPU...")
            device_manager.switch_to_cpu()
            device_manager.clear_cache()
            
            # Retry on CPU
            if model_type == "HuggingFace folder":
                model = AutoModelForCausalLM.from_pretrained(
                    model_path,
                    torch_dtype=torch.float16,
                    low_cpu_mem_usage=True,
                    device_map="cpu"
                )
                model = model.to(torch.device("cpu"))
                tokenizer = AutoTokenizer.from_pretrained(model_path)
                return model, tokenizer
            else:
                raise
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
    Save a model and tokenizer to the specified directory.
    
    Args:
        model: The model to save
        tokenizer: The tokenizer to save (can be None)
        out_dir: Output directory path
        log_fn: Optional logging function
        label: Label for logging purposes
    """
    os.makedirs(out_dir, exist_ok=True)
    
    if log_fn:
        log_fn(f"Saving {label} model to: {out_dir}")
    
    if hasattr(model, "save_pretrained"):
        model.save_pretrained(out_dir)
    else:
        torch.save(model, os.path.join(out_dir, "model.pt"))
    
    if tokenizer is not None:
        try:
            tokenizer.save_pretrained(out_dir)
        except Exception as e:
            if log_fn:
                log_fn(f"Tokenizer save failed (ok if not HF): {e}")


def apply_quantization(model, quant_type: str, log_fn=None, device_manager=None):
    """
    Apply quantization to a model using TorchAO with device management.
    
    Args:
        model: The model to quantize
        quant_type: Type of quantization ("INT4 (weight-only)" or "INT8 (dynamic)")
        log_fn: Optional logging function
        device_manager: Optional DeviceManager instance
        
    Returns:
        Quantized model
    """
    if device_manager is None:
        device_manager = get_device_manager(log_fn)
    
    if log_fn:
        log_fn(f"Applying TorchAO quantization: {quant_type}")
        log_fn(f"Quantization running on: {device_manager.get_device_name()}")
    
    # Check VRAM before quantization
    if AUTO_DEVICE_SWITCHING:
        estimated_size = device_manager.estimate_model_size(model)
        if device_manager.should_use_cpu(estimated_size):
            device_manager.switch_to_cpu()
            model = model.to(torch.device("cpu"))
            if log_fn:
                log_fn("Using CPU for quantization due to VRAM limitations")
    
    try:
        config = get_quantization_config(quant_type)
        quantize_(model, config)
        
        # Clear cache after quantization
        device_manager.clear_cache()
        
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


def run_pipeline(model_path: str, model_type: str, quant_type: str, log_fn=None):
    """
    Run the complete distillation and quantization pipeline with GPU/CPU management.
    
    Args:
        model_path: Path to the input model
        model_type: Type of model
        quant_type: Type of quantization
        log_fn: Optional logging function
        
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
        quantized_model = apply_quantization(distilled_model, quant_type, log_fn, device_manager)
        
        # Clear memory before final save
        device_manager.clear_cache()
        
        # 5. Save quantized
        if log_fn:
            log_fn("-" * 50)
        quantized_out = QUANTIZED_DIR / "quantized_model"
        save_model(quantized_model, tokenizer, quantized_out, log_fn, "quantized")
        
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
