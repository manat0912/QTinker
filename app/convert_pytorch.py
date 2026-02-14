
import torch
import os

def convert_pytorch_fp16(model_path, output_path, log_fn):
    """
    Converts a PyTorch model to FP16 precision.
    """
    if not model_path:
        raise ValueError("Model path cannot be empty.")

    log_fn(f"Loading PyTorch weights from: {model_path}")
    weights = torch.load(model_path, map_location='cpu')

    log_fn("Converting weights to half precision (FP16)...")
    
    # Handle both state_dict and entire model
    if isinstance(weights, dict):
        for key in weights:
            if isinstance(weights[key], torch.Tensor):
                weights[key] = weights[key].half()
    else:
        weights.half()


    if not output_path:
        model_dir = os.path.dirname(model_path)
        model_name = os.path.basename(model_path).split('.')[0]
        output_path = os.path.join(model_dir, f"{model_name}_fp16.pth")

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    log_fn(f"Saving FP16 model to: {output_path}")
    torch.save(weights, output_path)
    log_fn("Conversion successful.")
    return output_path

def convert_pytorch_bf16(model_path, output_path, log_fn):
    """
    Converts a PyTorch model to BF16 precision.
    """
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("BF16 is not supported on this device.")

    if not model_path:
        raise ValueError("Model path cannot be empty.")

    log_fn(f"Loading PyTorch weights from: {model_path}")
    weights = torch.load(model_path, map_location='cpu')

    log_fn("Converting weights to bfloat16 precision (BF16)...")
    
    # Handle both state_dict and entire model
    if isinstance(weights, dict):
        for key in weights:
            if isinstance(weights[key], torch.Tensor):
                weights[key] = weights[key].to(torch.bfloat16)
    else:
        weights.to(torch.bfloat16)

    if not output_path:
        model_dir = os.path.dirname(model_path)
        model_name = os.path.basename(model_path).split('.')[0]
        output_path = os.path.join(model_dir, f"{model_name}_bf16.pth")

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    log_fn(f"Saving BF16 model to: {output_path}")
    torch.save(weights, output_path)
    log_fn("Conversion successful.")
    return output_path
