
import torch
import os
try:
    import torch_tensorrt
except (ImportError, OSError):
    torch_tensorrt = None

def convert_tensorrt(model_path, output_path, log_fn):
    """
    Converts a PyTorch model to TensorRT.
    """
    if torch_tensorrt is None:
        raise ImportError("torch_tensorrt is not installed or failed to load. Please verify your TensorRT installation.")

    if not torch.cuda.is_available():
        raise RuntimeError("TensorRT conversion requires a CUDA-enabled GPU.")

    if not model_path:
        raise ValueError("Model path cannot be empty.")

    log_fn(f"Loading PyTorch model from: {model_path}")
    model = torch.load(model_path).cuda()

    log_fn("Compiling model to TensorRT...")
    
    # Assuming the model is traceable and targeting common input shape
    # This might need adjustment depending on the model architecture
    example_inputs = [torch.randn(1, 3, 224, 224).cuda()]
    
    try:
        trt_model = torch_tensorrt.compile(model, 
            inputs=example_inputs,
            enabled_precisions={torch.half} # Enable FP16
        )

        if not output_path:
            model_dir = os.path.dirname(model_path)
            model_name = os.path.basename(model_path).split('.')[0]
            output_path = os.path.join(model_dir, f"{model_name}_tensorrt.ts")

        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        log_fn(f"Saving TensorRT model to: {output_path}")
        torch.jit.save(trt_model, output_path)
        log_fn("TensorRT conversion successful.")
        return output_path

    except Exception as e:
        log_fn(f"TensorRT conversion failed: {e}")
        raise

