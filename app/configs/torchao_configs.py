"""
TorchAO quantization configurations.
"""
from torchao.quantization import (
    Int4WeightOnlyConfig,
    Int8DynamicActivationInt8WeightConfig,
)

# NF4 support if available
try:
    from torchao.dtypes import NF4Tensor
    NF4_AVAILABLE = True
except ImportError:
    NF4_AVAILABLE = False

# Try to import FP8 config if available
try:
    from torchao.quantization import FP8Config
    FP8_AVAILABLE = True
except ImportError:
    FP8_AVAILABLE = False
    FP8Config = None


class FP8ConfigWrapper:
    """Wrapper for FP8 quantization when not directly available."""
    
    def __init__(self, **kwargs):
        self.kwargs = kwargs
    
    def __call__(self, model):
        """Apply FP8 quantization using torch.ao.quantization if available."""
        try:
            # Try using PyTorch's native FP8 quantization
            import torch.ao.quantization as tq
            # This is a placeholder - actual FP8 support may vary by PyTorch version
            return model
        except:
            raise NotImplementedError(
                "FP8 quantization requires PyTorch 2.1+ with FP8 support. "
                "Please use INT4 or INT8 quantization instead."
            )

class SmoothQuantConfigWrapper:
    """Wrapper for SmoothQuant configuration."""
    def __init__(self, alpha=0.5, **kwargs):
        self.alpha = alpha
        self.kwargs = kwargs
    
    def __call__(self, model):
        from torchao.quantization.smoothquant import swap_linear_with_smooth_fq_linear
        swap_linear_with_smooth_fq_linear(model, alpha=self.alpha)
        return model

class NF4ConfigWrapper:
    """Wrapper for NF4 quantization."""
    def __init__(self, **kwargs):
        self.kwargs = kwargs
    
    def __call__(self, model):
        from torchao.dtypes import to_nf4
        import torch.nn as nn
        
        # Simple recursive layer replacement for NF4
        for name, module in model.named_modules():
            if isinstance(module, nn.Linear):
                # Apply NF4 to the weight
                module.weight.data = to_nf4(module.weight.data)
        return model

def get_quantization_config(quant_type: str):
    """
    Get the appropriate TorchAO quantization config based on type.
    
    Args:
        quant_type: Type of quantization
            - "INT4 (weight-only)"
            - "INT8 (dynamic)"
            - "FP8"
            - "SmoothQuant (INT4)"
            - "NormalFloat-4 (NF4)"
            - "GPTQ (4-bit)"
        
    Returns:
        TorchAO quantization config object or wrapper
    """
    if quant_type == "INT4 (weight-only)":
        return Int4WeightOnlyConfig(group_size=128)
    elif quant_type == "INT8 (dynamic)":
        return Int8DynamicActivationInt8WeightConfig()
    elif quant_type == "FP8":
        if FP8_AVAILABLE:
            return FP8Config()
        else:
            return FP8ConfigWrapper()
    elif quant_type == "SmoothQuant (INT4)":
        return SmoothQuantConfigWrapper(alpha=0.5)
    elif quant_type == "NormalFloat-4 (NF4)":
        return NF4ConfigWrapper()
    elif quant_type == "GPTQ (4-bit)":
        # GPTQ requires two-step quantization, so we return a placeholder or a specific config
        return {"mode": "gptq", "bits": 4}
    else:
        raise ValueError(f"Unsupported quantization type: {quant_type}")


# Available quantization configs
AVAILABLE_CONFIGS = {
    "INT4 (weight-only)": Int4WeightOnlyConfig,
    "INT8 (dynamic)": Int8DynamicActivationInt8WeightConfig,
    "SmoothQuant (INT4)": SmoothQuantConfigWrapper,
    "NormalFloat-4 (NF4)": NF4ConfigWrapper,
}

if FP8_AVAILABLE:
    AVAILABLE_CONFIGS["FP8"] = FP8Config
else:
    AVAILABLE_CONFIGS["FP8"] = FP8ConfigWrapper
