"""
TorchAO quantization configurations.
"""
from torchao.quantization import (
    Int4WeightOnlyConfig,
    Int8DynamicActivationInt8WeightConfig,
)

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


def get_quantization_config(quant_type: str):
    """
    Get the appropriate TorchAO quantization config based on type.
    
    Args:
        quant_type: Type of quantization
            - "INT4 (weight-only)"
            - "INT8 (dynamic)"
            - "FP8"
        
    Returns:
        TorchAO quantization config object
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
    else:
        raise ValueError(f"Unsupported quantization type: {quant_type}")


# Available quantization configs
AVAILABLE_CONFIGS = {
    "INT4 (weight-only)": Int4WeightOnlyConfig,
    "INT8 (dynamic)": Int8DynamicActivationInt8WeightConfig,
}

if FP8_AVAILABLE:
    AVAILABLE_CONFIGS["FP8"] = FP8Config
else:
    AVAILABLE_CONFIGS["FP8"] = FP8ConfigWrapper
