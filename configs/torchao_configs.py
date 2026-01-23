"""
TorchAO quantization configurations.
"""
from torchao.quantization.configs import (
    Int4WeightOnlyConfig,
    Int8DynamicConfig,
)


def get_quantization_config(quant_type: str):
    """
    Get the appropriate TorchAO quantization config based on type.
    
    Args:
        quant_type: Type of quantization ("INT4 (weight-only)" or "INT8 (dynamic)")
        
    Returns:
        TorchAO quantization config object
    """
    if quant_type == "INT4 (weight-only)":
        return Int4WeightOnlyConfig(group_size=128)
    elif quant_type == "INT8 (dynamic)":
        return Int8DynamicConfig()
    else:
        raise ValueError(f"Unsupported quantization type: {quant_type}")


# Available quantization configs
AVAILABLE_CONFIGS = {
    "INT4 (weight-only)": Int4WeightOnlyConfig,
    "INT8 (dynamic)": Int8DynamicConfig,
}
