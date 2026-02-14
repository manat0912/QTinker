
import onnx
from onnxconverter_common import float16
import os

def convert_onnx_fp16(model_path, output_path, log_fn):
    """
    Converts an ONNX model to FP16 precision.
    """
    if not model_path:
        raise ValueError("Model path cannot be empty.")
    
    log_fn(f"Loading ONNX model from: {model_path}")
    model = onnx.load(model_path)
    
    log_fn("Converting model to FP16...")
    model_fp16 = float16.convert_float_to_float16(model)
    
    if not output_path:
        model_dir = os.path.dirname(model_path)
        model_name = os.path.basename(model_path).split('.')[0]
        output_path = os.path.join(model_dir, f"{model_name}_fp16.onnx")

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    log_fn(f"Saving FP16 model to: {output_path}")
    onnx.save(model_fp16, output_path)
    log_fn("Conversion successful.")
    return output_path

def convert_onnx_int8(model_path, output_path, log_fn):
    """
    Converts an ONNX model to INT8 precision.
    """
    from onnxruntime.quantization import quantize_dynamic, QuantType

    if not model_path:
        raise ValueError("Model path cannot be empty.")

    log_fn(f"Loading ONNX model from: {model_path} for INT8 quantization")

    if not output_path:
        model_dir = os.path.dirname(model_path)
        model_name = os.path.basename(model_path).split('.')[0]
        output_path = os.path.join(model_dir, f"{model_name}_int8.onnx")
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    quantize_dynamic(
        model_input=model_path,
        model_output=output_path,
        weight_type=QuantType.QInt8
    )

    log_fn(f"INT8 quantized model saved to: {output_path}")
    log_fn("INT8 Quantization successful.")
    return output_path
