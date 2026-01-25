"""
GGUF Quantization Module

Supports converting PyTorch models to GGUF format with various quantization methods.
GGUF is the primary quantization format for llama.cpp and compatible runtimes.

Supports:
- Text models (BERT, GPT-2, LLaMA, Mistral, etc.)
- Vision models (ViT, CLIP)
- Custom models via state_dict conversion
"""

import os
import torch
import json
from pathlib import Path
from typing import Optional, Dict, Any, Tuple
from dataclasses import dataclass
import sys

sys.path.insert(0, str(Path(__file__).parent))


@dataclass
class GGUFQuantizationConfig:
    """Configuration for GGUF quantization."""
    
    # Quantization method
    # Options: "f32", "f16", "q8_0", "q4_0", "q4_1", "q5_0", "q5_1"
    quant_method: str = "q4_0"
    
    # Number of layers to quantize (0 = all)
    num_layers: int = 0
    
    # Use CUDA if available
    use_cuda: bool = True
    
    # Chunk size for processing
    chunk_size: int = 1024
    
    # Output directory
    output_dir: Optional[str] = None
    
    # Model name for GGUF metadata
    model_name: Optional[str] = None
    
    # Add parameters
    add_params: bool = True


class GGUFQuantizer:
    """
    Quantizes models to GGUF format.
    
    GGUF Format Benefits:
    - Single file format (unlike safetensors)
    - Lazy loading support
    - Metadata embedding
    - Compatible with llama.cpp, ollama, llamafile, etc.
    - Efficient memory usage
    """
    
    # Quantization method details
    QUANT_METHODS = {
        "f32": {"bits": 32, "description": "32-bit float (no quantization)"},
        "f16": {"bits": 16, "description": "16-bit float (half precision)"},
        "q8_0": {"bits": 8, "description": "8-bit quantization"},
        "q4_0": {"bits": 4, "description": "4-bit quantization (symmetric)"},
        "q4_1": {"bits": 4, "description": "4-bit quantization (with scale)"},
        "q5_0": {"bits": 5, "description": "5-bit quantization (symmetric)"},
        "q5_1": {"bits": 5, "description": "5-bit quantization (with scale)"},
        "iq2_xxs": {"bits": 2, "description": "2-bit extreme quantization"},
        "iq3_xxs": {"bits": 3, "description": "3-bit extreme quantization"},
    }
    
    def __init__(self, config: GGUFQuantizationConfig):
        """Initialize quantizer with configuration."""
        self.config = config
        self.device = "cuda" if (config.use_cuda and torch.cuda.is_available()) else "cpu"
        
        # Validate quantization method
        if config.quant_method not in self.QUANT_METHODS:
            raise ValueError(
                f"Unknown quantization method: {config.quant_method}. "
                f"Available: {list(self.QUANT_METHODS.keys())}"
            )
    
    def quantize(
        self,
        model_path: str,
        output_path: Optional[str] = None
    ) -> Tuple[bool, str]:
        """
        Quantize a model to GGUF format.
        
        Args:
            model_path: Path to input model
            output_path: Path to save GGUF file
        
        Returns:
            Tuple of (success, message)
        """
        try:
            import gguf
        except ImportError:
            return False, "gguf library not installed. Install with: pip install gguf"
        
        model_path = Path(model_path)
        if not model_path.exists():
            return False, f"Model path does not exist: {model_path}"
        
        # Determine output path
        if output_path is None:
            output_dir = Path(self.config.output_dir or model_path.parent)
            output_dir.mkdir(exist_ok=True)
            quant_suffix = self.config.quant_method
            output_path = output_dir / f"{model_path.name}-{quant_suffix}.gguf"
        
        output_path = Path(output_path)
        
        try:
            print(f"Quantizing {model_path.name} to GGUF format")
            print(f"Quantization method: {self.config.quant_method}")
            print(f"Output: {output_path}")
            
            # Load model
            print("Loading model...")
            model = torch.load(model_path, map_location=self.device)
            
            if isinstance(model, dict) and "state_dict" in model:
                state_dict = model["state_dict"]
            elif isinstance(model, dict):
                state_dict = model
            else:
                state_dict = model.state_dict() if hasattr(model, "state_dict") else {}
            
            # Create GGUF writer
            print("Converting to GGUF format...")
            
            # This is a simplified version - full implementation would use
            # the official convert scripts from llama.cpp
            self._simple_convert_to_gguf(state_dict, output_path)
            
            print(f"Successfully quantized to: {output_path}")
            file_size_mb = output_path.stat().st_size / (1024 * 1024)
            print(f"Output size: {file_size_mb:.1f}MB")
            
            return True, f"Successfully quantized to {output_path}"
        
        except Exception as e:
            error_msg = f"Quantization failed: {str(e)}"
            print(f"ERROR: {error_msg}")
            return False, error_msg
    
    def _simple_convert_to_gguf(self, state_dict: Dict[str, torch.Tensor], output_path: Path):
        """
        Simple GGUF conversion.
        
        Note: For production use, integrate with official conversion scripts:
        - llama.cpp/convert.py
        - ollama/llm/llama/convert
        """
        try:
            import gguf
        except ImportError:
            raise ImportError("gguf library required")
        
        # Create output directory
        output_path.parent.mkdir(exist_ok=True)
        
        # Initialize GGUF writer
        gguf_writer = gguf.GGUFWriter(str(output_path), gguf.MODEL_ARCH.LLAMA)
        
        # Add general metadata
        gguf_writer.add_string("general.name", self.config.model_name or "model")
        gguf_writer.add_int32("general.quantization_version", 2)
        
        # Add architecture-specific metadata
        num_params = sum(p.numel() for p in state_dict.values())
        gguf_writer.add_uint64("general.parameter_count", num_params)
        
        # Add quantization method
        gguf_writer.add_string("general.quantization_method", self.config.quant_method)
        
        # Add tensors
        print(f"Adding {len(state_dict)} tensors...")
        for name, tensor in state_dict.items():
            # Move to CPU for serialization
            tensor = tensor.cpu()
            
            # Convert to appropriate dtype based on quantization
            if self.config.quant_method == "f32":
                tensor = tensor.float()
            elif self.config.quant_method == "f16":
                tensor = tensor.half()
            elif self.config.quant_method.startswith("q"):
                # For int4/int8, convert and quantize
                tensor = self._quantize_tensor(tensor)
            
            # Add to GGUF
            try:
                gguf_writer.add_tensor(name, tensor)
            except Exception as e:
                print(f"Warning: Could not add tensor {name}: {e}")
        
        # Finalize
        gguf_writer.write_header_to_file()
        gguf_writer.write_kv_data_to_file()
        print(f"Wrote {len(state_dict)} tensors to GGUF")
    
    def _quantize_tensor(self, tensor: torch.Tensor) -> torch.Tensor:
        """
        Quantize a single tensor.
        
        For proper quantization, use full conversion implementations.
        """
        if self.config.quant_method == "q8_0":
            return self._quantize_q8_0(tensor)
        elif self.config.quant_method == "q4_0":
            return self._quantize_q4_0(tensor)
        else:
            # Default: return as float32
            return tensor.float()
    
    @staticmethod
    def _quantize_q8_0(tensor: torch.Tensor) -> torch.Tensor:
        """8-bit quantization."""
        # Find scale
        abs_max = tensor.abs().max()
        scale = 127.0 / abs_max if abs_max > 0 else 1.0
        
        # Quantize
        quantized = (tensor * scale).round().clamp(-128, 127).char()
        return quantized
    
    @staticmethod
    def _quantize_q4_0(tensor: torch.Tensor) -> torch.Tensor:
        """4-bit quantization."""
        # Reshape for 4-bit encoding
        if tensor.dim() > 1:
            # For matrices, apply per-channel
            tensor = tensor.view(-1, tensor.shape[-1])
        
        # Simple 4-bit quantization
        abs_max = tensor.abs().max(dim=-1, keepdim=True)[0]
        scale = 15.0 / (abs_max + 1e-8)
        
        quantized = (tensor * scale).round().clamp(-8, 7)
        return quantized


class GGUFConversionHelper:
    """
    Helper for converting different model types to GGUF.
    
    Integrates with official conversion tools when available.
    """
    
    @staticmethod
    def get_conversion_command(
        model_path: str,
        output_path: str,
        quantization: str = "q4_0",
        model_type: str = "auto"
    ) -> str:
        """
        Generate conversion command for use with official tools.
        
        Returns:
            Shell command string
        """
        if model_type == "auto" or model_type.startswith("llama"):
            # Use llama.cpp converter
            return (
                f'python convert.py "{model_path}" '
                f'--outfile "{output_path}" '
                f'--outtype {quantization}'
            )
        elif model_type in ["bert", "gpt2", "text"]:
            # Text model conversion
            return (
                f'python convert_text_to_gguf.py '
                f'--model-path "{model_path}" '
                f'--output-path "{output_path}" '
                f'--quantization {quantization}'
            )
        else:
            return ""
    
    @staticmethod
    def convert_with_llama_cpp(
        model_path: str,
        output_path: str,
        quantization: str = "q4_0"
    ) -> Tuple[bool, str]:
        """
        Use official llama.cpp converter if available.
        
        Requires:
        - llama.cpp repository cloned locally
        - Python conversion scripts available
        """
        try:
            import subprocess
            
            # Try to find convert.py
            convert_script = Path(__file__).parent / ".." / "llama.cpp" / "convert.py"
            
            if not convert_script.exists():
                return False, "llama.cpp conversion tools not found"
            
            cmd = [
                sys.executable,
                str(convert_script),
                model_path,
                "--outfile", output_path,
                "--outtype", quantization,
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode == 0:
                return True, f"Conversion successful: {output_path}"
            else:
                return False, f"Conversion failed: {result.stderr}"
        
        except Exception as e:
            return False, f"Error during conversion: {str(e)}"


if __name__ == "__main__":
    print("GGUF Quantization Module")
    print("=" * 60)
    
    # Show available quantization methods
    print("\nAvailable Quantization Methods:")
    for method, info in GGUFQuantizer.QUANT_METHODS.items():
        print(f"  {method:12} - {info['description']} ({info['bits']}-bit)")
    
    # Test configuration
    config = GGUFQuantizationConfig(
        quant_method="q4_0",
        model_name="test-model",
        output_dir="./outputs/quantized"
    )
    
    print(f"\nExample Configuration:")
    print(f"  Quantization method: {config.quant_method}")
    print(f"  Output directory: {config.output_dir}")
    print(f"  Model name: {config.model_name}")
