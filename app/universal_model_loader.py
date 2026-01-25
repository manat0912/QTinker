"""
Universal Model Loader supporting all model types:
- Text Models (BERT, GPT-2, LLaMA, etc.)
- Stable Diffusion (UNet, VAE, Text Encoder)
- Vision Models (ViT, CLIP, etc.)
- Audio Models (Whisper, etc.)
- Custom models (state_dict based)
- GGUF models

Cross-platform path detection for Pinokio environments.
"""

import os
import sys
import json
import torch
import torch.nn as nn
from pathlib import Path
from typing import Dict, Tuple, Optional, Any, Union
from abc import ABC, abstractmethod
import warnings

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))


class PinokioPathDetector:
    """Detects Pinokio root directory across different systems and drive letters."""
    
    @staticmethod
    def find_pinokio_root() -> Optional[Path]:
        """Find Pinokio root directory automatically."""
        # Check if PINOKIO_ROOT environment variable is set
        pinokio_root = os.getenv("PINOKIO_ROOT")
        if pinokio_root and os.path.isdir(pinokio_root):
            return Path(pinokio_root)
        
        # Check common Pinokio installation locations
        common_paths = [
            Path.home() / "pinokio",
            Path("C:/pinokio") if sys.platform == "win32" else Path("/pinokio"),
            Path("D:/pinokio") if sys.platform == "win32" else None,
            Path("E:/pinokio") if sys.platform == "win32" else None,
            # Check relative to current script
            Path(__file__).parent.parent.parent,
        ]
        
        for path in common_paths:
            if path and path.exists() and (path / "api").exists():
                return path
        
        # If not found, assume default structure
        return Path(__file__).parent.parent.parent
    
    @staticmethod
    def resolve_path(path_str: str) -> Path:
        """
        Resolve a path with Pinokio variables.
        
        Supports:
        - $PINOKIO_ROOT: Pinokio installation root
        - $PINOKIO_API: Pinokio API directory
        - $PROJECT_ROOT: Current project root
        - Absolute paths
        - Relative paths
        """
        if not path_str:
            return None
        
        pinokio_root = PinokioPathDetector.find_pinokio_root()
        
        # Replace Pinokio variables
        expanded = path_str.replace("$PINOKIO_ROOT", str(pinokio_root))
        expanded = expanded.replace("$PINOKIO_API", str(pinokio_root / "api"))
        expanded = expanded.replace("$PROJECT_ROOT", str(Path(__file__).parent.parent))
        
        # Convert to Path and resolve
        resolved = Path(expanded).expanduser()
        if not resolved.is_absolute():
            resolved = Path(__file__).parent.parent / resolved
        
        return resolved


class BaseModelLoader(ABC):
    """Abstract base class for different model types."""
    
    def __init__(self, model_path: Union[str, Path]):
        self.model_path = PinokioPathDetector.resolve_path(str(model_path))
        self.model = None
        self.tokenizer = None
        self.processor = None
        self.config = None
        self.model_type = None
    
    @abstractmethod
    def load(self) -> torch.nn.Module:
        """Load the model and return it."""
        pass
    
    def _detect_model_type(self) -> str:
        """Detect model type from directory structure."""
        if isinstance(self.model_path, str):
            self.model_path = Path(self.model_path)
        
        # Check for config files
        if (self.model_path / "config.json").exists():
            with open(self.model_path / "config.json") as f:
                config = json.load(f)
                return config.get("model_type", "unknown")
        
        if (self.model_path / "model_config.json").exists():
            with open(self.model_path / "model_config.json") as f:
                config = json.load(f)
                return config.get("_class_name", "unknown")
        
        return "unknown"
    
    def validate(self) -> bool:
        """Validate that model path exists and contains required files."""
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model path does not exist: {self.model_path}")
        return True


class HuggingFaceModelLoader(BaseModelLoader):
    """Loader for HuggingFace transformers models (BERT, GPT-2, LLaMA, etc.)."""
    
    def load(self) -> torch.nn.Module:
        """Load HuggingFace model with auto-detection."""
        try:
            from transformers import (
                AutoModel, AutoModelForCausalLM, AutoModelForMaskedLM,
                AutoTokenizer, AutoConfig
            )
        except ImportError:
            raise ImportError("transformers library is required. Install with: pip install transformers")
        
        print(f"Loading HuggingFace model from: {self.model_path}")
        
        # Load config
        self.config = AutoConfig.from_pretrained(str(self.model_path))
        self.model_type = self.config.model_type
        
        # Load tokenizer
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(str(self.model_path))
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
        except Exception as e:
            print(f"Warning: Could not load tokenizer: {e}")
            self.tokenizer = None
        
        # Detect model architecture and load appropriately
        if any(x in self.model_type.lower() for x in ["bert", "roberta", "distilbert", "albert"]):
            self.model = AutoModelForMaskedLM.from_pretrained(str(self.model_path))
        elif any(x in self.model_type.lower() for x in ["gpt", "llama", "mistral"]):
            self.model = AutoModelForCausalLM.from_pretrained(str(self.model_path))
        else:
            self.model = AutoModel.from_pretrained(str(self.model_path))
        
        print(f"Successfully loaded {self.model_type} model")
        return self.model


class StableDiffusionModelLoader(BaseModelLoader):
    """Loader for Stable Diffusion models (UNet, VAE, Text Encoder, full pipeline)."""
    
    def load(self, component: str = "unet") -> torch.nn.Module:
        """
        Load Stable Diffusion component.
        
        Args:
            component: Which component to load:
                - "unet": Load UNet2D model
                - "vae": Load VAE encoder/decoder
                - "text_encoder": Load text encoder (CLIP)
                - "full": Load full pipeline
        """
        try:
            from diffusers import (
                UNet2DConditionModel, AutoencoderKL, CLIPTextModel,
                StableDiffusionPipeline
            )
        except ImportError:
            raise ImportError("diffusers library is required. Install with: pip install diffusers")
        
        print(f"Loading Stable Diffusion {component} from: {self.model_path}")
        
        try:
            if component == "unet":
                self.model = UNet2DConditionModel.from_pretrained(
                    str(self.model_path), subfolder="unet"
                )
                self.model_type = "unet_2d_condition"
            
            elif component == "vae":
                self.model = AutoencoderKL.from_pretrained(
                    str(self.model_path), subfolder="vae"
                )
                self.model_type = "autoencoder_kl"
            
            elif component == "text_encoder":
                self.model = CLIPTextModel.from_pretrained(
                    str(self.model_path), subfolder="text_encoder"
                )
                self.model_type = "clip_text_model"
            
            elif component == "full":
                self.model = StableDiffusionPipeline.from_pretrained(
                    str(self.model_path)
                )
                self.model_type = "stable_diffusion_pipeline"
            
            else:
                raise ValueError(f"Unknown component: {component}")
            
            print(f"Successfully loaded Stable Diffusion {component}")
            return self.model
        
        except Exception as e:
            print(f"Failed to load SD {component}: {e}")
            # Try loading from state_dict if model repo structure
            return self._load_from_state_dict(component)
    
    def _load_from_state_dict(self, component: str) -> torch.nn.Module:
        """Load from raw state_dict if standard loading fails."""
        print(f"Attempting to load {component} from state_dict...")
        
        state_dict_path = None
        if component == "unet":
            state_dict_path = self.model_path / "unet" / "diffusion_pytorch_model.bin"
            if not state_dict_path.exists():
                state_dict_path = self.model_path / "pytorch_model.bin"
        
        if state_dict_path and state_dict_path.exists():
            state_dict = torch.load(state_dict_path, map_location="cpu")
            print(f"Loaded state_dict keys: {list(state_dict.keys())[:5]}...")
            return state_dict
        
        raise RuntimeError(f"Could not load {component} from {self.model_path}")


class VisionModelLoader(BaseModelLoader):
    """Loader for Vision models (ViT, CLIP, DINOv2, etc.)."""
    
    def load(self) -> torch.nn.Module:
        """Load vision model."""
        try:
            from transformers import AutoModel, AutoProcessor, AutoImageProcessor
        except ImportError:
            raise ImportError("transformers library is required")
        
        print(f"Loading Vision model from: {self.model_path}")
        
        self.config = AutoConfig.from_pretrained(str(self.model_path))
        self.model = AutoModel.from_pretrained(str(self.model_path))
        
        # Try to load processor
        try:
            self.processor = AutoImageProcessor.from_pretrained(str(self.model_path))
        except:
            try:
                self.processor = AutoProcessor.from_pretrained(str(self.model_path))
            except:
                print("Warning: Could not load processor")
        
        print(f"Successfully loaded Vision model")
        return self.model


class AudioModelLoader(BaseModelLoader):
    """Loader for Audio models (Whisper, Wav2Vec, etc.)."""
    
    def load(self) -> torch.nn.Module:
        """Load audio model."""
        try:
            from transformers import AutoModel, AutoProcessor, AutoFeatureExtractor
        except ImportError:
            raise ImportError("transformers library is required")
        
        print(f"Loading Audio model from: {self.model_path}")
        
        self.config = AutoConfig.from_pretrained(str(self.model_path))
        self.model = AutoModel.from_pretrained(str(self.model_path))
        
        # Try to load processor
        try:
            self.processor = AutoProcessor.from_pretrained(str(self.model_path))
        except:
            try:
                self.processor = AutoFeatureExtractor.from_pretrained(str(self.model_path))
            except:
                print("Warning: Could not load processor")
        
        print(f"Successfully loaded Audio model")
        return self.model


class GGUFModelLoader(BaseModelLoader):
    """Loader for GGUF quantized models."""
    
    def load(self) -> Dict[str, Any]:
        """
        Load GGUF model metadata and handle appropriately.
        
        Returns:
            Dictionary with GGUF metadata and path reference
        """
        try:
            import gguf
        except ImportError:
            raise ImportError(
                "gguf library is required. Install with: "
                "pip install gguf or pip install llama-cpp-python"
            )
        
        # Find GGUF file
        gguf_files = list(self.model_path.glob("*.gguf"))
        if not gguf_files:
            raise FileNotFoundError(f"No GGUF files found in {self.model_path}")
        
        gguf_path = gguf_files[0]
        print(f"Loading GGUF model from: {gguf_path}")
        
        # Read GGUF metadata
        try:
            reader = gguf.GGUFReader(str(gguf_path))
            self.config = {
                "model_name": reader.fields.get("general.name", "unknown"),
                "architecture": reader.fields.get("general.architecture", "unknown"),
                "file_size": gguf_path.stat().st_size,
                "parameters": reader.fields.get("general.parameter_count", 0),
            }
        except Exception as e:
            print(f"Warning: Could not read GGUF metadata: {e}")
            self.config = {"file_path": str(gguf_path)}
        
        self.model_type = "gguf"
        self.model = {
            "type": "gguf",
            "path": str(gguf_path),
            "metadata": self.config
        }
        
        print(f"Successfully loaded GGUF model: {gguf_path.name}")
        return self.model


class CustomStateDictModelLoader(BaseModelLoader):
    """Loader for custom models stored as state_dict."""
    
    def load(self) -> Dict[str, torch.Tensor]:
        """Load model from state_dict file."""
        # Find state_dict file
        state_dict_files = []
        for pattern in ["*.bin", "*.pt", "*.pth", "*.safetensors"]:
            state_dict_files.extend(self.model_path.glob(pattern))
        
        if not state_dict_files:
            raise FileNotFoundError(f"No model files found in {self.model_path}")
        
        state_dict_path = state_dict_files[0]
        print(f"Loading state_dict from: {state_dict_path}")
        
        if state_dict_path.suffix == ".safetensors":
            try:
                from safetensors.torch import load_file
                state_dict = load_file(str(state_dict_path))
            except ImportError:
                raise ImportError("safetensors library is required for .safetensors files")
        else:
            state_dict = torch.load(str(state_dict_path), map_location="cpu")
        
        self.model_type = "state_dict"
        self.model = state_dict
        
        print(f"Successfully loaded state_dict with {len(state_dict)} keys")
        return self.model


class UniversalModelLoader:
    """
    Universal model loader that automatically detects and loads any model type.
    """
    
    LOADERS = {
        "text": HuggingFaceModelLoader,
        "stable_diffusion": StableDiffusionModelLoader,
        "vision": VisionModelLoader,
        "audio": AudioModelLoader,
        "gguf": GGUFModelLoader,
        "state_dict": CustomStateDictModelLoader,
    }
    
    @staticmethod
    def detect_model_type(model_path: Union[str, Path]) -> str:
        """
        Detect model type from directory structure and files.
        
        Returns:
            One of: "text", "stable_diffusion", "vision", "audio", "gguf", "state_dict"
        """
        model_path = Path(model_path)
        
        # Check for GGUF files
        if list(model_path.glob("*.gguf")):
            return "gguf"
        
        # Check for Stable Diffusion structure
        if (model_path / "unet" / "config.json").exists() or \
           (model_path / "vae" / "config.json").exists() or \
           (model_path / "text_encoder" / "config.json").exists():
            return "stable_diffusion"
        
        # Check for config.json to determine architecture
        if (model_path / "config.json").exists():
            try:
                with open(model_path / "config.json") as f:
                    config = json.load(f)
                    model_type = config.get("model_type", "").lower()
                    
                    if any(x in model_type for x in ["bert", "gpt", "llama", "mistral", "roberta"]):
                        return "text"
                    elif any(x in model_type for x in ["vit", "dino", "clip"]):
                        return "vision"
                    elif any(x in model_type for x in ["wav2vec", "whisper", "hubert"]):
                        return "audio"
            except:
                pass
        
        # Default to state_dict if files exist
        if any(model_path.glob("*.bin")) or any(model_path.glob("*.safetensors")):
            return "state_dict"
        
        return "state_dict"
    
    @staticmethod
    def load(
        model_path: Union[str, Path],
        model_type: Optional[str] = None,
        device: str = "cpu",
        **kwargs
    ) -> Tuple[Any, Optional[Any]]:
        """
        Load any model with automatic type detection.
        
        Args:
            model_path: Path to model directory
            model_type: Optional explicit model type (auto-detected if None)
            device: Device to load model on ("cpu", "cuda", etc.)
            **kwargs: Additional arguments for specific loaders
        
        Returns:
            Tuple of (model, tokenizer/processor or None)
        """
        # Detect model type if not provided
        if model_type is None:
            model_type = UniversalModelLoader.detect_model_type(model_path)
        
        print(f"Detected model type: {model_type}")
        
        # Select appropriate loader
        loader_class = UniversalModelLoader.LOADERS.get(model_type)
        if not loader_class:
            raise ValueError(f"Unknown model type: {model_type}")
        
        # Instantiate loader
        loader = loader_class(model_path)
        
        # Load model
        if model_type == "stable_diffusion":
            component = kwargs.get("component", "unet")
            model = loader.load(component=component)
        else:
            model = loader.load()
        
        # Move to device if it's a torch module
        if isinstance(model, nn.Module):
            model = model.to(device)
        
        # Return model and tokenizer/processor
        return model, loader.tokenizer or loader.processor


class ModelLoaderPipeline:
    """Pipeline for loading teacher, student, and custom models."""
    
    def __init__(
        self,
        teacher_path: str,
        student_path: str,
        custom_path: Optional[str] = None,
        device: str = "cpu"
    ):
        self.teacher_path = teacher_path
        self.student_path = student_path
        self.custom_path = custom_path
        self.device = device
        
        self.teacher_model = None
        self.teacher_tokenizer = None
        self.student_model = None
        self.student_tokenizer = None
        self.custom_model = None
        self.custom_tokenizer = None
    
    def load_all(self):
        """Load all models."""
        print("Loading teacher model...")
        self.teacher_model, self.teacher_tokenizer = UniversalModelLoader.load(
            self.teacher_path, device=self.device
        )
        
        print("Loading student model...")
        self.student_model, self.student_tokenizer = UniversalModelLoader.load(
            self.student_path, device=self.device
        )
        
        if self.custom_path:
            print("Loading custom model...")
            self.custom_model, self.custom_tokenizer = UniversalModelLoader.load(
                self.custom_path, device=self.device
            )
        
        print("All models loaded successfully!")
    
    def get_models(self) -> Dict[str, Tuple]:
        """Get loaded models and tokenizers."""
        return {
            "teacher": (self.teacher_model, self.teacher_tokenizer),
            "student": (self.student_model, self.student_tokenizer),
            "custom": (self.custom_model, self.custom_tokenizer),
        }


if __name__ == "__main__":
    # Test cross-platform path detection
    pinokio_root = PinokioPathDetector.find_pinokio_root()
    print(f"Detected Pinokio root: {pinokio_root}")
    
    # Test path resolution
    test_paths = [
        "$PINOKIO_ROOT/api/QTinker/app/bert_models",
        "$PINOKIO_API",
        "$PROJECT_ROOT",
    ]
    
    for path in test_paths:
        resolved = PinokioPathDetector.resolve_path(path)
        print(f"{path} -> {resolved}")
