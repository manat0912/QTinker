"""
Model Registry

Comprehensive registry of all supported model types and libraries.
Supports auto-discovery and categorization of models.
"""

from typing import Dict, List, Set, Optional
from dataclasses import dataclass, asdict
from enum import Enum
import json
from pathlib import Path


class ModelFramework(Enum):
    """Supported model frameworks."""
    PYTORCH = "pytorch"
    TENSORFLOW = "tensorflow"
    JAX = "jax"
    ONNX = "onnx"
    GGUF = "gguf"


class ModelCategory(Enum):
    """Model categories."""
    TEXT = "text"
    VISION = "vision"
    AUDIO = "audio"
    DIFFUSION = "diffusion"
    MULTIMODAL = "multimodal"
    REINFORCEMENT_LEARNING = "reinforcement_learning"
    TABULAR = "tabular"
    CUSTOM = "custom"


class ModelType(Enum):
    """Specific model types."""
    # Text Models
    BERT = "bert"
    GPT2 = "gpt2"
    GPT3 = "gpt3"
    LLAMA = "llama"
    MISTRAL = "mistral"
    QWEN = "qwen"
    CLAUDE = "claude"
    GEMINI = "gemini"
    DISTILBERT = "distilbert"
    ROBERTA = "roberta"
    ELECTRA = "electra"
    
    # Vision Models
    VIT = "vit"
    CLIP = "clip"
    DINOV2 = "dinov2"
    RESNET = "resnet"
    YOLO = "yolo"
    EFFICIENTNET = "efficientnet"
    
    # Audio Models
    WHISPER = "whisper"
    WAV2VEC = "wav2vec"
    HUBERT = "hubert"
    WAVENET = "wavenet"
    
    # Diffusion Models
    STABLE_DIFFUSION = "stable_diffusion"
    UNET = "unet"
    VAE = "vae"
    CONTROLNET = "controlnet"
    LORA = "lora"
    INPAINTING = "inpainting"
    
    # Multimodal
    BLIP = "blip"
    FLAMINGO = "flamingo"
    GPT4_VISION = "gpt4_vision"
    
    # Custom
    CUSTOM = "custom"


@dataclass
class ModelLibraryEntry:
    """Entry for a model in the registry."""
    name: str
    model_type: ModelType
    category: ModelCategory
    frameworks: List[ModelFramework]
    description: str
    url: Optional[str] = None
    requires_auth: bool = False
    supports_quantization: bool = True
    supports_distillation: bool = True
    min_disk_space_gb: float = 0.5
    recommended_vram_gb: Optional[float] = None


class ModelRegistry:
    """
    Comprehensive registry of all supported models.
    
    Organized by:
    - Framework (PyTorch, TensorFlow, JAX, ONNX, GGUF)
    - Category (Text, Vision, Audio, Diffusion, etc.)
    - Type (specific model architectures)
    """
    
    def __init__(self):
        """Initialize the registry with all supported models."""
        self.entries: Dict[str, ModelLibraryEntry] = {}
        self._build_registry()
    
    def _build_registry(self):
        """Build the complete model registry."""
        
        # ===== TEXT MODELS =====
        text_models = [
            ModelLibraryEntry(
                name="BERT",
                model_type=ModelType.BERT,
                category=ModelCategory.TEXT,
                frameworks=[ModelFramework.PYTORCH, ModelFramework.TENSORFLOW],
                description="Bidirectional Encoder Representations from Transformers",
                url="https://huggingface.co/google-bert",
                supports_quantization=True,
                supports_distillation=True,
                min_disk_space_gb=0.3,
                recommended_vram_gb=2.0
            ),
            ModelLibraryEntry(
                name="DistilBERT",
                model_type=ModelType.DISTILBERT,
                category=ModelCategory.TEXT,
                frameworks=[ModelFramework.PYTORCH, ModelFramework.TENSORFLOW],
                description="Distilled BERT (40% smaller, 60% faster)",
                url="https://huggingface.co/distilbert",
                supports_quantization=True,
                supports_distillation=True,
                min_disk_space_gb=0.2,
                recommended_vram_gb=1.5
            ),
            ModelLibraryEntry(
                name="RoBERTa",
                model_type=ModelType.ROBERTA,
                category=ModelCategory.TEXT,
                frameworks=[ModelFramework.PYTORCH, ModelFramework.TENSORFLOW],
                description="Robustly Optimized BERT Approach",
                url="https://huggingface.co/roberta",
                supports_quantization=True,
                supports_distillation=True,
                min_disk_space_gb=0.3,
                recommended_vram_gb=2.0
            ),
            ModelLibraryEntry(
                name="GPT-2",
                model_type=ModelType.GPT2,
                category=ModelCategory.TEXT,
                frameworks=[ModelFramework.PYTORCH, ModelFramework.TENSORFLOW],
                description="Generative Pre-trained Transformer 2",
                url="https://huggingface.co/gpt2",
                supports_quantization=True,
                supports_distillation=True,
                min_disk_space_gb=0.5,
                recommended_vram_gb=2.0
            ),
            ModelLibraryEntry(
                name="LLaMA",
                model_type=ModelType.LLAMA,
                category=ModelCategory.TEXT,
                frameworks=[ModelFramework.PYTORCH, ModelFramework.GGUF],
                description="Large Language Model Meta AI",
                url="https://huggingface.co/meta-llama",
                requires_auth=False,
                supports_quantization=True,
                supports_distillation=True,
                min_disk_space_gb=3.5,
                recommended_vram_gb=8.0
            ),
            ModelLibraryEntry(
                name="Mistral",
                model_type=ModelType.MISTRAL,
                category=ModelCategory.TEXT,
                frameworks=[ModelFramework.PYTORCH, ModelFramework.GGUF],
                description="Mistral 7B Language Model",
                url="https://huggingface.co/mistralai",
                supports_quantization=True,
                supports_distillation=True,
                min_disk_space_gb=3.5,
                recommended_vram_gb=8.0
            ),
            ModelLibraryEntry(
                name="Qwen",
                model_type=ModelType.QWEN,
                category=ModelCategory.TEXT,
                frameworks=[ModelFramework.PYTORCH, ModelFramework.GGUF],
                description="Qwen Series Language Models",
                url="https://huggingface.co/Qwen",
                supports_quantization=True,
                supports_distillation=True,
                min_disk_space_gb=3.5,
                recommended_vram_gb=8.0
            ),
        ]
        
        # ===== VISION MODELS =====
        vision_models = [
            ModelLibraryEntry(
                name="Vision Transformer (ViT)",
                model_type=ModelType.VIT,
                category=ModelCategory.VISION,
                frameworks=[ModelFramework.PYTORCH, ModelFramework.TENSORFLOW],
                description="Transformer architecture for image classification",
                url="https://huggingface.co/google/vit",
                supports_quantization=True,
                supports_distillation=True,
                min_disk_space_gb=0.3,
                recommended_vram_gb=4.0
            ),
            ModelLibraryEntry(
                name="CLIP",
                model_type=ModelType.CLIP,
                category=ModelCategory.MULTIMODAL,
                frameworks=[ModelFramework.PYTORCH],
                description="Contrastive Language-Image Pre-training",
                url="https://huggingface.co/openai/clip",
                supports_quantization=True,
                supports_distillation=True,
                min_disk_space_gb=0.5,
                recommended_vram_gb=4.0
            ),
            ModelLibraryEntry(
                name="DINOv2",
                model_type=ModelType.DINOV2,
                category=ModelCategory.VISION,
                frameworks=[ModelFramework.PYTORCH],
                description="Self-supervised vision transformer",
                url="https://huggingface.co/facebook/dinov2",
                supports_quantization=True,
                supports_distillation=True,
                min_disk_space_gb=0.3,
                recommended_vram_gb=4.0
            ),
            ModelLibraryEntry(
                name="ResNet",
                model_type=ModelType.RESNET,
                category=ModelCategory.VISION,
                frameworks=[ModelFramework.PYTORCH, ModelFramework.TENSORFLOW],
                description="Deep Residual Networks",
                url="https://pytorch.org/vision",
                supports_quantization=True,
                supports_distillation=True,
                min_disk_space_gb=0.1,
                recommended_vram_gb=2.0
            ),
        ]
        
        # ===== AUDIO MODELS =====
        audio_models = [
            ModelLibraryEntry(
                name="Whisper",
                model_type=ModelType.WHISPER,
                category=ModelCategory.AUDIO,
                frameworks=[ModelFramework.PYTORCH],
                description="Robust Speech Recognition",
                url="https://huggingface.co/openai/whisper",
                supports_quantization=True,
                supports_distillation=True,
                min_disk_space_gb=1.5,
                recommended_vram_gb=4.0
            ),
            ModelLibraryEntry(
                name="Wav2Vec 2.0",
                model_type=ModelType.WAV2VEC,
                category=ModelCategory.AUDIO,
                frameworks=[ModelFramework.PYTORCH],
                description="Self-supervised speech representation",
                url="https://huggingface.co/facebook/wav2vec2",
                supports_quantization=True,
                supports_distillation=True,
                min_disk_space_gb=0.3,
                recommended_vram_gb=4.0
            ),
        ]
        
        # ===== STABLE DIFFUSION =====
        diffusion_models = [
            ModelLibraryEntry(
                name="Stable Diffusion v1.5",
                model_type=ModelType.STABLE_DIFFUSION,
                category=ModelCategory.DIFFUSION,
                frameworks=[ModelFramework.PYTORCH],
                description="Text-to-image diffusion model",
                url="https://huggingface.co/runwayml/stable-diffusion-v1-5",
                supports_quantization=True,
                supports_distillation=True,
                min_disk_space_gb=3.5,
                recommended_vram_gb=6.0
            ),
            ModelLibraryEntry(
                name="Stable Diffusion XL",
                model_type=ModelType.STABLE_DIFFUSION,
                category=ModelCategory.DIFFUSION,
                frameworks=[ModelFramework.PYTORCH],
                description="Latest Stable Diffusion model with improved quality",
                url="https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0",
                supports_quantization=True,
                supports_distillation=True,
                min_disk_space_gb=6.0,
                recommended_vram_gb=10.0
            ),
            ModelLibraryEntry(
                name="UNet2D",
                model_type=ModelType.UNET,
                category=ModelCategory.DIFFUSION,
                frameworks=[ModelFramework.PYTORCH],
                description="2D UNet for diffusion models",
                url="https://huggingface.co/docs/diffusers",
                supports_quantization=True,
                supports_distillation=True,
                min_disk_space_gb=1.0,
                recommended_vram_gb=4.0
            ),
            ModelLibraryEntry(
                name="ControlNet",
                model_type=ModelType.CONTROLNET,
                category=ModelCategory.DIFFUSION,
                frameworks=[ModelFramework.PYTORCH],
                description="Spatial control for diffusion models",
                url="https://huggingface.co/lllyasviel/ControlNet",
                supports_quantization=True,
                supports_distillation=True,
                min_disk_space_gb=1.5,
                recommended_vram_gb=6.0
            ),
        ]
        
        # Register all models
        all_models = text_models + vision_models + audio_models + diffusion_models
        for model in all_models:
            self.entries[f"{model.model_type.value}"] = model
    
    def get_by_category(self, category: ModelCategory) -> List[ModelLibraryEntry]:
        """Get all models in a category."""
        return [
            model for model in self.entries.values()
            if model.category == category
        ]
    
    def get_by_framework(self, framework: ModelFramework) -> List[ModelLibraryEntry]:
        """Get all models supporting a framework."""
        return [
            model for model in self.entries.values()
            if framework in model.frameworks
        ]
    
    def get_quantizable_models(self) -> List[ModelLibraryEntry]:
        """Get all models that support quantization."""
        return [
            model for model in self.entries.values()
            if model.supports_quantization
        ]
    
    def get_distillable_models(self) -> List[ModelLibraryEntry]:
        """Get all models that support distillation."""
        return [
            model for model in self.entries.values()
            if model.supports_distillation
        ]
    
    def to_dict(self) -> Dict:
        """Convert registry to dictionary."""
        result = {}
        for key, entry in self.entries.items():
            result[key] = {
                **asdict(entry),
                "model_type": entry.model_type.value,
                "category": entry.category.value,
                "frameworks": [fw.value for fw in entry.frameworks]
            }
        return result
    
    def to_json(self, path: Optional[str] = None) -> str:
        """Convert registry to JSON."""
        data = self.to_dict()
        json_str = json.dumps(data, indent=2)
        
        if path:
            Path(path).write_text(json_str)
        
        return json_str


# Global registry instance
_registry = None


def get_registry() -> ModelRegistry:
    """Get or create the global model registry."""
    global _registry
    if _registry is None:
        _registry = ModelRegistry()
    return _registry


if __name__ == "__main__":
    print("Model Registry")
    print("=" * 60)
    
    registry = get_registry()
    
    # Show statistics
    print(f"\nTotal models: {len(registry.entries)}")
    
    # By category
    print("\nModels by Category:")
    for category in ModelCategory:
        models = registry.get_by_category(category)
        if models:
            print(f"  {category.value}: {len(models)} models")
            for model in models:
                print(f"    - {model.name}")
    
    # Quantizable models
    quantizable = registry.get_quantizable_models()
    print(f"\nQuantizable models: {len(quantizable)}")
    
    # Distillable models
    distillable = registry.get_distillable_models()
    print(f"Distillable models: {len(distillable)}")
