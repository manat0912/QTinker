"""
Enhanced cross-platform file browser for model selection.
Supports:
- Complete directory tree display
- Filter by file type
- Pinokio path variables
- Cross-platform compatibility
"""

import os
import json
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
import sys

sys.path.insert(0, str(Path(__file__).parent))
from universal_model_loader import PinokioPathDetector


@dataclass
class FileInfo:
    """Information about a file or directory."""
    path: Path
    name: str
    is_dir: bool
    size: Optional[int] = None
    model_type: Optional[str] = None
    contains_model: bool = False
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "path": str(self.path),
            "name": self.name,
            "is_dir": self.is_dir,
            "size": self.size,
            "model_type": self.model_type,
            "contains_model": self.contains_model,
        }


class EnhancedFileBrowser:
    """Enhanced file browser with model detection and filtering."""
    
    # Model file extensions
    MODEL_EXTENSIONS = {
        "torch": [".bin", ".pt", ".pth"],
        "safetensors": [".safetensors"],
        "gguf": [".gguf"],
        "config": ["config.json", ".json"],
    }
    
    # Directories to skip
    SKIP_DIRS = {
        "__pycache__", ".git", ".venv", "env", "node_modules",
        ".cache", "cache", "__MACOSX", ".pytest_cache"
    }
    
    def __init__(self, root_path: Optional[str] = None, max_depth: int = 5):
        """
        Initialize browser.
        
        Args:
            root_path: Root directory to browse (defaults to Pinokio API)
            max_depth: Maximum directory depth to scan
        """
        if root_path is None:
            pinokio_root = PinokioPathDetector.find_pinokio_root()
            root_path = str(pinokio_root / "api")
        
        self.root_path = PinokioPathDetector.resolve_path(root_path)
        self.max_depth = max_depth
        self._cache = {}
    
    def get_directory_tree(
        self,
        path: Optional[str] = None,
        depth: int = 0,
        show_hidden: bool = False
    ) -> Dict:
        """
        Get complete directory tree structure.
        
        Args:
            path: Specific path to scan (defaults to root)
            depth: Current depth in recursion
            show_hidden: Whether to show hidden files
        
        Returns:
            Dictionary with directory structure
        """
        if path is None:
            path = str(self.root_path)
        
        target_path = PinokioPathDetector.resolve_path(path)
        
        if not target_path.exists():
            return {"error": f"Path does not exist: {path}"}
        
        tree = {
            "name": target_path.name or str(target_path),
            "path": str(target_path),
            "is_dir": target_path.is_dir(),
            "children": []
        }
        
        if not target_path.is_dir():
            return tree
        
        # Check if depth limit reached
        if depth >= self.max_depth:
            tree["truncated"] = True
            return tree
        
        try:
            items = sorted(target_path.iterdir())
        except (PermissionError, OSError) as e:
            tree["error"] = str(e)
            return tree
        
        for item in items:
            # Skip hidden files unless requested
            if not show_hidden and item.name.startswith("."):
                continue
            
            # Skip certain directories
            if item.is_dir() and item.name in self.SKIP_DIRS:
                continue
            
            if item.is_dir():
                # Recurse into subdirectories
                child = self.get_directory_tree(
                    str(item), depth + 1, show_hidden
                )
            else:
                # Create file entry
                child = {
                    "name": item.name,
                    "path": str(item),
                    "is_dir": False,
                    "size": item.stat().st_size if item.exists() else 0,
                }
                
                # Check if it's a model file
                if self._is_model_file(item):
                    child["is_model"] = True
            
            tree["children"].append(child)
        
        return tree
    
    def get_flat_file_list(
        self,
        path: Optional[str] = None,
        include_dirs: bool = True,
        model_type: Optional[str] = None,
        search_term: Optional[str] = None
    ) -> List[FileInfo]:
        """
        Get flat list of files and directories.
        
        Args:
            path: Path to scan
            include_dirs: Whether to include directories
            model_type: Filter by model type
            search_term: Filter by name containing term
        
        Returns:
            List of FileInfo objects
        """
        if path is None:
            path = str(self.root_path)
        
        target_path = PinokioPathDetector.resolve_path(path)
        results = []
        
        try:
            for item in target_path.rglob("*"):
                # Skip certain patterns
                if any(skip in item.parts for skip in self.SKIP_DIRS):
                    continue
                
                if item.is_dir():
                    if not include_dirs:
                        continue
                    contains_model = self._dir_contains_model(item)
                    file_info = FileInfo(
                        path=item,
                        name=item.name,
                        is_dir=True,
                        contains_model=contains_model
                    )
                else:
                    is_model = self._is_model_file(item)
                    if model_type and model_type not in self._get_model_type(item):
                        continue
                    
                    file_info = FileInfo(
                        path=item,
                        name=item.name,
                        is_dir=False,
                        size=item.stat().st_size,
                        model_type=self._get_model_type(item),
                        contains_model=is_model
                    )
                
                # Apply search filter
                if search_term and search_term.lower() not in file_info.name.lower():
                    continue
                
                results.append(file_info)
        
        except (PermissionError, OSError) as e:
            print(f"Error scanning directory: {e}")
        
        return results
    
    def find_models(
        self,
        path: Optional[str] = None,
        model_type: Optional[str] = None
    ) -> List[Dict]:
        """
        Find all model directories in a path.
        
        Args:
            path: Root path to search
            model_type: Specific model type to find
        
        Returns:
            List of model directories with metadata
        """
        if path is None:
            path = str(self.root_path)
        
        target_path = PinokioPathDetector.resolve_path(path)
        models = []
        
        for item in target_path.rglob("*"):
            if not item.is_dir():
                continue
            
            # Check if this is a model directory
            model_info = self._get_model_info(item)
            if model_info:
                if model_type is None or model_type in model_info["types"]:
                    models.append(model_info)
        
        return sorted(models, key=lambda x: x["path"])
    
    def _is_model_file(self, path: Path) -> bool:
        """Check if a file is a model file."""
        if not path.is_file():
            return False
        
        name = path.name.lower()
        
        # Check for model extensions
        for extensions in self.MODEL_EXTENSIONS.values():
            if any(name.endswith(ext) for ext in extensions):
                return True
        
        return False
    
    def _get_model_type(self, path: Path) -> Optional[str]:
        """Determine model type from file extension."""
        name = path.name.lower()
        
        for model_type, extensions in self.MODEL_EXTENSIONS.items():
            if any(name.endswith(ext) for ext in extensions):
                return model_type
        
        return None
    
    def _dir_contains_model(self, path: Path) -> bool:
        """Check if directory contains model files."""
        try:
            for item in path.iterdir():
                if self._is_model_file(item):
                    return True
        except (PermissionError, OSError):
            pass
        
        return False
    
    def _get_model_info(self, path: Path) -> Optional[Dict]:
        """
        Get information about a model directory.
        
        Returns:
            Dictionary with model info or None
        """
        try:
            config_path = path / "config.json"
            model_config_path = path / "model_config.json"
            
            model_types = set()
            model_name = path.name
            
            # Check for standard config
            if config_path.exists():
                try:
                    with open(config_path) as f:
                        config = json.load(f)
                        model_type = config.get("model_type", "").lower()
                        if model_type:
                            model_types.add(model_type)
                        model_name = config.get("architectures", [model_name])[0]
                except:
                    pass
            
            # Check for Stable Diffusion config
            if model_config_path.exists():
                try:
                    with open(model_config_path) as f:
                        config = json.load(f)
                        class_name = config.get("_class_name", "").lower()
                        if class_name:
                            model_types.add(class_name)
                except:
                    pass
            
            # Check for model files
            has_model_files = self._dir_contains_model(path)
            
            if model_types or has_model_files:
                return {
                    "path": str(path),
                    "name": model_name,
                    "types": list(model_types) or ["unknown"],
                    "has_config": config_path.exists() or model_config_path.exists(),
                    "size": self._get_dir_size(path)
                }
        
        except Exception as e:
            print(f"Error getting model info for {path}: {e}")
        
        return None
    
    @staticmethod
    def _get_dir_size(path: Path) -> int:
        """Calculate directory size in bytes."""
        try:
            total = 0
            for item in path.rglob("*"):
                if item.is_file():
                    total += item.stat().st_size
            return total
        except:
            return 0


class ModelPathSelector:
    """Helper class for selecting teacher/student model paths."""
    
    @staticmethod
    def get_bert_models_path() -> Path:
        """Get the bert_models directory path."""
        pinokio_root = PinokioPathDetector.find_pinokio_root()
        bert_models = pinokio_root / "api" / "QTinker" / "app" / "bert_models"
        return bert_models
    
    @staticmethod
    def get_default_paths() -> Dict[str, Path]:
        """Get default paths for teacher and student models."""
        bert_models = ModelPathSelector.get_bert_models_path()
        
        return {
            "teacher_root": bert_models,
            "student_root": bert_models,
            "custom_root": PinokioPathDetector.find_pinokio_root() / "api",
        }
    
    @staticmethod
    def browse_models(
        path: Optional[str] = None,
        model_type: Optional[str] = None
    ) -> List[Dict]:
        """
        Browse available models.
        
        Args:
            path: Path to browse (defaults to bert_models)
            model_type: Filter by model type
        
        Returns:
            List of model directories
        """
        if path is None:
            path = str(ModelPathSelector.get_bert_models_path())
        
        browser = EnhancedFileBrowser(path)
        return browser.find_models(model_type=model_type)
    
    @staticmethod
    def validate_model_path(path: str) -> Tuple[bool, str]:
        """
        Validate that a path contains a valid model.
        
        Returns:
            Tuple of (is_valid, message)
        """
        resolved = PinokioPathDetector.resolve_path(path)
        
        if not resolved.exists():
            return False, f"Path does not exist: {resolved}"
        
        if not resolved.is_dir():
            return False, f"Path is not a directory: {resolved}"
        
        # Check for model files or config
        has_model = False
        for pattern in ["*.bin", "*.pt", "*.pth", "*.safetensors", "config.json"]:
            if list(resolved.glob(pattern)):
                has_model = True
                break
        
        if not has_model:
            return False, f"No model files found in: {resolved}"
        
        return True, "Valid model path"


if __name__ == "__main__":
    # Test browser
    print("=" * 60)
    print("Enhanced File Browser Test")
    print("=" * 60)
    
    # Test Pinokio path detection
    pinokio_root = PinokioPathDetector.find_pinokio_root()
    print(f"\nDetected Pinokio root: {pinokio_root}")
    
    # Test default paths
    default_paths = ModelPathSelector.get_default_paths()
    print(f"\nDefault paths:")
    for name, path in default_paths.items():
        print(f"  {name}: {path}")
        print(f"    Exists: {path.exists()}")
    
    # Test browsing models
    print(f"\nBrowsing bert_models:")
    models = ModelPathSelector.browse_models()
    for model in models[:5]:  # Show first 5
        print(f"  - {model['name']} ({model['types']})")
        print(f"    Path: {model['path']}")
        print(f"    Size: {model['size'] / (1024*1024):.1f}MB")
    
    # Test file browser
    print(f"\nEnhanced File Browser:")
    browser = EnhancedFileBrowser()
    files = browser.get_flat_file_list(
        include_dirs=True,
        model_type="torch",
        search_term="bert"
    )
    print(f"Found {len(files)} items matching filters")
