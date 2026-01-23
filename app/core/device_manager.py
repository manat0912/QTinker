"""
Device management for GPU/CPU with automatic fallback based on VRAM availability.
"""
import torch
import gc
from typing import Optional, Tuple


class DeviceManager:
    """Manages device selection and memory monitoring."""
    
    def __init__(self, log_fn=None):
        """
        Initialize device manager.
        
        Args:
            log_fn: Optional logging function
        """
        self.log_fn = log_fn
        self.device = self._get_best_device()
        self.original_device = self.device
        
    def _log(self, msg: str):
        """Log a message if log function is available."""
        if self.log_fn:
            self.log_fn(msg)
    
    def _get_best_device(self) -> torch.device:
        """
        Detect and return the best available device.
        
        Returns:
            torch.device object (cuda, mps, or cpu)
        """
        if torch.cuda.is_available():
            return torch.device("cuda")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        else:
            return torch.device("cpu")
    
    def get_device(self) -> torch.device:
        """Get the current device."""
        return self.device
    
    def get_device_name(self) -> str:
        """Get human-readable device name."""
        if self.device.type == "cuda":
            return f"CUDA ({torch.cuda.get_device_name(0)})"
        elif self.device.type == "mps":
            return "Apple Silicon (MPS)"
        else:
            return "CPU"
    
    def get_vram_info(self) -> Optional[Tuple[float, float, float]]:
        """
        Get VRAM information if using CUDA.
        
        Returns:
            Tuple of (total_gb, allocated_gb, free_gb) or None if not CUDA
        """
        if self.device.type == "cuda":
            total = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            allocated = torch.cuda.memory_allocated(0) / (1024**3)
            reserved = torch.cuda.memory_reserved(0) / (1024**3)
            free = total - reserved
            return (total, allocated, free)
        return None
    
    def log_device_info(self):
        """Log current device information."""
        device_name = self.get_device_name()
        self._log(f"Using device: {device_name}")
        
        if self.device.type == "cuda":
            vram_info = self.get_vram_info()
            if vram_info:
                total, allocated, free = vram_info
                self._log(f"VRAM: {allocated:.2f}GB allocated / {total:.2f}GB total ({free:.2f}GB free)")
    
    def check_vram_available(self, required_gb: float = 2.0) -> bool:
        """
        Check if enough VRAM is available.
        
        Args:
            required_gb: Required VRAM in GB
            
        Returns:
            True if enough VRAM is available
        """
        if self.device.type == "cuda":
            vram_info = self.get_vram_info()
            if vram_info:
                _, _, free = vram_info
                return free >= required_gb
        return True  # CPU or MPS - assume available
    
    def should_use_cpu(self, model_size_gb: Optional[float] = None) -> bool:
        """
        Determine if CPU should be used instead of GPU.
        
        Args:
            model_size_gb: Estimated model size in GB (optional)
            
        Returns:
            True if CPU should be used
        """
        if self.device.type == "cpu":
            return True
        
        if self.device.type == "cuda":
            vram_info = self.get_vram_info()
            if vram_info:
                total, allocated, free = vram_info
                # Use CPU if less than 2GB free, or if model is larger than available VRAM
                if free < 2.0:
                    self._log(f"⚠️  Low VRAM detected ({free:.2f}GB free). Switching to CPU.")
                    return True
                if model_size_gb and model_size_gb > free * 0.9:  # 90% threshold
                    self._log(f"⚠️  Model size ({model_size_gb:.2f}GB) exceeds available VRAM ({free:.2f}GB). Using CPU.")
                    return True
        
        return False
    
    def switch_to_cpu(self):
        """Switch to CPU device."""
        if self.device.type != "cpu":
            self._log(f"Switching from {self.get_device_name()} to CPU")
            self.device = torch.device("cpu")
            # Clear GPU cache
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                gc.collect()
    
    def switch_to_gpu(self):
        """Switch back to GPU device if available."""
        if self.device.type == "cpu" and torch.cuda.is_available():
            self._log(f"Switching back to GPU: {torch.cuda.get_device_name(0)}")
            self.device = torch.device("cuda")
        elif self.device.type == "cpu" and hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            self._log("Switching back to Apple Silicon (MPS)")
            self.device = torch.device("mps")
    
    def clear_cache(self):
        """Clear GPU cache and run garbage collection."""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
    
    def move_model_to_device(self, model, force_cpu: bool = False):
        """
        Move model to appropriate device.
        
        Args:
            model: PyTorch model
            force_cpu: Force CPU even if GPU is available
            
        Returns:
            Model on the appropriate device
        """
        if force_cpu:
            target_device = torch.device("cpu")
        else:
            target_device = self.device
        
        try:
            model = model.to(target_device)
            self._log(f"Model moved to {target_device}")
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                self._log(f"⚠️  GPU OOM error: {e}")
                self._log("Falling back to CPU...")
                self.clear_cache()
                target_device = torch.device("cpu")
                model = model.to(target_device)
            else:
                raise
        
        return model
    
    def estimate_model_size(self, model) -> float:
        """
        Estimate model size in GB.
        
        Args:
            model: PyTorch model
            
        Returns:
            Estimated size in GB
        """
        param_size = sum(p.numel() * p.element_size() for p in model.parameters())
        buffer_size = sum(b.numel() * b.element_size() for b in model.buffers())
        total_size = (param_size + buffer_size) / (1024**3)
        return total_size


# Global device manager instance
_device_manager: Optional[DeviceManager] = None


def get_device_manager(log_fn=None) -> DeviceManager:
    """
    Get or create the global device manager.
    
    Args:
        log_fn: Optional logging function
        
    Returns:
        DeviceManager instance
    """
    global _device_manager
    if _device_manager is None:
        _device_manager = DeviceManager(log_fn)
    elif log_fn and _device_manager.log_fn != log_fn:
        _device_manager.log_fn = log_fn
    return _device_manager


def reset_device_manager():
    """Reset the global device manager."""
    global _device_manager
    _device_manager = None
