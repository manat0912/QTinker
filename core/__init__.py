"""Core logic module for distillation and quantization."""
from .device_manager import DeviceManager, get_device_manager

__all__ = ['DeviceManager', 'get_device_manager']