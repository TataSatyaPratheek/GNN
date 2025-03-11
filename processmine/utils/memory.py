import gc
import logging
import psutil
import torch
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)

def clear_memory(full_clear=False):
    """
    Free up memory by clearing caches
    
    Args:
        full_clear: Whether to perform more aggressive memory clearing
    """
    # Python garbage collection
    gc.collect()
    
    # PyTorch CUDA cache
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        
        if full_clear:
            # More aggressive clearing on CUDA devices
            torch.cuda.synchronize()

def get_memory_stats() -> Dict[str, Any]:
    """
    Get current memory usage statistics
    
    Returns:
        Dictionary with memory statistics
    """
    stats = {}
    
    # CPU memory
    vm = psutil.virtual_memory()
    stats['cpu_percent'] = vm.percent
    stats['cpu_used_gb'] = vm.used / (1024 ** 3)
    stats['cpu_available_gb'] = vm.available / (1024 ** 3)
    
    # GPU memory if available
    if torch.cuda.is_available():
        try:
            device = torch.cuda.current_device()
            stats['gpu_allocated_gb'] = torch.cuda.memory_allocated(device) / (1024 ** 3)
            stats['gpu_reserved_gb'] = torch.cuda.memory_reserved(device) / (1024 ** 3)
            stats['gpu_max_gb'] = torch.cuda.get_device_properties(device).total_memory / (1024 ** 3)
        except Exception as e:
            stats['gpu_error'] = str(e)
    
    return stats

def log_memory_usage(level=logging.INFO):
    """
    Log current memory usage
    
    Args:
        level: Logging level to use
    """
    stats = get_memory_stats()
    
    # Prepare log message
    msg_parts = [f"CPU: {stats['cpu_percent']:.1f}% ({stats['cpu_used_gb']:.2f} GB)"]
    
    if 'gpu_allocated_gb' in stats:
        gpu_percent = (stats['gpu_allocated_gb'] / stats['gpu_max_gb']) * 100
        msg_parts.append(f"GPU: {gpu_percent:.1f}% ({stats['gpu_allocated_gb']:.2f} GB)")
    
    logger.log(level, f"Memory usage: {' | '.join(msg_parts)}")
    
    return stats

def estimate_batch_size(input_size: int, model_complexity: str = 'medium') -> int:
    """
    Estimate optimal batch size based on available memory
    
    Args:
        input_size: Size of input features
        model_complexity: Model complexity ('low', 'medium', 'high')
        
    Returns:
        Estimated optimal batch size
    """
    # Get available memory
    stats = get_memory_stats()
    
    # Use conservative estimate of available memory (30% of free memory)
    if torch.cuda.is_available() and 'gpu_max_gb' in stats:
        free_memory_gb = stats['gpu_max_gb'] - stats['gpu_allocated_gb']
        available_memory_gb = free_memory_gb * 0.3
    else:
        available_memory_gb = stats['cpu_available_gb'] * 0.3
    
    # Convert to bytes
    available_memory = available_memory_gb * (1024 ** 3)
    
    # Memory requirements per sample (bytes)
    # This is a heuristic based on model complexity
    complexity_factors = {
        'low': 1.0,
        'medium': 2.0,
        'high': 4.0
    }
    factor = complexity_factors.get(model_complexity, 2.0)
    
    # Estimate memory per sample (input, activations, gradients)
    # Each float is 4 bytes
    bytes_per_sample = input_size * 4 * factor
    
    # Calculate batch size with min/max limits
    batch_size = max(1, min(1024, int(available_memory / bytes_per_sample)))
    
    logger.info(f"Estimated optimal batch size: {batch_size} "
               f"(Available memory: {available_memory_gb:.2f} GB)")
    
    return batch_size

def get_model_size(model: torch.nn.Module) -> Dict[str, Any]:
    """
    Calculate model memory usage
    
    Args:
        model: PyTorch model
        
    Returns:
        Dictionary with model size information
    """
    # Get parameter count
    param_count = sum(p.numel() for p in model.parameters())
    param_bytes = sum(p.numel() * p.element_size() for p in model.parameters())
    
    # Get buffer size (for batch norm, etc.)
    buffer_bytes = sum(b.numel() * b.element_size() for b in model.buffers())
    
    # Total size
    total_bytes = param_bytes + buffer_bytes
    
    # Convert to MB
    param_mb = param_bytes / (1024 * 1024)
    buffer_mb = buffer_bytes / (1024 * 1024)
    total_mb = total_bytes / (1024 * 1024)
    
    return {
        'param_count': param_count,
        'param_mb': param_mb,
        'buffer_mb': buffer_mb,
        'total_mb': total_mb
    }