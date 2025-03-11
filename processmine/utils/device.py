# utils/device.py - New centralized device management
import torch
import logging
import platform
import psutil

logger = logging.getLogger(__name__)

def setup_device():
    """Universal device setup function for all modules"""
    logger.info("Detecting optimal device...")
    
    # Check CUDA availability
    if torch.cuda.is_available():
        # Get GPU memory information
        device_name = torch.cuda.get_device_name(0)
        memory_total = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        memory_reserved = torch.cuda.memory_reserved(0) / (1024**3)
        memory_free = memory_total - memory_reserved
            
        logger.info(f"Found GPU: {device_name}")
        logger.info(f"GPU Memory: {memory_total:.2f} GB total, {memory_free:.2f} GB free")
            
        # Enable performance optimizations
        torch.backends.cudnn.benchmark = True
        if hasattr(torch.backends.cudnn, 'allow_tf32'):
            torch.backends.cudnn.allow_tf32 = True
        if hasattr(torch, 'set_float32_matmul_precision'):
            torch.set_float32_matmul_precision('high')
                
        return torch.device("cuda")
            
    # Check for Apple Silicon MPS support
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        logger.info("Using Apple Silicon GPU (MPS)")
        return torch.device("mps")
    
    # Use CPU as fallback
    logger.info("No GPU available. Using CPU for computation.")
    
    # Get CPU info for logging
    cpu_info = f"{platform.processor()}, {psutil.cpu_count(logical=False)} cores"
    mem_info = f"{psutil.virtual_memory().available / (1024**3):.2f} GB"
    logger.info(f"CPU: {cpu_info}, Available memory: {mem_info}")
    
    return torch.device("cpu")