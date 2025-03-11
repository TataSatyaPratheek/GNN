#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Device setup for ProcessMine CLI
"""

import torch
from termcolor import colored
import logging

logger = logging.getLogger(__name__)

def setup_device():
    """
    Setup computing device with enhanced detection
    
    Returns:
        torch.device: Optimal device for computation
    """
    print(colored("🔍 Detecting and configuring optimal device...", "cyan"))
    
    # Check CUDA availability with memory requirements
    if torch.cuda.is_available():
        # Get GPU memory information
        try:
            device_name = torch.cuda.get_device_name(0)
            memory_total = torch.cuda.get_device_properties(0).total_memory / (1024**3)  # GB
            memory_reserved = torch.cuda.memory_reserved(0) / (1024**3)  # GB
            memory_allocated = torch.cuda.memory_allocated(0) / (1024**3)  # GB
            memory_free = memory_total - memory_reserved
            
            # Print GPU details
            print(colored(f"✅ Found GPU: {device_name}", "green"))
            print(f"   CUDA Version: {torch.version.cuda}")
            print(f"   GPU Memory: {memory_total:.2f} GB total, {memory_free:.2f} GB free")
            
            # Ensure sufficient memory is available (at least 1GB)
            if memory_free < 1.0:
                print(colored(f"⚠️ Low GPU memory: {memory_free:.2f} GB available", "yellow"))
                print(f"   Will use CPU for model parameters to avoid OOM errors")
            
            # Use CUDA with memory tracking
            device = torch.device("cuda")
            
            # Try to improve performance
            torch.backends.cudnn.benchmark = True
            if hasattr(torch.backends.cudnn, 'allow_tf32'):
                torch.backends.cudnn.allow_tf32 = True
            if hasattr(torch, 'set_float32_matmul_precision'):
                torch.set_float32_matmul_precision('high')
                
            print(f"   Enabled performance optimizations for CUDA")
                
            return device
            
        except RuntimeError as e:
            print(colored(f"⚠️ CUDA error: {e}", "yellow"))
            print("   Falling back to CPU")
    
    # Check for Apple Silicon MPS support
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        try:
            device = torch.device("mps")
            print(colored("✅ Using Apple Silicon GPU (MPS)", "green"))
            return device
        except Exception as e:
            print(colored(f"⚠️ MPS error: {e}", "yellow"))
            print("   Falling back to CPU")
    
    # Use CPU as fallback
    device = torch.device("cpu")
    print(colored("⚠️ No GPU available. Using CPU for computation.", "yellow"))
    
    # Get CPU info
    import platform
    import psutil
    print(f"   CPU: {platform.processor()}")
    print(f"   Available cores: {psutil.cpu_count(logical=False)} physical, {psutil.cpu_count()} logical")
    print(f"   Available memory: {psutil.virtual_memory().available / (1024**3):.2f} GB")
    
    return device