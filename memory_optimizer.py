#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Memory optimization utilities for process mining models
"""

import gc
import torch
import os
import psutil
import numpy as np
from colorama import Fore, Style

class MemoryOptimizer:
    """
    Memory optimization utilities for model training and inference
    with aggressive garbage collection and memory tracking
    """
    
    @staticmethod
    def get_memory_stats():
        """Get current memory usage statistics"""
        stats = {
            "cpu_percent": psutil.virtual_memory().percent,
            "cpu_used_gb": psutil.virtual_memory().used / (1024**3),
            "cpu_available_gb": psutil.virtual_memory().available / (1024**3),
        }
        
        # Add GPU stats if available
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                stats[f"gpu{i}_used_mb"] = torch.cuda.memory_allocated(i) / (1024**2)
                stats[f"gpu{i}_cached_mb"] = torch.cuda.memory_reserved(i) / (1024**2)
                stats[f"gpu{i}_total_mb"] = torch.cuda.get_device_properties(i).total_memory / (1024**2)
        
        return stats
    
    @staticmethod
    def print_memory_stats(prefix=""):
        """Print current memory usage"""
        stats = MemoryOptimizer.get_memory_stats()
        
        print(f"{Fore.CYAN}{prefix} Memory Stats:{Style.RESET_ALL}")
        print(f"  CPU: {stats['cpu_percent']:.1f}% used, {stats['cpu_available_gb']:.2f} GB available")
        
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                used = stats[f"gpu{i}_used_mb"]
                total = stats[f"gpu{i}_total_mb"]
                percent = (used / total) * 100
                print(f"  GPU {i}: {used:.1f} MB / {total:.1f} MB ({percent:.1f}%)")
    
    @staticmethod
    def clear_memory(full_clear=True):
        """
        Clear memory with aggressive garbage collection
        
        Args:
            full_clear: Whether to perform aggressive GPU memory clearing
        """
        # Force garbage collection
        gc.collect()
        
        # Clear GPU cache if available
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            
            if full_clear:
                # More aggressive GPU memory clearing
                for i in range(torch.cuda.device_count()):
                    torch.cuda.synchronize(i)
    
    @staticmethod
    def estimate_tensor_memory(shape, dtype=torch.float32):
        """
        Estimate memory usage of a tensor
        
        Args:
            shape: Tensor shape
            dtype: Tensor data type
            
        Returns:
            Memory size in MB
        """
        # Get element size for data type
        if dtype == torch.float32 or dtype == torch.int32:
            elem_size = 4
        elif dtype == torch.float64 or dtype == torch.int64:
            elem_size = 8
        elif dtype == torch.float16 or dtype == torch.int16:
            elem_size = 2
        elif dtype == torch.uint8 or dtype == torch.int8:
            elem_size = 1
        else:
            elem_size = 4  # Default to 4 bytes
        
        # Calculate size
        num_elements = np.prod(shape)
        bytes_size = num_elements * elem_size
        mb_size = bytes_size / (1024**2)
        
        return mb_size
    
    @staticmethod
    def check_model_size(model):
        """
        Calculate model size in parameters and memory
        
        Args:
            model: PyTorch model
            
        Returns:
            Dict with parameter count and size in MB
        """
        # Count parameters
        param_count = sum(p.numel() for p in model.parameters())
        
        # Calculate parameter memory (assuming float32)
        param_size_mb = param_count * 4 / (1024**2)
        
        return {
            "param_count": param_count,
            "param_size_mb": param_size_mb
        }

    @staticmethod
    def optimize_batch_size(model, sample_input, device, start_batch=32, step=4, target_mem_percent=80):
        """
        Find optimal batch size that fits in GPU memory
        
        Args:
            model: Model to test
            sample_input: Sample input tensor (single sample)
            device: Target device
            start_batch: Starting batch size
            step: Batch size decrement step
            target_mem_percent: Target memory utilization percentage
            
        Returns:
            Optimal batch size
        """
        if not torch.cuda.is_available() or device.type != 'cuda':
            # For CPU, just return the starting batch size
            return start_batch
        
        # Get total GPU memory
        total_mem = torch.cuda.get_device_properties(device).total_memory
        
        batch_size = start_batch
        while batch_size > 1:
            try:
                # Clear memory
                MemoryOptimizer.clear_memory()
                
                # Create batch
                if isinstance(sample_input, torch.Tensor):
                    batch = sample_input.expand(batch_size, *sample_input.shape)
                elif isinstance(sample_input, (list, tuple)):
                    batch = [x.expand(batch_size, *x.shape) for x in sample_input]
                else:
                    # Can't create batch, return default
                    return max(1, start_batch // 2)
                
                # Try forward pass
                with torch.no_grad():
                    _ = model(batch)
                
                # Check memory usage
                used_mem = torch.cuda.memory_allocated(device)
                mem_percent = (used_mem / total_mem) * 100
                
                if mem_percent < target_mem_percent:
                    print(f"{Fore.GREEN}Found working batch size: {batch_size} ({mem_percent:.1f}% GPU){Style.RESET_ALL}")
                    return batch_size
                
            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    print(f"{Fore.YELLOW}Batch size {batch_size} too large, reducing...{Style.RESET_ALL}")
                else:
                    # Some other error, not memory related
                    print(f"{Fore.RED}Error testing batch size: {e}{Style.RESET_ALL}")
                
            # Reduce batch size and try again
            batch_size -= step
        
        # Return minimum batch size
        return 1
    
    @staticmethod
    def safe_to_device(model, device, max_gpu_mem_percent=90):
        """
        Safely move model to device with memory check
        
        Args:
            model: PyTorch model
            device: Target device
            max_gpu_mem_percent: Maximum GPU memory percentage to allow
            
        Returns:
            Model on device (or CPU if GPU memory insufficient)
        """
        # If device is CPU, just move and return
        if device.type == 'cpu':
            return model.to(device)
        
        # Check GPU memory before moving
        if torch.cuda.is_available() and device.type == 'cuda':
            # Clear memory first
            MemoryOptimizer.clear_memory()
            
            # Get memory stats
            total_mem = torch.cuda.get_device_properties(device).total_memory
            used_mem = torch.cuda.memory_allocated(device)
            
            # Estimate model memory from parameters
            model_size = MemoryOptimizer.check_model_size(model)
            param_size_bytes = model_size["param_size_mb"] * 1024 * 1024
            
            # Model will use ~2x parameter size in memory due to gradients, optimizer states, etc.
            estimated_usage = used_mem + (param_size_bytes * 2)
            estimated_percent = (estimated_usage / total_mem) * 100
            
            if estimated_percent > max_gpu_mem_percent:
                print(f"{Fore.YELLOW}Warning: Model too large for GPU ({estimated_percent:.1f}% > {max_gpu_mem_percent}%), keeping on CPU{Style.RESET_ALL}")
                return model.to('cpu')  # Keep on CPU
            
        # Safe to move to device
        return model.to(device)


# Example usage
if __name__ == "__main__":
    # Test memory optimization
    optimizer = MemoryOptimizer()
    optimizer.print_memory_stats("Initial")
    optimizer.clear_memory()
    optimizer.print_memory_stats("After clearing")