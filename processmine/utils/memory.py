# src/processmine/utils/memory.py
"""
Memory management utilities for process mining.
Provides memory-efficient data loaders and memory monitoring tools.
"""

import gc
import os
import random
import time
import threading
import warnings
from typing import List, Dict, Any, Callable, Optional, Iterator, Union, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset, Sampler
from torch_geometric.data import Data, Batch
import psutil

# Set up logging
import logging
logger = logging.getLogger(__name__)


class MemoryMonitor:
    """
    Memory usage monitoring with support for CPU and GPU.
    Can run as a background thread to periodically log memory usage.
    """
    def __init__(self, device=None, interval=5.0, log_level=logging.INFO):
        """
        Initialize memory monitor
        Args:
            device: torch device or device index
            interval: monitoring interval in seconds when running as thread
            log_level: logging level for memory reporting
        """
        self.device = device
        self.interval = interval
        self.log_level = log_level
        self.peak_cpu_memory = 0
        self.peak_gpu_memory = 0
        self.keep_running = False
        self.monitor_thread = None
        
        # Determine if GPU is available
        self.has_gpu = torch.cuda.is_available()
        if self.has_gpu and device is None:
            self.device = torch.device('cuda:0')
        elif device is None:
            self.device = torch.device('cpu')
            
        # Try to detect Apple Silicon MPS
        self.has_mps = hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
        if self.has_mps and device is None:
            self.device = torch.device('mps')
    
    def get_memory_usage(self) -> Dict[str, float]:
        """Get current memory usage for CPU and GPU (if available)"""
        result = {}
        
        # CPU memory
        cpu_percent = psutil.virtual_memory().percent
        cpu_used_gb = psutil.virtual_memory().used / (1024 ** 3)
        result['cpu_percent'] = cpu_percent
        result['cpu_used_gb'] = cpu_used_gb
        
        # Update peak CPU memory
        self.peak_cpu_memory = max(self.peak_cpu_memory, cpu_used_gb)
        result['peak_cpu_gb'] = self.peak_cpu_memory
        
        # GPU memory if available
        if self.has_gpu and str(self.device).startswith('cuda'):
            try:
                gpu_id = 0 if self.device == torch.device('cuda') else int(str(self.device).split(':')[1])
                gpu_percent = torch.cuda.memory_allocated(gpu_id) / torch.cuda.max_memory_allocated(gpu_id) * 100
                gpu_used_gb = torch.cuda.memory_allocated(gpu_id) / (1024 ** 3)
                result['gpu_percent'] = gpu_percent
                result['gpu_used_gb'] = gpu_used_gb
                
                # Update peak GPU memory
                self.peak_gpu_memory = max(self.peak_gpu_memory, gpu_used_gb)
                result['peak_gpu_gb'] = self.peak_gpu_memory
            except Exception as e:
                result['gpu_error'] = str(e)
        
        # MPS (Apple Silicon) memory
        elif self.has_mps and str(self.device) == 'mps':
            # Apple Silicon doesn't provide memory stats through PyTorch yet
            result['mps_device'] = True
        
        return result
    
    def log_memory_usage(self) -> Dict[str, float]:
        """Log current memory usage"""
        mem_info = self.get_memory_usage()
        
        # Prepare log message
        msg_parts = [f"CPU: {mem_info['cpu_percent']:.1f}% ({mem_info['cpu_used_gb']:.2f}GB)"]
        
        if 'gpu_used_gb' in mem_info:
            msg_parts.append(f"GPU: {mem_info['gpu_percent']:.1f}% ({mem_info['gpu_used_gb']:.2f}GB)")
        
        if 'mps_device' in mem_info:
            msg_parts.append("MPS: Apple Silicon (memory stats not available)")
        
        # Log the message
        logger.log(self.log_level, f"Memory usage: {' | '.join(msg_parts)}")
        
        return mem_info
    
    def start_monitoring(self):
        """Start monitoring thread for periodic logging"""
        if self.monitor_thread is not None and self.monitor_thread.is_alive():
            logger.warning("Memory monitor thread is already running")
            return
        
        self.keep_running = True
        self.monitor_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        self.monitor_thread.start()
        logger.info(f"Started memory monitoring thread with interval {self.interval}s")
    
    def stop_monitoring(self):
        """Stop monitoring thread"""
        self.keep_running = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=1.0)
            self.monitor_thread = None
            logger.info("Stopped memory monitoring thread")
    
    def _monitoring_loop(self):
        """Internal monitoring loop for the background thread"""
        while self.keep_running:
            self.log_memory_usage()
            time.sleep(self.interval)
    
    def optimize_batch_size(self, 
                          initial_batch_size: int, 
                          sample_input_fn: Callable[[int], Any],
                          forward_fn: Callable[[Any], Any], 
                          max_memory_percent: float = 80.0,
                          step_size: int = 4) -> int:
        """
        Find optimal batch size that fits in memory
        
        Args:
            initial_batch_size: Starting batch size to try
            sample_input_fn: Function that takes batch size and returns sample input
            forward_fn: Function that takes input and performs forward pass
            max_memory_percent: Maximum allowed memory usage percentage
            step_size: How much to decrease batch size in each step
            
        Returns:
            Optimal batch size
        """
        batch_size = initial_batch_size
        
        while batch_size > 0:
            try:
                # Clear memory before test
                gc.collect()
                if self.has_gpu:
                    torch.cuda.empty_cache()
                
                # Try batch size
                inputs = sample_input_fn(batch_size)
                outputs = forward_fn(inputs)
                
                # Check memory usage
                mem_info = self.get_memory_usage()
                if self.has_gpu and 'gpu_percent' in mem_info:
                    current_percent = mem_info['gpu_percent']
                else:
                    current_percent = mem_info['cpu_percent']
                
                if current_percent < max_memory_percent:
                    # This batch size works and is under the memory limit
                    logger.info(f"Optimal batch size found: {batch_size} "
                                f"(Memory usage: {current_percent:.1f}%)")
                    return batch_size
                
                # This batch size works but uses too much memory, try smaller
                logger.info(f"Batch size {batch_size} uses {current_percent:.1f}% memory, "
                            f"trying smaller batch")
                
            except RuntimeError as e:
                if "CUDA out of memory" in str(e) or "DefaultCPUAllocator: can't allocate memory" in str(e):
                    # Out of memory error, reduce batch size
                    logger.warning(f"Batch size {batch_size} causes OOM, reducing")
                else:
                    # Some other error
                    raise
            
            # Reduce batch size and try again
            batch_size -= step_size
        
        # If we got here, even batch_size=1 doesn't work
        raise RuntimeError("Unable to find a working batch size, "
                           "even batch_size=1 causes out of memory errors")


class MemoryOptimizer:
    """
    Utilities for optimizing memory usage during model training
    """
    def __init__(self, device=None):
        """
        Initialize memory optimizer
        Args:
            device: torch device to optimize for
        """
        self.device = device
        self.memory_monitor = MemoryMonitor(device=device)
    
    @staticmethod
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
            # More aggressive memory clearing
            if torch.cuda.is_available():
                # Clear unused memory blocks
                torch.cuda.synchronize()
                
                # On some systems, we can clear more aggressively
                try:
                    for i in range(torch.cuda.device_count()):
                        with torch.cuda.device(i):
                            torch.cuda.memory.empty_cache()
                except AttributeError:
                    # Older PyTorch versions don't have torch.cuda.memory.empty_cache
                    pass
    
    def get_model_size(self, model: torch.nn.Module) -> Dict[str, float]:
        """
        Calculate model size and memory usage
        
        Args:
            model: PyTorch model to analyze
            
        Returns:
            Dictionary with model size metrics
        """
        # Get parameter count
        param_count = sum(p.numel() for p in model.parameters())
        param_size_bytes = sum(p.numel() * p.element_size() for p in model.parameters())
        
        # Get buffer size (for batch norm, etc.)
        buffer_size_bytes = sum(b.numel() * b.element_size() for b in model.buffers())
        
        # Calculate total size
        total_size_bytes = param_size_bytes + buffer_size_bytes
        
        # Convert to more readable units
        mb = 1024 * 1024
        
        return {
            'param_count': param_count,
            'param_size_mb': param_size_bytes / mb,
            'buffer_size_mb': buffer_size_bytes / mb,
            'total_size_mb': total_size_bytes / mb
        }
    
    def log_model_size(self, model: torch.nn.Module):
        """Log model size information"""
        size_info = self.get_model_size(model)
        
        logger.info(f"Model size analysis:")
        logger.info(f"- Parameters: {size_info['param_count']:,}")
        logger.info(f"- Parameter size: {size_info['param_size_mb']:.2f} MB")
        logger.info(f"- Buffer size: {size_info['buffer_size_mb']:.2f} MB")
        logger.info(f"- Total size: {size_info['total_size_mb']:.2f} MB")
        
        return size_info

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
            
            # Get approximate model size
            model_size = sum(p.numel() * p.element_size() for p in model.parameters())
            
            # Model will use ~2x parameter size in memory due to gradients, optimizer states, etc.
            estimated_usage = used_mem + (model_size * 2)
            estimated_percent = (estimated_usage / total_mem) * 100
            
            if estimated_percent > max_gpu_mem_percent:
                logger.warning(f"Model too large for GPU ({estimated_percent:.1f}% > {max_gpu_mem_percent}%), keeping on CPU")
                return model.to('cpu')  # Keep on CPU
            
        # Safe to move to device
        return model.to(device)


class MemoryEfficientDataLoader:
    """
    Memory-efficient data loader with dynamic batch sizing
    and memory monitoring capabilities
    """
    def __init__(self, dataset, batch_size=32, shuffle=True, pin_memory=True,
                 prefetch_factor=2, memory_threshold=80, num_workers=0,
                 collate_fn=None, drop_last=False):
        """
        Initialize memory-efficient data loader
        
        Args:
            dataset: Dataset to load from
            batch_size: Initial batch size (may be adjusted dynamically)
            shuffle: Whether to shuffle data
            pin_memory: Whether to pin memory for faster GPU transfer
            prefetch_factor: Number of batches to prefetch
            memory_threshold: Memory usage percentage threshold for GC
            num_workers: Number of workers for data loading
            collate_fn: Function to collate samples into batches
            drop_last: Whether to drop the last batch if incomplete
        """
        self.dataset = dataset
        self.initial_batch_size = batch_size
        self.batch_size = batch_size  # Current batch size (may change dynamically)
        self.shuffle = shuffle
        self.pin_memory = pin_memory
        self.prefetch_factor = prefetch_factor
        self.memory_threshold = memory_threshold
        self.num_workers = num_workers
        self.drop_last = drop_last
        
        # Default collate function for graph data if not specified
        if collate_fn is None and isinstance(dataset[0], Data):
            self.collate_fn = self._collate_graph_batch
        else:
            self.collate_fn = collate_fn if collate_fn else torch.utils.data._utils.collate.default_collate
        
        # Set up memory monitor
        self.memory_monitor = MemoryMonitor(interval=5.0)
        
        # Adjust batch size before starting if needed
        if hasattr(dataset, '__getitem__') and len(dataset) > 0:
            # Only try to optimize if we can get individual items
            self._adjust_initial_batch_size()
    
    def _adjust_initial_batch_size(self):
        """Adjust initial batch size based on memory constraints"""
        try:
            # Get a single sample to test memory usage
            sample = self.dataset[0]
            
            def sample_input_fn(batch_size):
                """Create a sample batch of the given size"""
                # Just repeat the first sample for simplicity
                samples = [self.dataset[0] for _ in range(batch_size)]
                return self.collate_fn(samples)
            
            def forward_fn(batch):
                """Dummy forward pass to test memory usage"""
                # Just hold the batch in memory to see if it fits
                if isinstance(batch, Batch):
                    # For PyG batches, access some attributes
                    x = batch.x
                    if hasattr(batch, 'edge_index'):
                        edge_index = batch.edge_index
                    return x.shape[0]  # Just return something
                elif isinstance(batch, torch.Tensor):
                    return batch.shape[0]
                elif isinstance(batch, (list, tuple)):
                    return len(batch)
                else:
                    return None
            
            # Try to optimize batch size
            optimal_batch_size = self.memory_monitor.optimize_batch_size(
                initial_batch_size=self.initial_batch_size,
                sample_input_fn=sample_input_fn,
                forward_fn=forward_fn,
                max_memory_percent=self.memory_threshold,
                step_size=max(1, self.initial_batch_size // 8)
            )
            
            if optimal_batch_size != self.initial_batch_size:
                logger.info(f"Adjusted batch size from {self.initial_batch_size} to {optimal_batch_size}")
                self.batch_size = optimal_batch_size
            
        except Exception as e:
            # If optimization fails, keep the initial batch size
            logger.warning(f"Batch size optimization failed: {str(e)}")
            logger.warning(f"Using initial batch size: {self.initial_batch_size}")
    
    def _collate_graph_batch(self, data_list: List[Data]) -> Batch:
        """Collate PyG Data objects into a batch"""
        return Batch.from_data_list(data_list)
    
    def __iter__(self) -> Iterator[Any]:
        """Iterator over batches with memory management"""
        indices = list(range(len(self.dataset)))
        
        if self.shuffle:
            random.shuffle(indices)
        
        # Group indices into batches
        batch_indices = []
        for i in range(0, len(indices), self.batch_size):
            if i + self.batch_size <= len(indices) or not self.drop_last:
                batch_indices.append(indices[i:i + self.batch_size])
        
        # Process batches with memory checks
        for i, indices in enumerate(batch_indices):
            try:
                # Get samples for this batch
                samples = [self.dataset[idx] for idx in indices]
                
                # Collate samples into batch
                batch = self.collate_fn(samples)
                
                # Pin memory if needed
                if self.pin_memory and torch.cuda.is_available():
                    batch = self._pin_memory(batch)
                
                yield batch
                
                # Periodically check memory
                if i % 10 == 0:
                    mem_info = self.memory_monitor.get_memory_usage()
                    cpu_percent = mem_info['cpu_percent']
                    
                    if cpu_percent > self.memory_threshold:
                        # Memory usage high, force garbage collection
                        logger.info(f"Memory usage high ({cpu_percent:.1f}%), triggering GC")
                        gc.collect()
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                
            except RuntimeError as e:
                if "CUDA out of memory" in str(e):
                    # Out of memory error, emergency measures
                    logger.warning(f"CUDA OOM during batch loading. Emergency GC and reducing batch size")
                    gc.collect()
                    torch.cuda.empty_cache()
                    
                    # Reduce batch size for future batches
                    self.batch_size = max(1, self.batch_size // 2)
                    logger.warning(f"Reduced batch size to {self.batch_size}")
                    
                    # Try again with a smaller portion of this batch
                    if len(indices) > 1:
                        mid = len(indices) // 2
                        small_indices = indices[:mid]
                        samples = [self.dataset[idx] for idx in small_indices]
                        batch = self.collate_fn(samples)
                        
                        if self.pin_memory and torch.cuda.is_available():
                            batch = self._pin_memory(batch)
                        
                        yield batch
                else:
                    # Other RuntimeError, re-raise
                    raise
    
    def _pin_memory(self, batch):
        """Recursively pin memory in batch components"""
        if hasattr(batch, 'pin_memory'):
            return batch.pin_memory()
        elif isinstance(batch, (list, tuple)):
            return type(batch)([self._pin_memory(x) for x in batch])
        elif isinstance(batch, dict):
            return {k: self._pin_memory(v) for k, v in batch.items()}
        else:
            return batch
    
    def __len__(self) -> int:
        """Return the number of batches"""
        if self.drop_last:
            return len(self.dataset) // self.batch_size
        else:
            return (len(self.dataset) + self.batch_size - 1) // self.batch_size