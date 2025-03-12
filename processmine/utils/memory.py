# processmine/utils/memory.py (UPDATED FILE)

import gc
import logging
import psutil
import torch
import os
from typing import Dict, Any, Optional, Union

logger = logging.getLogger(__name__)

def clear_memory(full_clear=False):
    """
    Free up memory by clearing caches and forcing garbage collection
    
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
            try:
                # For newer PyTorch versions
                torch.cuda._sleep(2000)  # Wait for pending operations
                torch.cuda.empty_cache()
            except:
                pass

def get_memory_stats() -> Dict[str, Any]:
    """
    Get detailed current memory usage statistics
    
    Returns:
        Dictionary with memory statistics
    """
    stats = {}
    
    # CPU memory
    vm = psutil.virtual_memory()
    stats['cpu_percent'] = vm.percent
    stats['cpu_used_gb'] = vm.used / (1024 ** 3)
    stats['cpu_available_gb'] = vm.available / (1024 ** 3)
    stats['cpu_total_gb'] = vm.total / (1024 ** 3)
    
    # GPU memory if available
    if torch.cuda.is_available():
        try:
            device = torch.cuda.current_device()
            stats['gpu_allocated_gb'] = torch.cuda.memory_allocated(device) / (1024 ** 3)
            stats['gpu_reserved_gb'] = torch.cuda.memory_reserved(device) / (1024 ** 3)
            stats['gpu_max_gb'] = torch.cuda.get_device_properties(device).total_memory / (1024 ** 3)
            
            # Add per-device stats if multiple GPUs
            if torch.cuda.device_count() > 1:
                stats['gpu_per_device'] = {}
                for i in range(torch.cuda.device_count()):
                    stats['gpu_per_device'][i] = {
                        'allocated_gb': torch.cuda.memory_allocated(i) / (1024 ** 3),
                        'reserved_gb': torch.cuda.memory_reserved(i) / (1024 ** 3),
                        'max_gb': torch.cuda.get_device_properties(i).total_memory / (1024 ** 3),
                        'name': torch.cuda.get_device_name(i)
                    }
        except Exception as e:
            stats['gpu_error'] = str(e)
    
    # Process-specific memory
    process = psutil.Process(os.getpid())
    stats['process_memory_gb'] = process.memory_info().rss / (1024 ** 3)
    stats['process_percent'] = process.memory_percent()
    
    return stats

def log_memory_usage(level=logging.INFO, include_gpu=True):
    """
    Log current memory usage
    
    Args:
        level: Logging level to use
        include_gpu: Whether to include GPU stats
        
    Returns:
        Memory statistics dictionary
    """
    stats = get_memory_stats()
    
    # Prepare log message
    msg_parts = [f"CPU: {stats['cpu_percent']:.1f}% ({stats['cpu_used_gb']:.2f}/{stats['cpu_total_gb']:.2f} GB)"]
    msg_parts.append(f"Process: {stats['process_memory_gb']:.2f} GB ({stats['process_percent']:.1f}%)")
    
    if include_gpu and 'gpu_allocated_gb' in stats:
        gpu_percent = (stats['gpu_allocated_gb'] / stats['gpu_max_gb']) * 100
        msg_parts.append(f"GPU: {gpu_percent:.1f}% ({stats['gpu_allocated_gb']:.2f}/{stats['gpu_max_gb']:.2f} GB)")
    
    logger.log(level, f"Memory usage: {' | '.join(msg_parts)}")
    
    return stats

def estimate_batch_size(
    feature_dim: int, 
    sample_size: int = 1000, 
    complexity: str = 'medium', 
    available_memory_fraction: float = 0.7,
    device: Optional[Union[torch.device, str]] = None
) -> int:
    """
    Estimate optimal batch size based on model complexity and available memory
    
    Args:
        feature_dim: Dimension of input features
        sample_size: Size of test sample to use
        complexity: Model complexity ('low', 'medium', 'high', 'very_high')
        available_memory_fraction: Fraction of available memory to use
        device: Computing device (default: auto-detect)
        
    Returns:
        Estimated optimal batch size
    """
    # Determine device
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    elif isinstance(device, str):
        device = torch.device(device)
    
    # Get available memory
    stats = get_memory_stats()
    
    if device.type == 'cuda' and 'gpu_max_gb' in stats:
        # Use GPU memory if available
        available_memory = (stats['gpu_max_gb'] - stats['gpu_allocated_gb']) * available_memory_fraction
        memory_type = "GPU"
    else:
        # Use CPU memory
        available_memory = stats['cpu_available_gb'] * available_memory_fraction
        memory_type = "CPU"
    
    # Memory needed per sample based on complexity
    complexity_factors = {
        'low': 2.0,
        'medium': 4.0,
        'high': 8.0,
        'very_high': 16.0
    }
    complexity_factor = complexity_factors.get(complexity, complexity_factors['medium'])
    
    # Memory per sample (in GB):
    # (feature_dim * 4 bytes for float32) * complexity_factor for activations and gradients
    memory_per_sample_gb = (feature_dim * 4 * complexity_factor) / (1024 ** 3)
    
    # Calculate batch size
    batch_size = int(available_memory / memory_per_sample_gb)
    
    # Apply reasonable bounds
    min_batch = 4
    max_batch = 512
    
    # Adjust batch size to be a power of 2 (often more efficient)
    batch_size = max(min_batch, min(max_batch, 2 ** int(np.log2(batch_size))))
    
    logger.info(f"Estimated batch size: {batch_size} for {complexity} model " +
               f"(using {available_memory:.2f} GB of {memory_type} memory)")
    
    return batch_size

class MemoryTracker:
    """Memory usage tracker for monitoring memory during operations"""
    
    def __init__(self, check_gpu=True, verbose=True):
        """
        Initialize memory tracker
        
        Args:
            check_gpu: Whether to track GPU memory
            verbose: Whether to print messages
        """
        self.check_gpu = check_gpu and torch.cuda.is_available()
        self.verbose = verbose
        self.snapshots = []
        self.start_time = time.time()
        
        # Take initial snapshot
        self.take_snapshot("Initial")
    
    def take_snapshot(self, label):
        """
        Take a memory snapshot
        
        Args:
            label: Label for this snapshot
        
        Returns:
            Snapshot data
        """
        stats = get_memory_stats()
        timestamp = time.time() - self.start_time
        
        snapshot = {
            'label': label,
            'timestamp': timestamp,
            'cpu_used_gb': stats['cpu_used_gb'],
            'process_memory_gb': stats['process_memory_gb'],
        }
        
        if self.check_gpu and 'gpu_allocated_gb' in stats:
            snapshot['gpu_used_gb'] = stats['gpu_allocated_gb']
        
        self.snapshots.append(snapshot)
        
        if self.verbose:
            mem_parts = [f"CPU: {stats['cpu_used_gb']:.2f} GB"]
            if self.check_gpu and 'gpu_allocated_gb' in stats:
                mem_parts.append(f"GPU: {stats['gpu_allocated_gb']:.2f} GB")
            logger.info(f"Memory snapshot '{label}' ({timestamp:.2f}s): {' | '.join(mem_parts)}")
        
        return snapshot
    
    def summary(self):
        """
        Get memory usage summary
        
        Returns:
            Dictionary with summary statistics
        """
        if not self.snapshots:
            return {}
        
        # Calculate diffs between snapshots
        for i in range(1, len(self.snapshots)):
            prev = self.snapshots[i-1]
            curr = self.snapshots[i]
            
            curr['cpu_diff_gb'] = curr['cpu_used_gb'] - prev['cpu_used_gb']
            curr['process_diff_gb'] = curr['process_memory_gb'] - prev['process_memory_gb']
            
            if self.check_gpu and 'gpu_used_gb' in curr and 'gpu_used_gb' in prev:
                curr['gpu_diff_gb'] = curr['gpu_used_gb'] - prev['gpu_used_gb']
        
        # Get peak values
        cpu_peak = max(s['cpu_used_gb'] for s in self.snapshots)
        process_peak = max(s['process_memory_gb'] for s in self.snapshots)
        
        summary = {
            'snapshots': self.snapshots,
            'initial_cpu_gb': self.snapshots[0]['cpu_used_gb'],
            'peak_cpu_gb': cpu_peak,
            'initial_process_gb': self.snapshots[0]['process_memory_gb'],
            'peak_process_gb': process_peak,
            'cpu_growth_gb': cpu_peak - self.snapshots[0]['cpu_used_gb'],
            'process_growth_gb': process_peak - self.snapshots[0]['process_memory_gb'],
            'duration_sec': self.snapshots[-1]['timestamp']
        }
        
        if self.check_gpu and all('gpu_used_gb' in s for s in self.snapshots):
            gpu_peak = max(s['gpu_used_gb'] for s in self.snapshots)
            summary['initial_gpu_gb'] = self.snapshots[0]['gpu_used_gb']
            summary['peak_gpu_gb'] = gpu_peak
            summary['gpu_growth_gb'] = gpu_peak - self.snapshots[0]['gpu_used_gb']
        
        return summary
    
    def print_summary(self):
        """Print memory usage summary to console"""
        summary = self.summary()
        if not summary:
            print("No memory snapshots recorded")
            return
        
        print("\nMEMORY USAGE SUMMARY:")
        print(f"Duration: {summary['duration_sec']:.2f} seconds")
        print(f"CPU: {summary['initial_cpu_gb']:.2f} GB → Peak: {summary['peak_cpu_gb']:.2f} GB (+" +
              f"{summary['cpu_growth_gb']:.2f} GB)")
        print(f"Process: {summary['initial_process_gb']:.2f} GB → Peak: {summary['peak_process_gb']:.2f} GB (+" +
              f"{summary['process_growth_gb']:.2f} GB)")
        
        if 'peak_gpu_gb' in summary:
            print(f"GPU: {summary['initial_gpu_gb']:.2f} GB → Peak: {summary['peak_gpu_gb']:.2f} GB (+" +
                  f"{summary['gpu_growth_gb']:.2f} GB)")
        
        print("\nMEMORY SNAPSHOTS:")
        for i, snap in enumerate(summary['snapshots']):
            parts = [f"CPU: {snap['cpu_used_gb']:.2f} GB"]
            if i > 0:
                parts[0] += f" ({snap['cpu_diff_gb']:+.2f} GB)"
            
            parts.append(f"Process: {snap['process_memory_gb']:.2f} GB")
            if i > 0:
                parts[-1] += f" ({snap['process_diff_gb']:+.2f} GB)"
            
            if 'gpu_used_gb' in snap:
                parts.append(f"GPU: {snap['gpu_used_gb']:.2f} GB")
                if i > 0 and 'gpu_diff_gb' in snap:
                    parts[-1] += f" ({snap['gpu_diff_gb']:+.2f} GB)"
            
            print(f"  [{snap['timestamp']:.2f}s] {snap['label']}: {' | '.join(parts)}")