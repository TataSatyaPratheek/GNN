"""
Experiment management utilities for ProcessMine, including tracking,
result directory setup, and metrics logging.
"""
import os
import json
import logging
import time
import torch
import numpy as np
import psutil
from pathlib import Path
from typing import Dict, Any, Optional, Union, List

logger = logging.getLogger(__name__)

class ExperimentManager:
    """
    Manager for process mining experiments with tracking and result storage
    """
    def __init__(
        self,
        experiment_name: str,
        output_dir: Optional[Union[str, Path]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        track_memory: bool = True,
        auto_create_dirs: bool = True
    ):
        """
        Initialize experiment manager
        
        Args:
            experiment_name: Name of the experiment
            output_dir: Root directory for experiment outputs
            metadata: Additional metadata to store with experiment
            track_memory: Whether to track memory usage
            auto_create_dirs: Whether to automatically create output directories
        """
        self.experiment_name = experiment_name
        self.start_time = time.time()
        self.metrics = {}
        self.results = {}
        self.metadata = metadata or {}
        self.track_memory = track_memory
        
        # Set up output directory
        if output_dir:
            self.output_dir = Path(output_dir) / self.experiment_name
        else:
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            self.output_dir = Path("results") / f"{self.experiment_name}_{timestamp}"
        
        # Create directories if requested
        if auto_create_dirs:
            self.setup_directories()
        
        # Initialize with system info
        self._capture_system_info()
        
        logger.info(f"Experiment '{experiment_name}' initialized with output directory: {self.output_dir}")
    
    def setup_directories(self):
        """Set up experiment directory structure"""
        # Create main directory
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Create subdirectories
        subdirs = ["models", "metrics", "visualizations", "logs", "checkpoints"]
        for subdir in subdirs:
            os.makedirs(self.output_dir / subdir, exist_ok=True)
        
        # Set up log file
        log_file = self.output_dir / "logs" / "experiment.log"
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
        logger.addHandler(file_handler)
        
        return self.output_dir
    
    def _capture_system_info(self):
        """Capture system information for reproducibility"""
        import platform
        
        self.metadata["system"] = {
            "platform": platform.platform(),
            "python": platform.python_version(),
        }
        
        # Add GPU info if available
        if self.track_memory and torch.cuda.is_available():
            self.metadata["system"]["gpu"] = torch.cuda.get_device_name(0)
        
        # Add memory info if tracking enabled
        if self.track_memory:
            try:
                import psutil
                mem = psutil.virtual_memory()
                self.metadata["system"]["memory"] = {
                    "total_gb": mem.total / (1024**3),
                    "available_gb": mem.available / (1024**3),
                }
            except ImportError:
                pass
    
    def log_metrics(self, metrics: Dict[str, Any], step: Optional[int] = None):
        """
        Log metrics for current experiment
        
        Args:
            metrics: Dictionary of metrics to log
            step: Optional step/epoch number
        """
        # Update metrics store
        for k, v in metrics.items():
            if k not in self.metrics:
                self.metrics[k] = []
            
            # Ensure value is JSON serializable
            if hasattr(v, 'item'):
                v = v.item()
            
            self.metrics[k].append((step, v) if step is not None else v)
        
        # Log to console
        log_items = [f"{k}: {v:.4f}" if isinstance(v, float) else f"{k}: {v}" 
                    for k, v in metrics.items()]
        
        if step is not None:
            logger.info(f"Step {step}: " + ", ".join(log_items))
        else:
            logger.info("Metrics: " + ", ".join(log_items))
    
    def save_metrics(self, filename: str = "metrics.json"):
        """
        Save current metrics to JSON file
        
        Args:
            filename: Output filename
        
        Returns:
            Path to saved metrics file
        """
        # Create metrics directory
        metrics_dir = self.output_dir / "metrics"
        os.makedirs(metrics_dir, exist_ok=True)
        
        # Save to JSON
        metrics_path = metrics_dir / filename
        
        # Ensure metrics are JSON serializable
        serializable_metrics = _make_serializable(self.metrics)
        
        with open(metrics_path, 'w') as f:
            json.dump(serializable_metrics, f, indent=2)
        
        logger.info(f"Saved metrics to {metrics_path}")
        return metrics_path
    
    def save_results(self, results: Dict[str, Any], name: str = "results"):
        """
        Save experiment results
        
        Args:
            results: Results to save
            name: Name for results file
            
        Returns:
            Path to saved results file
        """
        # Store in instance
        self.results[name] = results
        
        # Save to file
        results_path = self.output_dir / f"{name}.json"
        
        # Make results serializable
        clean_results = _make_serializable(results)
        
        with open(results_path, 'w') as f:
            json.dump(clean_results, f, indent=2)
        
        logger.info(f"Saved {name} results to {results_path}")
        return results_path
    
    def save_model(self, model, filename: str = "model.pt"):
        """
        Save model checkpoint
        
        Args:
            model: PyTorch model to save
            filename: Checkpoint filename
            
        Returns:
            Path to saved model
        """
        # Create models directory
        models_dir = self.output_dir / "models"
        os.makedirs(models_dir, exist_ok=True)
        
        # Save model
        model_path = models_dir / filename
        torch.save(model.state_dict(), model_path)
        
        logger.info(f"Saved model to {model_path}")
        return model_path
    
    def finish(self):
        """
        Finish experiment and save summary
        
        Returns:
            Dictionary with experiment summary
        """
        duration = time.time() - self.start_time
        
        # Create summary
        summary = {
            "experiment_name": self.experiment_name,
            "duration_seconds": duration,
            "metadata": self.metadata,
            "metrics": {k: v[-1] if isinstance(v, list) and len(v) > 0 else v 
                      for k, v in self.metrics.items()}
        }
        
        # Save summary
        summary_path = self.output_dir / "summary.json"
        with open(summary_path, 'w') as f:
            json.dump(_make_serializable(summary), f, indent=2)
        
        logger.info(f"Experiment '{self.experiment_name}' completed in {duration:.2f} seconds")
        logger.info(f"Results saved to {self.output_dir}")
        
        return summary


def setup_results_dir(
    base_dir: Optional[Union[str, Path]] = None,
    experiment_name: Optional[str] = None,
    subdirs: Optional[List[str]] = None
) -> Path:
    """
    Set up directories for experiment results
    
    Args:
        base_dir: Base directory for results (default: 'results')
        experiment_name: Experiment name (default: timestamp)
        subdirs: List of subdirectories to create
        
    Returns:
        Path to created results directory
    """
    if base_dir is None:
        base_dir = Path("results")
    else:
        base_dir = Path(base_dir)
    
    if experiment_name is None:
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        experiment_name = f"experiment_{timestamp}"
    
    # Create main directory
    result_dir = base_dir / experiment_name
    os.makedirs(result_dir, exist_ok=True)
    
    # Create subdirectories if provided
    if subdirs:
        for subdir in subdirs:
            os.makedirs(result_dir / subdir, exist_ok=True)
    
    logger.info(f"Created results directory: {result_dir}")
    return result_dir


def save_metrics(
    metrics: Dict[str, Any],
    output_dir: Union[str, Path],
    filename: str = "metrics.json"
) -> Path:
    """
    Save metrics to JSON file
    
    Args:
        metrics: Dictionary of metrics to save
        output_dir: Output directory
        filename: Output filename
        
    Returns:
        Path to saved metrics file
    """
    output_dir = Path(output_dir)
    os.makedirs(output_dir, exist_ok=True)
    
    # Process metrics for JSON serialization
    processed_metrics = _make_serializable(metrics)
    
    # Save to JSON
    metrics_path = output_dir / filename
    with open(metrics_path, 'w') as f:
        json.dump(processed_metrics, f, indent=2)
    
    logger.info(f"Saved metrics to {metrics_path}")
    return metrics_path


def print_section_header(title: str, width: int = 80):
    """
    Print a formatted section header
    
    Args:
        title: Section title
        width: Width of header
    """
    print("\n" + "=" * width)
    print(f"{title.center(width)}")
    print("=" * width + "\n")


def _make_serializable(obj):
    """
    Make an object JSON serializable
    
    Args:
        obj: Input object
        
    Returns:
        JSON serializable version of the object
    """
    if isinstance(obj, (int, float, str, bool, type(None))):
        # Handle NumPy types
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        return obj
    elif isinstance(obj, (list, tuple)):
        return [_make_serializable(item) for item in obj]
    elif isinstance(obj, dict):
        return {key: _make_serializable(value) for key, value in obj.items()}
    elif hasattr(obj, 'item'):
        # Handle PyTorch tensors and NumPy arrays with single items
        try:
            return obj.item()
        except (ValueError, AttributeError):
            pass
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    
    # Default: convert to string
    return str(obj)