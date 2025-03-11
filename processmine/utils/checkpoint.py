# utils/checkpoint.py
"""
Model checkpointing utilities
"""
import os
import torch
import logging
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)

class CheckpointManager:
    """
    Manages model checkpoints and resumption
    """
    
    def __init__(self, checkpoint_dir: str, max_to_keep: int = 3):
        """
        Initialize checkpoint manager
        
        Args:
            checkpoint_dir: Directory to store checkpoints
            max_to_keep: Maximum number of checkpoints to retain
        """
        self.checkpoint_dir = checkpoint_dir
        self.max_to_keep = max_to_keep
        self.best_metric = float('inf')
        self.checkpoints = []
        
        os.makedirs(checkpoint_dir, exist_ok=True)
    
    def save(self, model: torch.nn.Module, optimizer: torch.optim.Optimizer, 
            epoch: int, metrics: Dict[str, float], is_best: bool = False) -> str:
        """
        Save a checkpoint
        
        Args:
            model: PyTorch model
            optimizer: PyTorch optimizer
            epoch: Current epoch
            metrics: Dictionary of metrics
            is_best: Whether this is the best model so far
            
        Returns:
            Path to saved checkpoint
        """
        # Create checkpoint data
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'metrics': metrics
        }
        
        # Create checkpoint filename
        filename = f"checkpoint_epoch_{epoch}.pt"
        filepath = os.path.join(self.checkpoint_dir, filename)
        
        # Save checkpoint
        torch.save(checkpoint, filepath)
        logger.info(f"Saved checkpoint to {filepath}")
        
        # Add to checkpoints list
        self.checkpoints.append(filepath)
        
        # Keep only max_to_keep checkpoints
        if len(self.checkpoints) > self.max_to_keep:
            # Remove oldest checkpoint
            old_checkpoint = self.checkpoints.pop(0)
            if os.path.exists(old_checkpoint):
                os.remove(old_checkpoint)
                logger.info(f"Removed old checkpoint: {old_checkpoint}")
        
        # Save best model if needed
        if is_best:
            best_path = os.path.join(self.checkpoint_dir, "best_model.pt")
            torch.save(checkpoint, best_path)
            logger.info(f"Saved best model to {best_path}")
        
        return filepath
    
    def load_latest(self, model: torch.nn.Module, 
                   optimizer: Optional[torch.optim.Optimizer] = None) -> Optional[Dict[str, Any]]:
        """
        Load latest checkpoint
        
        Args:
            model: PyTorch model
            optimizer: Optional PyTorch optimizer
            
        Returns:
            Checkpoint data or None if no checkpoint exists
        """
        # Find latest checkpoint
        checkpoints = [f for f in os.listdir(self.checkpoint_dir) 
                      if f.startswith("checkpoint_epoch_")]
        
        if not checkpoints:
            logger.info("No checkpoints found")
            return None
        
        # Get latest checkpoint
        latest = sorted(checkpoints, 
                      key=lambda x: int(x.split("_epoch_")[1].split(".")[0]))[-1]
        checkpoint_path = os.path.join(self.checkpoint_dir, latest)
        
        return self.load(model, optimizer, checkpoint_path)
    
    def load(self, model: torch.nn.Module, optimizer: Optional[torch.optim.Optimizer], 
            checkpoint_path: str) -> Optional[Dict[str, Any]]:
        """
        Load a specific checkpoint
        
        Args:
            model: PyTorch model
            optimizer: Optional PyTorch optimizer
            checkpoint_path: Path to checkpoint
            
        Returns:
            Checkpoint data or None if loading fails
        """
        if not os.path.exists(checkpoint_path):
            logger.error(f"Checkpoint {checkpoint_path} not found")
            return None
        
        try:
            checkpoint = torch.load(checkpoint_path)
            model.load_state_dict(checkpoint['model_state_dict'])
            
            if optimizer is not None and 'optimizer_state_dict' in checkpoint:
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            
            logger.info(f"Loaded checkpoint from {checkpoint_path}")
            return checkpoint
        except Exception as e:
            logger.error(f"Error loading checkpoint: {e}")
            return None
    
    def load_best(self, model: torch.nn.Module, 
                 optimizer: Optional[torch.optim.Optimizer] = None) -> Optional[Dict[str, Any]]:
        """
        Load best model checkpoint
        
        Args:
            model: PyTorch model
            optimizer: Optional PyTorch optimizer
            
        Returns:
            Checkpoint data or None if no best model exists
        """
        best_path = os.path.join(self.checkpoint_dir, "best_model.pt")
        
        if not os.path.exists(best_path):
            logger.info("No best model checkpoint found")
            return None
        
        return self.load(model, optimizer, best_path)