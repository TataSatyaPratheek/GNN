#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Mixed precision training utilities for process mining models
"""

import torch
import time
import logging
from tqdm import tqdm
import numpy as np
from typing import Dict, Any, Optional, Tuple, Callable, List, Union

logger = logging.getLogger(__name__)

def train_with_amp(
    model: torch.nn.Module,
    train_loader: Any,
    val_loader: Any,
    criterion: Any,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epochs: int = 20,
    patience: int = 5,
    model_path: Optional[str] = None,
    loss_calculator: Optional[Callable] = None,
    callbacks: Optional[List[Callable]] = None
) -> Tuple[torch.nn.Module, Dict[str, Any]]:
    """
    Train model with Automatic Mixed Precision (AMP) for better performance on GPUs
    
    Args:
        model: PyTorch model to train
        train_loader: Training data loader
        val_loader: Validation data loader
        criterion: Loss function
        optimizer: PyTorch optimizer
        device: Device to train on (CPU, CUDA, MPS)
        epochs: Number of training epochs
        patience: Patience for early stopping
        model_path: Path to save best model
        loss_calculator: Custom function to calculate loss
        callbacks: List of callback functions to call after each epoch
        
    Returns:
        Tuple of (trained model, training metrics)
    """
    # Check if AMP is supported (CUDA only)
    use_amp = device.type == 'cuda' and hasattr(torch.cuda, 'amp')
    
    if use_amp:
        logger.info("Using Automatic Mixed Precision (AMP) for faster training")
        from torch.cuda.amp import autocast, GradScaler
        scaler = GradScaler()
    else:
        logger.info(f"AMP not available on {device.type}, using standard precision")
    
    # Default loss calculator function if none provided
    if loss_calculator is None:
        def default_loss_calculator(outputs, batch_data, criterion):
            # Handle different output formats
            if isinstance(outputs, dict):
                # Dictionary output
                if "task_pred" in outputs:
                    # Get graph-level targets if needed
                    if hasattr(batch_data, 'batch') and hasattr(batch_data, 'y'):
                        from torch_geometric.utils import scatter
                        # Use scatter to get most common target per graph
                        graph_targets = scatter(
                            batch_data.y,
                            batch_data.batch,
                            dim=0,
                            reduce="mode"
                        )
                    else:
                        graph_targets = batch_data.y
                    
                    # Check for multi-task models
                    if "time_pred" in outputs and hasattr(criterion, '__code__') and \
                       len(criterion.__code__.co_varnames) > 3:
                        # Dual-task loss
                        time_target = batch_data.time if hasattr(batch_data, 'time') else None
                        return criterion(outputs["task_pred"], graph_targets, 
                                         outputs["time_pred"], time_target)
                    
                    # Regular task prediction loss
                    task_loss = criterion(outputs["task_pred"], graph_targets)
                    
                    # Add diversity loss if available
                    if "diversity_loss" in outputs:
                        return task_loss + outputs["diversity_loss"]
                    
                    return task_loss
                
                # Generic dictionary output
                return criterion(outputs["logits"], batch_data.y)
            
            elif isinstance(outputs, tuple) and len(outputs) == 2:
                # Tuple of (logits, aux_loss)
                logits, aux_loss = outputs
                return criterion(logits, batch_data.y) + aux_loss
            
            # Standard output
            return criterion(outputs, batch_data.y)
        
        loss_calculator = default_loss_calculator
    
    # Initialize tracking variables
    best_val_loss = float('inf')
    patience_counter = 0
    train_losses = []
    val_losses = []
    
    # For early stopping
    best_model_state = None
    
    # Training loop
    for epoch in range(1, epochs + 1):
        epoch_start_time = time.time()
        
        # Training phase
        model.train()
        train_loss = 0.0
        train_batch_count = 0
        
        train_progress = tqdm(
            train_loader, 
            desc=f"Epoch {epoch}/{epochs} [Train]",
            leave=False
        )
        
        for batch_data in train_progress:
            # Move batch to device
            batch_data = batch_data.to(device)
            
            # Zero gradients
            optimizer.zero_grad()
            
            if use_amp:
                # Forward pass with AMP
                with autocast():
                    outputs = model(batch_data)
                    loss = loss_calculator(outputs, batch_data, criterion)
                
                # Backward pass with gradient scaling
                scaler.scale(loss).backward()
                
                # Unscale before gradient clipping
                scaler.unscale_(optimizer)
                
                # Gradient clipping - helps with stability
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                
                # Optimizer step with scaling
                scaler.step(optimizer)
                scaler.update()
            else:
                # Standard forward and backward pass
                outputs = model(batch_data)
                loss = loss_calculator(outputs, batch_data, criterion)
                loss.backward()
                
                # Gradient clipping for stability
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                
                optimizer.step()
            
            # Track loss
            train_loss += loss.item()
            train_batch_count += 1
            
            # Update progress
            train_progress.set_postfix({"loss": f"{loss.item():.4f}"})
        
        # Calculate average training loss
        avg_train_loss = train_loss / max(1, train_batch_count)
        train_losses.append(avg_train_loss)
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        val_batch_count = 0
        
        with torch.no_grad():
            val_progress = tqdm(
                val_loader, 
                desc=f"Epoch {epoch}/{epochs} [Valid]",
                leave=False
            )
            
            for batch_data in val_progress:
                # Move batch to device
                batch_data = batch_data.to(device)
                
                # Forward pass
                outputs = model(batch_data)
                loss = loss_calculator(outputs, batch_data, criterion)
                
                # Track loss
                val_loss += loss.item()
                val_batch_count += 1
                
                # Update progress
                val_progress.set_postfix({"loss": f"{loss.item():.4f}"})
        
        # Calculate average validation loss
        avg_val_loss = val_loss / max(1, val_batch_count)
        val_losses.append(avg_val_loss)
        
        # Calculate epoch time
        epoch_time = time.time() - epoch_start_time
        
        # Print epoch summary
        logger.info(
            f"Epoch {epoch}/{epochs}: "
            f"train_loss={avg_train_loss:.4f}, "
            f"val_loss={avg_val_loss:.4f}, "
            f"time={epoch_time:.2f}s"
        )
        
        # Early stopping check
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            
            # Save model state for later use
            best_model_state = model.state_dict().copy()
            
            # Save best model to disk if path provided
            if model_path:
                torch.save(model.state_dict(), model_path)
                logger.info(f"Saved best model to {model_path}")
        else:
            patience_counter += 1
            logger.info(f"No improvement for {patience_counter}/{patience} epochs")
            
            if patience_counter >= patience:
                logger.info(f"Early stopping triggered after {epoch} epochs")
                break
        
        # Run callbacks if provided
        if callbacks:
            for callback in callbacks:
                callback(epoch=epoch, model=model, train_loss=avg_train_loss, 
                         val_loss=avg_val_loss, optimizer=optimizer)
    
    # Load best model state if available
    if best_model_state:
        model.load_state_dict(best_model_state)
        logger.info("Restored best model state")
    
    # Return model and training metrics
    metrics = {
        "train_losses": train_losses,
        "val_losses": val_losses,
        "best_val_loss": best_val_loss,
        "epochs_trained": len(train_losses),
        "early_stopped": patience_counter >= patience
    }
    
    return model, metrics

def calculate_loss(outputs, batch_data, criterion):
    """
    Calculate loss based on model outputs and batch data
    
    Args:
        outputs: Model outputs (tensor, dict, or tuple)
        batch_data: PyTorch Geometric data batch
        criterion: Loss function
        
    Returns:
        Loss tensor
    """
    # Handle different output formats
    if isinstance(outputs, dict):
        # Dictionary output
        if "task_pred" in outputs:
            # Get graph-level targets if needed
            if hasattr(batch_data, 'batch') and hasattr(batch_data, 'y'):
                from torch_geometric.utils import scatter
                graph_targets = scatter(
                    batch_data.y,
                    batch_data.batch,
                    dim=0,
                    reduce="mode"  # Get most common target in each graph
                )
            else:
                graph_targets = batch_data.y
            
            # Check for multi-task criteria
            if "time_pred" in outputs and hasattr(criterion, '__code__') and \
               len(criterion.__code__.co_varnames) > 3:
                # Dual-task loss
                time_target = batch_data.time if hasattr(batch_data, 'time') else None
                return criterion(outputs["task_pred"], graph_targets, 
                                outputs["time_pred"], time_target)
            
            # Regular task prediction loss
            task_loss = criterion(outputs["task_pred"], graph_targets)
            
            # Add diversity loss if available
            if "diversity_loss" in outputs:
                return task_loss + outputs["diversity_loss"]
            
            return task_loss
        
        # Generic dictionary output
        return criterion(outputs["logits"], batch_data.y)
    
    elif isinstance(outputs, tuple) and len(outputs) == 2:
        # Output is (logits, aux_loss)
        logits, aux_loss = outputs
        return criterion(logits, batch_data.y) + aux_loss
    
    # Standard output format
    if hasattr(batch_data, 'batch') and hasattr(batch_data, 'y'):
        from torch_geometric.utils import scatter
        graph_targets = scatter(
            batch_data.y,
            batch_data.batch,
            dim=0,
            reduce="mode"
        )
        return criterion(outputs, graph_targets)
    else:
        # Standard classification loss
        return criterion(outputs, batch_data.y)

def create_lr_scheduler(optimizer, scheduler_type="cosine", **kwargs):
    """
    Create a learning rate scheduler
    
    Args:
        optimizer: PyTorch optimizer
        scheduler_type: Type of scheduler (cosine, step, plateau)
        **kwargs: Additional arguments for the scheduler
        
    Returns:
        PyTorch learning rate scheduler
    """
    if scheduler_type == "cosine":
        # Cosine annealing
        T_max = kwargs.get("T_max", 10)
        eta_min = kwargs.get("eta_min", 0)
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=T_max, eta_min=eta_min
        )
    
    elif scheduler_type == "step":
        # Step LR
        step_size = kwargs.get("step_size", 5)
        gamma = kwargs.get("gamma", 0.1)
        return torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=step_size, gamma=gamma
        )
    
    elif scheduler_type == "plateau":
        # Reduce on plateau
        factor = kwargs.get("factor", 0.1)
        patience = kwargs.get("patience", 2)
        min_lr = kwargs.get("min_lr", 1e-6)
        return torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=factor, 
            patience=patience, min_lr=min_lr
        )
    
    elif scheduler_type == "one_cycle":
        # One cycle policy
        max_lr = kwargs.get("max_lr", 0.01)
        total_steps = kwargs.get("total_steps", 100)
        return torch.optim.lr_scheduler.OneCycleLR(
            optimizer, max_lr=max_lr, total_steps=total_steps
        )
    
    else:
        logger.warning(f"Unknown scheduler type: {scheduler_type}, no scheduler will be used")
        return None