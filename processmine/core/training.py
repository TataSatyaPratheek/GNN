# core/training.py
import torch
import time
import logging
import numpy as np
from tqdm import tqdm
from typing import Dict, Any, Optional, Union, Tuple, Callable
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

logger = logging.getLogger(__name__)

def train_model(
    model: torch.nn.Module,
    train_loader: Any,
    val_loader: Optional[Any] = None,
    optimizer: Optional[torch.optim.Optimizer] = None,
    criterion: Optional[Any] = None,
    device: Optional[torch.device] = None,
    epochs: int = 10,
    patience: int = 5,
    model_path: Optional[str] = None,
    callback: Optional[Callable] = None,
    use_amp: bool = False
) -> Tuple[torch.nn.Module, Dict[str, Any]]:
    """
    Unified training function for all model types
    
    Args:
        model: PyTorch model to train
        train_loader: Training data loader
        val_loader: Validation data loader (optional)
        optimizer: PyTorch optimizer (default: AdamW)
        criterion: Loss function (default: CrossEntropyLoss)
        device: Computing device (default: CUDA if available, else CPU)
        epochs: Number of training epochs
        patience: Early stopping patience
        model_path: Path to save best model
        callback: Optional callback function after each epoch
        use_amp: Whether to use automatic mixed precision
        
    Returns:
        Tuple of (trained model, training metrics)
    """
    # Set up device
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Move model to device
    model = model.to(device)
    
    # Set up optimizer if not provided
    if optimizer is None:
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=5e-4)
    
    # Set up loss function if not provided
    if criterion is None:
        criterion = torch.nn.CrossEntropyLoss()
    
    # Set up AMP if requested and available
    scaler = None
    if use_amp and device.type == 'cuda' and hasattr(torch.cuda, 'amp'):
        from torch.cuda.amp import autocast, GradScaler
        scaler = GradScaler()
        logger.info("Using automatic mixed precision training")
    
    # Initialize tracking variables
    best_val_loss = float('inf')
    patience_counter = 0
    metrics_history = {
        'train_loss': [],
        'val_loss': [],
        'epoch_times': []
    }
    best_model_state = None
    
    # Training loop
    for epoch in range(1, epochs + 1):
        epoch_start = time.time()
        
        # Training phase
        model.train()
        train_loss = 0.0
        train_samples = 0
        
        # Progress bar for training
        train_bar = tqdm(train_loader, desc=f"Epoch {epoch}/{epochs} [Train]")
        
        for batch in train_bar:
            # Move batch to device
            batch = batch.to(device)
            
            # Reset gradients
            optimizer.zero_grad()
            
            if use_amp and scaler is not None:
                # Forward pass with AMP
                with autocast():
                    outputs = model(batch)
                    loss = _compute_loss(outputs, batch, criterion)
                
                # Backward pass with gradient scaling
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                # Standard forward and backward pass
                outputs = model(batch)
                loss = _compute_loss(outputs, batch, criterion)
                loss.backward()
                optimizer.step()
            
            # Track loss
            batch_size = _get_batch_size(batch)
            train_loss += loss.item() * batch_size
            train_samples += batch_size
            
            # Update progress bar
            train_bar.set_postfix({'loss': f"{loss.item():.4f}"})
        
        # Calculate average training loss
        avg_train_loss = train_loss / max(train_samples, 1)
        metrics_history['train_loss'].append(avg_train_loss)
        
        # Validation phase (if validation loader provided)
        val_loss = 0.0
        val_samples = 0
        
        if val_loader is not None:
            model.eval()
            val_bar = tqdm(val_loader, desc=f"Epoch {epoch}/{epochs} [Valid]")
            
            with torch.no_grad():
                for batch in val_bar:
                    batch = batch.to(device)
                    outputs = model(batch)
                    loss = _compute_loss(outputs, batch, criterion)
                    
                    # Track loss
                    batch_size = _get_batch_size(batch)
                    val_loss += loss.item() * batch_size
                    val_samples += batch_size
                    
                    # Update progress bar
                    val_bar.set_postfix({'loss': f"{loss.item():.4f}"})
            
            # Calculate average validation loss
            avg_val_loss = val_loss / max(val_samples, 1)
            metrics_history['val_loss'].append(avg_val_loss)
            
            # Early stopping check
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                patience_counter = 0
                
                # Save best model state
                best_model_state = {key: value.cpu().clone() for key, value in model.state_dict().items()}
                
                # Save to disk if path provided
                if model_path:
                    torch.save(model.state_dict(), model_path)
                    logger.info(f"Saved best model to {model_path}")
            else:
                patience_counter += 1
                logger.info(f"No improvement for {patience_counter}/{patience} epochs")
                
                if patience_counter >= patience:
                    logger.info(f"Early stopping triggered after {epoch} epochs")
                    break
        
        # Record epoch time
        epoch_time = time.time() - epoch_start
        metrics_history['epoch_times'].append(epoch_time)
        
        # Log epoch summary
        if val_loader is not None:
            logger.info(f"Epoch {epoch}/{epochs}: train_loss={avg_train_loss:.4f}, "
                       f"val_loss={avg_val_loss:.4f}, time={epoch_time:.2f}s")
        else:
            logger.info(f"Epoch {epoch}/{epochs}: train_loss={avg_train_loss:.4f}, "
                       f"time={epoch_time:.2f}s")
        
        # Call callback if provided
        if callback is not None:
            callback(epoch=epoch, model=model, train_loss=avg_train_loss,
                    val_loss=avg_val_loss if val_loader is not None else None)
    
    # Restore best model if validation was used
    if val_loader is not None and best_model_state is not None:
        model.load_state_dict(best_model_state)
        logger.info("Restored best model state")
    
    # Return model and metrics
    return model, metrics_history

def _compute_loss(outputs, batch, criterion):
    """
    Compute loss based on model outputs and batch data.
    Handles different output formats (tensor, dict, tuple).
    """
    # Handle dictionary outputs (e.g., from diverse attention models)
    if isinstance(outputs, dict):
        if "task_pred" in outputs:
            # Get predictions and targets
            preds = outputs["task_pred"]
            
            # Handle PyG batch targets
            if hasattr(batch, 'batch') and hasattr(batch, 'y'):
                from torch_geometric.nn import global_mean_pool
                # Get graph-level targets (assuming first node's target per graph)
                node_targets = batch.y
                graph_indices = batch.batch
                unique_graphs = torch.unique(graph_indices)
                targets = []
                
                for g in unique_graphs:
                    graph_mask = (graph_indices == g)
                    graph_targets = node_targets[graph_mask]
                    # Use most common target (mode) for the graph
                    if len(graph_targets) > 0:
                        values, counts = torch.unique(graph_targets, return_counts=True)
                        targets.append(values[torch.argmax(counts)])
                    else:
                        targets.append(torch.tensor(0, device=node_targets.device))
                
                targets = torch.stack(targets)
            else:
                targets = batch.y
            
            # Compute task loss
            task_loss = criterion(preds, targets)
            
            # Add diversity loss if available
            if "diversity_loss" in outputs:
                return task_loss + outputs["diversity_loss"]
            
            return task_loss
    
    # Handle tuple outputs (e.g., output and auxiliary loss)
    elif isinstance(outputs, tuple) and len(outputs) == 2:
        preds, aux_loss = outputs
        if hasattr(batch, 'batch') and hasattr(batch, 'y'):
            from torch_geometric.nn import global_mean_pool
            # Same graph-level target extraction as above
            node_targets = batch.y
            graph_indices = batch.batch
            unique_graphs = torch.unique(graph_indices)
            targets = []
            
            for g in unique_graphs:
                graph_mask = (graph_indices == g)
                graph_targets = node_targets[graph_mask]
                if len(graph_targets) > 0:
                    values, counts = torch.unique(graph_targets, return_counts=True)
                    targets.append(values[torch.argmax(counts)])
                else:
                    targets.append(torch.tensor(0, device=node_targets.device))
            
            targets = torch.stack(targets)
        else:
            targets = batch.y
        
        return criterion(preds, targets) + aux_loss
    
    # Standard case (direct output tensor)
    else:
        if hasattr(batch, 'batch') and hasattr(batch, 'y'):
            # Same graph-level target extraction
            node_targets = batch.y
            graph_indices = batch.batch
            unique_graphs = torch.unique(graph_indices)
            targets = []
            
            for g in unique_graphs:
                graph_mask = (graph_indices == g)
                graph_targets = node_targets[graph_mask]
                if len(graph_targets) > 0:
                    values, counts = torch.unique(graph_targets, return_counts=True)
                    targets.append(values[torch.argmax(counts)])
                else:
                    targets.append(torch.tensor(0, device=node_targets.device))
            
            targets = torch.stack(targets)
        else:
            targets = batch.y
        
        return criterion(outputs, targets)

def _get_batch_size(batch):
    """Extract batch size from different batch formats"""
    if hasattr(batch, 'batch'):
        # PyG batch
        return torch.unique(batch.batch).size(0)
    elif hasattr(batch, 'size') and isinstance(batch.size(), tuple):
        # Regular tensor batch
        return batch.size(0)
    else:
        # Fallback
        return 1

def evaluate_model(
    model: torch.nn.Module,
    test_loader: Any,
    device: Optional[torch.device] = None
) -> Dict[str, Any]:
    """
    Evaluate model on test data
    
    Args:
        model: PyTorch model to evaluate
        test_loader: Test data loader
        device: Computing device
        
    Returns:
        Dictionary of evaluation metrics
    """
    # Set up device
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Move model to device
    model = model.to(device)
    
    # Set model to evaluation mode
    model.eval()
    
    # Initialize result arrays
    all_preds = []
    all_labels = []
    
    # Evaluate without gradient tracking
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Evaluating"):
            batch = batch.to(device)
            outputs = model(batch)
            
            # Extract predictions
            if isinstance(outputs, dict):
                outputs = outputs.get("task_pred", outputs)
            elif isinstance(outputs, tuple):
                outputs = outputs[0]
            
            # Get predicted classes
            _, preds = torch.max(outputs, dim=1)
            
            # Get targets (same as in _compute_loss)
            if hasattr(batch, 'batch') and hasattr(batch, 'y'):
                node_targets = batch.y
                graph_indices = batch.batch
                unique_graphs = torch.unique(graph_indices)
                targets = []
                
                for g in unique_graphs:
                    graph_mask = (graph_indices == g)
                    graph_targets = node_targets[graph_mask]
                    if len(graph_targets) > 0:
                        values, counts = torch.unique(graph_targets, return_counts=True)
                        targets.append(values[torch.argmax(counts)])
                    else:
                        targets.append(torch.tensor(0, device=node_targets.device))
                
                targets = torch.stack(targets)
            else:
                targets = batch.y
            
            # Collect predictions and labels
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(targets.cpu().numpy())
    
    # Convert to numpy arrays
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    
    # Calculate metrics
    metrics = {
        'accuracy': accuracy_score(all_labels, all_preds),
        'f1_macro': f1_score(all_labels, all_preds, average='macro', zero_division=0),
        'f1_weighted': f1_score(all_labels, all_preds, average='weighted', zero_division=0),
        'precision': precision_score(all_labels, all_preds, average='macro', zero_division=0),
        'recall': recall_score(all_labels, all_preds, average='macro', zero_division=0)
    }
    
    return metrics