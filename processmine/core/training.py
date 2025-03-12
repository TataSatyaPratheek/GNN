"""
Memory-optimized training utilities with efficient CUDA management, batching strategies, 
and mixed precision training.
"""
import torch
import time
import logging
import numpy as np
from tqdm import tqdm
import gc
from typing import Dict, Any, Optional, Union, Tuple, Callable, List
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, matthews_corrcoef

logger = logging.getLogger(__name__)

class MemoryTracker:
    """Utility class to track memory usage during training"""
    
    def __init__(self, logging_interval: int = 5, device: Optional[torch.device] = None):
        """
        Initialize memory tracker
        
        Args:
            logging_interval: How often to log memory usage (in iterations)
            device: Torch device to track
        """
        self.logging_interval = logging_interval
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.is_cuda = self.device.type == 'cuda'
        self.current_step = 0
        self.peak_memory = 0
        self.history = []
    
    def step(self, manual_log: bool = False):
        """
        Track memory for current step
        
        Args:
            manual_log: Whether to force logging regardless of interval
        """
        self.current_step += 1
        
        if self.is_cuda:
            current = torch.cuda.memory_allocated(self.device) / (1024 ** 2)
            peak = torch.cuda.max_memory_allocated(self.device) / (1024 ** 2)
            self.peak_memory = max(self.peak_memory, peak)
            
            self.history.append(current)
            
            if manual_log or (self.current_step % self.logging_interval == 0):
                logger.debug(f"Step {self.current_step}: {current:.1f} MB, Peak: {peak:.1f} MB")
    
    def reset_peak(self):
        """Reset peak memory stats"""
        if self.is_cuda:
            torch.cuda.reset_peak_memory_stats(self.device)
    
    def summary(self):
        """Print memory usage summary"""
        if self.is_cuda:
            current = torch.cuda.memory_allocated(self.device) / (1024 ** 2)
            peak = self.peak_memory
            logger.info(f"Memory usage - Current: {current:.1f} MB, Peak: {peak:.1f} MB")
            return {"current_mb": current, "peak_mb": peak}
        return {"current_mb": 0, "peak_mb": 0}

def clear_memory(full_clear: bool = False):
    """
    Clear memory caches and unused objects
    
    Args:
        full_clear: Whether to perform a more aggressive memory clearing
    """
    # Python garbage collection
    gc.collect()
    
    # PyTorch CUDA cache
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        
        if full_clear:
            # Force synchronization
            torch.cuda.synchronize()

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
    use_amp: bool = False,
    clip_grad_norm: Optional[float] = None,
    lr_scheduler: Optional[Any] = None,
    memory_efficient: bool = True,
    track_memory: bool = False
) -> Tuple[torch.nn.Module, Dict[str, Any]]:
    """
    Unified training function with memory optimization, mixed precision, and advanced features
    
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
        clip_grad_norm: Max norm for gradient clipping (None to disable)
        lr_scheduler: Learning rate scheduler (None to disable)
        memory_efficient: Whether to use memory-efficient training
        track_memory: Whether to track memory usage
        
    Returns:
        Tuple of (trained model, training metrics)
    """
    # Set up device
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Memory tracker
    mem_tracker = MemoryTracker() if track_memory else None
    
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
        'learning_rates': [],
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
        
        # Clear memory before epoch
        if memory_efficient:
            clear_memory()
        
        for batch_idx, batch in enumerate(train_bar):
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
                
                # Gradient clipping if enabled
                if clip_grad_norm is not None:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad_norm)
                
                # Optimizer step with scaler
                scaler.step(optimizer)
                scaler.update()
            else:
                # Standard forward and backward pass
                outputs = model(batch)
                loss = _compute_loss(outputs, batch, criterion)
                loss.backward()
                
                # Gradient clipping if enabled
                if clip_grad_norm is not None:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad_norm)
                
                optimizer.step()
            
            # Track loss
            batch_size = _get_batch_size(batch)
            train_loss += loss.item() * batch_size
            train_samples += batch_size
            
            # Update progress bar
            train_bar.set_postfix({'loss': f"{loss.item():.4f}"})
            
            # Memory tracking
            if mem_tracker is not None:
                mem_tracker.step(batch_idx % 50 == 0)  # Log every 50 batches
            
            # Aggressive memory clearing for very large models
            if memory_efficient and batch_idx % 50 == 0:
                # Clear unnecessary memory
                del loss, outputs
                if torch.cuda.is_available():
                    # Don't empty cache too often as it can slow down training
                    if batch_idx % 200 == 0:
                        torch.cuda.empty_cache()
        
        # Calculate average training loss
        avg_train_loss = train_loss / max(train_samples, 1)
        metrics_history['train_loss'].append(avg_train_loss)
        
        # Update learning rate scheduler if provided
        if lr_scheduler is not None:
            current_lr = optimizer.param_groups[0]['lr']
            metrics_history['learning_rates'].append(current_lr)
            
            if isinstance(lr_scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                # This type needs validation loss
                if val_loader is not None:
                    # Will update after validation
                    pass
                else:
                    lr_scheduler.step(avg_train_loss)
            else:
                # Other schedulers update per epoch
                lr_scheduler.step()
        
        # Validation phase (if validation loader provided)
        val_loss = 0.0
        val_samples = 0
        
        if val_loader is not None:
            model.eval()
            
            # Clear memory before validation
            if memory_efficient:
                clear_memory()
            
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
            
            # Update learning rate scheduler if it's ReduceLROnPlateau
            if lr_scheduler is not None and isinstance(lr_scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                lr_scheduler.step(avg_val_loss)
            
            # Early stopping check
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                patience_counter = 0
                
                # Save best model state
                # Use memory-efficient copy to CPU
                best_model_state = {
                    key: value.cpu().clone() 
                    for key, value in model.state_dict().items()
                }
                
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
        
        # Memory usage summary
        if mem_tracker is not None:
            mem_tracker.summary()
        
        # Call callback if provided
        if callback is not None:
            callback(epoch=epoch, model=model, train_loss=avg_train_loss,
                    val_loss=avg_val_loss if val_loader is not None else None)
    
    # Restore best model if validation was used
    if val_loader is not None and best_model_state is not None:
        model.load_state_dict(best_model_state)
        logger.info("Restored best model state")
    
    # Final memory cleanup
    if memory_efficient:
        clear_memory(full_clear=True)
    
    # Add memory tracking data if available
    if mem_tracker is not None:
        metrics_history['memory'] = mem_tracker.summary()
    
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
            
            # Get graph-level targets efficiently
            if hasattr(batch, 'batch') and hasattr(batch, 'y'):
                targets = _get_graph_targets(batch.y, batch.batch)
            else:
                targets = batch.y
            
            # Compute task loss
            task_loss = criterion(preds, targets)
            
            # Add diversity loss if available
            if "diversity_loss" in outputs:
                return task_loss + outputs["diversity_loss"] * outputs.get("diversity_weight", 0.1)
            
            return task_loss
    
    # Handle tuple outputs (e.g., output and auxiliary loss)
    elif isinstance(outputs, tuple) and len(outputs) == 2:
        preds, aux_loss = outputs
        if hasattr(batch, 'batch') and hasattr(batch, 'y'):
            targets = _get_graph_targets(batch.y, batch.batch)
        else:
            targets = batch.y
        
        return criterion(preds, targets) + aux_loss
    
    # Standard case (direct output tensor)
    else:
        if hasattr(batch, 'batch') and hasattr(batch, 'y'):
            targets = _get_graph_targets(batch.y, batch.batch)
        else:
            targets = batch.y
        
        return criterion(outputs, targets)

def _get_graph_targets(node_targets, batch_indices):
    """
    Extract graph-level targets from node-level targets efficiently
    
    Args:
        node_targets: Node-level targets
        batch_indices: Batch assignment indices
        
    Returns:
        Graph-level targets
    """
    # Get unique graphs
    unique_graphs = torch.unique(batch_indices)
    graph_targets = []
    
    # Extract most common target for each graph
    for g in unique_graphs:
        # Get targets for this graph
        graph_mask = (batch_indices == g)
        graph_node_targets = node_targets[graph_mask]
        
        # Find most common target (mode)
        if len(graph_node_targets) > 0:
            values, counts = torch.unique(graph_node_targets, return_counts=True)
            mode_idx = torch.argmax(counts)
            graph_targets.append(values[mode_idx])
        else:
            # Fallback if no targets (should not happen)
            graph_targets.append(torch.tensor(0, device=node_targets.device))
    
    # Convert to tensor
    return torch.stack(graph_targets)

def _get_batch_size(batch):
    """Extract batch size from different batch formats"""
    if hasattr(batch, 'batch'):
        # PyG batch - use unique batch indices
        return torch.unique(batch.batch).size(0)
    elif hasattr(batch, 'size') and isinstance(batch.size(), tuple):
        # Regular tensor batch - use first dimension
        return batch.size(0)
    else:
        # Fallback
        return 1

def evaluate_model(
    model: torch.nn.Module,
    test_loader: Any,
    device: Optional[torch.device] = None,
    criterion: Optional[Any] = None,
    detailed: bool = True
) -> Dict[str, Any]:
    """
    Evaluate model on test data with expanded metrics
    
    Args:
        model: PyTorch model to evaluate
        test_loader: Test data loader
        device: Computing device
        criterion: Loss function (optional)
        detailed: Whether to compute detailed metrics
        
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
    test_loss = 0.0 if criterion is not None else None
    
    # Evaluate without gradient tracking
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Evaluating"):
            batch = batch.to(device)
            outputs = model(batch)
            
            # Calculate loss if criterion provided
            if criterion is not None:
                loss = _compute_loss(outputs, batch, criterion)
                test_loss += loss.item() * _get_batch_size(batch)
            
            # Extract predictions
            if isinstance(outputs, dict):
                outputs = outputs.get("task_pred", outputs)
            elif isinstance(outputs, tuple):
                outputs = outputs[0]
            
            # Get predicted classes
            _, preds = torch.max(outputs, dim=1)
            
            # Get targets
            if hasattr(batch, 'batch') and hasattr(batch, 'y'):
                targets = _get_graph_targets(batch.y, batch.batch)
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
        'recall': recall_score(all_labels, all_preds, average='macro', zero_division=0),
        'mcc': matthews_corrcoef(all_labels, all_preds)
    }
    
    # Add loss if calculated
    if test_loss is not None:
        metrics['test_loss'] = test_loss / len(test_loader) if len(test_loader) > 0 else 0.0
    
    # Add class-wise metrics if detailed
    if detailed:
        # Class-wise F1, precision, and recall
        unique_classes = np.unique(np.concatenate([all_labels, all_preds]))
        class_metrics = {}
        
        for cls in unique_classes:
            class_metrics[int(cls)] = {
                'f1': f1_score(all_labels, all_preds, average=None, zero_division=0)[int(cls)],
                'precision': precision_score(all_labels, all_preds, average=None, zero_division=0)[int(cls)],
                'recall': recall_score(all_labels, all_preds, average=None, zero_division=0)[int(cls)]
            }
        
        metrics['class_metrics'] = class_metrics
        
        # Confusion matrix
        from sklearn.metrics import confusion_matrix
        metrics['confusion_matrix'] = confusion_matrix(all_labels, all_preds).tolist()
    
    # Log summary of results
    logger.info(f"Evaluation results:")
    logger.info(f"  Accuracy: {metrics['accuracy']:.4f}")
    logger.info(f"  F1 (weighted): {metrics['f1_weighted']:.4f}")
    logger.info(f"  MCC: {metrics['mcc']:.4f}")
    
    return metrics, all_preds, all_labels

def create_optimizer(
    model: torch.nn.Module, 
    optimizer_type: str = 'adamw',
    lr: float = 0.001,
    weight_decay: float = 5e-4,
    momentum: float = 0.9,
    layer_decay: Optional[float] = None
) -> torch.optim.Optimizer:
    """
    Create optimizer with advanced options
    
    Args:
        model: Model to optimize
        optimizer_type: Type of optimizer ('adam', 'adamw', 'sgd', 'rmsprop')
        lr: Learning rate
        weight_decay: Weight decay factor
        momentum: Momentum factor (SGD only)
        layer_decay: Optional layer-wise learning rate decay factor
        
    Returns:
        PyTorch optimizer
    """
    # Get model parameters
    if layer_decay is not None:
        # Group parameters by layer
        params_with_decay = []
        params_without_decay = []
        
        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue
                
            # Skip batch norm and bias terms for weight decay
            if name.endswith('.bias') or 'norm' in name or 'bn' in name:
                params_without_decay.append(param)
            else:
                params_with_decay.append(param)
        
        # Create optimizer with param groups
        param_groups = [
            {'params': params_with_decay, 'weight_decay': weight_decay},
            {'params': params_without_decay, 'weight_decay': 0.0}
        ]
    else:
        # Single param group
        param_groups = model.parameters()
    
    # Create optimizer based on type
    if optimizer_type.lower() == 'adam':
        return torch.optim.Adam(param_groups, lr=lr, weight_decay=weight_decay)
    elif optimizer_type.lower() == 'adamw':
        return torch.optim.AdamW(param_groups, lr=lr, weight_decay=weight_decay)
    elif optimizer_type.lower() == 'sgd':
        return torch.optim.SGD(param_groups, lr=lr, momentum=momentum, weight_decay=weight_decay)
    elif optimizer_type.lower() == 'rmsprop':
        return torch.optim.RMSprop(param_groups, lr=lr, weight_decay=weight_decay, momentum=momentum)
    else:
        raise ValueError(f"Unknown optimizer type: {optimizer_type}")

def create_lr_scheduler(
    optimizer: torch.optim.Optimizer,
    scheduler_type: str = 'cosine',
    epochs: int = 100,
    warmup_epochs: int = 5,
    min_lr: float = 1e-6,
    patience: int = 10,
    factor: float = 0.1
) -> torch.optim.lr_scheduler._LRScheduler:
    """
    Create learning rate scheduler with advanced options
    
    Args:
        optimizer: Optimizer to schedule
        scheduler_type: Type of scheduler ('cosine', 'step', 'plateau', 'linear', 'constant')
        epochs: Total epochs
        warmup_epochs: Epochs for linear warmup
        min_lr: Minimum learning rate
        patience: Patience for ReduceLROnPlateau
        factor: Reduction factor for step and plateau schedulers
        
    Returns:
        PyTorch learning rate scheduler
    """
    if scheduler_type == 'cosine':
        # Cosine decay with warmup
        if warmup_epochs > 0:
            # Linear warmup followed by cosine decay
            warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
                optimizer, 
                start_factor=0.1, 
                end_factor=1.0, 
                total_iters=warmup_epochs
            )
            
            cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=epochs - warmup_epochs,
                eta_min=min_lr
            )
            
            # Chain schedulers
            return torch.optim.lr_scheduler.SequentialLR(
                optimizer,
                schedulers=[warmup_scheduler, cosine_scheduler],
                milestones=[warmup_epochs]
            )
        else:
            # Just cosine decay
            return torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=epochs,
                eta_min=min_lr
            )
    
    elif scheduler_type == 'step':
        # Step decay
        step_size = max(epochs // 3, 1)  # Default: 3 steps over the training
        return torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=step_size,
            gamma=factor
        )
    
    elif scheduler_type == 'plateau':
        # Reduce on plateau
        return torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=factor,
            patience=patience,
            min_lr=min_lr
        )
    
    elif scheduler_type == 'linear':
        # Linear decay
        return torch.optim.lr_scheduler.LinearLR(
            optimizer,
            start_factor=1.0,
            end_factor=min_lr,
            total_iters=epochs
        )
    
    elif scheduler_type == 'constant':
        # Constant LR
        return torch.optim.lr_scheduler.LambdaLR(
            optimizer,
            lambda epoch: 1.0
        )
    
    else:
        raise ValueError(f"Unknown scheduler type: {scheduler_type}")

def compute_class_weights(df, num_classes, method='balanced'):
    """
    Compute class weights to handle imbalanced datasets
    
    Args:
        df: Process data dataframe
        num_classes: Number of classes
        method: Weight calculation method ('balanced', 'log', 'sqrt')
        
    Returns:
        Tensor of class weights
    """
    from sklearn.utils.class_weight import compute_class_weight
    
    # Extract labels
    labels = df["next_task"].values
    
    # Get unique classes
    unique_classes = np.unique(labels)
    
    # Create weight array (default to 1.0)
    weights = np.ones(num_classes, dtype=np.float32)
    
    # Compute weights based on method
    if method == 'balanced':
        # Use sklearn's balanced method
        class_weights = compute_class_weight('balanced', classes=unique_classes, y=labels)
        weights[unique_classes] = class_weights
    
    elif method == 'log':
        # Log-based weighting (less aggressive than balanced)
        class_counts = np.bincount(labels, minlength=num_classes)
        valid_counts = class_counts[class_counts > 0]
        total = len(labels)
        
        # Log-based inverse weighting
        for cls in unique_classes:
            count = class_counts[cls]
            weights[cls] = np.log(total / max(count, 1))
        
        # Normalize weights
        weights = weights / np.min(weights[weights > 0])
    
    elif method == 'sqrt':
        # Square root based weighting (even less aggressive)
        class_counts = np.bincount(labels, minlength=num_classes)
        total = len(labels)
        
        for cls in unique_classes:
            count = class_counts[cls]
            weights[cls] = np.sqrt(total / max(count, 1))
        
        # Normalize weights
        weights = weights / np.min(weights[weights > 0])
    
    # Convert to tensor
    return torch.tensor(weights, dtype=torch.float32)

def setup_training(
    model: torch.nn.Module,
    df: pd.DataFrame,
    batch_size: int = 32,
    val_ratio: float = 0.2,
    test_ratio: float = 0.1,
    device: Optional[torch.device] = None,
    seed: int = 42,
    class_weight_method: str = 'balanced',
    optimizer_type: str = 'adamw',
    lr: float = 0.001,
    weight_decay: float = 5e-4,
    scheduler_type: str = 'cosine',
    epochs: int = 100,
    warmup_epochs: int = 5,
    memory_efficient: bool = True
) -> Dict[str, Any]:
    """
    Complete training setup with data split, model, optimizer, and loss function
    
    Args:
        model: PyTorch model to train
        df: Process data dataframe
        batch_size: Batch size
        val_ratio: Validation set ratio
        test_ratio: Test set ratio
        device: Computing device
        seed: Random seed
        class_weight_method: Method for class weight calculation
        optimizer_type: Type of optimizer
        lr: Learning rate
        weight_decay: Weight decay factor
        scheduler_type: Type of learning rate scheduler
        epochs: Total epochs
        warmup_epochs: Epochs for linear warmup
        memory_efficient: Whether to use memory-efficient training
        
    Returns:
        Dictionary with training setup
    """
    # Set random seed for reproducibility
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    
    # Set device
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Determine number of classes
    num_classes = df["next_task"].nunique()
    
    # Compute class weights for handling imbalance
    class_weights = compute_class_weights(df, num_classes, method=class_weight_method)
    class_weights = class_weights.to(device)
    
    # Create criterion with class weights
    criterion = torch.nn.CrossEntropyLoss(weight=class_weights)
    
    # Create optimizer
    optimizer = create_optimizer(
        model,
        optimizer_type=optimizer_type,
        lr=lr,
        weight_decay=weight_decay
    )
    
    # Create learning rate scheduler
    lr_scheduler = create_lr_scheduler(
        optimizer,
        scheduler_type=scheduler_type,
        epochs=epochs,
        warmup_epochs=warmup_epochs
    )
    
    # Return complete setup
    return {
        'model': model,
        'device': device,
        'criterion': criterion,
        'optimizer': optimizer,
        'lr_scheduler': lr_scheduler,
        'class_weights': class_weights,
        'num_classes': num_classes,
        'memory_efficient': memory_efficient,
        'batch_size': batch_size,
        'epochs': epochs
    }