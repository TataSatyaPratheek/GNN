#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Training utilities for process mining models with memory optimization
"""

import gc
import os
import time
import torch
import numpy as np
from tqdm import tqdm
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score, matthews_corrcoef
)
from colorama import Fore, Style

from processmine.utils.memory import MemoryOptimizer
# Import optimized components
from processmine.utils.dependency import check_dependency
from processmine.utils.checkpoint import CheckpointManager, TrainingCheckpoint

def setup_device():
    """
    Setup computing device with robust fallback mechanisms
    
    Returns:
        Torch device (CUDA, MPS, or CPU)
    """
    print(f"{Fore.CYAN}🔍 Detecting and configuring optimal device...{Style.RESET_ALL}")
    
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
            print(f"{Fore.GREEN}✅ Found GPU: {device_name}{Style.RESET_ALL}")
            print(f"   CUDA Version: {torch.version.cuda}")
            print(f"   GPU Memory: {memory_total:.2f} GB total, {memory_free:.2f} GB free")
            
            # Ensure sufficient memory is available (at least 1GB)
            if memory_free < 1.0:
                print(f"{Fore.YELLOW}⚠️ Low GPU memory: {memory_free:.2f} GB available{Style.RESET_ALL}")
                print(f"   Will use CPU for model parameters to avoid OOM errors")
                # Return GPU but will manage tensors carefully
            
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
            print(f"{Fore.YELLOW}⚠️ CUDA error: {e}{Style.RESET_ALL}")
            print("   Falling back to CPU")
    
    # Check for Apple Silicon MPS support
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        try:
            device = torch.device("mps")
            print(f"{Fore.GREEN}✅ Using Apple Silicon GPU (MPS){Style.RESET_ALL}")
            return device
        except Exception as e:
            print(f"{Fore.YELLOW}⚠️ MPS error: {e}{Style.RESET_ALL}")
            print("   Falling back to CPU")
    
    # Use CPU as fallback
    device = torch.device("cpu")
    print(f"{Fore.YELLOW}⚠️ No GPU available. Using CPU for computation.{Style.RESET_ALL}")
    
    # Get CPU info
    import platform
    import psutil
    print(f"   CPU: {platform.processor()}")
    print(f"   Available cores: {psutil.cpu_count(logical=False)} physical, {psutil.cpu_count()} logical")
    print(f"   Available memory: {psutil.virtual_memory().available / (1024**3):.2f} GB")
    
    return device

def setup_model(args, df, task_encoder, resource_encoder, device):
    """
    Setup model with memory optimization
    
    Args:
        args: Command-line arguments
        df: Process data dataframe
        task_encoder: Task label encoder
        resource_encoder: Resource label encoder
        device: Torch device
        
    Returns:
        Instantiated model (on appropriate device)
    """
    print(f"{Fore.CYAN}🧠 Setting up {args.model_type} model...{Style.RESET_ALL}")
    
    # Clear memory before creating model
    MemoryOptimizer.clear_memory()
    
    # Calculate input dimension
    num_features = len([col for col in df.columns if col.startswith('feat_')])
    num_classes = len(task_encoder.classes_)
    num_resources = len(resource_encoder.classes_)
    
    print(f"   Input features: {num_features}")
    print(f"   Output classes: {num_classes}")
    
    # Get model parameters
    hidden_dim = args.hidden_dim
    num_layers = args.num_layers
    dropout = args.dropout
    
    # Import appropriate model based on model type
    if args.model_type == 'mlp':
        from processmine.models.baseline.mlp import OptimizedMLP
        model = OptimizedMLP(
            input_dim=num_features,
            hidden_dims=[hidden_dim, hidden_dim // 2],
            output_dim=num_classes,
            dropout=dropout
        )
    
    elif args.model_type == 'lstm':
        from processmine.models.sequence.lstm import OptimizedLSTM
        model = OptimizedLSTM(
            num_classes=num_classes,
            embedding_dim=hidden_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            dropout=dropout
        )
    
    elif args.model_type == 'basic_gat':
        from processmine.models.gnn.gat import OptimizedGAT
        model = OptimizedGAT(
            input_dim=num_features,
            hidden_dim=hidden_dim,
            output_dim=num_classes,
            num_layers=num_layers,
            heads=args.num_heads,
            dropout=dropout
        )
    
    elif args.model_type == 'positional_gat':
        from processmine.models.gnn.positional import OptimizedPositionalGAT
        model = OptimizedPositionalGAT(
            input_dim=num_features,
            hidden_dim=hidden_dim,
            output_dim=num_classes,
            pos_dim=args.pos_dim,
            num_layers=num_layers,
            heads=args.num_heads,
            dropout=dropout
        )
    
    elif args.model_type == 'diverse_gat':
        from processmine.models.gnn.diverse import OptimizedDiverseGAT
        model = OptimizedDiverseGAT(
            input_dim=num_features,
            hidden_dim=hidden_dim,
            output_dim=num_classes,
            num_layers=num_layers,
            heads=args.num_heads,
            dropout=dropout,
            diversity_weight=args.diversity_weight
        )
    
    elif args.model_type == 'enhanced_gnn':
        from processmine.models.gnn.enhanced import OptimizedEnhancedGNN
        model = OptimizedEnhancedGNN(
            input_dim=num_features,
            hidden_dim=hidden_dim,
            output_dim=num_classes,
            pos_dim=args.pos_dim,
            num_layers=num_layers,
            heads=args.num_heads,
            dropout=dropout,
            diversity_weight=args.diversity_weight,
            predict_time=args.predict_time
        )
    
    else:
        raise ValueError(f"Unknown model type: {args.model_type}")
    
    # Safely move model to device
    model = MemoryOptimizer.safe_to_device(model, device)
    
    return model

def setup_optimizer_and_loss(model, args, class_weights=None, device=None):
    """
    Set up optimizer and loss function with memory optimization
    
    Args:
        model: The model
        args: Command-line arguments
        class_weights: Optional class weights tensor
        device: Torch device
        
    Returns:
        Tuple of (optimizer, criterion)
    """
    print(f"{Fore.CYAN}⚙️ Setting up optimizer and loss function...{Style.RESET_ALL}")
    
    # Default learning rate and weight decay if not specified
    lr = args.lr if hasattr(args, 'lr') else 0.001
    weight_decay = args.weight_decay if hasattr(args, 'weight_decay') else 5e-4
    
    # Determine if class weights should be on CPU to save GPU memory
    weights_on_cpu = False
    if device is not None and device.type == 'cuda' and class_weights is not None:
        # Check if GPU is low on memory
        free_memory = torch.cuda.get_device_properties(device).total_memory - torch.cuda.memory_allocated(device)
        weights_on_cpu = free_memory < (1024**3)  # Less than 1GB free
        
        if weights_on_cpu:
            print(f"{Fore.YELLOW}   Using class weights on CPU to save GPU memory{Style.RESET_ALL}")
            # Keep class weights on CPU
            class_weights = class_weights.cpu()
    
    # Set up criterion based on model type
    if args.model_type == 'enhanced_gnn' and args.predict_time:
        # Multi-objective loss for dual task
        from processmine.utils.losses import ProcessLoss
        
        criterion = ProcessLoss(
            task_weight=0.7,
            time_weight=0.3,
            class_weights=class_weights
        )
    else:
        # Standard cross-entropy for classification
        criterion = torch.nn.CrossEntropyLoss(weight=class_weights)
    
    # Move criterion to device if needed (and it's not using CPU weights)
    if not weights_on_cpu and device is not None:
        criterion = criterion.to(device)
    
    # Set up optimizer with model parameters
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=lr,
        weight_decay=weight_decay
    )
    
    # Print optimizer information
    param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"   Optimizing {param_count:,} parameters")
    print(f"   Learning rate: {lr}, Weight decay: {weight_decay}")
    
    return optimizer, criterion

def train_and_evaluate_model(model, graphs, args, criterion, optimizer, device, run_dir):
    """
    Train and evaluate model with memory optimization
    
    Args:
        model: Model to train
        graphs: List of graph data objects
        args: Command-line arguments
        criterion: Loss function
        optimizer: Optimizer
        device: Torch device
        run_dir: Output directory
        
    Returns:
        Tuple of (trained model, evaluation metrics)
    """
    from torch_geometric.loader import DataLoader
    from sklearn.model_selection import train_test_split
    
    print(f"{Fore.CYAN}🏋️ Training and evaluating {args.model_type} model...{Style.RESET_ALL}")
    
    # Clear memory before training
    MemoryOptimizer.clear_memory()
    
    # Split data into train/val/test
    train_val_graphs, test_graphs = train_test_split(graphs, test_size=0.2, random_state=42)
    train_graphs, val_graphs = train_test_split(train_val_graphs, test_size=0.25, random_state=42)
    
    print(f"   Training set: {len(train_graphs)} graphs")
    print(f"   Validation set: {len(val_graphs)} graphs")
    print(f"   Testing set: {len(test_graphs)} graphs")
    
    # Determine optimal batch size for device
    if device.type == 'cuda':
        batch_size = min(args.batch_size, 16)  # Start with smaller batch size
        print(f"   Starting with batch_size={batch_size} (will be optimized)")
    else:
        batch_size = args.batch_size
    
    # Create data loaders
    train_loader = DataLoader(train_graphs, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_graphs, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_graphs, batch_size=batch_size, shuffle=False)
    
    # Training parameters
    epochs = args.epochs
    best_val_loss = float('inf')
    best_model_path = os.path.join(run_dir, 'models', 'best_model.pth')
    patience = 5
    patience_counter = 0
    
    # Initialize automatic mixed precision (AMP) scaler if available
    scaler = torch.cuda.amp.GradScaler() if device.type == 'cuda' else None
    use_amp = scaler is not None
    
    if use_amp:
        print(f"{Fore.GREEN}   Using automatic mixed precision (AMP) for faster training{Style.RESET_ALL}")
    
    # Track metrics
    train_losses = []
    val_losses = []
    val_accuracies = []
    
    # Get graph-level targets from node-level targets (helper function)
    def get_graph_targets(node_targets, batch):
        unique_graphs = torch.unique(batch)
        graph_targets = []
        
        for g in unique_graphs:
            # Get targets for this graph
            graph_mask = (batch == g)
            graph_node_targets = node_targets[graph_mask]
            
            # Find most common target (mode)
            if len(graph_node_targets) > 0:
                values, counts = torch.unique(graph_node_targets, return_counts=True)
                mode_idx = torch.argmax(counts)
                graph_targets.append(values[mode_idx])
            else:
                # Fallback if no targets (should not happen)
                graph_targets.append(torch.tensor(0, device=node_targets.device))
        
        return torch.tensor(graph_targets, dtype=torch.long, device=node_targets.device)
    
    # Determine if this is a neural model (vs scikit-learn)
    is_neural = isinstance(model, torch.nn.Module)
    
    if is_neural:
        # Training loop for neural models
        print(f"{Fore.CYAN}   Training for {epochs} epochs...{Style.RESET_ALL}")
        epoch_progress = tqdm(range(1, epochs+1), desc="Training progress")
        
        # Try to determine optimal batch size from first batch
        if device.type == 'cuda':
            try:
                first_batch = next(iter(train_loader))
                first_batch = first_batch.to(device)
                
                # Adapt batch size based on memory usage
                # Track memory before and after processing a batch
                before_mem = torch.cuda.memory_allocated(device)
                
                # Forward and backward pass with a single batch
                model.train()
                optimizer.zero_grad()
                
                if use_amp:
                    with torch.cuda.amp.autocast():
                        # Handle different model output formats
                        if args.model_type == 'enhanced_gnn':
                            outputs = model(first_batch)
                            graph_targets = get_graph_targets(first_batch.y, first_batch.batch)
                            loss, _ = criterion(outputs["task_pred"], graph_targets)
                        elif args.model_type == 'diverse_gat':
                            logits, diversity_loss = model(first_batch)
                            graph_targets = get_graph_targets(first_batch.y, first_batch.batch)
                            loss = criterion(logits, graph_targets) + diversity_loss
                        else:
                            logits = model(first_batch)
                            graph_targets = get_graph_targets(first_batch.y, first_batch.batch)
                            loss = criterion(logits, graph_targets)
                    
                    # Scale gradients
                    scaler.scale(loss).backward()
                else:
                    # Regular processing
                    if args.model_type == 'enhanced_gnn':
                        outputs = model(first_batch)
                        graph_targets = get_graph_targets(first_batch.y, first_batch.batch)
                        loss, _ = criterion(outputs["task_pred"], graph_targets)
                    elif args.model_type == 'diverse_gat':
                        logits, diversity_loss = model(first_batch)
                        graph_targets = get_graph_targets(first_batch.y, first_batch.batch)
                        loss = criterion(logits, graph_targets) + diversity_loss
                    else:
                        logits = model(first_batch)
                        graph_targets = get_graph_targets(first_batch.y, first_batch.batch)
                        loss = criterion(logits, graph_targets)
                    
                    loss.backward()
                
                # Check memory after backward pass
                after_mem = torch.cuda.memory_allocated(device)
                mem_per_sample = (after_mem - before_mem) / batch_size
                
                # Calculate optimal batch size (aiming for 70% GPU utilization)
                total_mem = torch.cuda.get_device_properties(device).total_memory
                target_mem = total_mem * 0.7
                free_mem = total_mem - after_mem
                
                # Calculate how many more samples we can fit
                additional_samples = int(free_mem / mem_per_sample)
                optimal_batch_size = min(batch_size + additional_samples, args.batch_size)
                
                # Ensure batch size is at least 1
                optimal_batch_size = max(1, optimal_batch_size)
                
                if optimal_batch_size != batch_size:
                    print(f"{Fore.GREEN}   Optimized batch size: {optimal_batch_size} (was {batch_size}){Style.RESET_ALL}")
                    
                    # Recreate data loaders with optimal batch size
                    batch_size = optimal_batch_size
                    train_loader = DataLoader(train_graphs, batch_size=batch_size, shuffle=True)
                    val_loader = DataLoader(val_graphs, batch_size=batch_size, shuffle=False)
                    test_loader = DataLoader(test_graphs, batch_size=batch_size, shuffle=False)
                
                # Clear gradients
                optimizer.zero_grad()
                
                # Clear memory after batch size determination
                torch.cuda.empty_cache()
                
            except Exception as e:
                print(f"{Fore.YELLOW}   Error determining optimal batch size: {e}{Style.RESET_ALL}")
                print(f"   Continuing with batch_size={batch_size}")
                
                # Clear memory after error
                MemoryOptimizer.clear_memory(True)
        
        # Main training loop
        for epoch in epoch_progress:
            # Training phase
            model.train()
            train_loss = 0.0
            
            batch_progress = tqdm(train_loader, desc=f"Epoch {epoch}/{epochs} [Train]", 
                                leave=False, ncols=80)
            
            for batch_data in batch_progress:
                # Move data to device
                batch_data = batch_data.to(device)
                optimizer.zero_grad()
                
                # Forward and backward pass with AMP if available
                if use_amp:
                    with torch.cuda.amp.autocast():
                        # Handle different model output formats
                        if args.model_type == 'enhanced_gnn':
                            outputs = model(batch_data)
                            graph_targets = get_graph_targets(batch_data.y, batch_data.batch)
                            loss, _ = criterion(outputs["task_pred"], graph_targets)
                            
                            # Add diversity loss if available
                            if "diversity_loss" in outputs:
                                loss = loss + outputs["diversity_loss"]
                                
                        elif args.model_type == 'diverse_gat':
                            logits, diversity_loss = model(batch_data)
                            graph_targets = get_graph_targets(batch_data.y, batch_data.batch)
                            loss = criterion(logits, graph_targets) + diversity_loss
                        else:
                            logits = model(batch_data)
                            graph_targets = get_graph_targets(batch_data.y, batch_data.batch)
                            loss = criterion(logits, graph_targets)
                    
                    # Scale gradients
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    # Regular processing
                    if args.model_type == 'enhanced_gnn':
                        outputs = model(batch_data)
                        graph_targets = get_graph_targets(batch_data.y, batch_data.batch)
                        loss, _ = criterion(outputs["task_pred"], graph_targets)
                        
                        # Add diversity loss if available
                        if "diversity_loss" in outputs:
                            loss = loss + outputs["diversity_loss"]
                            
                    elif args.model_type == 'diverse_gat':
                        logits, diversity_loss = model(batch_data)
                        graph_targets = get_graph_targets(batch_data.y, batch_data.batch)
                        loss = criterion(logits, graph_targets) + diversity_loss
                    else:
                        logits = model(batch_data)
                        graph_targets = get_graph_targets(batch_data.y, batch_data.batch)
                        loss = criterion(logits, graph_targets)
                    
                    loss.backward()
                    optimizer.step()
                
                # Track loss
                train_loss += loss.item()
                batch_progress.set_postfix({"loss": f"{loss.item():.4f}"})
                
                # Periodic memory cleanup (every 10 batches)
                if batch_progress.n % 10 == 0 and device.type == 'cuda':
                    # Just empty cache, don't do full garbage collection
                    torch.cuda.empty_cache()
            
            # Calculate average training loss
            avg_train_loss = train_loss / len(train_loader) if len(train_loader) > 0 else 0
            train_losses.append(avg_train_loss)
            
            # Validation phase
            model.eval()
            val_loss = 0.0
            correct = 0
            total = 0
            
            # Perform validation with no gradient tracking
            with torch.no_grad():
                val_progress = tqdm(val_loader, desc=f"Epoch {epoch}/{epochs} [Valid]", 
                                   leave=False, ncols=80)
                
                for batch_data in val_progress:
                    batch_data = batch_data.to(device)
                    
                    # Handle different model output formats
                    if args.model_type == 'enhanced_gnn':
                        outputs = model(batch_data)
                        graph_targets = get_graph_targets(batch_data.y, batch_data.batch)
                        loss, _ = criterion(outputs["task_pred"], graph_targets)
                        logits = outputs["task_pred"]
                    elif args.model_type == 'diverse_gat':
                        logits, _ = model(batch_data)
                        graph_targets = get_graph_targets(batch_data.y, batch_data.batch)
                        loss = criterion(logits, graph_targets)
                    else:
                        logits = model(batch_data)
                        graph_targets = get_graph_targets(batch_data.y, batch_data.batch)
                        loss = criterion(logits, graph_targets)
                    
                    # Track validation metrics
                    val_loss += loss.item()
                    
                    # Calculate accuracy
                    _, predicted = torch.max(logits, 1)
                    total += graph_targets.size(0)
                    correct += (predicted == graph_targets).sum().item()
                    
                    # Update validation progress
                    val_progress.set_postfix({"loss": f"{loss.item():.4f}"})
            
            # Calculate average validation metrics
            avg_val_loss = val_loss / len(val_loader) if len(val_loader) > 0 else 0
            val_losses.append(avg_val_loss)
            
            val_accuracy = correct / total if total > 0 else 0
            val_accuracies.append(val_accuracy)
            
            # Update epoch progress
            epoch_progress.set_postfix({
                "train_loss": f"{avg_train_loss:.4f}",
                "val_loss": f"{avg_val_loss:.4f}",
                "val_acc": f"{val_accuracy:.4f}"
            })
            
            # Check for improvement for early stopping
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                patience_counter = 0
                
                # Save best model
                torch.save(model.state_dict(), best_model_path)
                print(f"{Fore.GREEN}   Saved best model (val_loss={best_val_loss:.4f}){Style.RESET_ALL}")
            else:
                patience_counter += 1
                print(f"{Fore.YELLOW}   No improvement for {patience_counter}/{patience} epochs{Style.RESET_ALL}")
                
                if patience_counter >= patience:
                    print(f"{Fore.YELLOW}   Early stopping triggered after {epoch} epochs{Style.RESET_ALL}")
                    break
            
            # Full memory cleanup at end of epoch
            MemoryOptimizer.clear_memory()
        
        # Load best model for evaluation
        try:
            model.load_state_dict(torch.load(best_model_path))
            print(f"{Fore.GREEN}   Loaded best model from {best_model_path}{Style.RESET_ALL}")
        except Exception as e:
            print(f"{Fore.YELLOW}   Error loading best model: {e}{Style.RESET_ALL}")
        
        # Plot training curves
        try:
            import matplotlib.pyplot as plt
            plt.figure(figsize=(12, 4))
            
            plt.subplot(1, 2, 1)
            plt.plot(train_losses, label='Train Loss')
            plt.plot(val_losses, label='Validation Loss')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.legend()
            plt.title('Training and Validation Loss')
            
            plt.subplot(1, 2, 2)
            plt.plot(val_accuracies, label='Validation Accuracy')
            plt.xlabel('Epoch')
            plt.ylabel('Accuracy')
            plt.legend()
            plt.title('Validation Accuracy')
            
            plt.tight_layout()
            plt.savefig(os.path.join(run_dir, 'visualizations', f'{args.model_type}_training_curves.png'))
            plt.close()
        except Exception as e:
            print(f"{Fore.YELLOW}   Error plotting training curves: {e}{Style.RESET_ALL}")
    
    else:
        # Training for non-neural models (sklearn-based)
        print(f"{Fore.CYAN}   Training {args.model_type} model (non-neural)...{Style.RESET_ALL}")
        
        # Extract features and labels for tabular models
        X_train, y_train = [], []
        
        for graph in train_graphs:
            X_train.extend(graph.x.cpu().numpy())
            y_train.extend(graph.y.cpu().numpy())
        
        # Convert to numpy arrays
        X_train = np.array(X_train)
        y_train = np.array(y_train)
        
        print(f"   Training with {len(X_train)} samples")
        
        # Train model
        model.fit(X_train, y_train)
        print(f"{Fore.GREEN}   Training completed{Style.RESET_ALL}")
    
    # Evaluate on test set
    print(f"{Fore.CYAN}📊 Evaluating model on test set...{Style.RESET_ALL}")
    
    # Clear memory before evaluation
    MemoryOptimizer.clear_memory()
    
    test_metrics = evaluate_model(model, test_loader, criterion, device, args)
    
    # Print metrics
    print(f"{Fore.GREEN}   Test metrics:{Style.RESET_ALL}")
    for name, value in test_metrics.items():
        print(f"      {name}: {value:.4f}")
    
    # Final memory cleanup
    MemoryOptimizer.clear_memory(True)
    
    return model, test_metrics

def evaluate_model(model, data_loader, criterion, device, args):
    """
    Evaluate model on test data with memory optimization
    
    Args:
        model: Model to evaluate
        data_loader: Test data loader
        criterion: Loss function
        device: Torch device
        args: Command line arguments
        
    Returns:
        Dictionary of evaluation metrics
    """
    # Helper function for graph-level targets
    def get_graph_targets(node_targets, batch):
        unique_graphs = torch.unique(batch)
        graph_targets = []
        
        for g in unique_graphs:
            # Get targets for this graph
            graph_mask = (batch == g)
            graph_node_targets = node_targets[graph_mask]
            
            # Find most common target (mode)
            if len(graph_node_targets) > 0:
                values, counts = torch.unique(graph_node_targets, return_counts=True)
                mode_idx = torch.argmax(counts)
                graph_targets.append(values[mode_idx])
            else:
                # Fallback if no targets (should not happen)
                graph_targets.append(torch.tensor(0, device=node_targets.device))
        
        return torch.tensor(graph_targets, dtype=torch.long, device=node_targets.device)
    
    # Check if model is neural network
    is_neural = isinstance(model, torch.nn.Module)
    
    if is_neural:
        # Evaluate neural model
        model.eval()
        test_loss = 0.0
        y_true = []
        y_pred = []
        
        # Process batches with no gradient tracking
        with torch.no_grad():
            for batch_data in data_loader:
                batch_data = batch_data.to(device)
                
                # Handle different model output formats
                if args.model_type == 'enhanced_gnn':
                    outputs = model(batch_data)
                    graph_targets = get_graph_targets(batch_data.y, batch_data.batch)
                    
                    if args.predict_time:
                        loss, _ = criterion(outputs["task_pred"], graph_targets)
                    else:
                        loss = criterion(outputs["task_pred"], graph_targets)
                    
                    logits = outputs["task_pred"]
                
                elif args.model_type == 'diverse_gat':
                    logits, _ = model(batch_data)
                    graph_targets = get_graph_targets(batch_data.y, batch_data.batch)
                    loss = criterion(logits, graph_targets)
                
                else:
                    logits = model(batch_data)
                    graph_targets = get_graph_targets(batch_data.y, batch_data.batch)
                    loss = criterion(logits, graph_targets)
                
                # Track loss
                test_loss += loss.item()
                
                # Get predictions
                _, predicted = torch.max(logits, 1)
                
                # Collect targets and predictions
                y_true.extend(graph_targets.cpu().numpy())
                y_pred.extend(predicted.cpu().numpy())
                
                # Clear memory after each batch if on GPU
                if device.type == 'cuda':
                    torch.cuda.empty_cache()
        
        # Convert to numpy arrays
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
        
    else:
        # Evaluate non-neural model
        # Extract features and labels
        X_test, y_true = [], []
        
        for batch_data in data_loader:
            X_test.extend(batch_data.x.cpu().numpy())
            y_true.extend(batch_data.y.cpu().numpy())
        
        # Convert to numpy arrays
        X_test = np.array(X_test)
        y_true = np.array(y_true)
        
        # Make predictions
        y_pred = model.predict(X_test)
    
    # Calculate metrics
    metrics = {}
    metrics['accuracy'] = accuracy_score(y_true, y_pred)
    metrics['f1_macro'] = f1_score(y_true, y_pred, average='macro', zero_division=0)
    metrics['f1_weighted'] = f1_score(y_true, y_pred, average='weighted', zero_division=0)
    metrics['precision_macro'] = precision_score(y_true, y_pred, average='macro', zero_division=0)
    metrics['recall_macro'] = recall_score(y_true, y_pred, average='macro', zero_division=0)
    metrics['mcc'] = matthews_corrcoef(y_true, y_pred)
    
    if is_neural:
        metrics['test_loss'] = test_loss / len(data_loader) if len(data_loader) > 0 else 0
    
    return metrics