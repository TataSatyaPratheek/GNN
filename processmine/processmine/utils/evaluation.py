# src/processmine/utils/evaluation.py
"""
Evaluation utilities for process mining models.
"""

import gc
import logging
import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score, 
    matthews_corrcoef, confusion_matrix
)
from sklearn.utils.class_weight import compute_class_weight as sk_compute_class_weight

logger = logging.getLogger(__name__)

def compute_class_weights(df, num_classes):
    """
    Compute balanced class weights for training with improved efficiency.
    
    Args:
        df: Preprocessed dataframe
        num_classes: Number of classes
        
    Returns:
        Class weights tensor
    """
    print("\nComputing class weights...")
    
    # Extract labels more efficiently
    train_labels = df["next_task"].values
    
    # Count class frequencies
    unique_labels, counts = np.unique(train_labels, return_counts=True)
    
    # Report class distribution
    total = len(train_labels)
    print(f"Class distribution ({len(unique_labels)} classes):")
    for label, count in zip(unique_labels[:5], counts[:5]):
        print(f"  Class {label}: {count:,} samples ({count/total*100:.2f}%)")
    if len(unique_labels) > 5:
        print(f"  ... and {len(unique_labels)-5} more classes")
    
    # Compute weights
    class_weights = np.ones(num_classes, dtype=np.float32)
    present = np.unique(train_labels)
    cw = sk_compute_class_weight("balanced", classes=present, y=train_labels)
    
    for i, cval in enumerate(present):
        class_weights[cval] = cw[i]
    
    # Report weight range
    min_weight = np.min(class_weights[class_weights > 0])
    max_weight = np.max(class_weights)
    print(f"Class weight range: {min_weight:.4f} - {max_weight:.4f}")
    
    # Keep weights on CPU initially - will be moved to device in setup_optimizer_and_loss if needed
    return torch.tensor(class_weights, dtype=torch.float32)

def get_graph_targets(g):
    """
    Extract graph-level targets from a DGL graph or batched graph
    
    Args:
        g: DGL graph or batched graph
        
    Returns:
        Graph-level target tensor
    """
    if 'label' in g.ndata:
        # Get node labels
        node_labels = g.ndata['label']
        
        if hasattr(g, 'batch_size') and g.batch_size > 1:
            # For batched graph, extract one label per graph
            batch_num_nodes = g.batch_num_nodes()
            graph_labels = []
            
            node_offset = 0
            for num_nodes in batch_num_nodes:
                # Get labels for this graph
                graph_node_labels = node_labels[node_offset:node_offset + num_nodes]
                
                # Use mode (most common label) as graph label
                if len(graph_node_labels) > 0:
                    values, counts = torch.unique(graph_node_labels, return_counts=True)
                    mode_idx = torch.argmax(counts)
                    graph_labels.append(values[mode_idx])
                else:
                    # Fallback if no labels
                    graph_labels.append(torch.tensor(0, device=node_labels.device))
                
                # Update offset
                node_offset += num_nodes
            
            return torch.stack(graph_labels)
        else:
            # For single graph, use mode of node labels
            values, counts = torch.unique(node_labels, return_counts=True)
            mode_idx = torch.argmax(counts)
            return values[mode_idx].unsqueeze(0)
    else:
        # No labels found
        return None
    
def evaluate_model(model, data_loader, criterion=None, device=None, detailed=True, memory_efficient=True):
    """
    Evaluate model on test data with improved DGL compatibility
    
    Args:
        model: PyTorch model to evaluate
        data_loader: Data loader for test data
        criterion: Loss function (optional)
        device: Computing device
        detailed: Whether to compute detailed metrics
        memory_efficient: Whether to use memory-efficient evaluation
        
    Returns:
        Tuple of (metrics, predictions, labels)
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
    test_samples = 0
    
    # Import utility function for getting graph targets
    from processmine.utils.dataloader import get_graph_targets
    
    # Clear memory before evaluation
    if memory_efficient and torch.cuda.is_available():
        torch.cuda.empty_cache()
        gc.collect()
    
    # Get progress bar
    try:
        from tqdm import tqdm
        progress_bar = tqdm(data_loader, desc="Evaluating", leave=False)
    except ImportError:
        progress_bar = data_loader
    
    # Evaluate without gradient tracking
    with torch.no_grad():
        for batch_data in progress_bar:
            # Move batch to device
            batch_data = batch_data.to(device)
            
            # Forward pass
            outputs = model(batch_data)
            
            # Handle different model output formats
            if isinstance(outputs, dict):
                logits = outputs.get("task_pred", next(iter(outputs.values())))
            elif isinstance(outputs, tuple):
                logits = outputs[0]
            else:
                logits = outputs
            
            # Get graph-level targets using DGL approach
            graph_targets = get_graph_targets(batch_data)
            
            if graph_targets is None:
                logger.warning("No target labels found in graph data")
                continue
            
            # Compute loss if criterion provided
            if criterion is not None:
                loss = criterion(logits, graph_targets)
                batch_size = batch_data.batch_size if hasattr(batch_data, 'batch_size') else 1
                test_loss += loss.item() * batch_size
                test_samples += batch_size
            
            # Get predicted classes
            _, predicted = torch.max(logits, dim=1)
            
            # Collect predictions and labels
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(graph_targets.cpu().numpy())
            
            # Free batch memory for very large models
            if memory_efficient:
                del outputs, batch_data
                if criterion is not None:
                    del loss
    
    # Convert to numpy arrays
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    
    # Calculate metrics
    from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, matthews_corrcoef
    
    metrics = {}
    
    if len(all_labels) > 0:
        metrics['accuracy'] = accuracy_score(all_labels, all_preds)
        metrics['f1_macro'] = f1_score(all_labels, all_preds, average='macro', zero_division=0)
        metrics['f1_weighted'] = f1_score(all_labels, all_preds, average='weighted', zero_division=0)
        metrics['precision_macro'] = precision_score(all_labels, all_preds, average='macro', zero_division=0)
        metrics['recall_macro'] = recall_score(all_labels, all_preds, average='macro', zero_division=0)
        
        # Add MCC for binary and multiclass (not multilabel)
        try:
            metrics['mcc'] = matthews_corrcoef(all_labels, all_preds)
        except ValueError:
            # Skip MCC for incompatible data
            pass
    else:
        logger.warning("No labels collected during evaluation, metrics will be zeros")
        metrics = {
            'accuracy': 0.0, 'f1_macro': 0.0, 'f1_weighted': 0.0,
            'precision_macro': 0.0, 'recall_macro': 0.0
        }
    
    # Add test loss
    if test_loss is not None:
        metrics['test_loss'] = test_loss / max(test_samples, 1)
    
    # Final memory cleanup
    if memory_efficient and torch.cuda.is_available():
        torch.cuda.empty_cache()
        gc.collect()
    
    return metrics, all_preds, all_labels