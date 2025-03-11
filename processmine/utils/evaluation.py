# src/processmine/utils/evaluation.py
"""
Evaluation utilities for process mining models.
"""

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score, 
    matthews_corrcoef, confusion_matrix
)
from sklearn.utils.class_weight import compute_class_weight as sk_compute_class_weight

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

def get_graph_targets(node_targets, batch):
    """
    Convert node-level targets to graph-level targets.
    
    Args:
        node_targets: Node-level target tensor
        batch: Batch assignment tensor
        
    Returns:
        Graph-level target tensor
    """
    # Use the most common target for each graph
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

def evaluate_model(model, data_loader, criterion, device, is_neural=True):
    """
    Evaluate model performance with comprehensive metrics.
    
    Args:
        model: Model to evaluate
        data_loader: Data loader for test data
        criterion: Loss function
        device: Computing device
        is_neural: Whether the model is a neural network
        
    Returns:
        Dictionary of evaluation metrics
    """
    if is_neural:
        # Evaluate neural model
        model.eval()
        test_loss = 0.0
        y_true = []
        y_pred = []
        
        with torch.no_grad():
            for batch_data in data_loader:
                batch_data = batch_data.to(device)
                
                # Handle different model output formats
                if hasattr(model, 'predict'):
                    # Model has a predict method
                    predictions = model.predict(batch_data)
                    if isinstance(predictions, tuple):
                        logits = predictions[0]
                    else:
                        logits = predictions
                else:
                    # Standard forward pass
                    logits = model(batch_data)
                
                # Get graph-level targets
                graph_targets = get_graph_targets(batch_data.y, batch_data.batch)
                
                # Compute loss
                loss = criterion(logits, graph_targets)
                test_loss += loss.item()
                
                # Get predictions
                _, predicted = torch.max(logits, 1)
                
                # Collect targets and predictions
                y_true.extend(graph_targets.cpu().numpy())
                y_pred.extend(predicted.cpu().numpy())
        
        # Convert to numpy arrays
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
    
    else:
        # Evaluate sklearn-based model
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
    
    if is_neural and 'test_loss' not in metrics:
        metrics['test_loss'] = test_loss / len(data_loader) if len(data_loader) > 0 else 0.0
    
    # Add confusion matrix
    metrics['confusion_matrix'] = confusion_matrix(y_true, y_pred).tolist()
    
    return metrics, y_true, y_pred

def calculate_f1_scores(y_true, y_pred, average='weighted'):
    """
    Calculate F1 scores with proper handling of edge cases.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        average: Averaging method ('micro', 'macro', 'weighted', None)
        
    Returns:
        F1 score
    """
    return f1_score(y_true, y_pred, average=average, zero_division=0)