#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
MLP models for process mining with DGL support
Basic neural network baseline for comparison with more advanced models
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import logging
import dgl
from tqdm import tqdm
from typing import Dict, List, Tuple, Optional, Union, Any

from processmine.models.base import ProcessModel

# Set up logging
logger = logging.getLogger(__name__)



def train_mlp_model(model, train_loader, val_loader, criterion, optimizer, 
                  device, num_epochs=20, patience=5, model_path=None):
    """
    Train MLP model with early stopping and DGL support
    
    Args:
        model: MLP model
        train_loader: Training data loader
        val_loader: Validation data loader
        criterion: Loss function
        optimizer: Optimizer
        device: Torch device
        num_epochs: Maximum number of epochs
        patience: Early stopping patience
        model_path: Path to save best model
        
    Returns:
        Trained model and training history
    """
    logger.info(f"Training MLP model for {num_epochs} epochs (patience={patience})")
    
    best_val_loss = float('inf')
    patience_counter = 0
    train_losses = []
    val_losses = []
    
    # Import utility for getting graph targets
    from processmine.utils.dataloader import get_graph_targets
    
    # Training loop
    for epoch in range(1, num_epochs + 1):
        model.train()
        epoch_loss = 0.0
        
        # Training phase
        train_bar = tqdm(train_loader, desc=f"Epoch {epoch}/{num_epochs} [Train]")
        for batch in train_bar:
            # Move batch to device
            if hasattr(batch, 'to'):
                batch = batch.to(device)
            
            # Extract targets using DGL utility for DGL graphs
            if hasattr(batch, 'ndata') and 'label' in batch.ndata:
                # DGL graph - extract targets
                targets = get_graph_targets(batch)
            elif hasattr(batch, 'y'):
                # PyG Data (backward compatibility)
                targets = batch.y
            else:
                # Assume tensor inputs with separate targets
                batch, targets = batch
                
            # Forward pass
            optimizer.zero_grad()
            outputs = model(batch)
            
            # Handle different output formats
            if isinstance(outputs, dict):
                logits = outputs.get("task_pred", next(iter(outputs.values())))
            else:
                logits = outputs
            
            # Handle different target formats
            if hasattr(targets, 'shape') and len(targets.shape) > 1 and targets.shape[1] > 1:
                # Multi-dimensional targets
                loss = criterion(logits, targets)
            else:
                # Class indices
                loss = criterion(logits, targets.long())
            
            # Backward pass and optimize
            loss.backward()
            optimizer.step()
            
            # Track loss
            epoch_loss += loss.item()
            train_bar.set_postfix({"loss": f"{loss.item():.4f}"})
        
        avg_train_loss = epoch_loss / len(train_loader)
        train_losses.append(avg_train_loss)
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        
        with torch.no_grad():
            val_bar = tqdm(val_loader, desc=f"Epoch {epoch}/{num_epochs} [Valid]")
            for batch in val_bar:
                # Move batch to device
                if hasattr(batch, 'to'):
                    batch = batch.to(device)
                
                # Extract targets using DGL utility for DGL graphs
                if hasattr(batch, 'ndata') and 'label' in batch.ndata:
                    # DGL graph - extract targets
                    targets = get_graph_targets(batch)
                elif hasattr(batch, 'y'):
                    # PyG Data (backward compatibility)
                    targets = batch.y
                else:
                    # Assume tensor inputs with separate targets
                    batch, targets = batch
                
                # Forward pass
                outputs = model(batch)
                
                # Handle different output formats
                if isinstance(outputs, dict):
                    logits = outputs.get("task_pred", next(iter(outputs.values())))
                else:
                    logits = outputs
                
                # Handle different target formats
                if hasattr(targets, 'shape') and len(targets.shape) > 1 and targets.shape[1] > 1:
                    # Multi-dimensional targets
                    loss = criterion(logits, targets)
                else:
                    # Class indices
                    loss = criterion(logits, targets.long())
                
                # Track loss
                val_loss += loss.item()
                val_bar.set_postfix({"loss": f"{loss.item():.4f}"})
        
        avg_val_loss = val_loss / len(val_loader)
        val_losses.append(avg_val_loss)
        
        # Print epoch summary
        logger.info(f"Epoch {epoch}/{num_epochs}: "
                   f"train_loss={avg_train_loss:.4f}, "
                   f"val_loss={avg_val_loss:.4f}")
        
        # Check for improvement for early stopping
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            
            # Save best model
            if model_path:
                torch.save(model.state_dict(), model_path)
                logger.info(f"Saved best model with val_loss={best_val_loss:.4f}")
        else:
            patience_counter += 1
            logger.info(f"No improvement for {patience_counter}/{patience} epochs")
            
            if patience_counter >= patience:
                logger.info(f"Early stopping triggered after {epoch} epochs")
                break
    
    # Load best model if path provided
    if model_path:
        try:
            model.load_state_dict(torch.load(model_path))
            logger.info(f"Loaded best model from {model_path}")
        except Exception as e:
            logger.error(f"Error loading best model: {e}")
    
    return model, {'train_losses': train_losses, 'val_losses': val_losses}


def evaluate_mlp_model(model, test_loader, device, criterion=None):
    """
    Evaluate MLP model on test data with DGL support
    
    Args:
        model: MLP model
        test_loader: Test data loader
        device: Torch device
        criterion: Optional loss function
        
    Returns:
        Tuple of (metrics, predictions, true_labels)
    """
    model.eval()
    all_preds = []
    all_labels = []
    test_loss = 0.0 if criterion else None
    
    # Import utility for getting graph targets
    from processmine.utils.dataloader import get_graph_targets
    
    with torch.no_grad():
        for batch in test_loader:
            # Move batch to device
            if hasattr(batch, 'to'):
                batch = batch.to(device)
            
            # Extract targets using DGL utility for DGL graphs
            if hasattr(batch, 'ndata') and 'label' in batch.ndata:
                # DGL graph - extract targets
                targets = get_graph_targets(batch)
            elif hasattr(batch, 'y'):
                # PyG Data (backward compatibility)
                targets = batch.y
            else:
                # Assume tensor inputs with separate targets
                batch, targets = batch
            
            # Forward pass
            outputs = model(batch)
            
            # Handle different output formats
            if isinstance(outputs, dict):
                logits = outputs.get("task_pred", next(iter(outputs.values())))
            else:
                logits = outputs
            
            # Calculate loss if criterion provided
            if criterion is not None:
                if hasattr(targets, 'shape') and len(targets.shape) > 1 and targets.shape[1] > 1:
                    loss = criterion(logits, targets)
                else:
                    loss = criterion(logits, targets.long())
                test_loss += loss.item()
            
            # Get predictions
            _, preds = torch.max(logits, dim=1)
            
            # Collect results
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(targets.cpu().numpy())
    
    # Calculate metrics
    from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, matthews_corrcoef
    
    metrics = {
        'accuracy': accuracy_score(all_labels, all_preds),
        'f1_macro': f1_score(all_labels, all_preds, average='macro', zero_division=0),
        'f1_weighted': f1_score(all_labels, all_preds, average='weighted', zero_division=0),
        'precision': precision_score(all_labels, all_preds, average='macro', zero_division=0),
        'recall': recall_score(all_labels, all_preds, average='macro', zero_division=0)
    }
    
    # Add MCC for binary and multiclass (not multilabel)
    try:
        metrics['mcc'] = matthews_corrcoef(all_labels, all_preds)
    except:
        pass
    
    # Add test loss if calculated
    if test_loss is not None:
        metrics['test_loss'] = test_loss / len(test_loader)
    
    return metrics, np.array(all_preds), np.array(all_labels)

class BasicMLP(ProcessModel):
    """
    Basic MLP model for process mining
    """
    def __init__(self, input_dim, hidden_dims, output_dim, dropout=0.3,
                 activation=F.relu):
        """
        Initialize MLP model
        
        Args:
            input_dim: Input dimension
            hidden_dims: List of hidden layer dimensions
            output_dim: Output dimension
            dropout: Dropout probability
            activation: Activation function
        """
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.output_dim = output_dim
        self.dropout = dropout
        self.activation = activation
        
        # Create layers
        layers = []
        prev_dim = input_dim
        
        for h_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, h_dim))
            layers.append(nn.BatchNorm1d(h_dim))
            layers.append(nn.Dropout(dropout))
            prev_dim = h_dim
        
        self.hidden_layers = nn.ModuleList(layers)
        self.output_layer = nn.Linear(prev_dim, output_dim)
    
    def forward(self, x):
        """
        Forward pass
        
        Args:
            x: Input tensor [batch_size, input_dim] or DGL graph
            
        Returns:
            Output logits [batch_size, output_dim]
        """
        # Handle DGL Graph objects
        if hasattr(x, 'ndata') and 'feat' in x.ndata:
            # Extract node features and perform pooling
            node_x = x.ndata['feat']
            
            # Apply hidden layers to nodes
            for i in range(0, len(self.hidden_layers), 3):
                node_x = self.hidden_layers[i](node_x)  # Linear
                node_x = self.hidden_layers[i+1](node_x)  # BatchNorm
                node_x = self.activation(node_x)  # Activation
                node_x = self.hidden_layers[i+2](node_x)  # Dropout
            
            # Global mean pooling using DGL's readout
            x = dgl.readout_nodes(x, 'feat', op='mean')
        else:
            # Standard tensor processing
            for i in range(0, len(self.hidden_layers), 3):
                x = self.hidden_layers[i](x)  # Linear
                x = self.hidden_layers[i+1](x)  # BatchNorm
                x = self.activation(x)  # Activation
                x = self.hidden_layers[i+2](x)  # Dropout
        
        # Apply output layer
        logits = self.output_layer(x)
        
        return logits