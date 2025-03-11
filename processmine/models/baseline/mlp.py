#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
MLP models for process mining
Basic neural network baseline for comparison with more advanced models
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import logging
from tqdm import tqdm
from typing import Dict, List, Tuple, Optional, Union, Any

from processmine.models.base import ProcessMiningModel

# Set up logging
logger = logging.getLogger(__name__)


class BasicMLP(ProcessMiningModel):
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
            x: Input tensor [batch_size, input_dim] or PyG Data
            
        Returns:
            Output logits [batch_size, output_dim]
        """
        # Handle PyG Data objects
        if hasattr(x, 'x') and hasattr(x, 'batch'):
            # Extract node features and perform pooling
            node_x = x.x
            batch = x.batch
            
            # Apply hidden layers to nodes
            for i in range(0, len(self.hidden_layers), 3):
                node_x = self.hidden_layers[i](node_x)  # Linear
                node_x = self.hidden_layers[i+1](node_x)  # BatchNorm
                node_x = self.activation(node_x)  # Activation
                node_x = self.hidden_layers[i+2](node_x)  # Dropout
            
            # Global mean pooling
            from torch_geometric.nn import global_mean_pool
            x = global_mean_pool(node_x, batch)
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


def train_mlp_model(model, train_loader, val_loader, criterion, optimizer, 
                 device, num_epochs=20, patience=5, model_path=None):
    """
    Train MLP model with early stopping
    
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
        Trained model
    """
    logger.info(f"Training MLP model for {num_epochs} epochs (patience={patience})")
    
    best_val_loss = float('inf')
    patience_counter = 0
    train_losses = []
    val_losses = []
    
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
            elif isinstance(batch, dict):
                batch = {k: v.to(device) if hasattr(v, 'to') else v 
                         for k, v in batch.items()}
            
            # Extract features and targets
            if hasattr(batch, 'x') and hasattr(batch, 'y'):
                # PyG Data batch
                x = batch
                y = batch.y
            elif isinstance(batch, dict):
                # Dictionary batch
                x = batch['x']
                y = batch['y']
            else:
                # Tuple/list batch
                x, y = batch
            
            # Forward pass
            optimizer.zero_grad()
            outputs = model(x)
            
            # Handle different output and target formats
            if hasattr(y, 'shape') and len(y.shape) > 1 and y.shape[1] > 1:
                # Multi-dimensional targets
                loss = criterion(outputs, y)
            else:
                # Class indices
                loss = criterion(outputs, y.long())
            
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
                elif isinstance(batch, dict):
                    batch = {k: v.to(device) if hasattr(v, 'to') else v 
                             for k, v in batch.items()}
                
                # Extract features and targets
                if hasattr(batch, 'x') and hasattr(batch, 'y'):
                    # PyG Data batch
                    x = batch
                    y = batch.y
                elif isinstance(batch, dict):
                    # Dictionary batch
                    x = batch['x']
                    y = batch['y']
                else:
                    # Tuple/list batch
                    x, y = batch
                
                # Forward pass
                outputs = model(x)
                
                # Handle different output and target formats
                if hasattr(y, 'shape') and len(y.shape) > 1 and y.shape[1] > 1:
                    # Multi-dimensional targets
                    loss = criterion(outputs, y)
                else:
                    # Class indices
                    loss = criterion(outputs, y.long())
                
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


def evaluate_mlp_model(model, test_loader, device):
    """
    Evaluate MLP model on test data
    
    Args:
        model: MLP model
        test_loader: Test data loader
        device: Torch device
        
    Returns:
        Tuple of (accuracy, predictions, true_labels)
    """
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for batch in test_loader:
            # Move batch to device
            if hasattr(batch, 'to'):
                batch = batch.to(device)
            elif isinstance(batch, dict):
                batch = {k: v.to(device) if hasattr(v, 'to') else v 
                         for k, v in batch.items()}
            
            # Extract features and targets
            if hasattr(batch, 'x') and hasattr(batch, 'y'):
                # PyG Data batch
                x = batch
                y = batch.y
            elif isinstance(batch, dict):
                # Dictionary batch
                x = batch['x']
                y = batch['y']
            else:
                # Tuple/list batch
                x, y = batch
            
            # Forward pass
            outputs = model(x)
            
            # Get predictions
            _, preds = torch.max(outputs, 1)
            
            # Collect results
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(y.cpu().numpy())
    
    # Calculate accuracy
    correct = sum(1 for pred, label in zip(all_preds, all_labels) if pred == label)
    total = len(all_labels)
    accuracy = correct / total if total > 0 else 0
    
    return accuracy, np.array(all_preds), np.array(all_labels)