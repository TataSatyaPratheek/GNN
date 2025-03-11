#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Graph Attention Network (GAT) model for process mining
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, global_mean_pool

from ..base import TorchProcessModel, get_graph_labels
import logging

logger = logging.getLogger(__name__)

class NextTaskGAT(TorchProcessModel):
    """
    Graph Attention Network for next task prediction
    """
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=2, heads=4, dropout=0.5):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_layers = num_layers
        self.heads = heads
        self.dropout = dropout
        
        # Build model architecture
        self.convs = nn.ModuleList()
        self.convs.append(GATConv(input_dim, hidden_dim, heads=heads, concat=True))
        
        # Add residual connections to combat over-smoothing
        self.residuals = nn.ModuleList()
        for _ in range(num_layers-1):
            self.convs.append(GATConv(hidden_dim*heads, hidden_dim, heads=heads, concat=True))
            self.residuals.append(nn.Linear(hidden_dim*heads, hidden_dim*heads))
            
        # Use batch normalization for more stable training
        self.batch_norms = nn.ModuleList([
            nn.BatchNorm1d(hidden_dim*heads) for _ in range(num_layers)
        ])
        
        self.fc = nn.Linear(hidden_dim*heads, output_dim)
    
    def forward(self, x, edge_index, batch):
        """
        Forward pass
        
        Args:
            x: Node features [num_nodes, input_dim]
            edge_index: Graph edge indices [2, num_edges]
            batch: Batch assignments [num_nodes]
            
        Returns:
            Classification logits [num_graphs, output_dim]
        """
        for i, conv in enumerate(self.convs):
            # Apply convolution
            new_x = conv(x, edge_index)
            
            # Apply batch normalization
            new_x = self.batch_norms[i](new_x)
            
            # Apply activation
            new_x = F.elu(new_x)
            
            # Apply dropout
            new_x = F.dropout(new_x, p=self.dropout, training=self.training)
            
            # Apply residual connection if not the first layer
            if i > 0:
                x = new_x + self.residuals[i-1](x)
            else:
                x = new_x
        
        # Global pooling
        x = global_mean_pool(x, batch)
        return self.fc(x)
    
    def _prepare_batch(self, batch_data, device):
        """Prepare batch for model"""
        return batch_data.to(device)
    
    def _compute_loss(self, batch_data, criterion):
        """Compute loss for batch"""
        # Get model outputs
        logits = self(batch_data.x, batch_data.edge_index, batch_data.batch)
        
        # Get graph-level labels
        graph_labels = get_graph_labels(batch_data.y, batch_data.batch)
        
        # Compute loss
        loss = criterion(logits, graph_labels)
        return loss
    
    def _extract_predictions(self, outputs, batch_data):
        """Extract predictions from outputs"""
        # Get logits
        logits = outputs
        
        # Get probabilities
        probs = F.softmax(logits, dim=1)
        
        # Get predictions
        _, preds = torch.max(logits, dim=1)
        
        # Get graph-level labels
        graph_labels = get_graph_labels(batch_data.y, batch_data.batch)
        
        return preds.cpu().numpy(), probs.cpu().numpy(), graph_labels.cpu().numpy()


def train_gat_model(model, train_loader, val_loader, criterion, optimizer, 
                    device, num_epochs=20, model_path="best_gnn_model.pth", viz_dir=None):
    """
    Train the GAT model with enhanced progress tracking
    
    Args:
        model: GAT model
        train_loader: Training data loader
        val_loader: Validation data loader
        criterion: Loss function
        optimizer: Optimizer
        device: Device to use (cpu/cuda)
        num_epochs: Number of training epochs
        model_path: Path to save best model
        viz_dir: Directory to save visualizations
        
    Returns:
        Trained model
    """
    # Use the fit method from TorchProcessModel
    model.fit(
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        criterion=criterion,
        epochs=num_epochs,
        device=device,
        early_stopping=True,
        patience=5,
        model_path=model_path
    )
    
    # Create visualization if viz_dir provided
    if viz_dir and hasattr(model, 'loss_history'):
        try:
            import matplotlib.pyplot as plt
            import os
            
            plt.figure(figsize=(10, 5))
            plt.plot(model.loss_history)
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.title('Training Loss')
            
            os.makedirs(viz_dir, exist_ok=True)
            loss_path = os.path.join(viz_dir, 'gat_training_loss.png')
            plt.savefig(loss_path)
            plt.close()
            logger.info(f"Saved loss curve to {loss_path}")
        except Exception as e:
            logger.warning(f"Failed to create loss curve visualization: {e}")
    
    return model


def evaluate_gat_model(model, val_loader, device):
    """
    Evaluate GAT model and return predictions and probabilities
    
    Args:
        model: GAT model
        val_loader: Validation data loader
        device: Device to use (cpu/cuda)
        
    Returns:
        Tuple of (true labels, predictions, probabilities)
    """
    # Use predict method from TorchProcessModel
    y_pred, y_prob, y_true = model.predict(val_loader, device)
    
    # Calculate and print accuracy
    from sklearn.metrics import accuracy_score
    accuracy = accuracy_score(y_true, y_pred)
    logger.info(f"Evaluation accuracy: {accuracy:.4f}")
    
    return y_true, y_pred, y_prob