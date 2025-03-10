#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Enhanced GNN model for process mining
Combines positional encoding, multi-head attention diversity, and residual connections
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import global_mean_pool, global_add_pool, global_max_pool
import gc
import time
import numpy as np
from typing import Dict, List, Tuple, Optional, Union

# Import custom components
# These need to be in your Python path or in the same directory
try:
    from position_enhanced_gat import PositionalGATConv, PositionalEncoding
    from diverse_attention import DiverseGATConv
except ImportError:
    from models.position_enhanced_gat import PositionalGATConv, PositionalEncoding
    from models.diverse_attention import DiverseGATConv


class ExpressiveGATConv(nn.Module):
    """
    Enhanced GAT convolution that combines positional encoding and diverse attention
    """
    def __init__(self, in_channels, out_channels, heads=4, pos_dim=16,
                 concat=True, dropout=0.0, diversity_weight=0.1):
        """
        Initialize enhanced GAT convolution
        
        Args:
            in_channels: Input feature dimension
            out_channels: Output feature dimension per head
            heads: Number of attention heads
            pos_dim: Dimension of positional encoding
            concat: Whether to concatenate head outputs (True) or average them (False)
            dropout: Dropout probability
            diversity_weight: Weight for head diversity loss
        """
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.heads = heads
        self.concat = concat
        self.dropout = dropout
        
        # Position-enhanced attention
        self.pos_gat = PositionalGATConv(
            in_channels=in_channels,
            out_channels=out_channels,
            heads=heads,
            pos_dim=pos_dim,
            concat=concat,
            dropout=dropout
        )
        
        # Diverse attention with orthogonal regularization
        self.diverse_gat = DiverseGATConv(
            in_channels=in_channels,
            out_channels=out_channels,
            heads=heads,
            concat=concat,
            dropout=dropout,
            diversity_weight=diversity_weight,
            orthogonal_reg=True
        )
        
        # Feature fusion layer (combines outputs from both attention mechanisms)
        if concat:
            fusion_dim = out_channels * heads * 2
        else:
            fusion_dim = out_channels * 2
            
        self.fusion = nn.Linear(fusion_dim, out_channels * heads if concat else out_channels)
    
    def forward(self, x, edge_index, pos=None):
        """
        Forward pass with both position-enhanced and diverse attention
        
        Args:
            x: Input node features [num_nodes, in_channels]
            edge_index: Graph edge indices [2, num_edges]
            pos: Optional node positions [num_nodes, 2]
            
        Returns:
            Tuple of (fused features, diversity loss)
        """
        # Apply position-enhanced GAT
        if pos is not None:
            pos_out = self.pos_gat(x, edge_index, pos=pos)
        else:
            pos_out = self.pos_gat(x, edge_index)
        
        # Apply diverse GAT
        div_out, div_loss = self.diverse_gat(x, edge_index)
        
        # Concatenate outputs from both attention mechanisms
        combined = torch.cat([pos_out, div_out], dim=-1)
        
        # Apply fusion layer
        fused = self.fusion(combined)
        
        # Return fused features and diversity loss
        return fused, div_loss


class EnhancedGNN(nn.Module):
    """
    Enhanced GNN model that combines positional encoding, 
    diverse multi-head attention, and residual connections
    """
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=2, 
                 heads=4, dropout=0.5, pos_dim=16, diversity_weight=0.1,
                 pooling='mean', task_type='classification',
                 predict_time=False, use_batch_norm=True):
        """
        Initialize enhanced GNN model
        
        Args:
            input_dim: Input feature dimension
            hidden_dim: Hidden layer dimension
            output_dim: Output dimension (num classes for classification)
            num_layers: Number of GNN layers
            heads: Number of attention heads
            dropout: Dropout probability
            pos_dim: Dimension of positional encoding
            diversity_weight: Weight for attention diversity loss
            pooling: Graph pooling method ('mean', 'max', or 'sum')
            task_type: Task type ('classification' or 'regression')
            predict_time: Whether to also predict time (dual task)
            use_batch_norm: Whether to use batch normalization
        """
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_layers = num_layers
        self.heads = heads
        self.dropout = dropout
        self.diversity_weight = diversity_weight
        self.task_type = task_type
        self.predict_time = predict_time
        
        # Position encoding for input features
        self.pos_encoder = PositionalEncoding(input_dim, pos_dim=pos_dim)
        self.input_pos_dim = input_dim + pos_dim
        
        # GNN layers
        self.convs = nn.ModuleList()
        
        # Batch normalization layers
        self.batch_norms = nn.ModuleList() if use_batch_norm else None
        
        # Residual connection linear projections
        self.residuals = nn.ModuleList()
        
        # First layer
        self.convs.append(ExpressiveGATConv(
            in_channels=self.input_pos_dim,
            out_channels=hidden_dim,
            heads=heads,
            pos_dim=pos_dim,
            concat=True,
            dropout=dropout,
            diversity_weight=diversity_weight
        ))
        
        if use_batch_norm:
            self.batch_norms.append(nn.BatchNorm1d(hidden_dim * heads))
        
        # Hidden layers with residual connections
        for i in range(num_layers - 1):
            # Residual connection linear projection
            self.residuals.append(nn.Linear(
                hidden_dim * heads,
                hidden_dim * heads
            ))
            
            # GNN layer
            self.convs.append(ExpressiveGATConv(
                in_channels=hidden_dim * heads,
                out_channels=hidden_dim,
                heads=heads,
                pos_dim=pos_dim,
                concat=True,
                dropout=dropout,
                diversity_weight=diversity_weight
            ))
            
            if use_batch_norm:
                self.batch_norms.append(nn.BatchNorm1d(hidden_dim * heads))
        
        # Set pooling function
        if pooling == 'mean':
            self.pool = global_mean_pool
        elif pooling == 'max':
            self.pool = global_max_pool
        elif pooling == 'sum':
            self.pool = global_add_pool
        else:
            raise ValueError(f"Unknown pooling method: {pooling}")
        
        # Output layers
        self.output_dim = hidden_dim * heads
        
        # Task prediction layer
        self.task_pred = nn.Linear(self.output_dim, output_dim)
        
        # Time prediction layer (optional)
        if predict_time:
            self.time_pred = nn.Sequential(
                nn.Linear(self.output_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, 1)
            )
        else:
            self.time_pred = None
    
    def forward(self, data):
        """
        Forward pass through the enhanced GNN
        
        Args:
            data: PyG data object with x, edge_index, and batch attributes
            
        Returns:
            Dictionary with model outputs and losses
        """
        x, edge_index, batch = data.x, data.edge_index, data.batch
        num_nodes = x.size(0)
        
        # Generate node positions if needed
        pos = self._get_normalized_positions(batch, num_nodes)
        
        # Apply position encoding to input
        x = self.pos_encoder(x, pos=pos)
        
        # Track diversity losses from all layers
        diversity_losses = []
        
        # Process through GNN layers
        for i, conv in enumerate(self.convs):
            # Apply convolution with diversity
            conv_out, div_loss = conv(x, edge_index, pos=pos)
            
            # Track diversity loss
            diversity_losses.append(div_loss)
            
            # Apply batch normalization if enabled
            if self.batch_norms is not None:
                conv_out = self.batch_norms[i](conv_out)
            
            # Apply non-linearity
            conv_out = F.elu(conv_out)
            
            # Apply residual connection if not the first layer
            if i > 0:
                # Ensure dimensions match with projection
                residual = self.residuals[i-1](x)
                x = conv_out + residual
            else:
                x = conv_out
            
            # Apply dropout
            x = F.dropout(x, p=self.dropout, training=self.training)
        
        # Global pooling to get graph-level representations
        pooled = self.pool(x, batch)
        
        # Task prediction
        task_pred = self.task_pred(pooled)
        
        # Time prediction (if enabled)
        time_pred = None
        if self.predict_time and self.time_pred is not None:
            time_pred = self.time_pred(pooled).squeeze(-1)
        
        # Calculate total diversity loss
        total_diversity_loss = sum(diversity_losses)
        
        # Prepare output dictionary
        outputs = {
            'task_pred': task_pred,
            'time_pred': time_pred,
            'diversity_loss': total_diversity_loss,
            'node_embeddings': x,  # Return node embeddings for structural loss if needed
            'graph_embeddings': pooled  # Return graph embeddings for downstream tasks
        }
        
        return outputs
    
    def _get_normalized_positions(self, batch, num_nodes):
        """Generate normalized positions based on batch assignments"""
        # Get unique batches and counts
        unique_batches, counts = torch.unique(batch, return_counts=True)
        num_batches = len(unique_batches)
        
        # Create positions tensor
        pos = torch.zeros((num_nodes, 2), device=batch.device)
        
        # Calculate position for each node
        start_idx = 0
        for b, count in zip(unique_batches, counts):
            # Create a grid-like positioning for nodes in each batch
            grid_size = int(np.ceil(np.sqrt(count)))
            for i in range(count):
                row = i // grid_size
                col = i % grid_size
                
                # Normalize positions to [0, 1]
                norm_row = row / max(grid_size - 1, 1)
                norm_col = col / max(grid_size - 1, 1)
                
                # Assign position
                pos[start_idx + i, 0] = norm_row
                pos[start_idx + i, 1] = norm_col
            
            start_idx += count
        
        return pos


class EnhancedGNNWithCritic(nn.Module):
    """
    Enhanced GNN model with internal critic for process mining
    Contains both a main model for prediction and a critic model for self-evaluation
    """
    def __init__(self, input_dim, hidden_dim, output_dim, critic_dim=32,
                 num_layers=2, heads=4, dropout=0.5, pos_dim=16, 
                 diversity_weight=0.1, **kwargs):
        """
        Initialize enhanced GNN with critic
        
        Args:
            input_dim: Input feature dimension
            hidden_dim: Hidden layer dimension
            output_dim: Output dimension (num classes for classification)
            critic_dim: Critic network hidden dimension
            num_layers: Number of GNN layers
            heads: Number of attention heads
            dropout: Dropout probability
            pos_dim: Dimension of positional encoding
            diversity_weight: Weight for attention diversity loss
            **kwargs: Additional parameters for EnhancedGNN
        """
        super().__init__()
        
        # Main model for prediction
        self.main_model = EnhancedGNN(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            output_dim=output_dim,
            num_layers=num_layers,
            heads=heads,
            dropout=dropout,
            pos_dim=pos_dim,
            diversity_weight=diversity_weight,
            **kwargs
        )
        
        # Critic model for confidence estimation
        self.critic = nn.Sequential(
            nn.Linear(hidden_dim * heads, critic_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(critic_dim, 1),
            nn.Sigmoid()  # Output confidence score between 0 and 1
        )
    
    def forward(self, data):
        """
        Forward pass with critic evaluation
        
        Args:
            data: PyG data object
            
        Returns:
            Dictionary with main outputs and critic's confidence score
        """
        # Get main model outputs
        outputs = self.main_model(data)
        
        # Extract graph embeddings
        graph_embeddings = outputs['graph_embeddings']
        
        # Apply critic to get confidence score
        confidence = self.critic(graph_embeddings)
        outputs['confidence'] = confidence
        
        return outputs


def create_enhanced_gnn(input_dim, num_classes, use_critic=False, **kwargs):
    """
    Factory function to create an enhanced GNN model
    
    Args:
        input_dim: Input feature dimension
        num_classes: Number of output classes
        use_critic: Whether to use the critic version
        **kwargs: Additional parameters for the model
        
    Returns:
        EnhancedGNN or EnhancedGNNWithCritic model
    """
    # Default parameters
    default_params = {
        'hidden_dim': 64,
        'num_layers': 2,
        'heads': 4,
        'dropout': 0.5,
        'pos_dim': 16,
        'diversity_weight': 0.1,
        'pooling': 'mean',
        'task_type': 'classification',
        'predict_time': False,
        'use_batch_norm': True
    }
    
    # Update with provided parameters
    params = {**default_params, **kwargs}
    
    if use_critic:
        return EnhancedGNNWithCritic(
            input_dim=input_dim, 
            output_dim=num_classes,
            **params
        )
    else:
        return EnhancedGNN(
            input_dim=input_dim, 
            output_dim=num_classes,
            **params
        )


# Utility function for testing
def test_enhanced_gnn():
    """Test the enhanced GNN implementation"""
    from torch_geometric.data import Data, Batch
    
    # Create a simple test graph
    num_nodes = 10
    num_features = 8
    num_classes = 3
    
    # Random node features
    x = torch.randn(num_nodes, num_features)
    
    # Random edges (ensure connected graph)
    edge_index = torch.zeros(2, num_nodes*2, dtype=torch.long)
    for i in range(num_nodes):
        # Each node connects to next node (circular)
        edge_index[0, i] = i
        edge_index[1, i] = (i + 1) % num_nodes
        
        # And to a random node
        edge_index[0, num_nodes + i] = i
        edge_index[1, num_nodes + i] = torch.randint(0, num_nodes, (1,)).item()
    
    # Create sample batch assignment (2 graphs)
    batch = torch.zeros(num_nodes, dtype=torch.long)
    batch[num_nodes//2:] = 1
    
    # Create test data
    data = Data(x=x, edge_index=edge_index, batch=batch)
    
    # Test EnhancedGNN model
    print("Testing EnhancedGNN model...")
    
    # Create model
    model = EnhancedGNN(
        input_dim=num_features,
        hidden_dim=16,
        output_dim=num_classes,
        num_layers=2,
        heads=4,
        dropout=0.1,
        pos_dim=8,
        diversity_weight=0.1
    )
    
    # Forward pass
    outputs = model(data)
    
    # Check outputs
    print(f"Task prediction shape: {outputs['task_pred'].shape}")
    print(f"Diversity loss: {outputs['diversity_loss'].item()}")
    print(f"Node embeddings shape: {outputs['node_embeddings'].shape}")
    print(f"Graph embeddings shape: {outputs['graph_embeddings'].shape}")
    
    # Test with time prediction
    print("\nTesting EnhancedGNN with time prediction...")
    
    model_with_time = EnhancedGNN(
        input_dim=num_features,
        hidden_dim=16,
        output_dim=num_classes,
        predict_time=True
    )
    
    outputs_with_time = model_with_time(data)
    print(f"Time prediction shape: {outputs_with_time['time_pred'].shape}")
    
    # Test EnhancedGNNWithCritic
    print("\nTesting EnhancedGNNWithCritic...")
    
    critic_model = EnhancedGNNWithCritic(
        input_dim=num_features,
        hidden_dim=16,
        output_dim=num_classes,
        critic_dim=32
    )
    
    critic_outputs = critic_model(data)
    print(f"Confidence score shape: {critic_outputs['confidence'].shape}")
    print(f"Confidence score: {critic_outputs['confidence'].squeeze().tolist()}")
    
    # Test factory function
    print("\nTesting factory function...")
    
    factory_model = create_enhanced_gnn(
        input_dim=num_features,
        num_classes=num_classes,
        use_critic=True,
        hidden_dim=32,
        dropout=0.2
    )
    
    factory_outputs = factory_model(data)
    print(f"Factory model outputs: {list(factory_outputs.keys())}")
    
    print("All tests passed!")
    return True


if __name__ == "__main__":
    # Test the implementation
    success = test_enhanced_gnn()
    print(f"Enhanced GNN testing {'succeeded' if success else 'failed'}")