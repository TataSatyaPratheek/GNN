#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Position-Enhanced Graph Attention Network for process mining
Adds spatial awareness to node representations
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing, global_mean_pool
from torch_geometric.utils import add_self_loops, softmax
import torch_geometric.utils as utils
from torch_scatter import scatter_add
import math
import numpy as np

from ..base import TorchProcessModel, get_graph_labels
import logging

logger = logging.getLogger(__name__)


class PositionalEncoding(nn.Module):
    """
    Positional encoding module for incorporating spatial information
    """
    def __init__(self, input_dim, pos_dim=16, max_len=1000):
        super().__init__()
        self.pos_dim = pos_dim
        self.input_dim = input_dim
        
        # Position encoding projection
        self.pos_encoder = nn.Linear(2, pos_dim)
        
        # Parameter to control how position information is combined with input
        self.alpha = nn.Parameter(torch.tensor(0.5))
        
        # Create learnable parameters for sequence positions
        self.register_buffer('position_table', self._get_position_table(max_len, pos_dim // 2))
        
    def _get_position_table(self, max_len, feature_dim):
        """Generate sinusoidal position encodings like in Transformer"""
        position = torch.arange(max_len).unsqueeze(1)
        
        # Create position table with correct total dimensions
        pos_table = torch.zeros(max_len, feature_dim * 2)
        
        # Calculate divisor terms for each position dimension individually
        # to avoid broadcasting issues
        for i in range(feature_dim):
            # Calculate angle rates based on position
            angle_rate = 1.0 / (10000 ** (2 * i / float(feature_dim * 2)))
            
            # Apply sine to even indices and cosine to odd indices
            pos_table[:, 2*i] = torch.sin(position[:, 0] * angle_rate)
            pos_table[:, 2*i+1] = torch.cos(position[:, 0] * angle_rate)
        
        return pos_table
    
    def forward(self, x, pos=None, seq_idx=None):
        """
        Apply positional encoding to input features
        Args:
            x: Input features [num_nodes, input_dim]
            pos: Optional explicit positions [num_nodes, 2]
            seq_idx: Optional sequence indices for nodes [num_nodes]
        """
        batch_size = x.size(0)
        
        if pos is not None:
            # If explicit positions are provided, encode them
            pos_embedding = self.pos_encoder(pos)
            
            # Combine with input features
            x_with_pos = torch.cat([x, pos_embedding], dim=-1)
            
        elif seq_idx is not None:
            # If sequence indices are provided, use position table
            seq_idx = seq_idx.clamp(0, self.position_table.size(0) - 1)
            pos_embedding = self.position_table[seq_idx]
            
            # Project to match input dimension if needed
            if pos_embedding.size(1) != self.pos_dim:
                pos_embedding = self.pos_encoder(pos_embedding)
                
            # Combine with input features
            x_with_pos = torch.cat([x, pos_embedding], dim=-1)
            
        else:
            # Default: create implicit positions based on node order
            batch_size = x.size(0)
            default_pos = torch.arange(batch_size, device=x.device).float() / batch_size
            default_pos = default_pos.view(-1, 1).repeat(1, 2)
            
            # Encode default positions
            pos_embedding = self.pos_encoder(default_pos)
            
            # Combine with input features
            x_with_pos = torch.cat([x, pos_embedding], dim=-1)
        
        return x_with_pos


class PositionalGATConv(MessagePassing):
    """
    Position-enhanced Graph Attention layer
    Extends GAT with positional information in the attention mechanism
    """
    def __init__(self, in_channels, out_channels, heads=1, concat=True, 
                 pos_dim=16, dropout=0.0, negative_slope=0.2, bias=True, **kwargs):
        super().__init__(aggr='add', node_dim=0, **kwargs)
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.heads = heads
        self.concat = concat
        self.dropout = dropout
        self.negative_slope = negative_slope
        self.pos_dim = pos_dim
        
        # Linear transformation for node features
        self.lin = nn.Linear(in_channels, heads * out_channels, bias=False)
        
        # Position-aware attention mechanism
        self.att = nn.Parameter(torch.Tensor(1, heads, 2 * out_channels + pos_dim))
        
        # Optional bias
        if bias and concat:
            self.bias = nn.Parameter(torch.Tensor(heads * out_channels))
        elif bias and not concat:
            self.bias = nn.Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)
            
        # Position encoder for edge positions
        self.pos_encoder = nn.Linear(2, pos_dim, bias=False)
        
        # Initialize parameters
        self.reset_parameters()
        
    def reset_parameters(self):
        """Initialize parameters with Xavier/Glorot initialization"""
        gain = nn.init.calculate_gain('relu')
        nn.init.xavier_normal_(self.lin.weight, gain=gain)
        nn.init.xavier_normal_(self.att, gain=gain)
        nn.init.xavier_normal_(self.pos_encoder.weight, gain=gain)
        if self.bias is not None:
            nn.init.zeros_(self.bias)
    
    def forward(self, x, edge_index, pos=None, return_attention_weights=False):
        """
        Forward pass for position-enhanced GAT layer
        Args:
            x: Node features [num_nodes, in_channels]
            edge_index: Graph edge indices [2, num_edges]
            pos: Optional node position features [num_nodes, 2]
            return_attention_weights: Whether to return attention weights
        """
        if pos is None:
            # If positions not provided, create default ones based on node indices
            num_nodes = x.size(0)
            pos = torch.arange(num_nodes, device=x.device).float() / max(num_nodes, 1)
            pos = pos.view(-1, 1).repeat(1, 2)  # [num_nodes, 2]
        
        # Add self-loops to edge_index
        edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))
        
        # Linear transformation
        x_transformed = self.lin(x).view(-1, self.heads, self.out_channels)
        
        # Compute attention
        out, attention_weights = self._compute_attention(x_transformed, edge_index, pos)
        
        # Apply bias if available
        if self.bias is not None:
            if self.concat:
                out = out.view(-1, self.heads * self.out_channels)
            else:
                out = out.mean(dim=1)
            out = out + self.bias
        
        # Return with or without attention weights
        if return_attention_weights:
            return out, attention_weights
        else:
            return out
    
    def _compute_attention(self, x, edge_index, pos):
        """Compute attention scores with positional information"""
        # Prepare position features for edges
        src_idx, dst_idx = edge_index
        
        # Get position differences between connected nodes
        pos_src = pos[src_idx]  # [num_edges, 2]
        pos_dst = pos[dst_idx]  # [num_edges, 2]
        pos_diff = pos_dst - pos_src  # [num_edges, 2]
        
        # Encode position differences
        pos_embedding = self.pos_encoder(pos_diff)  # [num_edges, pos_dim]
        
        # Prepare node features for message passing
        x_i = x[src_idx]  # [num_edges, heads, out_channels]
        x_j = x[dst_idx]  # [num_edges, heads, out_channels]
        
        # Concatenate features for attention computation
        x_combined = torch.cat([x_i, x_j], dim=-1)  # [num_edges, heads, 2*out_channels]
        
        # Expand position embeddings for multi-head attention
        pos_embedding = pos_embedding.unsqueeze(1).expand(-1, self.heads, -1)
        
        # Concatenate with position embeddings
        x_with_pos = torch.cat([x_combined, pos_embedding], dim=-1)  # [num_edges, heads, 2*out_channels+pos_dim]
        
        # Compute attention scores
        alpha = (x_with_pos * self.att).sum(dim=-1)  # [num_edges, heads]
        alpha = F.leaky_relu(alpha, self.negative_slope)
        
        # Normalize attention scores (softmax over destination nodes)
        alpha = softmax(alpha, dst_idx, num_nodes=x.size(0))
        
        # Apply dropout to attention weights
        alpha = F.dropout(alpha, p=self.dropout, training=self.training)
        
        # Weight source features by attention scores
        out = x_j * alpha.view(-1, self.heads, 1)
        
        # Aggregate messages at destination nodes
        out = scatter_add(out, dst_idx, dim=0, dim_size=x.size(0))
        
        # Return output and attention weights
        return out, alpha
    
    def message(self, x_j, alpha):
        """Message function for propagating features"""
        # Weight features by attention and reshape
        return alpha.unsqueeze(-1) * x_j


class PositionalGATModel(TorchProcessModel):
    """
    Position-enhanced Graph Attention Network model for process mining
    """
    def __init__(self, input_dim, hidden_dim, output_dim, pos_dim=16, 
                 num_layers=2, heads=4, dropout=0.5):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_layers = num_layers
        self.heads = heads
        self.dropout = dropout
        self.pos_dim = pos_dim
        
        # Position encoding for input features
        self.pos_encoder = PositionalEncoding(input_dim, pos_dim=pos_dim)
        
        # Enhanced input dimension due to positional encoding
        input_dim_with_pos = input_dim + pos_dim
        
        # GNN layers
        self.convs = nn.ModuleList()
        self.batch_norms = nn.ModuleList()
        
        # First layer with positional input
        self.convs.append(PositionalGATConv(
            input_dim_with_pos, hidden_dim, 
            heads=heads, concat=True, pos_dim=pos_dim, dropout=dropout
        ))
        self.batch_norms.append(nn.BatchNorm1d(hidden_dim * heads))
        
        # Intermediate layers
        for _ in range(num_layers - 1):
            self.convs.append(PositionalGATConv(
                hidden_dim * heads, hidden_dim, 
                heads=heads, concat=True, pos_dim=pos_dim, dropout=dropout
            ))
            self.batch_norms.append(nn.BatchNorm1d(hidden_dim * heads))
        
        # Output layer
        self.fc = nn.Linear(hidden_dim * heads, output_dim)
    
    def forward(self, data):
        """
        Forward pass
        Args:
            data: PyG data object with x, edge_index, and batch attributes
        """
        x, edge_index, batch = data.x, data.edge_index, data.batch
        
        # Create positional features
        num_nodes = x.size(0)
        pos = self._get_normalized_positions(batch, num_nodes)
        
        # Apply position encoding to input
        x = self.pos_encoder(x, pos=pos)
        
        # Process through GAT layers
        for i, conv in enumerate(self.convs):
            # Apply positional GAT
            x = conv(x, edge_index, pos=pos)
            
            # Apply batch normalization
            x = self.batch_norms[i](x)
            
            # Apply activation and dropout
            x = F.elu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
            
            # Update positions based on node features if needed
            # This allows position refinement during message passing
            if i < len(self.convs) - 1:
                pos = self._refine_positions(x, pos, edge_index)
        
        # Global pooling to get graph-level representations
        x = global_mean_pool(x, batch)
        
        # Final prediction
        x = self.fc(x)
        
        return x
    
    def _get_normalized_positions(self, batch, num_nodes):
        """Generate normalized positions based on batch assignments"""
        # Get unique batches and counts
        unique_batches, counts = torch.unique(batch, return_counts=True)
        
        # Create positions tensor
        pos = torch.zeros((num_nodes, 2), device=batch.device)
        
        # Calculate position for each node
        start_idx = 0
        for b, count in zip(unique_batches, counts):
            # Create a grid-like positioning for nodes in each batch
            grid_size = int(math.ceil(math.sqrt(count)))
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
    
    def _refine_positions(self, x, pos, edge_index):
        """Refine node positions based on current embeddings"""
        # Optional: This can be used to adaptively update positions
        # For now, just return the original positions
        return pos
    
    def _prepare_batch(self, batch_data, device):
        """Prepare batch for model"""
        return batch_data.to(device)
    
    def _compute_loss(self, batch_data, criterion):
        """Compute loss for batch"""
        # Get model outputs
        logits = self(batch_data)
        
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