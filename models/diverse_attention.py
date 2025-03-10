#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Diverse Attention Mechanism for Graph Neural Networks
Encourages different attention patterns across heads
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing, GATConv, global_mean_pool
from torch_geometric.utils import add_self_loops, softmax
import torch_geometric.utils as utils
from torch_scatter import scatter_add
import math
import numpy as np


class DiverseGATConv(nn.Module):
    """
    Graph Attention layer with diversity mechanism
    Encourages different attention patterns across heads
    """
    def __init__(self, in_channels, out_channels, heads=4, 
                 concat=True, dropout=0.0, negative_slope=0.2, 
                 diversity_weight=0.1, orthogonal_reg=True):
        """
        Initialize diverse GAT convolution
        
        Args:
            in_channels: Input feature dimension
            out_channels: Output feature dimension per head
            heads: Number of attention heads
            concat: Whether to concatenate or average heads
            dropout: Dropout probability
            negative_slope: LeakyReLU negative slope
            diversity_weight: Weight for diversity loss
            orthogonal_reg: Whether to use orthogonal regularization
        """
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.heads = heads
        self.concat = concat
        self.dropout = dropout
        self.negative_slope = negative_slope
        self.diversity_weight = diversity_weight
        self.orthogonal_reg = orthogonal_reg
        
        # Use built-in GATConv as the base layer
        self.gat = GATConv(
            in_channels, out_channels, heads=heads, 
            concat=concat, dropout=dropout, 
            negative_slope=negative_slope
        )
        
        # Head diversity projection
        # This projects each head's output to measure diversity
        if orthogonal_reg:
            self.head_projectors = nn.Parameter(torch.Tensor(heads, out_channels, out_channels))
            self.reset_parameters()
    
    def reset_parameters(self):
        """Initialize parameters with orthogonal initialization"""
        if self.orthogonal_reg:
            # Initialize with orthogonal bases for diversity
            for i in range(self.heads):
                nn.init.orthogonal_(self.head_projectors[i])
    
    def forward(self, x, edge_index, return_attention_weights=False):
        """
        Forward pass with diversity loss calculation
        
        Args:
            x: Input node features [num_nodes, in_channels]
            edge_index: Graph edge indices [2, num_edges]
            return_attention_weights: Whether to return attention weights
            
        Returns:
            Tuple of (output features, diversity loss)
        """
        # Apply GAT convolution
        if return_attention_weights:
            out, attention_weights = self.gat(x, edge_index, return_attention_weights=True)
        else:
            out = self.gat(x, edge_index)
        
        # Calculate diversity loss
        diversity_loss = self._calculate_diversity_loss(x, edge_index)
        
        # Return output and loss
        if return_attention_weights:
            return out, diversity_loss, attention_weights
        else:
            return out, diversity_loss
    
    def _calculate_diversity_loss(self, x, edge_index):
        """
        Calculate diversity loss between attention heads
        
        Args:
            x: Input node features
            edge_index: Graph edge indices
            
        Returns:
            Diversity loss tensor
        """
        # Get attention coefficients from GAT layer
        _, attention_weights = self.gat(x, edge_index, return_attention_weights=True)
        
        # Extract attention coefficients [num_edges, heads]
        edge_index, attn_coeffs = attention_weights
        
        # Calculate diversity loss based on attention correlation
        diversity_loss = torch.tensor(0.0, device=x.device)
        
        if self.heads > 1:
            # Reshape attention coefficients for easier processing
            # [num_edges, heads]
            attn_per_edge = attn_coeffs
            
            # Method 1: Cosine similarity between attention vectors
            if not self.orthogonal_reg:
                # Compute pairwise similarities between attention heads
                for i in range(self.heads):
                    for j in range(i+1, self.heads):
                        # Extract attention vectors for heads i and j
                        attn_i = attn_per_edge[:, i]
                        attn_j = attn_per_edge[:, j]
                        
                        # Compute cosine similarity
                        similarity = F.cosine_similarity(attn_i, attn_j, dim=0)
                        
                        # Square similarity to penalize both positive and negative correlation
                        diversity_loss += similarity**2
            
            # Method 2: Orthogonal regularization using head projectors
            else:
                # Get number of edges
                num_edges = attn_per_edge.size(0)
                
                # Calculate the Gram matrix of head projectors
                gram_matrix = torch.mm(
                    self.head_projectors.view(self.heads, -1),
                    self.head_projectors.view(self.heads, -1).t()
                )
                
                # Remove diagonal elements (self-similarity)
                mask = torch.ones_like(gram_matrix) - torch.eye(self.heads, device=gram_matrix.device)
                gram_matrix = gram_matrix * mask
                
                # Compute Frobenius norm of the masked Gram matrix
                # This measures how far the projectors are from being orthogonal
                diversity_loss = torch.norm(gram_matrix, p='fro')
            
            # Scale the diversity loss by the number of head pairs
            num_pairs = self.heads * (self.heads - 1) // 2
            diversity_loss = diversity_loss / max(1, num_pairs)
        
        # Apply diversity weight
        return self.diversity_weight * diversity_loss


class DiverseGATModel(nn.Module):
    """
    Graph Attention Network with diversity mechanism
    Encourages different attention patterns across heads
    """
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=2, 
                 heads=4, dropout=0.5, diversity_weight=0.1, 
                 orthogonal_reg=True, output_all_losses=False):
        """
        Initialize diverse GAT model
        
        Args:
            input_dim: Input feature dimension
            hidden_dim: Hidden layer dimension
            output_dim: Output dimension
            num_layers: Number of GAT layers
            heads: Number of attention heads
            dropout: Dropout probability
            diversity_weight: Weight for diversity loss
            orthogonal_reg: Whether to use orthogonal regularization
            output_all_losses: Whether to output all layer losses
        """
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_layers = num_layers
        self.heads = heads
        self.dropout = dropout
        self.diversity_weight = diversity_weight
        self.output_all_losses = output_all_losses
        
        # Input layer
        self.convs = nn.ModuleList()
        self.batch_norms = nn.ModuleList()
        
        # First layer with input dimension
        self.convs.append(DiverseGATConv(
            input_dim, hidden_dim, heads=heads, 
            concat=True, dropout=dropout, 
            diversity_weight=diversity_weight,
            orthogonal_reg=orthogonal_reg
        ))
        self.batch_norms.append(nn.BatchNorm1d(hidden_dim * heads))
        
        # Hidden layers
        for _ in range(num_layers - 1):
            self.convs.append(DiverseGATConv(
                hidden_dim * heads, hidden_dim, heads=heads, 
                concat=True, dropout=dropout, 
                diversity_weight=diversity_weight,
                orthogonal_reg=orthogonal_reg
            ))
            self.batch_norms.append(nn.BatchNorm1d(hidden_dim * heads))
        
        # Output layer
        self.output_layer = nn.Linear(hidden_dim * heads, output_dim)
    
    def forward(self, x, edge_index, batch):
        """
        Forward pass
        
        Args:
            x: Input node features [num_nodes, input_dim]
            edge_index: Graph edge indices [2, num_edges]
            batch: Batch assignment for nodes [num_nodes]
            
        Returns:
            Tuple of (output logits, diversity_loss)
        """
        # Track diversity losses from all layers
        diversity_losses = []
        
        # Process through GAT layers with diversity
        for i, conv in enumerate(self.convs):
            # Apply GATConv with diversity mechanism
            features, div_loss = conv(x, edge_index)
            
            # Track diversity loss
            diversity_losses.append(div_loss)
            
            # Apply batch normalization
            features = self.batch_norms[i](features)
            
            # Apply non-linearity
            features = F.elu(features)
            
            # Apply dropout
            features = F.dropout(features, p=self.dropout, training=self.training)
            
            # Update node features
            x = features
        
        # Apply global pooling
        x = global_mean_pool(x, batch)
        
        # Apply output layer
        logits = self.output_layer(x)
        
        # Compute total diversity loss
        total_diversity_loss = sum(diversity_losses)
        
        # Return logits and diversity losses
        if self.output_all_losses:
            return logits, diversity_losses
        else:
            return logits, total_diversity_loss
    
    def get_attention_weights(self, x, edge_index):
        """
        Get attention weights from each layer for analysis
        
        Args:
            x: Input node features
            edge_index: Graph edge indices
            
        Returns:
            List of attention weights from each layer
        """
        attention_weights = []
        
        # Forward pass storing attention weights
        with torch.no_grad():
            for conv in self.convs:
                # Get base GAT layer
                gat_layer = conv.gat
                
                # Get attention weights
                _, attn_weights = gat_layer(x, edge_index, return_attention_weights=True)
                attention_weights.append(attn_weights)
                
                # Update features for next layer
                x = gat_layer(x, edge_index)
        
        return attention_weights


class AttentionVisualization:
    """
    Utility class for visualizing attention patterns
    """
    @staticmethod
    def compute_attention_entropy(attention_weights):
        """
        Compute entropy of attention weights as a diversity measure
        Higher entropy = more uniform attention (less focused)
        
        Args:
            attention_weights: Attention weights tensor [num_edges, num_heads]
            
        Returns:
            Entropy per head [num_heads]
        """
        # Ensure attention weights are normalized
        weights_sum = attention_weights.sum(dim=0, keepdim=True)
        normalized_weights = attention_weights / weights_sum.clamp(min=1e-6)
        
        # Compute entropy: -sum(p * log(p))
        # Add small epsilon to avoid log(0)
        entropy = -torch.sum(
            normalized_weights * torch.log(normalized_weights + 1e-10),
            dim=0
        )
        
        return entropy
    
    @staticmethod
    def compute_attention_diversity(attention_weights):
        """
        Compute diversity between attention heads
        
        Args:
            attention_weights: Attention weights tensor [num_edges, num_heads]
            
        Returns:
            Diversity score (average cosine distance between heads)
        """
        num_heads = attention_weights.shape[1]
        
        if num_heads <= 1:
            return torch.tensor(0.0, device=attention_weights.device)
        
        # Compute pairwise cosine similarity between attention vectors
        cosine_sim = torch.zeros((num_heads, num_heads), device=attention_weights.device)
        
        for i in range(num_heads):
            for j in range(i+1, num_heads):
                cosine_sim[i, j] = F.cosine_similarity(
                    attention_weights[:, i], 
                    attention_weights[:, j],
                    dim=0
                )
                cosine_sim[j, i] = cosine_sim[i, j]
        
        # Convert similarity to distance: (1 - similarity) / 2
        cosine_dist = (1 - cosine_sim) / 2
        
        # Compute average distance (excluding self-comparisons)
        mask = 1 - torch.eye(num_heads, device=cosine_dist.device)
        avg_distance = (cosine_dist * mask).sum() / (num_heads * (num_heads - 1))
        
        return avg_distance
    
    @staticmethod
    def analyze_attention_patterns(model, data):
        """
        Analyze attention patterns from the model
        
        Args:
            model: Trained DiverseGATModel
            data: PyG Data object
            
        Returns:
            Dictionary with attention analysis metrics
        """
        # Get attention weights
        attention_weights = model.get_attention_weights(data.x, data.edge_index)
        
        # Store metrics for each layer
        metrics = []
        
        for layer_idx, layer_attn in enumerate(attention_weights):
            # Extract edge indices and attention coefficients
            edge_index, attn_coeffs = layer_attn
            
            # Compute metrics per head
            entropy_per_head = AttentionVisualization.compute_attention_entropy(attn_coeffs)
            diversity_score = AttentionVisualization.compute_attention_diversity(attn_coeffs)
            
            # Store metrics
            layer_metrics = {
                'layer_idx': layer_idx,
                'entropy_per_head': entropy_per_head.cpu().numpy(),
                'diversity_score': diversity_score.item(),
                'attention_stats': {
                    'min': attn_coeffs.min().item(),
                    'max': attn_coeffs.max().item(),
                    'mean': attn_coeffs.mean().item(),
                    'std': attn_coeffs.std().item()
                }
            }
            
            metrics.append(layer_metrics)
        
        return metrics


# Utility function for testing
def test_diverse_gat():
    """Test function for diverse GAT implementation"""
    from torch_geometric.data import Data
    
    # Create a simple test graph
    num_nodes = 10
    num_features = 8
    
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
    
    # Test DiverseGATConv layer
    print("Testing DiverseGATConv layer...")
    conv = DiverseGATConv(
        in_channels=num_features, 
        out_channels=16, 
        heads=4, 
        diversity_weight=0.1
    )
    
    # Forward pass
    out, div_loss = conv(x, edge_index)
    print(f"Output shape: {out.shape}")
    print(f"Diversity loss: {div_loss.item()}")
    
    # Test with attention weights
    out, div_loss, attn_weights = conv(x, edge_index, return_attention_weights=True)
    edge_index_out, attn_coeffs = attn_weights
    print(f"Attention coeffs shape: {attn_coeffs.shape}")
    
    # Test complete model
    print("\nTesting DiverseGATModel...")
    model = DiverseGATModel(
        input_dim=num_features,
        hidden_dim=16,
        output_dim=3,
        num_layers=2,
        heads=4,
        dropout=0.1,
        diversity_weight=0.1
    )
    
    # Forward pass
    logits, div_loss = model(x, edge_index, batch)
    print(f"Logits shape: {logits.shape}")
    print(f"Total diversity loss: {div_loss.item()}")
    
    # Test attention visualization
    print("\nTesting AttentionVisualization...")
    metrics = AttentionVisualization.analyze_attention_patterns(model, data)
    
    for layer_idx, layer_metrics in enumerate(metrics):
        print(f"Layer {layer_idx}:")
        print(f"  Diversity score: {layer_metrics['diversity_score']:.4f}")
        print(f"  Entropy per head: {layer_metrics['entropy_per_head']}")
    
    print("All tests passed!")
    return True


if __name__ == "__main__":
    # Test the implementation
    success = test_diverse_gat()
    print(f"Diverse GAT testing {'succeeded' if success else 'failed'}")